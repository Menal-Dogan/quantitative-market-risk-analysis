import numpy as np
import pandas as pd
from scipy import stats
from risk_measures import historical_var, parametric_gaussian_var


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 250,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Computes one-day-ahead rolling volatility without look-ahead bias.
    At date t, volatility is estimated strictly using returns from [t - window, t - 1].
    """
    if window < 2:
        raise ValueError("Rolling window length must be at least 2.")
    if len(returns) <= window:
        raise ValueError("Insufficient observations for the requested rolling window.")
    if returns.isna().any():
        raise ValueError("Input returns contain missing values (NaN).")

    rolling_std = returns.rolling(window=window).std(ddof=1).shift(1).dropna()
    if annualize:
        rolling_std = rolling_std * np.sqrt(trading_days)

    rolling_std.name = "Rolling_Volatility"
    return rolling_std


def rolling_var_backtest(
    losses: pd.Series,
    window: int = 250,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> pd.DataFrame:
    """
    Executes a strictly out-of-sample one-day-ahead VaR backtest without look-ahead bias.
    At each evaluation date t, VaR_{t|t-1} is estimated exclusively on [t - window, t - 1]
    and compared against the realized loss L_t.
    """
    if not isinstance(losses, pd.Series):
        raise TypeError("Input losses must be a pandas Series with a date index.")
    if losses.isna().any():
        raise ValueError("Input losses contain missing values (NaN).")
    if window < 2:
        raise ValueError("Estimation window length must be at least 2.")
    if len(losses) <= window:
        raise ValueError(
            f"Insufficient observations ({len(losses)}) for estimation window ({window})."
        )
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")

    method_key = method.lower()
    if method_key == "historical":
        var_fn = historical_var
    elif method_key == "gaussian":
        var_fn = parametric_gaussian_var
    else:
        raise ValueError(f"Unsupported VaR method '{method}'. Choose 'historical' or 'gaussian'.")

    loss_values = losses.to_numpy(dtype=float)
    n_total = len(loss_values)
    n_oos = n_total - window

    var_forecasts = np.empty(n_oos, dtype=float)
    realized_losses = loss_values[window:]

    # Strict causal slicing: window [i - window : i] ends at i - 1, strictly prior to date i
    for idx, i in enumerate(range(window, n_total)):
        estimation_slice = loss_values[i - window : i]
        var_forecasts[idx] = var_fn(estimation_slice, confidence_level=confidence_level)

    violations = realized_losses > var_forecasts

    results = pd.DataFrame(
        {
            "Realized_Loss": realized_losses,
            "VaR_Forecast": var_forecasts,
            "Violation": violations,
        },
        index=losses.index[window:],
    )
    return results


def kupiec_pof_test(
    violations: pd.Series | np.ndarray,
    confidence_level: float = 0.95,
    significance_level: float = 0.05,
) -> dict[str, float | int | bool]:
    """
    Performs Kupiec's (1995) Proportion of Failures (POF) unconditional coverage test.
    Tests H0: E[I_t] = alpha = 1 - confidence_level under i.i.d. Bernoulli violations.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")

    viol_array = np.asarray(violations, dtype=bool)
    if viol_array.ndim != 1 or len(viol_array) == 0:
        raise ValueError("Violations must be a non-empty 1-dimensional sequence.")

    n_obs = int(len(viol_array))
    n_viol = int(np.sum(viol_array))
    alpha = 1.0 - confidence_level
    expected_viol = n_obs * alpha
    observed_rate = n_viol / n_obs

    # Log-likelihood under H0: p = alpha
    ll_null = (n_obs - n_viol) * np.log(1.0 - alpha) + n_viol * np.log(alpha)

    # Log-likelihood under H1: p = observed_rate (handling boundary cases n_viol = 0 or n_obs)
    if n_viol == 0 or n_viol == n_obs:
        ll_alt = 0.0
    else:
        ll_alt = (n_obs - n_viol) * np.log(1.0 - observed_rate) + n_viol * np.log(observed_rate)

    lr_stat = float(max(0.0, 2.0 * (ll_alt - ll_null)))
    p_value = float(stats.chi2.sf(lr_stat, df=1))

    return {
        "n_observations": n_obs,
        "observed_violations": n_viol,
        "expected_violations": float(expected_viol),
        "observed_rate": float(observed_rate),
        "expected_rate": float(alpha),
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_h0": bool(p_value < significance_level),
    }