import numpy as np
import pandas as pd
from scipy import stats


def _validate_inputs(losses: pd.Series | np.ndarray, confidence_level: float) -> np.ndarray:
    """
    Validates the loss array and confidence level, returning a clean 1D NumPy float array.
    """
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"Confidence level must be strictly between 0 and 1, got {confidence_level}.")

    loss_array = np.asarray(losses, dtype=float)

    if loss_array.ndim != 1:
        raise ValueError("Input losses must be a 1-dimensional series or array.")

    if len(loss_array) < 2:
        raise ValueError("At least 2 observations are required to estimate risk measures.")

    if np.isnan(loss_array).any():
        raise ValueError("Input losses contain missing values (NaN).")

    return loss_array


def compute_losses(returns: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Converts portfolio returns into portfolio losses using the positive-loss convention:
    L_t = -R_t.
    """
    if isinstance(returns, pd.Series):
        if returns.isna().any():
            raise ValueError("Input returns contain missing values (NaN).")
        losses = -returns
        losses.name = "Portfolio_Loss"
        return losses

    ret_array = np.asarray(returns, dtype=float)
    if np.isnan(ret_array).any():
        raise ValueError("Input returns contain missing values (NaN).")
    return -ret_array


def historical_var(losses: pd.Series | np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Computes the non-parametric Historical Value-at-Risk (VaR) at the given confidence level.
    Under the positive-loss convention, VaR_{1-alpha} is the (1 - alpha) empirical quantile of losses.
    """
    loss_array = _validate_inputs(losses, confidence_level)
    return float(np.quantile(loss_array, confidence_level))


def historical_es(losses: pd.Series | np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Computes the non-parametric Historical Expected Shortfall (CVaR) at the given confidence level,
    defined as E[L | L >= VaR_{1-alpha}(L)].
    """
    loss_array = _validate_inputs(losses, confidence_level)
    var_threshold = historical_var(loss_array, confidence_level)
    tail_losses = loss_array[loss_array >= var_threshold]
    return float(np.mean(tail_losses))


def parametric_gaussian_var(losses: pd.Series | np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Computes the parametric Gaussian Value-at-Risk (VaR) assuming L ~ N(mu_L, sigma_L^2):
    VaR_{1-alpha} = mu_L + z_{1-alpha} * sigma_L.
    """
    loss_array = _validate_inputs(losses, confidence_level)
    mu_loss = float(np.mean(loss_array))
    sigma_loss = float(np.std(loss_array, ddof=1))
    z_score = float(stats.norm.ppf(confidence_level))
    return mu_loss + z_score * sigma_loss


def parametric_gaussian_es(losses: pd.Series | np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Computes the parametric Gaussian Expected Shortfall (CVaR) assuming L ~ N(mu_L, sigma_L^2):
    ES_{1-alpha} = mu_L + sigma_L * phi(z_{1-alpha}) / (1 - confidence_level).
    """
    loss_array = _validate_inputs(losses, confidence_level)
    alpha = 1.0 - confidence_level
    mu_loss = float(np.mean(loss_array))
    sigma_loss = float(np.std(loss_array, ddof=1))
    z_score = float(stats.norm.ppf(confidence_level))
    return mu_loss + sigma_loss * (float(stats.norm.pdf(z_score)) / alpha)


def compare_risk_measures(
    losses: pd.Series | np.ndarray,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Compares Historical and Gaussian Parametric VaR and Expected Shortfall across confidence levels.
    """
    records = []
    for cl in confidence_levels:
        h_var = historical_var(losses, cl)
        h_es = historical_es(losses, cl)
        g_var = parametric_gaussian_var(losses, cl)
        g_es = parametric_gaussian_es(losses, cl)
        records.append(
            {
                "Confidence_Level": cl,
                "Historical_VaR": h_var,
                "Gaussian_VaR": g_var,
                "Historical_ES": h_es,
                "Gaussian_ES": g_es,
            }
        )
    return pd.DataFrame(records).set_index("Confidence_Level")