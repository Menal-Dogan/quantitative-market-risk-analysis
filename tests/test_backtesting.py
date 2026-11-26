import numpy as np
import pandas as pd
import pytest
from backtesting import compute_rolling_volatility, rolling_var_backtest, kupiec_pof_test


def test_rolling_volatility_timing_and_no_lookahead():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, 0.01, -0.01, 0.005, -0.015, 0.02], index=dates)

    vol = compute_rolling_volatility(returns, window=3, annualize=False)

    # First forecast corresponds to index 3 (4th business day), computed strictly from indices [0, 1, 2]
    assert len(vol) == 7
    assert vol.index[0] == dates[3]
    assert np.isclose(vol.iloc[0], returns.iloc[0:3].std(ddof=1))

    # Perturbing date 3 must NOT affect the forecast for date 3
    returns_perturbed = returns.copy()
    returns_perturbed.iloc[3] = 0.50
    vol_perturbed = compute_rolling_volatility(returns_perturbed, window=3, annualize=False)
    assert np.isclose(vol_perturbed.iloc[0], vol.iloc[0])


def test_rolling_var_backtest_zero_lookahead_bias():
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    rng = np.random.default_rng(7)
    losses = pd.Series(rng.normal(0.0, 0.01, size=30), index=dates)

    window = 10
    bt_original = rolling_var_backtest(losses, window=window, confidence_level=0.95, method="gaussian")

    # Inject a massive shock on evaluation day index 15 (6th out-of-sample day)
    losses_shocked = losses.copy()
    losses_shocked.iloc[15] = 0.25
    bt_shocked = rolling_var_backtest(losses_shocked, window=window, confidence_level=0.95, method="gaussian")

    # Forecasts up to and including index 15 MUST be identical (strictly no look-ahead)
    assert np.allclose(
        bt_original["VaR_Forecast"].iloc[:6].to_numpy(),
        bt_shocked["VaR_Forecast"].iloc[:6].to_numpy(),
    )
    # Forecast at index 16 MUST change because index 15 enters its [6:16] window
    assert not np.isclose(bt_original["VaR_Forecast"].iloc[6], bt_shocked["VaR_Forecast"].iloc[6])
    # Index 15 must be flagged as a VaR violation in the shocked series
    assert bool(bt_shocked["Violation"].iloc[5]) is True


def test_rolling_var_backtest_gaussian_calibration_on_synthetic_data():
    rng = np.random.default_rng(12345)
    dates = pd.date_range("2015-01-01", periods=2_500, freq="B")
    # True i.i.d. Gaussian losses: Gaussian VaR should pass Kupiec POF test
    losses = pd.Series(rng.normal(loc=0.0, scale=0.01, size=2_500), index=dates)

    bt = rolling_var_backtest(losses, window=250, confidence_level=0.95, method="gaussian")
    kupiec = kupiec_pof_test(bt["Violation"], confidence_level=0.95)

    assert len(bt) == 2_250
    assert np.isclose(kupiec["expected_violations"], 2_250 * 0.05)
    assert kupiec["reject_h0"] is False


def test_kupiec_pof_test_exact_and_miscalibrated():
    # Exact 5% violation rate -> LR = 0, p-value = 1.0
    exact_violations = np.array([True] * 50 + [False] * 950)
    res_exact = kupiec_pof_test(exact_violations, confidence_level=0.95)
    assert np.isclose(res_exact["lr_stat"], 0.0)
    assert np.isclose(res_exact["p_value"], 1.0)
    assert res_exact["reject_h0"] is False

    # Severely under-estimated risk: 15% violations instead of 5% -> reject H0
    bad_violations = np.array([True] * 150 + [False] * 850)
    res_bad = kupiec_pof_test(bad_violations, confidence_level=0.95)
    assert res_bad["p_value"] < 0.01
    assert res_bad["reject_h0"] is True

    # Zero violations edge case (0 * log(0) handled cleanly without NaN)
    zero_violations = np.zeros(100, dtype=bool)
    res_zero = kupiec_pof_test(zero_violations, confidence_level=0.95)
    assert np.isfinite(res_zero["lr_stat"])
    assert np.isfinite(res_zero["p_value"])


def test_backtesting_edge_cases():
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    short_losses = pd.Series([0.01, -0.01, 0.02, 0.0, -0.02], index=dates)

    with pytest.raises(ValueError, match="Insufficient observations"):
        rolling_var_backtest(short_losses, window=5, confidence_level=0.95)

    with pytest.raises(ValueError, match="Unsupported VaR method"):
        rolling_var_backtest(short_losses, window=2, confidence_level=0.95, method="cornish_fisher")