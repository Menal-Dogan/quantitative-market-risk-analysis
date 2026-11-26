import numpy as np
import pandas as pd
import pytest
from scipy import stats
from risk_measures import (
    compute_losses,
    historical_var,
    historical_es,
    parametric_gaussian_var,
    parametric_gaussian_es,
    compare_risk_measures,
)


def test_compute_losses_sign_convention():
    returns = pd.Series([0.02, -0.03, 0.0], name="Portfolio_Return")
    losses = compute_losses(returns)

    assert losses.name == "Portfolio_Loss"
    assert np.allclose(losses.to_numpy(), np.array([-0.02, 0.03, 0.0]))


def test_historical_var_and_es_deterministic():
    # 20 ordered integer losses from 1 to 20
    losses = np.arange(1, 21, dtype=float)
    # 95% quantile with default linear interpolation: 1 + 0.95 * 19 = 19.05
    var_95 = historical_var(losses, confidence_level=0.95)
    assert np.isclose(var_95, 19.05)

    # Only observation >= 19.05 is 20.0
    es_95 = historical_es(losses, confidence_level=0.95)
    assert np.isclose(es_95, 20.0)


def test_gaussian_formulas_against_simulated_normal_distribution():
    rng = np.random.default_rng(42)
    true_mu_loss = 0.0005
    true_sigma_loss = 0.015
    simulated_losses = rng.normal(loc=true_mu_loss, scale=true_sigma_loss, size=250_000)

    cl = 0.99
    alpha = 1.0 - cl
    z_99 = stats.norm.ppf(cl)
    theoretical_var = true_mu_loss + z_99 * true_sigma_loss
    theoretical_es = true_mu_loss + true_sigma_loss * (stats.norm.pdf(z_99) / alpha)

    assert np.isclose(parametric_gaussian_var(simulated_losses, cl), theoretical_var, atol=1e-4)
    assert np.isclose(parametric_gaussian_es(simulated_losses, cl), theoretical_es, atol=1e-4)
    assert np.isclose(historical_var(simulated_losses, cl), theoretical_var, atol=5e-4)
    assert np.isclose(historical_es(simulated_losses, cl), theoretical_es, atol=5e-4)


def test_risk_measure_consistency_and_monotonicity():
    rng = np.random.default_rng(123)
    losses = rng.standard_t(df=4, size=5_000) * 0.01

    comparison = compare_risk_measures(losses, confidence_levels=(0.95, 0.99))

    # ES must always be >= VaR for both methods
    assert comparison.loc[0.95, "Historical_ES"] >= comparison.loc[0.95, "Historical_VaR"]
    assert comparison.loc[0.99, "Historical_ES"] >= comparison.loc[0.99, "Historical_VaR"]
    assert comparison.loc[0.95, "Gaussian_ES"] >= comparison.loc[0.95, "Gaussian_VaR"]
    assert comparison.loc[0.99, "Gaussian_ES"] >= comparison.loc[0.99, "Gaussian_VaR"]

    # Higher confidence level must yield higher or equal VaR and ES
    assert comparison.loc[0.99, "Historical_VaR"] >= comparison.loc[0.95, "Historical_VaR"]
    assert comparison.loc[0.99, "Historical_ES"] >= comparison.loc[0.95, "Historical_ES"]
    assert comparison.loc[0.99, "Gaussian_VaR"] >= comparison.loc[0.95, "Gaussian_VaR"]
    assert comparison.loc[0.99, "Gaussian_ES"] >= comparison.loc[0.95, "Gaussian_ES"]


def test_var_violates_subadditivity_while_es_respects_it():
    """
    Demonstrates on a classic two-defaultable-bond distribution (4% default prob each, independent)
    that VaR_95(L1 + L2) > VaR_95(L1) + VaR_95(L2), whereas generalized ES satisfies sub-additivity.
    """
    # Exact 625-state product space (25 x 25): 1/25 = 4% default probability per bond
    states_1 = np.array([0.0] * 24 + [100.0])
    states_2 = np.array([0.0] * 24 + [100.0])
    l1, l2 = np.meshgrid(states_1, states_2)
    l1_flat = l1.ravel()
    l2_flat = l2.ravel()
    l_sum = l1_flat + l2_flat

    # Using lower empirical quantile (exact inverse CDF at 95% for discrete distribution)
    var_l1 = float(np.quantile(l1_flat, 0.95, method="lower"))
    var_l2 = float(np.quantile(l2_flat, 0.95, method="lower"))
    var_sum = float(np.quantile(l_sum, 0.95, method="lower"))

    # Individual 95% VaRs are 0 (since P(L_i = 0) = 96% >= 95%), but combined VaR is 100!
    assert var_l1 == 0.0
    assert var_l2 == 0.0
    assert var_sum == 100.0
    assert var_sum > var_l1 + var_l2

    # Generalized ES (integral of upper 5% quantiles) satisfies sub-additivity: ES(L1+L2) <= ES(L1) + ES(L2)
    u_grid = np.linspace(0.95, 1.0, 10_000, endpoint=False)
    es_l1 = float(np.mean(np.quantile(l1_flat, u_grid, method="lower")))
    es_l2 = float(np.mean(np.quantile(l2_flat, u_grid, method="lower")))
    es_sum = float(np.mean(np.quantile(l_sum, u_grid, method="lower")))

    assert es_sum <= es_l1 + es_l2


def test_risk_measures_edge_cases():
    # Insufficient observations
    with pytest.raises(ValueError, match="At least 2 observations"):
        historical_var(np.array([0.01]), 0.95)

    # Missing values
    with pytest.raises(ValueError, match="missing values"):
        parametric_gaussian_var(np.array([0.01, np.nan, -0.02]), 0.95)

    # Invalid confidence levels
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        historical_es(np.array([0.01, -0.02, 0.03]), 1.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        parametric_gaussian_es(np.array([0.01, -0.02, 0.03]), 0.0)