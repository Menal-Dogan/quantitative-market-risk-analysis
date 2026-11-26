import pandas as pd
import numpy as np
import pytest
from portfolio import compute_simple_returns, compute_log_returns, aggregate_portfolio_returns

def test_compute_returns():
    """
    Verify both simple and logarithmic returns are computed accurately.
    """
    prices = pd.DataFrame({'A': [100.0, 105.0, 102.9]})
    sim_ret = compute_simple_returns(prices)
    log_ret = compute_log_returns(prices)
    
    assert np.isclose(sim_ret['A'].iloc[0], 0.05)
    assert np.isclose(log_ret['A'].iloc[0], np.log(1.05))

def test_aggregate_portfolio_returns_correct_math():
    """
    Verify the linear combination of simple returns matches the portfolio return.
    """
    sim_ret = pd.DataFrame({
        'Asset_A': [0.05, -0.02],
        'Asset_B': [0.01, 0.03]
    })
    weights = np.array([0.6, 0.4])
    port_ret = aggregate_portfolio_returns(sim_ret, weights)
    
    assert np.isclose(port_ret.iloc[0], (0.05 * 0.6) + (0.01 * 0.4))
    assert np.isclose(port_ret.iloc[1], (-0.02 * 0.6) + (0.03 * 0.4))

def test_aggregate_portfolio_weights_error():
    """
    Ensure an error is raised if weights do not sum to 1.
    """
    sim_ret = pd.DataFrame({'Asset_A': [0.05], 'Asset_B': [0.01]})
    with pytest.raises(ValueError, match="sum to 1"):
        aggregate_portfolio_returns(sim_ret, np.array([0.5, 0.4]))