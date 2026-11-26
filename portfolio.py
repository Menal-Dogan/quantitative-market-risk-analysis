import pandas as pd
import numpy as np

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple arithmetic returns.
    Drops the first row (NaN) automatically.
    """
    return prices.pct_change().dropna()

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily logarithmic returns.
    Useful for time-additivity in specific statistical modeling.
    """
    return np.log(prices / prices.shift(1)).dropna()

def aggregate_portfolio_returns(simple_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Calculate the portfolio's daily simple returns using asset weights.
    Avoids the asset aggregation trap by strictly using simple returns.
    """
    if len(weights) != len(simple_returns.columns):
        raise ValueError("Weight dimension must match the number of assets.")
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Portfolio weights must sum to 1.0.")
    
    port_returns = simple_returns.dot(weights)
    port_returns.name = "Portfolio_Return"
    return port_returns