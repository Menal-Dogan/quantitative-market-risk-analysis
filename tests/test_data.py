import pandas as pd
import numpy as np
import pytest
from data import handle_missing_values, validate_data

def test_handle_missing_values_ffill_dropna_logic():
    """
    Ensure forward fill is applied and initial NaNs are strictly dropped, 
    preventing any backward information leakage.
    """
    df = pd.DataFrame({
        'SPY': [np.nan, 100.0, np.nan, 105.0],
        'CAC': [np.nan, np.nan, 200.0, 202.0]
    })
    
    cleaned_df = handle_missing_values(df)
    
    # Rows with any remaining NaNs are dropped, ensuring no bfill occurs
    assert cleaned_df.isna().sum().sum() == 0
    assert len(cleaned_df) == 2
    assert cleaned_df['SPY'].iloc[0] == 100.0
    assert cleaned_df['CAC'].iloc[0] == 200.0

def test_validate_data_raises_error_on_negative_prices():
    """
    Validation should fail if any price is zero or negative.
    """
    df = pd.DataFrame({'AAPL': [150.0, -5.0, 155.0]})
    
    with pytest.raises(ValueError, match="Non-positive prices"):
        validate_data(df)

def test_validate_data_raises_error_on_remaining_nans():
    """
    Validation should fail if NaNs are still present.
    """
    df = pd.DataFrame({'AAPL': [150.0, np.nan, 155.0]})
    
    with pytest.raises(ValueError, match="Missing values"):
        validate_data(df)