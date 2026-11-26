import yfinance as yf
import pandas as pd

def fetch_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download unadjusted close prices for a list of tickers.
    auto_adjust=False ensures consistent handling of splits and dividends
    when manually targeting the 'Adj Close' column.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"]
    else:
        prices = data.to_frame(name=tickers[0]) if isinstance(data, pd.Series) else data[["Adj Close"]]
        prices.columns = tickers
        
    return prices

def handle_missing_values(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing data by propagating the last valid observation forward (ffill).
    Drops any remaining NaNs (e.g., at the very beginning of the series) to absolutely 
    prevent look-ahead bias.
    """
    return prices.ffill().dropna()

def validate_data(prices: pd.DataFrame) -> bool:
    """
    Check the integrity of the price dataframe.
    Raises ValueError if missing values or non-positive prices are found.
    """
    if prices.isnull().values.any():
        raise ValueError("Data error: Missing values (NaN) remain in the dataset.")
    
    if (prices <= 0).values.any():
        raise ValueError("Data error: Non-positive prices detected, cannot compute returns.")
    
    return True