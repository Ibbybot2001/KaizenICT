import pandas as pd
import pytz
from .config import DATA_TIMEZONE, TRADING_TIMEZONE

def load_data(filepath):
    """
    Load MNQ 1-minute data, handle timezone conversion to America/New_York.
    """
    print(f"Loading data from {filepath}...")
    
    # Read CSV
    # The file format is: time,open,high,low,close,volume
    # 2023-11-26 17:00:00-06:00
    df = pd.read_csv(
        filepath, 
        parse_dates=['time'], 
        index_col='time'
    )
    
    # Ensure index is timezone aware. 
    # The CSV strings already have offset, so pandas should pick it up.
    # If not, we force conversion.
    
    # Force index to datetime (in case parse_dates failed or resulted in object)
    df.index = pd.to_datetime(df.index, utc=True)

    # Convert to Target TZ
    try:
        df.index = df.index.tz_convert(TRADING_TIMEZONE)
    except Exception as e:
        print(f"Timezone conversion error: {e}")
        # If naive, localize first
        df.index = df.index.tz_localize('UTC').tz_convert(TRADING_TIMEZONE)
        
    df.sort_index(inplace=True)
    
    # Validation
    _validate_data(df)
    
    print(f"Data loaded: {len(df)} rows. Range: {df.index[0]} to {df.index[-1]}")
    return df

def _validate_data(df):
    """
    Check for critical data integrity issues.
    """
    # Check for gaps > 5 minutes during trading hours (approximate)
    time_diff = df.index.to_series().diff()
    gaps = time_diff[time_diff > pd.Timedelta(minutes=5)]
    
    if not gaps.empty:
        print(f"WARNING: Found {len(gaps)} gaps larger than 5 minutes.")
        print(f"Largest gap: {gaps.max()}")
    
    # Check for duplicates
    if df.index.has_duplicates:
        print(f"WARNING: Dataset contains {df.index.duplicated().sum()} duplicate timestamps. Keeping first.")
        df = df[~df.index.duplicated(keep='first')]

if __name__ == "__main__":
    # Test run
    path = r"c:\Users\CEO\ICT reinforcement\kaizen_1m_data_ibkr_2yr.csv"
    df = load_data(path)
    print(df.head())
    print(df.tail())
