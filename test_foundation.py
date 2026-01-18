import sys
import os
import pandas as pd

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from strategies.legacy.data_loader import load_data
from strategies.legacy.ict_library import calculate_swings, detect_fair_value_gaps

def test_system():
    # 1. Load Data
    path = r"c:\Users\CEO\ICT reinforcement\kaizen_1m_data_ibkr_2yr.csv"
    print("--- Testing Data Loader ---")
    df = load_data(path)
    
    # Slice a small chunk for testing (e.g., 1 day)
    # 2023-11-27 is a Monday
    test_day = df.loc['2023-11-27 09:30':'2023-11-27 16:00']
    print(f"Test Subset: {len(test_day)} rows")
    
    # 2. Test Swings
    print("\n--- Testing Swing Points ---")
    sw_highs, sw_lows = calculate_swings(test_day, left=3, right=3)
    print(f"Found {sw_highs.count()} Swing Highs")
    print(f"Found {sw_lows.count()} Swing Lows")
    
    # Print first few
    for t, p in sw_highs.dropna().head(3).items():
        print(f"Swing High confirmed at {t}: {p}")
        
    # 3. Test FVGs
    print("\n--- Testing FVG Detection ---")
    fvgs = detect_fair_value_gaps(test_day)
    print(f"Found {len(fvgs)} FVGs")
    print(fvgs.head())
    
    # Simulating Loop Access
    print("\n--- Testing Access ---")
    for t in fvgs.index[:5]:
        print(f"Accessing {t}...")
        try:
            item = fvgs.loc[t]
            print(f"Type: {type(item)}")
            # Mimic Model logic
            if isinstance(item, pd.DataFrame):
                item = item.iloc[0]
            print(f"FVG Type: {item['type']}")
        except Exception as e:
            print(f"ERROR accessing {t}: {e}")

if __name__ == "__main__":
    test_system()
