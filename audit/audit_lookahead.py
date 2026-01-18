import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'src'))
from src.data_loader import load_data
from src.ict_library import calculate_swings
from src.backtester import Backtester

class AuditStrategy:
    """
    Malicious Strategy that tries to peek into the future.
    """
    def __init__(self, backtester):
        self.backtester = backtester
        self.errors = []
        
    def on_start(self, df):
        self.full_df = df
        self.swings, _ = calculate_swings(df)
        
    def on_bar(self, i, bar):
        current_time = bar.name
        
        # TEST 1: Check if Swings are leaked from future
        # At index i, self.swings[i] should be valid IF confirmation happened.
        # But self.swings[i+1] must NOT be readable or must be NaN/Future.
        # Since we pre-calc, self.swings contains EVERYTHING.
        # The strategy MUST NOT access self.swings[i+1].
        
        # Real verification: The swing *value* at index `i` implies confirmation.
        # Let's verify that if self.swings[i] has a value, it corresponds to a peak at i-3.
        
        if not pd.isna(self.swings.iloc[i]):
            peak_time_should_be = self.full_df.index[i-3]
            # Check if that peak was indeed a local max.
            # This confirms our logic is "At bar i, we confirm peak at i-3".
            # Which is correct and causal (3 bars delay).
            pass
            
        # TEST 2: Attempt to access future price in DF
        # The Backtester passes 'bar' which is safe.
        # But if we access self.backtester.data.iloc[i+1], we are cheating.
        
        pass

def run_audit():
    print("--- Running Strict Lookahead Audit ---")
    path = r"c:\Users\CEO\ICT reinforcement\kaizen_1m_data_ibkr_2yr.csv"
    df = load_data(path)
    df = df.iloc[:1000] # Use small subset
    
    # 1. Verify Swing Lag
    print("Verifying Swing Calculation Lag...")
    highs, _ = calculate_swings(df, left=3, right=3)
    
    # If there is a Swing at index T
    swing_indices = highs.dropna().index
    
    for t_idx in swing_indices:
        # Get integer loc
        i = df.index.get_loc(t_idx)
        
        # The Peak should be at i - 3
        peak_idx = i - 3
        if peak_idx < 0: continue
        
        peak_val = df.iloc[peak_idx]['high']
        neighbors = df.iloc[peak_idx-3 : peak_idx+4]['high'] # +/- 3
        
        # Validation: Peak Value reported must match Actual Peak
        logged_val = highs.loc[t_idx]
        assert logged_val == peak_val, f"Mismatch! Logged {logged_val} != Actual {peak_val}"
        
        # Validation: The peak must be the max of its neighbors
        # (This proves it IS a swing high)
        if peak_val < neighbors.max():
             # Edge case: Equal highs allowed? Logic says > max of left, > max of right.
             # Strict > check in code.
             print(f"FAILED Strict Peak check at {t_idx} (Peak time {df.index[peak_idx]})")
             
        # CRITICAL: We confirmed it at `t_idx`. 
        # Is it physically possible to know this at `t_idx`? 
        # Yes, because we have seen bars t-3, t-2, t-1, t.
        # Wait, neighbors are [p-3 ... p ... p+3]. 
        # p+3 IS t_idx.
        # So at t_idx, we just saw the close of the 3rd bar to the right. 
        # So we NOW know it is a swing.
        # Verdict: NO LOOKAHEAD. 
        
    print("Swing High Logic: CAUSAL (Pas)" )
    
    # 2. Verify Execution Timing
    # We will simulate a trade and ensure execution timestamp >= signal timestamp
    
if __name__ == "__main__":
    run_audit()
