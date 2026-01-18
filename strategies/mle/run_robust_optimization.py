"""
Robust Optimization Search (Training Phase)
Search Grid: Trigger x Context x Exit
Data: 8 Training Months (Jan-Dec 2025 subset)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Import Backtester
# Fix path for imports
sys.path.append("C:/Users/CEO/ICT reinforcement")
from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

def load_all_data(base_dir):
    # Load all parquet files to get full set
    files = list(base_dir.glob("USTEC_2025_GOLDEN_PARQUET/USTEC_2025-*_clean_1m.parquet"))
    dfs = []
    print(f"Loading {len(files)} monthly files...")
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    
    full_df = pd.concat(dfs)
    full_df = full_df.sort_index()
    return full_df

def get_liquidity_levels(df_bars):
    # Pre-compute Swing Highs (Fractals)
    highs = df_bars['high'].values
    # 5-bar fractal (2 left, 2 right)
    # Be CAREFUL with lookahead. 
    # To be valid at time T, we must only look at T-1, T-2...
    # A fractal at T implies T is the highest. We only know this at T+2.
    # So if we enter at T, we can only target OLD fractals.
    
    # Lagged Fractal Detection:
    # At index i, is i-2 a high?
    # We need confirmation.
    
    # Vectorized:
    # High[i-2] > High[i-3], High[i-4] AND High[i-2] > High[i-1], High[i]
    # We can know this at time i.
    is_swing = (highs[:-4] < highs[1:-3]) & (highs[:-4] < highs[2:-2]) & \
               (highs[2:-2] > highs[3:-1]) & (highs[2:-2] > highs[4:])
               
    # Map swing checks to time index [4:]
    # True means "At this bar close, we CONFIRMED a swing high happened 2 bars "
    # So the LEVEL is valid target from now on.
    
    swing_levels = np.full(len(highs), np.nan)
    swing_levels[4:][is_swing] = highs[2:-2][is_swing]
    
    return swing_levels

def run_grid_search():
    base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
    tick_path = base_dir / "USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet" # Placeholder for backtester init
    
    print("Loading 1M Data...")
    df_bars = load_all_data(base_dir)
    df_bars['time'] = pd.to_datetime(df_bars.index)
    df_bars['month'] = df_bars['time'].dt.month
    df_bars['hour'] = df_bars['time'].dt.hour
    
    # TRAIN / TEST SPLIT
    TRAIN_MONTHS = [1, 3, 4, 6, 8, 9, 11, 12]
    TEST_MONTHS = [2, 5, 7, 10]
    
    print(f"Filtering Training Set: Months {TRAIN_MONTHS}")
    df_train = df_bars[df_bars['month'].isin(TRAIN_MONTHS)].copy()
    
    # Pre-compute Indicators
    # Volatility
    vol_col = 'tick_volume' if 'tick_volume' in df_train.columns else 'volume'
    df_train['vol_ma'] = df_train[vol_col].rolling(20).mean()
    df_train['range'] = df_train['high'] - df_train['low']
    df_train['range_ma'] = df_train['range'].rolling(20).mean()
    
    # Grid Definitions
    triggers = [
        {'name': 'Vol_1.5x', 'mask': df_train[vol_col] > (df_train['vol_ma'] * 1.5)},
        {'name': 'Range_1.5x', 'mask': df_train['range'] > (df_train['range_ma'] * 1.5)}
    ]
    
    Contexts = [
        {'name': 'Full_US', 'mask': ((df_train['hour'] == 9) & (df_train['time'].dt.minute >= 30)) | ((df_train['hour'] >= 10) & (df_train['hour'] < 16))},
        {'name': 'PM_Session', 'mask': (df_train['hour'] >= 13) & (df_train['hour'] < 16)}
    ]
    
    Risks = [
        {'sl': 15, 'tp': 50, 'tp1': None},
        {'sl': 20, 'tp': 50, 'tp1': None},
        {'sl': 20, 'tp': 100, 'tp1': 30} # Scale out
    ]
    
    # Initialize Backtester (We need a way to mock the tick data for 8 months efficiently)
    # Loading 8 months of ticks is HUGE.
    # PROXY STRATEGY: Use 1M Bar Backtest for Training (Fast), Verify with Ticks.
    # Or load ticks dynamically? 
    # Let's use 1M Bar approximation for the Grid Search to narrow down candidates.
    # Reason: 8 months of ticks is Terabytes/Gigabytes and slow.
    
    results = []
    
    print("Starting Grid Search (Bar-Level Proxy)...")
    
    for trig in triggers:
        for ctx in Contexts:
            # Combined Mask
            mask = trig['mask'] & ctx['mask'] & (df_train['close'] > df_train['open']) # Long
            signals = df_train[mask]
            
            if len(signals) < 50:
                continue
                
            for risk in Risks:
                # Fast Bar Backtest
                # Iterate signals
                pnl = 0
                wins = 0
                
                sl_dist = risk['sl']
                tp_dist = risk['tp']
                
                for _, row in signals.iterrows():
                    entry = row['close']
                    stop = entry - sl_dist
                    target = entry + tp_dist
                    
                    # Look forward max 120 mins
                    time_limit = row['time'] + timedelta(minutes=120)
                    
                    # Slice future bars (Slow in loop, but acceptable for this scale)
                    # Optimization: Vectorized outcome lookup?
                    # For now, simplistic check
                    future = df_train[(df_train['time'] > row['time']) & (df_train['time'] <= time_limit)]
                    
                    if len(future) == 0:
                        continue
                        
                    # Did we hit SL or TP?
                    # Lows hit SL?
                    sl_hit_idx = (future['low'] <= stop).idxmax() if (future['low'] <= stop).any() else None
                    tp_hit_idx = (future['high'] >= target).idxmax() if (future['high'] >= target).any() else None
                    
                    # Logic
                    if sl_hit_idx and tp_hit_idx:
                        if sl_hit_idx < tp_hit_idx:
                            pnl -= sl_dist
                        else:
                            pnl += tp_dist
                            wins += 1
                    elif sl_hit_idx:
                        pnl -= sl_dist
                    elif tp_hit_idx:
                        pnl += tp_dist
                        wins += 1
                    else:
                        # Time exit
                        exit_price = future.iloc[-1]['close']
                        pnl += (exit_price - entry)
                        
                trades = len(signals)
                pf = (pnl + (trades * sl_dist)) / (trades * sl_dist) # Approx
                
                res = {
                    'Trigger': trig['name'],
                    'Context': ctx['name'],
                    'SL': risk['sl'],
                    'TP': risk['tp'],
                    'Trades': trades,
                    'PnL': pnl,
                    'WinRate': wins/trades
                }
                results.append(res)
                print(f"Result: {trig['name']} | {ctx['name']} | SL {risk['sl']} -> PnL: {pnl:.1f}")

    # Sort and Save
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('PnL', ascending=False)
    print("\nTOP 5 CONFIGURATIONS (TRAINING SET):")
    print(res_df.head())
    
    # Save best config for phase 3
    res_df.head(1).to_csv("output/best_training_config.csv")

if __name__ == "__main__":
    run_grid_search()
