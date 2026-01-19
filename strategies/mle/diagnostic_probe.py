"""
DIAGNOSTIC PROBE: MULTI-POOL VOLUME CHECK
Tests a single "All Lows" strategy to verify trade volume.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys 

BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]

# Load Data
print("Loading data...")
all_bars = []
for m in TRAIN_MONTHS:
    m_str = f"{m:02d}"
    try:
        df = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_1m.parquet")
        df['time'] = pd.to_datetime(df.index)
        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        all_bars.append(df)
    except:
        pass

df_bars = pd.concat(all_bars, ignore_index=False)
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase16_pj_engine import engineer_pools
df_bars = engineer_pools(df_bars)
print(f"Loaded {len(df_bars)} bars")

# TEST PARAMS: "All Lows" Strategy
params = {
    'pools': ['PDL', 'ONL', 'ASIA_L', 'LON_L'],
    'disp_threshold': 0.4,
    'direction': 'LONG',
    'session': (9, 30, 15, 30),
    'sl_buffer': 3.0,
    'max_trades': 100, # Uncapped
    'require_disp': False # Volume check
}

print("\n--- RUNNING BACKTEST ---")
print(f"Pools: {params['pools']}")
print(f"Session: {params['session']}")

trades = []
tracker = defaultdict(lambda: 'DEFINED')
pools_to_trade = params['pools']
h_start, m_start, h_end, m_end = params['session']
mask = ((df_bars['hour'] == h_start) & (df_bars['minute'] >= m_start)) | \
       ((df_bars['hour'] > h_start) & (df_bars['hour'] < h_end)) | \
       ((df_bars['hour'] == h_end) & (df_bars['minute'] <= m_end))

dates = df_bars['date'].unique()

for d in dates:
    tracker.clear()
    day_bars = df_bars[(df_bars['date'] == d) & mask]
    
    for idx, row in day_bars.iterrows():
        for pool in pools_to_trade:
            if tracker[pool] != 'DEFINED': continue
            
            level = row.get(pool)
            if pd.isna(level): continue
            
            # Logic: Sweep + Reclaim
            sweep = row['low'] < level
            reclaim = row['close'] > level
            
            if sweep and reclaim:
                trades.append({
                    'date': d,
                    'pool': pool,
                    'entry': row['close'],
                    'sl': row['low'] - 3.0,
                    'tp': row['high'] + 50 # simplified
                })
                tracker[pool] = 'TRADED'

print(f"\nTotal Trades: {len(trades)}")
print(f"Days: {len(dates)}")
print(f"Avg Trades/Day: {len(trades)/len(dates):.2f}")

if len(trades) > 0:
    print("\nSample Trades:")
    print(pd.DataFrame(trades).head())
