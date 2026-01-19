"""
VERIFICATION: POWER TRIO STRATEGY
Pools: PDL, ONL, ASIA_L (and Highs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys 

BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]  # Jan-Aug
TEST_MONTHS = [9, 10, 11, 12]           # Sep-Dec

# Load Data
print("Loading data...")
all_bars = []
for m in TRAIN_MONTHS + TEST_MONTHS:
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

# STRATEGY CONFIG
params = {
    'pools': ['PDL', 'ONL', 'ASIA_L', 'PDH', 'ONH', 'ASIA_H'],
    'disp_threshold': 0.4,
    'direction': 'BOTH',
    'session': (9, 30, 15, 30),
    'sl_buffer': 5.0,
    'max_trades': 10,
    'require_disp': False # Volume check
}

print("\n--- RUNNING VERIFICATION ---")
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
    daily_count = 0 
    day_bars = df_bars[(df_bars['date'] == d) & mask]
    
    for idx, row in day_bars.iterrows():
        if daily_count >= params['max_trades']: break

        for pool in pools_to_trade:
            if tracker[pool] != 'DEFINED': continue
            
            level = row.get(pool)
            if pd.isna(level): continue
            
            is_low = pool.endswith('L') or pool in ['PDL', 'ONL']
            
            if is_low:
                sweep = row['low'] < level
                reclaim = row['close'] > level
                direction = 1
                sl = row['low'] - params['sl_buffer']
                tp_col = pool.replace('L', 'H') if 'L' in pool else ('PDH' if pool=='PDL' else 'ONH')
                tp = row.get(tp_col, row['high'] + 50)
            else:
                sweep = row['high'] > level
                reclaim = row['close'] < level
                direction = -1
                sl = row['high'] + params['sl_buffer']
                tp_col = pool.replace('H', 'L') if 'H' in pool else ('PDL' if pool=='PDH' else 'ONL')
                tp = row.get(tp_col, row['low'] - 50)
            
            if pd.isna(tp): tp = row['close'] + (50 * direction)

            if sweep and reclaim:
                # Sim trade
                entry = row['close']
                pnl = 0
                future_bars = df_bars[(df_bars['date'] == d) & (df_bars.index > idx)]
                
                for _, fb in future_bars.iterrows():
                    if direction == 1:
                        if fb['low'] <= sl:
                            pnl = sl - entry
                            break
                        if fb['high'] >= tp:
                            pnl = tp - entry
                            break
                    else:
                        if fb['high'] >= sl:
                            pnl = entry - sl
                            break
                        if fb['low'] <= tp:
                            pnl = entry - tp
                            break
                
                trades.append({'pnl': pnl})
                tracker[pool] = 'TRADED'
                daily_count += 1

# REPORT
df = pd.DataFrame(trades)
print(f"\nTotal Trades: {len(df)}")
print(f"Days: {len(dates)}")
print(f"Trades/Day: {len(df)/len(dates):.2f}")

winners = df[df['pnl'] > 0]
losers = df[df['pnl'] < 0]
pf = winners['pnl'].sum() / abs(losers['pnl'].sum())
exp = df['pnl'].mean()

print(f"Profit Factor: {pf:.2f}")
print(f"Expectancy: ${exp:.2f}")
print(f"Total PnL: ${df['pnl'].sum():.2f}")
