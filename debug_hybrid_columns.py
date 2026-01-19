import pandas as pd
import sys
from pathlib import Path

# Setup
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase16_pj_engine import engineer_pools
from strategies.mle.phase17_fvg_engine import engineer_fvg

try:
    BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
    # Load 1 Month
    df = pd.read_parquet(BASE_DIR / "USTEC_2025-01_clean_1m.parquet")
    df['time'] = pd.to_datetime(df.index)
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    
    print(f"Loaded {len(df)} bars")
    
    # Engineer
    df = engineer_pools(df)
    df = engineer_fvg(df)
    
    print("Columns:", df.columns.tolist())
    
    # Check Counts
    if 'IB_H' in df.columns:
        print(f"IB_H populated rows: {df['IB_H'].count()}")
        print(f"IB_H Sample: {df['IB_H'].dropna().iloc[0] if df['IB_H'].count() > 0 else 'None'}")
    else:
        print("MISSING IB_H COLUMN")
        
    if 'fvg_bull' in df.columns:
        print(f"Bull FVGs: {df['fvg_bull'].sum()}")
    else:
        print("MISSING fvg_bull COLUMN")

    # Simulate Logic Trace
    print("\n--- TRACE SIMULATION ---")
    
    # params: Pools IB, TP 30, SL 5
    params = {'pools': ['IB_L', 'IB_H'], 'tp_target': 30, 'sl_buffer': 5.0, 'max_trades': 2}
    
    active_pools = set(params['pools'])
    mask_trade = (df['hour'] >= 10) & (df['hour'] < 16)
    day_bars = df[mask_trade][df['date'] == df['date'].unique()[1]] # 2nd day
    
    print(f"Simulating Day: {day_bars['date'].iloc[0]}")
    
    swept_low = False
    trades = 0
    
    for i, row in day_bars.iterrows():
        if not swept_low:
            for p in active_pools:
                if 'L' in p:
                    lvl = row.get(p)
                    # print(f"Checking {p}: {lvl} vs Low {row['low']}")
                    if pd.notna(lvl):
                        if row['low'] < lvl:
                            print(f"✅ SWEPT LOW {p} ({lvl}) at {row['time']} | Low: {row['low']}")
                            swept_low = True
                        else:
                            # Print 1st check only to avoid spam
                            if i == day_bars.index[0]:
                                print(f"First Check {p}: {lvl} vs {row['low']}")
                        
        if swept_low and row['fvg_bull']:
            print(f"🚀 SIGNAL BULL at {row['time']}")
            trades += 1
            
    print(f"Total Signals Found: {trades}")

except Exception as e:
    print(f"CRASH: {e}")
