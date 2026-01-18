"""
Run Generalized Validation
Validates "High Volatility" US Session Strategy with Tick Precision.
Tests "PJ Style" management: SL > 10, TP1 + BE.
"""

from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester
import pandas as pd
import numpy as np
from pathlib import Path
import time

def main():
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    tick_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet"
    bar_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    print("Loading Data...")
    df_bars = pd.read_parquet(bar_path)
    # Parse timestamps
    df_bars['time'] = pd.to_datetime(df_bars.index)
    df_bars['hour'] = df_bars['time'].dt.hour
    df_bars['minute'] = df_bars['time'].dt.minute
    
    # Compute Volatility Signal (High Vol)
    # Range > 1.5x Avg Range (20 period)
    df_bars['range'] = df_bars['high'] - df_bars['low']
    df_bars['avg_range'] = df_bars['range'].rolling(20).mean()
    
    # Signal Logic
    # 1. High Vol
    mask_vol = df_bars['range'] > (df_bars['avg_range'] * 1.5)
    # 2. US Session (09:30 - 16:00)
    mask_time = (
        ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) |
        ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    )
    # 3. Direction Long (from GPU finding)
    mask_green = df_bars['close'] > df_bars['open']
    
    final_mask = mask_vol & mask_time & mask_green
    
    signals_idx = np.where(final_mask)[0]
    signals_list = []
    
    # Calculate Liquidity Levels (PDH, London High, Swing Highs)
    
    # 1. Fractal Swing Highs
    highs = df_bars['high'].values
    is_swing_high = (highs[2:] < highs[1:-1]) & (highs[:-2] < highs[1:-1])
    swing_indices = np.where(is_swing_high)[0] + 1
    swing_highs = highs[swing_indices]
    
    # 2. Session Highs (PDH, London)
    # Start simply: Pre-compute daily highs
    df_bars['date'] = df_bars['time'].dt.date
    daily_highs = df_bars.groupby('date')['high'].max().shift(1) # Previous day high
    # Map back to dataframe
    df_bars['pdh'] = df_bars['date'].map(daily_highs)
    
    # London High (approx 2am-5am)
    # We can do a rolling calc or just strict window?
    # Strict window is easier: For each day, find max high between 2-5
    lon_mask = (df_bars['hour'] >= 2) & (df_bars['hour'] < 5)
    lon_highs = df_bars[lon_mask].groupby('date')['high'].max()
    df_bars['london_high'] = df_bars['date'].map(lon_highs)
    
    print(f"Identified {len(signals_idx)} High Vol Long Signals.")
    
    for idx in signals_idx:
        row = df_bars.iloc[idx]
        sig_time = row['time']
        entry_price = row['close']
        
        # Pool of Targets
        targets = []
        
        # A. Nearest Fractal Swing Highs > Entry
        swings_above = swing_highs[swing_highs > entry_price]
        if len(swings_above) > 0:
            targets.append(swings_above.min())
            
        # B. PDH > Entry
        if not pd.isna(row['pdh']) and row['pdh'] > entry_price:
            targets.append(row['pdh'])
            
        # C. London High > Entry
        if not pd.isna(row['london_high']) and row['london_high'] > entry_price:
            targets.append(row['london_high'])
            
        # Select Target: Simplest is 'Nearest Major Liquidity'
        # Or should we aim for the furthest (Run on Liquidity)?
        # Let's try the Nearest Valid Target (> 10 pts away)
        
        valid_targets = [t for t in targets if (t - entry_price) >= 10.0]
        
        if len(valid_targets) > 0:
            target_price = min(valid_targets)
        else:
            target_price = None # No DoL nearby
        
        signal_obj = {
            'time': sig_time,
            'target_price': target_price if target_price else (entry_price + 50.0)
        }
        signals_list.append(signal_obj)
        
        # If no DoL found, what? Fallback to fixed? Or skip?
        # User wants DoL. Let's start with DoL or Default 50.
        
        signal_obj = {
            'time': sig_time,
            'target_price': target_price if target_price else (entry_price + 50.0)
        }
        signals_list.append(signal_obj)
    
    # Initialize Backtester
    tester = TickGeneralizedBacktester(str(tick_path), str(bar_path))
    
    # Grid Search Parameters (PJ Style)
    stops = [12.0, 15.0, 20.0]
    
    # Configs: Now TP2 is mostly ignored if target_price is present in signal
    # But TP1 logic still applies relative to entry
    configs = [
        # (TP1, TP2_Default, TP1_Pct)
        (None, 50.0, 0.0),      # Baseline: DoL Only (or default 50)
        (None, 100.0, 0.0),     # Baseline: DoL Only (or default 100)
        (20.0, 50.0, 0.5),      # Scale 50% at 20pts, then DoL
        (30.0, 50.0, 0.5),      # Scale 50% at 30pts, then DoL
    ]
    
    results = []
    
    print("\nStarting Validation Grid...")
    print("-" * 80)
    print(f"{'SL':<5} | {'TP1':<5} | {'TP2':<5} | {'Mode':<20} | {'PnL':<8} | {'PF':<6} | {'WR%':<5} | {'Trades':<6}")
    print("-" * 80)
    
    for sl in stops:
        for (tp1, tp2, pct) in configs:
            mode = "Fixed Target" if tp1 is None else f"Scale {int(pct*100)}%@{int(tp1)}->BE"
            
            stats = tester.backtest_signals(
                signals_list, 
                direction=1, 
                stop_pts=sl, 
                target_pts=tp2,
                tp1_pts=tp1,
                tp1_pct=pct,
                move_to_be=True if tp1 else False
            )
            
            res_row = {
                'sl': sl, 'tp1': tp1, 'tp2': tp2,
                'pnl': stats['pnl'], 'pf': stats['pf'], 'wr': stats['wr'], 'trades': stats['trades']
            }
            results.append(res_row)
            
            print(f"{sl:<5} | {tp1 if tp1 else '-':<5} | {tp2:<5} | {mode:<20} | {stats['pnl']:<8.1f} | {stats['pf']:<6.2f} | {stats['wr']:<5.1f} | {stats['trades']:<6}")
            
    # Find Best by PnL
    best = sorted(results, key=lambda x: x['pnl'], reverse=True)[0]
    print("-" * 80)
    print(f"BEST CONFIG: SL {best['sl']}, TP1 {best['tp1']}, TP2 {best['tp2']} -> PnL: {best['pnl']:.1f}")

if __name__ == "__main__":
    main()
