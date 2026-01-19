
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time as pytime
import sys
import os

# Ensure we can import from the root /live folder
sys.path.append(str(Path(os.getcwd())))

from strategies.mle.phase16_pj_engine import engineer_pools, detect_pj_signals, PoolStateTracker

# --- CONFIGURATION ---
GOLDEN_DATA_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
# US Market Hour Filter for DATA PROCESSING (We need session highs/lows from before 9am)
# But we only TRADED during these hours.
AUDIT_OPEN = pytime(9, 30)
AUDIT_CLOSE = pytime(15, 30)

def run_audit():
    print("="*60)
    print("ICT PRODUCTION AUDIT: US MARKET HOURS (09:30 - 15:30)")
    print("="*60)

    all_trades = []
    months = [f"{m:02d}" for m in range(1, 13)]
    
    for m_str in months:
        bar_file = GOLDEN_DATA_DIR / f"USTEC_2025-{m_str}_clean_1m.parquet"
        if not bar_file.exists():
            continue
            
        print(f"-> Processing Month {m_str}...")
        df_bars = pd.read_parquet(bar_file)
        df_bars.index = pd.to_datetime(df_bars.index)
        
        # 1. Engineer Pools (Session H/L, PDH/PDL) - MUST BE DONE ON FULL DATA
        df_bars = engineer_pools(df_bars)
        
        # 2. Extract Data for Signal Detection
        df_bars['date'] = df_bars.index.date
        dates = df_bars['date'].unique()
        tracker = PoolStateTracker()
        
        for d in dates:
            daily_df = df_bars[df_bars['date'] == d]
            tracker.reset()
            
            # --- SIGNAL DETECTION ---
            # detect_pj_signals handles its own 9:30-15:30 mask internally
            signals = detect_pj_signals(daily_df, tracker, d)
            
            for sig in signals:
                # sig: (bar_time, pool_id, direction, sl_price, tp_price, displacement_flag)
                sig_time, pool_id, direction_val, sl, tp, disp_flag = sig
                
                entry_price = daily_df.loc[sig_time, 'close']
                
                trade = {
                    'date': d,
                    'pool': pool_id,
                    'direction': 'LONG' if direction_val == 1 else 'SHORT',
                    'entry': entry_price,
                    'pnl': 0,
                    'status': 'OPEN'
                }
                
                # Verify Exit using the full day's data
                future_bars = daily_df.loc[sig_time:]
                for f_ts, f_row in future_bars.iterrows():
                    if direction_val == 1: # LONG
                        if f_row['low'] <= sl:
                            trade['pnl'] = sl - entry_price
                            trade['status'] = 'CLOSED'
                            trade['exit_reason'] = 'SL'
                            break
                        if f_row['high'] >= tp:
                            trade['pnl'] = tp - entry_price
                            trade['status'] = 'CLOSED'
                            trade['exit_reason'] = 'TP'
                            break
                    else: # SHORT
                        if f_row['high'] >= sl:
                            trade['pnl'] = entry_price - sl
                            trade['status'] = 'CLOSED'
                            trade['exit_reason'] = 'SL'
                            break
                        if f_row['low'] <= tp:
                            trade['pnl'] = entry_price - tp
                            trade['status'] = 'CLOSED'
                            trade['exit_reason'] = 'TP'
                            break
                
                if trade['status'] == 'CLOSED':
                    all_trades.append(trade)

    # --- REPORTING ---
    df_results = pd.DataFrame(all_trades)
    if df_results.empty:
        print("Audit Complete: No trades found across 2025.")
        return

    print("\n" + "="*60)
    print("AUDIT RESULTS: 2025 US MARKET HOURS (09:30 - 15:30)")
    print("="*60)
    print(f"Total Trades: {len(df_results)}")
    print(f"Win Rate: {len(df_results[df_results['pnl'] > 0]) / len(df_results):.1%}")
    print(f"Gross PnL (Points): {df_results['pnl'].sum():.2f}")
    print(f"Avg PnL (Points): {df_results['pnl'].mean():.2f}")
    print(f"Total USD PnL: ${df_results['pnl'].sum() * 2.0:,.2f}")

if __name__ == "__main__":
    run_audit()
