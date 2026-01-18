"""
Robust Verification (The Exam)
Logic: Range_1.5x | PM_Session | SL 20 | TP 100 | TP1 30
Data: 4 Test Months (Feb, May, Jul, Oct 2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import timedelta

# Import Backtester
sys.path.append("C:/Users/CEO/ICT reinforcement")
from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

def load_test_data(base_dir, test_months):
    files = list(base_dir.glob("USTEC_2025_GOLDEN_PARQUET/USTEC_2025-*_clean_1m.parquet"))
    dfs = []
    print(f"Filtering for Test Months: {test_months}")
    for f in files:
        # Check filename month
        # Assuming format USTEC_2025-MM_...
        try:
            month_str = f.name.split('-')[1].split('_')[0]
            month = int(month_str)
            if month in test_months:
                print(f"  Loading Test File: {f.name}")
                df = pd.read_parquet(f)
                dfs.append(df)
        except:
            continue
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs)
    full_df = full_df.sort_index()
    return full_df

def main():
    base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
    tick_path = base_dir / "USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet" # Tick path needs to handle multi-month?
    # The Generic Backtester takes a single tick file path string in init.
    # BUT we need to test across 4 months.
    # We must instantiate the backtester dynamically for each month OR modify backtester to accept list.
    # Simplest: Loop through test months, run backtest per month, aggregate results.
    
    TEST_MONTHS = [2, 5, 7, 10]
    
    total_trades = []
    
    print("Starting Blind Verification Exam...")
    print(f"Logic: PM Session Range Expansion > 1.5x | Long Only | SL 20 | TP1 30 | TP2 100")
    
    for m in TEST_MONTHS:
        # Load Monthly Bar Data
        m_str = f"{m:02d}"
        bar_file = base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m_str}_clean_1m.parquet"
        tick_file = base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m_str}_clean_ticks.parquet"
        
        if not bar_file.exists() or not tick_file.exists():
            print(f"Skipping Month {m} (Files missing)")
            continue
            
        print(f"\nProcessing Month {m}...")
        df_bars = pd.read_parquet(bar_file)
        df_bars['time'] = pd.to_datetime(df_bars.index)
        df_bars['hour'] = df_bars['time'].dt.hour
        
        # Define Signal Logic (From Training Winner)
        df_bars['range'] = df_bars['high'] - df_bars['low']
        df_bars['range_ma'] = df_bars['range'].rolling(20).mean()
        
        mask_trigger = df_bars['range'] > (df_bars['range_ma'] * 1.5)
        mask_context = (df_bars['hour'] >= 13) & (df_bars['hour'] < 16) # PM Session
        mask_long = df_bars['close'] > df_bars['open']
        
        final_mask = mask_trigger & mask_context & mask_long
        signals = df_bars[final_mask]['time'].tolist()
        
        if not signals:
            print(f"  No signals found.")
            continue
            
        # Backtest (Tick Level)
        tester = TickGeneralizedBacktester(str(tick_file), str(bar_file))
        
        # Risk: SL 20, TP 100, TP1 30 (Scale 50%)
        # Note: best_training_config had SL 20, TP 100 which corresponds to Risk[2] in optimization
        # Risk[2] was {'sl': 20, 'tp': 100, 'tp1': 30}
        
        res = tester.backtest_signals(
            signals, 
            direction=1, 
            stop_pts=20, 
            target_pts=100, 
            tp1_pts=30, 
            tp1_pct=0.5, 
            move_to_be=True
        )
        
        # Collect Trades
        for t in res['trade_list']:
            total_trades.append(t)
            
        print(f"  PnL: {res['pnl']:.2f} | Trades: {res['trades']}")

    # Aggregate Results
    if not total_trades:
        print("\nNO TRADES generated in Test Set.")
        return

    df_res = pd.DataFrame(total_trades)
    
    gross_win = df_res[df_res['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_res[df_res['pnl'] < 0]['pnl'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    
    total_pnl = df_res['pnl'].sum()
    win_rate = (len(df_res[df_res['pnl'] > 0]) / len(df_res)) * 100
    
    print("\n" + "="*40)
    print("FINAL BLIND EXAM RESULTS (4 Months)")
    print("="*40)
    print(f"Total Trades: {len(df_res)}")
    print(f"Win Rate:     {win_rate:.2f}%")
    print(f"Profit Factor:{pf:.2f}")
    print(f"Total PnL:    {total_pnl:.2f}")
    print(f"Exp per Start:{total_pnl/4:.2f} pts/mo")
    
    if pf > 1.3:
        print("\n[PASS] STATUS: PASSED (Robust Strategy Confirmed)")
        print("This strategy has proven predictive power on unseen data.")
    else:
        print("\n[FAIL] STATUS: FAILED (Curve Fit Detected)")
        print("The strategy failed to perform on unseen data.")

if __name__ == "__main__":
    main()
