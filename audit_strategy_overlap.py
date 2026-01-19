import pandas as pd
import sys
from pathlib import Path

# Engines
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase16_pj_engine import engineer_pools
from strategies.mle.phase17_fvg_engine import engineer_fvg

# CONFIG
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# STRATEGIES TO TEST
STRATS = [
    {
        'name': 'IB_HYBRID_PF2.73',
        'params': {'pools': ['IB_L', 'IB_H'], 'tp_target': 55, 'sl_buffer': 5.0, 'max_trades': 2}
    },
    {
        'name': 'ASIA_HYBRID_PF2.53',
        'params': {'pools': ['ASIA_L', 'ASIA_H'], 'tp_target': 85, 'sl_buffer': 5.0, 'max_trades': 2}
    }
]

def run_backtest_audit(bars_df, params):
    # Returns DataFrame of Trades with Times
    trade_logs = []
    
    active_pools = set(params['pools'])
    mask_trade = (bars_df['hour'] >= 10) & (bars_df['hour'] < 16)
    
    grouped = bars_df[mask_trade].groupby('date')
    
    for date, day_bars in grouped:
        daily_trades = 0
        swept_low = False
        swept_high = False
        
        for i, row in day_bars.iterrows():
            if daily_trades >= params['max_trades']: break
            
            # Check Sweeps
            if not swept_low:
                for p in active_pools:
                    if 'L' in p and pd.notna(row.get(p)) and row['low'] < row[p]:
                        swept_low = True
            if not swept_high:
                for p in active_pools:
                    if 'H' in p and pd.notna(row.get(p)) and row['high'] > row[p]:
                        swept_high = True
            
            # Check FVG Signal
            signal = False
            direction = 0
            entry = 0
            sl = 0
            tp = 0
            
            if swept_low and row['fvg_bull']:
                entry = row['fvg_bull_top']
                sl = row['low'] - params['sl_buffer']
                tp = entry + params['tp_target']
                direction = 1
                signal = True
            elif swept_high and row['fvg_bear']:
                entry = row['fvg_bear_btm']
                sl = row['high'] + params['sl_buffer']
                tp = entry - params['tp_target']
                direction = -1
                signal = True
                
            if signal:
                # Simulate
                filled = False
                exit_time = None
                pnl = 0
                future = day_bars.loc[i:].iloc[1:]
                
                for _, f_row in future.iterrows():
                    if not filled:
                        if direction == 1 and f_row['low'] <= entry: filled = True
                        elif direction == -1 and f_row['high'] >= entry: filled = True
                    
                    if filled:
                        if direction == 1:
                            if f_row['low'] <= sl:
                                pnl = sl - entry
                                exit_time = f_row['time']
                                break
                            if f_row['high'] >= tp:
                                pnl = tp - entry
                                exit_time = f_row['time']
                                break
                        else:
                            if f_row['high'] >= sl:
                                pnl = entry - sl
                                exit_time = f_row['time']
                                break
                            if f_row['low'] <= tp:
                                pnl = entry - tp
                                exit_time = f_row['time']
                                break
                
                if filled and exit_time:
                    trade_logs.append({
                        'entry_time': row['time'],
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'date': date
                    })
                    daily_trades += 1
                    
    return pd.DataFrame(trade_logs)

# VALIDATION LOGIC
if __name__ == '__main__':
    try:
        print("Loading Data (Full Year)...")
        all_bars = []
        for m in MONTHS: 
            try:
                p = BASE_DIR / f"USTEC_2025-{m:02d}_clean_1m.parquet"
                if p.exists():
                    df = pd.read_parquet(p)
                    df['time'] = pd.to_datetime(df.index)
                    df['date'] = df['time'].dt.date
                    df['hour'] = df['time'].dt.hour
                    all_bars.append(df)
            except: pass
        
        full_df = pd.concat(all_bars)
        print(f"Loaded {len(full_df)} bars.")
        
        # Pre-Calc Features
        print("Engine: Pools...")
        full_df = engineer_pools(full_df)
        print("Engine: FVG...")
        full_df = engineer_fvg(full_df)
        
        results = {}
        
        for s in STRATS:
            print(f"Running {s['name']}...")
            df_res = run_backtest_audit(full_df, s['params'])
            results[s['name']] = df_res
            print(f"  Trades: {len(df_res)}")
        
        # Compare
        keys = list(results.keys())
        df_a = results[keys[0]]
        df_b = results[keys[1]]
        
        # Merge on Date (Same Day Overlap)
        # And check precise Time Overlap
        
        # Find identical Entry Times
        if len(df_a) > 0 and len(df_b) > 0:
            merged = pd.merge(df_a, df_b, on='date', suffixes=('_A', '_B'), how='inner')
            
            # Exact Same Entry?
            exact_duplicates = merged[merged['entry_time_A'] == merged['entry_time_B']]
            
            print("\n--- OVERLAP REPORT (JANUARY) ---")
            print(f"Strategy A ({keys[0]}): {len(df_a)} trades")
            print(f"Strategy B ({keys[1]}): {len(df_b)} trades")
            print(f"Days where BOTH traded: {len(merged)}")
            print(f"EXACT Simultaneous Entries: {len(exact_duplicates)}")
            
            if len(exact_duplicates) > 0:
                # Compare Outcomes
                dups = exact_duplicates.copy()
                dups['outcome_A'] = dups['pnl_A'] > 0
                dups['outcome_B'] = dups['pnl_B'] > 0
                
                both_win = len(dups[ dups['outcome_A'] & dups['outcome_B'] ])
                both_loss = len(dups[ (~dups['outcome_A']) & (~dups['outcome_B']) ])
                a_win_b_loss = len(dups[ dups['outcome_A'] & (~dups['outcome_B']) ])
                a_loss_b_win = len(dups[ (~dups['outcome_A']) & dups['outcome_B'] ])
                
                print("\n--- OUTCOME DIVERGENCE (Based on TP Diff) ---")
                print(f"Strategy A (TP 55): IB Hybrid")
                print(f"Strategy B (TP 85): ASIA Hybrid")
                print(f"1. Mutual Stop Out (Both Lose): {both_loss} ({both_loss/len(dups)*100:.1f}%)")
                print(f"2. Mutual Target (Both Win):    {both_win} ({both_win/len(dups)*100:.1f}%)")
                print(f"3. Greed Fail (IB Wins, ASIA Loses): {a_win_b_loss} ({a_win_b_loss/len(dups)*100:.1f}%)")
                print(f"4. Mystery (IB Loses, ASIA Wins):  {a_loss_b_win} (Should be 0)")
                
                # Check EV diff
                net_a = dups['pnl_A'].sum()
                net_b = dups['pnl_B'].sum()
                print(f"\nNet PnL on Overlap: A=${net_a*20:.0f} vs B=${net_b*20:.0f}")
                
                overlap_pct = (len(exact_duplicates) / len(df_a)) * 100
                print(f"\nOverlap Percentage: {overlap_pct:.1f}%")
            else:
                print("Zero Overlap.")
        else:
            print("No trades generated in test period.")

    except Exception as e:
        import traceback
        traceback.print_exc()
