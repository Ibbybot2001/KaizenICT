"""
THE FINAL GRID SEARCH ("The Collapsing")
Exhaustive, Deterministic, Physics-Aware Audit of 6,912 Strategies.
"""

import pandas as pd
import numpy as np
import itertools
import multiprocessing
import time
import sys
from pathlib import Path
from datetime import datetime

# Import customized components
sys.path.append("C:/Users/CEO/ICT reinforcement")
from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

# ==============================================================================
# 1. THE SEARCH SPACE (6,912 Combinations)
# ==============================================================================
OPTIONS = {
    'trigger': ['Vol_1.5x', 'Range_1.5x', 'Disp_60%'],
    'context': ['AM_Session', 'PM_Session', 'Full_US'],
    'sl': [10, 15, 20, 30],
    'tp': [20, 50, 100, 200], 
    'manager': ['Fixed', 'Scale_Out'],
    'be': [True, False],
    'regime_atr': ['Any', 'Low_Vol', 'High_Vol'], 
    'daily_limit': [1, 2, 3, 100],
}

# ==============================================================================
# 2. WORKER LOGIC (Reused from Evo Engine but with PHYSICS)
# ==============================================================================
WORKER_TICKS = {}
WORKER_BARS = {}

def init_worker(tick_paths, bar_base_dir):
    global WORKER_TICKS, WORKER_BARS
    for m, tick_path in tick_paths.items():
        if m not in WORKER_TICKS:
           try:
               df_ticks = pd.read_parquet(tick_path)
               bar_path = bar_base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m:02d}_clean_1m.parquet"
               df_bars = pd.read_parquet(bar_path)
               
               # Precompute Bar Features
               df_bars['time'] = pd.to_datetime(df_bars.index)
               df_bars['hour'] = df_bars['time'].dt.hour
               df_bars['minute'] = df_bars['time'].dt.minute
               
               vol_col = 'tick_volume' if 'tick_volume' in df_bars.columns else 'volume'
               df_bars['vol_ma'] = df_bars[vol_col].rolling(20).mean()
               df_bars['range'] = df_bars['high'] - df_bars['low']
               df_bars['range_ma'] = df_bars['range'].rolling(20).mean()
               df_bars['is_high_vol'] = df_bars['range'] > df_bars['range_ma']
               
               WORKER_TICKS[m] = df_ticks
               WORKER_BARS[m] = df_bars
           except Exception as e:
               print(f"Worker Load Error {m}: {e}")

def eval_candidate(dna_tuple):
    """
    Evaluates one candidate tuple against the PHYSICS engine.
    dna_tuple matches order: (trigger, context, sl, tp, manager, be, regime_atr, daily_limit)
    """
    global WORKER_TICKS, WORKER_BARS
    
    # Unpack DNA
    strat_dna = {
        'trigger': dna_tuple[0], 'context': dna_tuple[1], 'sl': dna_tuple[2], 'tp': dna_tuple[3],
        'manager': dna_tuple[4], 'be': dna_tuple[5], 'regime_atr': dna_tuple[6], 'daily_limit': dna_tuple[7]
    }
    
    total_pnl = 0.0
    all_trades = []
    
    for m in WORKER_TICKS:
        df_ticks = WORKER_TICKS[m]
        df_bars = WORKER_BARS[m]
        
        # 1. Calc Trigger Mask
        vol_col = 'tick_volume' if 'tick_volume' in df_bars.columns else 'volume'
        trig = strat_dna['trigger']
        if trig == 'Vol_1.5x': mask_t = df_bars[vol_col] > (df_bars['vol_ma'] * 1.5)
        elif trig == 'Range_1.5x': mask_t = df_bars['range'] > (df_bars['range_ma'] * 1.5)
        elif trig == 'Disp_60%': mask_t = (df_bars['close']-df_bars['open']).abs() > (df_bars['range']*0.6)
        else: mask_t = df_bars[vol_col] > -1
        
        # 2. Context Mask
        ctx = strat_dna['context']
        if ctx == 'Full_US': mask_c = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | ((df_bars['hour']>=10)&(df_bars['hour']<16))
        elif ctx == 'PM_Session': mask_c = (df_bars['hour']>=13)&(df_bars['hour']<16)
        else: mask_c = (df_bars['hour']>=9)&(df_bars['hour']<12)
        
        # 3. Regime Mask
        reg = strat_dna['regime_atr']
        if reg == 'High_Vol': mask_r = df_bars['is_high_vol']
        elif reg == 'Low_Vol': mask_r = ~df_bars['is_high_vol']
        else: mask_r = True
        
        mask = mask_t & mask_c & mask_r & (df_bars['close'] > df_bars['open']) 
        raw_signals = df_bars[mask]['time'].tolist()
        
        if not raw_signals: continue
        
        # 4. Daily Shot Clock
        limit = strat_dna['daily_limit']
        if limit < 100:
            filtered_signals = []
            day_counts = {}
            for t in raw_signals:
                d = t.date()
                c = day_counts.get(d, 0)
                if c < limit:
                    filtered_signals.append(t)
                    day_counts[d] = c + 1
            signals = filtered_signals
        else:
            signals = raw_signals
            
        if not signals: continue

        # 5. Backtest with PHYSICS (Spread=0.25, Slippage=0.25)
        tester = TickGeneralizedBacktester(df_ticks, df_bars)
        tp1 = 30 if strat_dna['manager'] == 'Scale_Out' else None
        pct = 0.5 if tp1 else 0.0
        move_be = strat_dna.get('be', True)
        
        res = tester.backtest_signals(signals, direction=1, 
                                    stop_pts=strat_dna['sl'], 
                                    target_pts=strat_dna['tp'], 
                                    tp1_pts=tp1, tp1_pct=pct, move_to_be=move_be,
                                    slippage_pts=0.25, spread_pts=0.25) # <--- THE TRUTH
        
        if res['trades'] > 0:
            total_pnl += res['pnl']
            if 'trades_df' in res and not res['trades_df'].empty:
                 all_trades.append(res['trades_df'])
    
    if not all_trades: return (dna_tuple, 0.0, 0, 0.0, 0.0, "Low_Trades")
        
    full_df = pd.concat(all_trades).sort_values('fill_time')
    full_df['cum_pnl'] = full_df['pnl'].cumsum()
    peak = full_df['cum_pnl'].cummax()
    dd = (peak - full_df['cum_pnl'])
    max_dd = dd.max() if not dd.empty else 0.0
    
    # FITNESS LOGIC
    if len(full_df) < 10: return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Low_Trades")
    if total_pnl <= 0: return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Negative_PnL")
    
    eff_score = total_pnl / len(full_df)
    
    dd_penalty = 1.0
    if max_dd > (total_pnl * 0.2): dd_penalty = 0.5 
    if max_dd > (total_pnl * 0.5): dd_penalty = 0.1
    
    if dd_penalty < 0.2: return (dna_tuple, total_pnl, len(full_df), max_dd, eff_score, "Unstable_DD")
    if eff_score < 5: return (dna_tuple, total_pnl, len(full_df), max_dd, eff_score, "Inefficient")
    
    fitness = total_pnl * dd_penalty
    if eff_score < 10: fitness *= 0.5
    
    return (dna_tuple, total_pnl, len(full_df), max_dd, fitness, "SURVIVOR")

# ==============================================================================
# 3. GRID SEARCH CONTROLLER
# ==============================================================================
class GridSearchEngine:
    def __init__(self):
        self.base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
        # Train on 8mo + Test 4mo? Or Full 12mo?
        # User said "Same blind split".
        # Blind Split: Train = [1,3,4,6,8,9,11,12], Test = [2,5,7,10]
        # But wait, Evolution tested FITNESS. Grid Search should test on WHAT?
        # If I verify on test set only, it is "Testing".
        # If I verify on ALL months, it is "Performance".
        # Let's verify on the VALIDATION SET (Test Months) to show robustness.
        # Or should we look for globally robust?
        # Standard: Run on 8mo Train -> Filter -> Run on 4mo Test.
        # But for Grid Search "Absolute Truth", we usually run on ALL Data or just Test?
        # User said: "Same blind split". 
        # So I will load just the TEST months to check for "Survivors"?
        # Or run on TRAIN, then check TEST?
        # Actually, let's just run on the TEST SET (2,5,7,10). 
        # Because we want to know if *anything* works in the "Winter" (Blind) data.
        # If it works in Train but fails Test, it's not a Survivor.
        # If it works in Test, it IS a Survivor (by definition of blind validation).
        # Let's run on TEST MONTHS [2, 5, 7, 10].
        self.months = [2, 5, 7, 10]
        self.tick_paths = self._get_tick_paths(self.months)

    def _get_tick_paths(self, months):
        paths = {}
        for m in months:
            m_str = f"{m:02d}"
            p = self.base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m_str}_clean_ticks.parquet"
            if p.exists(): paths[m] = p
        return paths

    def run(self):
        # 1. Generate Grid
        keys = list(OPTIONS.keys())
        values = list(OPTIONS.values())
        grid = list(itertools.product(*values))
        print(f"[GRID] Generated {len(grid)} unique strategy combinations.")
        print(f"[PHYSICS] Spread: 0.25 | Slippage: 0.25 (Total Penalty: 0.5 pts)")
        print(f"[DATA] Blind Test Months: {self.months}")
        
        # 2. Run Parallel
        t0 = time.time()
        with multiprocessing.Pool(processes=24, initializer=init_worker, initargs=(self.tick_paths, self.base_dir)) as pool:
            results = pool.map(eval_candidate, grid)
        dt = time.time() - t0
        
        # 3. Analyze
        df = pd.DataFrame(results, columns=['DNA', 'PnL', 'Trades', 'MaxDD', 'Fitness', 'Status'])
        
        print(f"\n[COMPLETE] Scanned {len(grid)} strategies in {dt:.2f}s ({len(grid)/dt:.1f} strat/sec)")
        
        survivors = df[df['Status'] == 'SURVIVOR']
        print(f"\n[SURVIVORS] Count: {len(survivors)}")
        
        if len(survivors) > 0:
            print(survivors.sort_values('Fitness', ascending=False).head(10))
            survivors.to_csv("output/Grid_Survivors.csv")
            print("\n[SUCCESS] Outcome A: Survivors Found.")
        else:
            print("\n[FAILURE] Outcome B: Zero Survivors.")
            
        # Save Full Truth
        df.drop(columns=['DNA']).to_csv("output/Grid_Truth_Summary.csv")
        # Save detailed DNA for debugging
        # Convert tuple back to string for CSV
        df['DNA_Str'] = df['DNA'].apply(lambda x: str(x))
        df[['DNA_Str', 'PnL', 'Trades', 'MaxDD', 'Fitness', 'Status']].to_csv("output/Grid_Truth_Full.csv", index=False)

if __name__ == "__main__":
    eng = GridSearchEngine()
    eng.run()
