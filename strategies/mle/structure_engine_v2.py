"""
STRUCTURE ENGINE V2 (Phase 12)
Tests Structural Alpha (Liquidity Sweeps, Draws on Liquidity) with Institutional Physics.
Target: ICT / PJ Models.
"""

import pandas as pd
import numpy as np
import itertools
import multiprocessing
import time
import sys
import importlib 
from pathlib import Path
from datetime import datetime, time as dtime

# Import Backtester
sys.path.append("C:/Users/CEO/ICT reinforcement")
from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

# ==============================================================================
# 1. STRUCTURAL DNA (The Gene Space)
# ==============================================================================
OPTIONS = {
    'pool': ['PDH_PDL', 'Session_HL', 'Fractal_1H'], 
    'trigger': ['Sweep_Reclaim', 'Sweep_Expansion'],
    'time_filter': ['Killzone_AM', 'Killzone_PM', 'Full_Session'],
    'stop_type': ['Swing_Fractal', 'Fixed_20'],
    'target_type': ['Opposing_Pool', 'Fixed_2R']
}

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def precompute_structure(df_bars):
    df = df_bars.copy()
    df['date'] = df['time'].dt.date
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min'})
    daily['PDH'] = daily['high'].shift(1)
    daily['PDL'] = daily['low'].shift(1)
    df = df.merge(daily[['PDH', 'PDL']], on='date', how='left')
    
    mask_overnight = (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30))
    overnight = df[mask_overnight].groupby('date').agg(ONH=('high', 'max'), ONL=('low', 'min'))
    df = df.merge(overnight, on='date', how='left')
    
    df['Last_1H_High'] = df['high'].rolling(60).max().shift(1)
    df['Last_1H_Low'] = df['low'].rolling(60).min().shift(1)
    return df

# ==============================================================================
# 3. WORKER LOGIC
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
               df_bars['time'] = pd.to_datetime(df_bars.index)
               df_bars['hour'] = df_bars['time'].dt.hour
               df_bars['minute'] = df_bars['time'].dt.minute
               df_bars = precompute_structure(df_bars)
               WORKER_TICKS[m] = df_ticks
               WORKER_BARS[m] = df_bars
           except Exception as e:
               print(f"Worker Load Error {m}: {e}")

def eval_structure_candidate(dna_tuple):
    # RELOAD BACKTESTER TO ENSURE TRADES_DF FIX IS ACTIVE
    import strategies.mle.tick_generalized_backtester
    importlib.reload(strategies.mle.tick_generalized_backtester)
    from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

    dna = {'pool': dna_tuple[0], 'trigger': dna_tuple[1], 'time_filter': dna_tuple[2], 'stop_type': dna_tuple[3], 'target_type': dna_tuple[4]}
    global WORKER_TICKS, WORKER_BARS
    
    total_pnl = 0.0
    all_trades = []
    
    for m in WORKER_TICKS:
        df_ticks = WORKER_TICKS[m]
        df_bars = WORKER_BARS[m]
        
        # Time Filter
        if dna['time_filter'] == 'Killzone_AM':
            mask_time = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | (df_bars['hour']==10)
        elif dna['time_filter'] == 'Killzone_PM':
            mask_time = (df_bars['hour']==13)&(df_bars['minute']>=30) | (df_bars['hour']==14)
        else:
            mask_time = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | ((df_bars['hour']>=10)&(df_bars['hour']<16))
            
        # Pool
        if dna['pool'] == 'PDH_PDL': col_high, col_low = 'PDH', 'PDL'
        elif dna['pool'] == 'Session_HL': col_high, col_low = 'ONH', 'ONL'
        else: col_high, col_low = 'Last_1H_High', 'Last_1H_Low'
            
        # Trigger
        sweep_low = df_bars['low'] < df_bars[col_low]
        sweep_high = df_bars['high'] > df_bars[col_high]
        
        if dna['trigger'] == 'Sweep_Reclaim':
            signal_long = mask_time & sweep_low & (df_bars['close'] > df_bars[col_low])
            signal_short = mask_time & sweep_high & (df_bars['close'] < df_bars[col_high])
        else:
            # Expansion (Breakout)
            # Long Breakout: High > PDH & Close > PDH
            signal_long = mask_time & sweep_high & (df_bars['close'] > df_bars[col_high])
            # Short Breakdown: Low < PDL & Close < PDL
            signal_short = mask_time & sweep_low & (df_bars['close'] < df_bars[col_low])

        times_long = df_bars[signal_long]['time'].tolist()
        times_short = df_bars[signal_short]['time'].tolist()
        
        sl_pts = 20 if dna['stop_type'] == 'Fixed_20' else 40 
        tp_pts = 40 if dna['target_type'] == 'Fixed_2R' else 100 
        
        tester = TickGeneralizedBacktester(df_ticks, df_bars)
        
        if times_long:
            res_l = tester.backtest_signals(times_long, direction=1, stop_pts=sl_pts, target_pts=tp_pts, slippage_pts=0.25, spread_pts=0.25)
            # DEFENSIVE EXTRACTION
            if res_l.get('trades', 0) > 0:
                total_pnl += res_l.get('pnl', 0.0)
                td = res_l.get('trades_df')
                if td is not None and not td.empty:
                    all_trades.append(td)
                elif 'trade_list' in res_l:
                    all_trades.append(pd.DataFrame(res_l['trade_list']))

        if times_short:
            res_s = tester.backtest_signals(times_short, direction=-1, stop_pts=sl_pts, target_pts=tp_pts, slippage_pts=0.25, spread_pts=0.25)
            # DEFENSIVE EXTRACTION
            if res_s.get('trades', 0) > 0:
                total_pnl += res_s.get('pnl', 0.0)
                td = res_s.get('trades_df')
                if td is not None and not td.empty:
                    all_trades.append(td)
                elif 'trade_list' in res_s:
                    all_trades.append(pd.DataFrame(res_s['trade_list']))

    if not all_trades: return (dna_tuple, 0.0, 0, 0.0, 0.0, "Low_Trades")
    
    full_df = pd.concat(all_trades).sort_values('fill_time')
    full_df['cum_pnl'] = full_df['pnl'].cumsum()
    peak = full_df['cum_pnl'].cummax()
    dd = (peak - full_df['cum_pnl'])
    max_dd = dd.max() if not dd.empty else 0.0
    
    eff_score = total_pnl / len(full_df)
    
    dd_penalty = 1.0
    if max_dd > (total_pnl * 0.5): return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Unstable_DD")
    
    fitness = total_pnl * dd_penalty
    if eff_score < 5: return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Inefficient")
    
    return (dna_tuple, total_pnl, len(full_df), max_dd, fitness, "SURVIVOR")

class StructureGridEngine:
    def __init__(self):
        self.base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
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
        keys = list(OPTIONS.keys())
        values = list(OPTIONS.values())
        grid = list(itertools.product(*values))
        print(f"[STRUCTURAL GRID V2] Testing {len(grid)} interactions.")
        print(f"[PHYSICS] Spread: 0.25 | Slippage: 0.25")
        
        with multiprocessing.Pool(processes=24, initializer=init_worker, initargs=(self.tick_paths, self.base_dir)) as pool:
            results = pool.map(eval_structure_candidate, grid)
            
        df = pd.DataFrame(results, columns=['DNA', 'PnL', 'Trades', 'MaxDD', 'Fitness', 'Status'])
        survivors = df[df['Status'] == 'SURVIVOR']
        
        print(f"[COMPLETE] Survivors: {len(survivors)}")
        df.sort_values('Fitness', ascending=False).to_csv("output/Structure_Grid_Results_V2.csv")
        
        if len(survivors) > 0:
            print(survivors.head())
        else:
            print("[FAILURE] No Structural Alpha found under these constraints.")

if __name__ == "__main__":
    eng = StructureGridEngine()
    eng.run()
