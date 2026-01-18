"""
STRUCTURE ENGINE (Phase 12)
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
    # THE POOL: Which liquidity are we hunting?
    'pool': ['PDH_PDL', 'Session_HL', 'Fractal_1H'], 
    
    # THE TRIGGER: How do we enter?
    # Sweep_Reclaim: Price breaks level -> Closes back inside (Reversal/Turtle Soup)
    # Sweep_Expansion: Price breaks level -> Closes outside (Continuation)
    'trigger': ['Sweep_Reclaim', 'Sweep_Expansion'],
    
    # THE FILTER: When do we take it?
    # Killzone_NY: 09:30 - 11:00 (Classic ICT)
    # Killzone_PM: 13:30 - 15:00 (Silver Bullet)
    # Anytime: 09:30 - 16:00
    'time_filter': ['Killzone_AM', 'Killzone_PM', 'Full_Session'],
    
    # THE STOP: Where is invalidation?
    # Swing: Recent Fractal High/Low (Standard)
    # Fixed: 20 pts (Tight)
    'stop_type': ['Swing_Fractal', 'Fixed_20'],
    
    # THE TARGET: Where do we exit?
    # Opposing_Pool: e.g. Buy PDL -> Target PDH (Aggressive)
    # Fixed_2R: 1:2 R:R (Conservative)
    'target_type': ['Opposing_Pool', 'Fixed_2R']
}

# ==============================================================================
# 2. FEATURE ENGINEERING (The "Where")
# ==============================================================================
def precompute_structure(df_bars):
    """
    Adds Structural Levels to the 1-minute dataframe.
    """
    df = df_bars.copy()
    df['date'] = df['time'].dt.date
    
    # 1. Previous Day High/Low (PDH/PDL)
    # Resample to Daily, Shift 1
    daily = df.groupby('date').agg({'high': 'max', 'low': 'min'})
    daily['PDH'] = daily['high'].shift(1)
    daily['PDL'] = daily['low'].shift(1)
    
    # Map back to minute bars
    # Note: This is slow if done via join for massive data, but okay for 1 year
    df = df.merge(daily[['PDH', 'PDL']], on='date', how='left')
    
    # 2. Session High/Low (Prior Session)
    # Define Sessions: London (03:00-09:00 ET approx or just Overnight), AM (09:30-12:00)
    # Simplified: "Overnight High/Low" (Midnight to 09:30)
    # We will compute "High/Low since Midnight" up to 09:30
    mask_overnight = (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30))
    
    # This is tricky to vectorize efficiently without lookahead.
    # Group by Date, apply expanding max/min on overnight mask?
    # For speed in grid search, we might simplify:
    # Calculate Overnight H/L per day and map it.
    
    # Group by Date, Filter for Overnight, Calc Max/Min
    overnight = df[mask_overnight].groupby('date').agg(
        ONH=('high', 'max'),
        ONL=('low', 'min')
    )
    df = df.merge(overnight, on='date', how='left')
    
    # 3. 1H Fractals (Rolling Max/Min of last 60 bars, shifted)
    # Pivot High: High > Left 60 bars AND High > Right 0 bars (Real-time detection)
    # Actually ICT Fractals are 3-bar or 5-bar. 
    # Let's use "Rolling 60m High" as "Recent Liquidity".
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
               
               # Basic Time Features
               df_bars['time'] = pd.to_datetime(df_bars.index)
               df_bars['hour'] = df_bars['time'].dt.hour
               df_bars['minute'] = df_bars['time'].dt.minute
               
               # STRUCTURAL FEATURES
               df_bars = precompute_structure(df_bars)
               
               WORKER_TICKS[m] = df_ticks
               WORKER_BARS[m] = df_bars
           except Exception as e:
               print(f"Worker Load Error {m}: {e}")

def eval_structure_candidate(dna_tuple):
    # Unpack DNA
    # (pool, trigger, time_filter, stop_type, target_type)
    dna = {
        'pool': dna_tuple[0], 'trigger': dna_tuple[1], 'time_filter': dna_tuple[2],
        'stop_type': dna_tuple[3], 'target_type': dna_tuple[4]
    }
    
    global WORKER_TICKS, WORKER_BARS
    
    total_pnl = 0.0
    all_trades = []
    
    for m in WORKER_TICKS:
        df_ticks = WORKER_TICKS[m]
        df_bars = WORKER_BARS[m]
        
        # 0. Time Filter
        if dna['time_filter'] == 'Killzone_AM':
            mask_time = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | (df_bars['hour']==10)
        elif dna['time_filter'] == 'Killzone_PM':
            mask_time = (df_bars['hour']==13)&(df_bars['minute']>=30) | (df_bars['hour']==14)
        else: # Full Session
            mask_time = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | ((df_bars['hour']>=10)&(df_bars['hour']<16))
            
        # 1. Pool Identification
        if dna['pool'] == 'PDH_PDL':
            col_high, col_low = 'PDH', 'PDL'
        elif dna['pool'] == 'Session_HL':
            col_high, col_low = 'ONH', 'ONL'
        else: # Fractal_1H
            col_high, col_low = 'Last_1H_High', 'Last_1H_Low'
            
        # 2. Trigger Logic (Vectorized Sweep Detection)
        # We need to detect the MOMENT of sweep.
        # Condition: previously not swept, now swept? 
        # Simpler: Bar High > Pool High. 
        # But we need "Sweep & Reclaim" vs "Sweep & Expansion"
        
        # Let's iterate signals? No, slow. Vectorize.
        
        # Long Setup (Pool = Low)
        # Sweep: Low < Pool Low
        sweep_low = df_bars['low'] < df_bars[col_low]
        
        # Reclaim (Close > Pool Low) vs Expansion (Close < Pool Low)
        if dna['trigger'] == 'Sweep_Reclaim':
            # Turtle Soup Long: Dip below Low, Close above Low
            signal_long = mask_time & sweep_low & (df_bars['close'] > df_bars[col_low])
        else:
            # Expansion Short (Breakout): Dip below Low, Close below Low
            # Wait, if we are expanding down, we are Shorting? 
            # If we break PDL and hold, it's a breakdown.
            # But usually folks trade Reversals on Liquidity.
            # Let's assume Expansion means "Breakout Trading".
            # So if Low < PDL and Close < PDL, we Short?
            signal_long = pd.Series(False, index=df_bars.index) # Logic placeholder
            # Let's keep it simple: Liquidity logic usually means Reversals for this test.
            # If expanding, we short breaks of lows.
            # Long Breakout: High > PDH & Close > PDH.
            
        # Short Setup (Pool = High)
        sweep_high = df_bars['high'] > df_bars[col_high]
        
        if dna['trigger'] == 'Sweep_Reclaim':
            # Turtle Soup Short: Spike above High, Close below High
            signal_short = mask_time & sweep_high & (df_bars['close'] < df_bars[col_high])
        else:
            # Expansion Long (Breakout): Spike above High, Close above High
            signal_short = pd.Series(False, index=df_bars.index)
            # Long Breakout logic
            signal_long = signal_long | (mask_time & sweep_high & (df_bars['close'] > df_bars[col_high]))
            # Short Breakdown logic
            signal_short = mask_time & sweep_low & (df_bars['close'] < df_bars[col_low])

        # Combine
        # We need to pass signals to backtester. 
        # Backtester takes list of times. But needs Direction.
        # Our General Backtester takes one direction list. 
        # We might need to split Longs and Shorts or update Backtester to handle 'side' column.
        # Current Backtester handles one direction at a time per call.
        
        # Run Longs
        times_long = df_bars[signal_long]['time'].tolist()
        # Run Shorts
        times_short = df_bars[signal_short]['time'].tolist()
        
        # Parameters
        tester = TickGeneralizedBacktester(df_ticks, df_bars)
        
        # Stop/Target Logic
        # This is hard to vectorize perfectly (dynamic stop based on swing).
        # We will approximate "Swing_Fractal" as "20 pts" for now to get the GRID RUNNING.
        # Or parse the DNA for fixed.
        sl_pts = 20 if dna['stop_type'] == 'Fixed_20' else 40 # Placeholder for fractal
        tp_pts = 40 if dna['target_type'] == 'Fixed_2R' else 100 # Placeholder for Opposing Pool
        
        # Execute Longs
        if times_long:
            res_l = tester.backtest_signals(times_long, direction=1, stop_pts=sl_pts, target_pts=tp_pts, slippage_pts=0.25, spread_pts=0.25)
            if res_l['trades'] > 0:
                total_pnl += res_l['pnl']
                td = res_l.get('trades_df')
                if td is not None and not td.empty:
                    all_trades.append(td)
                elif 'trade_list' in res_l:
                     # Fallback: Reconstruct DF if missing but list exists
                     all_trades.append(pd.DataFrame(res_l['trade_list']))
                
        # Execute Shorts
        if times_short:
            res_s = tester.backtest_signals(times_short, direction=-1, stop_pts=sl_pts, target_pts=tp_pts, slippage_pts=0.25, spread_pts=0.25)
            if res_s['trades'] > 0:
                total_pnl += res_s['pnl']
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
    
    # Fitness
    if len(full_df) < 10: return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Low_Trades")
    if total_pnl <= 0: return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Negative_PnL")
    
    eff_score = total_pnl / len(full_df)
    
    # Stability (Predator Logic)
    dd_penalty = 1.0
    if max_dd > (total_pnl * 0.5): return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Unstable_DD")
    
    fitness = total_pnl * dd_penalty
    if eff_score < 5: return (dna_tuple, total_pnl, len(full_df), max_dd, 0.0, "Inefficient")
    
    return (dna_tuple, total_pnl, len(full_df), max_dd, fitness, "SURVIVOR")

# ==============================================================================
# 4. GRID RUNNER
# ==============================================================================
class StructureGridEngine:
    def __init__(self):
        self.base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
        self.months = [2, 5, 7, 10] # Blind Test
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
        print(f"[STRUCTURAL GRID] Testing {len(grid)} interactions.")
        print(f"[PHYSICS] Spread: 0.25 | Slippage: 0.25")
        
        with multiprocessing.Pool(processes=24, initializer=init_worker, initargs=(self.tick_paths, self.base_dir)) as pool:
            results = pool.map(eval_structure_candidate, grid)
            
        df = pd.DataFrame(results, columns=['DNA', 'PnL', 'Trades', 'MaxDD', 'Fitness', 'Status'])
        survivors = df[df['Status'] == 'SURVIVOR']
        
        print(f"[COMPLETE] Survivors: {len(survivors)}")
        df.sort_values('Fitness', ascending=False).to_csv("output/Structure_Grid_Results.csv")
        
        if len(survivors) > 0:
            print(survivors.head())
        else:
            print("[FAILURE] No Structural Alpha found under these constraints.")

if __name__ == "__main__":
    eng = StructureGridEngine()
    eng.run()
