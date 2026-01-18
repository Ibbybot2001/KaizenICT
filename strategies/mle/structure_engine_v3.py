"""
STRUCTURE ENGINE V3 ("The Nuclear Option")
Tests Structural Alpha (Liquidity Sweeps) with INLINED Backtester to defeat Ghost Code.
"""

import pandas as pd
import numpy as np
import itertools
import multiprocessing
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, time as dtime

# ==============================================================================
# 0. INLINED BACKTESTER (NO IMPORTS = NO GHOSTS)
# ==============================================================================
class TickGeneralizedBacktester:
    def __init__(self, tick_data_source, bar_data_source):
        self.latency_ms = 500
        
        # Load Ticks
        if isinstance(tick_data_source, pd.DataFrame):
            self.df_ticks = tick_data_source.copy()
            if not isinstance(self.df_ticks.index, pd.DatetimeIndex):
                self.df_ticks.index = pd.to_datetime(self.df_ticks.index)
            self.tick_times = self.df_ticks.index.values
            if 'bid' in self.df_ticks.columns:
                 self.tick_bids = self.df_ticks['bid'].values
                 self.tick_asks = self.df_ticks['ask'].values
            else:
                 p = self.df_ticks['price'].values
                 self.tick_bids = p
                 self.tick_asks = p
        elif isinstance(tick_data_source, str):
            # SHOULD NOT HAPPEN IN WORKER (Passed DF)
            self.df_ticks = pd.read_parquet(tick_data_source)
            self.tick_times = pd.to_datetime(self.df_ticks.index).values
            p = self.df_ticks['price'].values
            self.tick_bids = p
            self.tick_asks = p

        # Load Bars
        if isinstance(bar_data_source, pd.DataFrame):
            self.df_bars = bar_data_source.copy()

    def find_fill_price(self, signal_time, direction):
        # BASIC LATENCY SIMULATION
        # Add latency
        entry_time_with_latency = signal_time + np.timedelta64(self.latency_ms, 'ms')
        
        # Binary Search
        idx = self.tick_times.searchsorted(entry_time_with_latency)
        if idx >= len(self.tick_times):
            return None, None
            
        # Get Price
        # Long (Buy): Fill at Ask
        # Short (Sell): Fill at Bid
        price = self.tick_asks[idx] if direction == 1 else self.tick_bids[idx]
        return price, self.tick_times[idx]

    def find_exit_price(self, entry_time, direction, stop_pts, target_pts, 
                        tp1_pts=None, tp1_pct=0.5, move_to_be=False,
                        slippage_pts=0.0, spread_pts=0.0):
        
        idx = self.tick_times.searchsorted(entry_time)
        if idx >= len(self.tick_times): return 0.0, "NO_DATA", entry_time
            
        # RAW TICK PRICE (Mid/Last) - assuming self.tick_asks IS the Ask
        raw_fill = self.tick_asks[idx] if direction == 1 else self.tick_bids[idx]
        
        # COST APPLICATION
        penalty = (spread_pts / 2.0) + slippage_pts
        real_entry_price = raw_fill + penalty if direction == 1 else raw_fill - penalty
        
        current_sl = real_entry_price - stop_pts if direction == 1 else real_entry_price + stop_pts
        current_tp = real_entry_price + target_pts if direction == 1 else real_entry_price - target_pts
        
        # Max Duration 4h
        end_time = entry_time + np.timedelta64(4, 'h')
        end_idx = self.tick_times.searchsorted(end_time)
        
        prices = self.tick_bids[idx:end_idx] if direction == 1 else self.tick_asks[idx:end_idx]
        times = self.tick_times[idx:end_idx]
        
        final_pnl = 0.0
        reason = "TIME_EXIT"
        exit_time = end_time
        
        # Vectorized Scan loop
        for i in range(len(prices)):
            p = prices[i]
            t = times[i]
            
            # SL Check (Market Exit -> Penalty)
            hit_sl = (p <= current_sl) if direction == 1 else (p >= current_sl)
            if hit_sl:
                exit_price = p - penalty if direction == 1 else p + penalty
                final_pnl = (exit_price - real_entry_price) if direction == 1 else (real_entry_price - exit_price)
                reason = "SL"
                exit_time = t
                break
                
            # TP Check (Limit Exit -> No Penalty)
            hit_tp = (p >= current_tp) if direction == 1 else (p <= current_tp)
            if hit_tp:
                final_pnl = target_pts # Limit fill
                reason = "TP"
                exit_time = t
                break
                
        else:
            # Force Close at End
            last_p = prices[-1] if len(prices) > 0 else real_entry_price
            exit_price = last_p - penalty if direction == 1 else last_p + penalty
            final_pnl = (exit_price - real_entry_price) if direction == 1 else (real_entry_price - exit_price)
            reason = "TIME"
            
        return final_pnl, reason, exit_time

    def backtest_signals(self, signals, direction, stop_pts, target_pts, slippage_pts=0.0, spread_pts=0.0):
        trades = []
        total_pnl = 0.0
        
        for sig_time in signals:
            if isinstance(sig_time, pd.Timestamp): sig_time = sig_time.to_datetime64()
            
            # Find Entry
            fill_price, fill_time = self.find_fill_price(sig_time, direction)
            if fill_price is None: continue
            
            pnl, reason, exit_time = self.find_exit_price(fill_time, direction, stop_pts, target_pts, 
                                                         slippage_pts=slippage_pts, spread_pts=spread_pts)
            
            trades.append({
                'entry_time': fill_time,
                'exit_time': exit_time,
                'pnl': pnl,
                'reason': reason
            })
            total_pnl += pnl
            
        return {
            'trades': len(trades),
            'pnl': total_pnl,
            'trades_df': pd.DataFrame(trades) if trades else pd.DataFrame()
        }

# ==============================================================================
# 1. STRUCTURAL DNA
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

def eval_candidate(dna_tuple):
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
            signal_long = mask_time & sweep_high & (df_bars['close'] > df_bars[col_high])
            signal_short = mask_time & sweep_low & (df_bars['close'] < df_bars[col_low])

        times_long = df_bars[signal_long]['time'].tolist()
        times_short = df_bars[signal_short]['time'].tolist()
        
        sl_pts = 20 if dna['stop_type'] == 'Fixed_20' else 40 
        tp_pts = 40 if dna['target_type'] == 'Fixed_2R' else 100 
        
        # USE INLINED BACKTESTER
        tester = TickGeneralizedBacktester(df_ticks, df_bars)
        
        if times_long:
            try:
                res_l = tester.backtest_signals(times_long, direction=1, stop_pts=sl_pts, target_pts=tp_pts, slippage_pts=0.25, spread_pts=0.25)
                if res_l['trades'] > 0:
                    total_pnl += res_l['pnl']
                    all_trades.append(res_l['trades_df'])
            except KeyError as e:
                print(f"CRASH DEBUG {m} LONG: Keys={res_l.keys() if 'res_l' in locals() else 'NoRes'} Error={e}")
                # return (dna, 0, 0, 0, 0, "CRASH")

        if times_short:
            try:
                res_s = tester.backtest_signals(times_short, direction=-1, stop_pts=sl_pts, target_pts=tp_pts, slippage_pts=0.25, spread_pts=0.25)
                if res_s['trades'] > 0:
                    total_pnl += res_s['pnl']
                    all_trades.append(res_s['trades_df'])
            except KeyError as e:
                print(f"CRASH DEBUG {m} SHORT: Keys={res_s.keys() if 'res_s' in locals() else 'NoRes'} Error={e}")

    if not all_trades: return (dna_tuple, 0.0, 0, 0.0, 0.0, "Low_Trades")
    
    full_df = pd.concat(all_trades).sort_values('entry_time')
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

    def run_serial(self):
        """
        Runs the grid search in the MAIN PROCESS (Single Core).
        Useful for debugging "Ghost Code" or Pickling errors.
        """
        keys = list(OPTIONS.keys())
        values = list(OPTIONS.values())
        grid = list(itertools.product(*values))
        print(f"[STRUCTURAL GRID V3 SERIAL] Testing {len(grid)} interactions in SINGLE PROCESS.")
        print(f"[PHYSICS] Spread: 0.25 | Slippage: 0.25")
        
        # Init Workers Locally
        init_worker(self.tick_paths, self.base_dir)
        
        results = []
        start_time = time.time()
        for i, dna in enumerate(grid):
            if i % 10 == 0: print(f"Processing {i}/{len(grid)}...")
            res = eval_candidate(dna)
            results.append(res)
            
            # Stop early if debugging
            if i > 50: 
                print("DEBUG STOP: processed 50 items.")
                break
                
        df = pd.DataFrame(results, columns=['DNA', 'PnL', 'Trades', 'MaxDD', 'Fitness', 'Status'])
        survivors = df[df['Status'] == 'SURVIVOR']
        
        print(f"[COMPLETE SERIAL] Survivors: {len(survivors)}")
        df.sort_values('Fitness', ascending=False).to_csv("output/Structure_Grid_Results_V3_Serial.csv")
        
        if len(survivors) > 0:
            print(survivors.head())
        else:
            print("[FAILURE] No Structural Alpha found under these constraints.")

if __name__ == "__main__":
    eng = StructureGridEngine()
    # FORCE SERIAL FOR DEBUGGING
    eng.run_serial()
