"""
PHASE 13: THE GOLDEN ANCHOR
Full-year stress test for the Session Sweep King.
DNA: Session_HL | Sweep & Reclaim | Full Session | Structural Stop | Opposing Target
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# ==============================================================================
# INLINED BACKTESTER (Fidelity Upgraded)
# ==============================================================================
class GoldenBacktester:
    def __init__(self, df_ticks, df_bars):
        self.latency_ms = 500
        self.tick_times = df_ticks.index.values
        self.tick_bids = df_ticks['bid'].values if 'bid' in df_ticks.columns else df_ticks['price'].values
        self.tick_asks = df_ticks['ask'].values if 'ask' in df_ticks.columns else df_ticks['price'].values

    def find_fill(self, signal_time, direction):
        if isinstance(signal_time, pd.Timestamp):
            signal_time = signal_time.to_datetime64()
        entry_time = signal_time + np.timedelta64(self.latency_ms, 'ms')
        idx = self.tick_times.searchsorted(entry_time)
        if idx >= len(self.tick_times): return None, None
        price = self.tick_asks[idx] if direction == 1 else self.tick_bids[idx]
        return price, self.tick_times[idx]

    def backtest_trade(self, entry_time, direction, sl_price, tp_price, slippage_pts=0.25, spread_pts=0.25):
        idx = self.tick_times.searchsorted(entry_time)
        if idx >= len(self.tick_times): return 0.0, "NO_DATA", entry_time
        
        penalty = (spread_pts / 2.0) + slippage_pts
        real_entry = self.tick_asks[idx] + penalty if direction == 1 else self.tick_bids[idx] - penalty
        
        # SL/TP must be absolute prices now
        # Check if sl/tp are valid (Long: Sl < Entry < TP)
        if direction == 1:
            if sl_price >= real_entry: sl_price = real_entry - 20 # Safety
        else:
            if sl_price <= real_entry: sl_price = real_entry + 20

        end_time = entry_time + np.timedelta64(4, 'h')
        end_idx = self.tick_times.searchsorted(end_time)
        
        prices = self.tick_bids[idx:end_idx] if direction == 1 else self.tick_asks[idx:end_idx]
        times = self.tick_times[idx:end_idx]
        
        for i in range(len(prices)):
            p = prices[i]
            t = times[i]
            
            # SL Check
            if (direction == 1 and p <= sl_price) or (direction == -1 and p >= sl_price):
                exit_price = p - penalty if direction == 1 else p + penalty
                return (exit_price - real_entry) if direction == 1 else (real_entry - exit_price), "SL", t
            
            # TP Check
            if (direction == 1 and p >= tp_price) or (direction == -1 and p <= tp_price):
                return (tp_price - real_entry) if direction == 1 else (real_entry - tp_price), "TP", t
                
        # Time Exit
        last_p = prices[-1] if len(prices) > 0 else real_entry
        exit_p = last_p - penalty if direction == 1 else last_p + penalty
        return (exit_p - real_entry) if direction == 1 else (real_entry - exit_p), "TIME", times[-1] if len(times) > 0 else entry_time

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")

def run_anchor():
    all_trades = []
    
    for m in range(1, 13):
        m_str = f"{m:02d}"
        print(f"--- Processing Month {m_str} ---")
        try:
            df_bars = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_1m.parquet")
            df_ticks = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_ticks.parquet")
            df_ticks.index = pd.to_datetime(df_ticks.index)
        except Exception as e:
            print(f"Skip {m_str}: {e}")
            continue
            
        # 1. Feature Prep
        df = df_bars.copy()
        df['time'] = pd.to_datetime(df.index)
        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        
        # Session HL (Midnight to 09:30)
        mask_on = (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30))
        on_hl = df[mask_on].groupby('date').agg(ONH=('high', 'max'), ONL=('low', 'min'))
        df = df.merge(on_hl, on='date', how='left')
        
        # Signals (Full Session 09:30-16:00)
        mask_trade = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] >= 10) & (df['hour'] < 16))
        
        # Long: Sweep ONL, Reclaim ONL
        sig_long = mask_trade & (df['low'] < df['ONL']) & (df['close'] > df['ONL'])
        # Short: Sweep ONH, Reclaim ONH
        sig_short = mask_trade & (df['high'] > df['ONH']) & (df['close'] < df['ONH'])
        
        tester = GoldenBacktester(df_ticks, df)
        
        # Execute Longs
        for idx_row, row in df[sig_long].iterrows():
            fill_p, fill_t = tester.find_fill(row['time'], 1)
            if fill_p:
                # Structural SL: Low of sweep candle
                sl = row['low'] - 2.0 # Buffer
                tp = row['ONH'] # Opposing target
                pnl, reason, t_ext = tester.backtest_trade(fill_t, 1, sl, tp)
                all_trades.append({'time': fill_t, 'month': m, 'pnl': pnl, 'reason': reason})
                
        # Execute Shorts
        for idx_row, row in df[sig_short].iterrows():
            fill_p, fill_t = tester.find_fill(row['time'], -1)
            if fill_p:
                sl = row['high'] + 2.0 
                tp = row['ONL']
                pnl, reason, t_ext = tester.backtest_trade(fill_t, -1, sl, tp)
                all_trades.append({'time': fill_t, 'month': m, 'pnl': pnl, 'reason': reason})

    if not all_trades:
        print("Zero trades found for the whole year!")
        return
        
    res = pd.DataFrame(all_trades).sort_values('time')
    res['cum_pnl'] = res['pnl'].cumsum()
    
    # Stats
    summary = []
    for m in range(1, 13):
        m_df = res[res['month'] == m]
        if m_df.empty:
            summary.append({'Month': m, 'PnL': 0, 'Trades': 0, 'PF': 0})
            continue
        gp = m_df[m_df['pnl'] > 0]['pnl'].sum()
        gl = abs(m_df[m_df['pnl'] < 0]['pnl'].sum())
        pf = gp/gl if gl > 0 else 9.9
        summary.append({'Month': m, 'PnL': m_df['pnl'].sum(), 'Trades': len(m_df), 'PF': pf})
    
    summary_df = pd.DataFrame(summary)
    print("\n--- ANNUAL PERFORMANCE SUMMARY ---")
    print(summary_df)
    
    summary_df.to_csv("output/Phase13_Full_Year_Bench.csv", index=False)
    res.to_csv("output/Phase13_Full_Trades.csv", index=False)
    print(f"\n[DONE] Cumulative PnL: {res['cum_pnl'].iloc[-1]:.2f}")

if __name__ == "__main__":
    run_anchor()
