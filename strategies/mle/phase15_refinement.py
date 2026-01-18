"""
PHASE 15: PORTFOLIO REFINEMENT
Test A: Wick Reclaim vs Close Reclaim (per pool expectancy impact).
Retained Pools: LUNCH_L, LUNCH_H, ASIA_L, LON_L, ONL.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================================
# INLINED BACKTESTER (From Phase 14)
# ==============================================================================
class GoldenBacktester:
    def __init__(self, df_ticks):
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
        if direction == 1:
            if sl_price >= real_entry: sl_price = real_entry - 20
        else:
            if sl_price <= real_entry: sl_price = real_entry + 20
        end_time = entry_time + np.timedelta64(4, 'h')
        end_idx = self.tick_times.searchsorted(end_time)
        prices = self.tick_bids[idx:end_idx] if direction == 1 else self.tick_asks[idx:end_idx]
        times = self.tick_times[idx:end_idx]
        for i in range(len(prices)):
            p = prices[i]
            t = times[i]
            if (direction == 1 and p <= sl_price) or (direction == -1 and p >= sl_price):
                exit_price = p - penalty if direction == 1 else p + penalty
                return (exit_price - real_entry) if direction == 1 else (real_entry - exit_price), "SL", t
            if (direction == 1 and p >= tp_price) or (direction == -1 and p <= tp_price):
                return (tp_price - real_entry) if direction == 1 else (real_entry - tp_price), "TP", t
        last_p = prices[-1] if len(prices) > 0 else real_entry
        exit_p = last_p - penalty if direction == 1 else last_p + penalty
        return (exit_p - real_entry) if direction == 1 else (real_entry - exit_p), "TIME", times[-1] if len(times) > 0 else entry_time

# ==============================================================================
# POOL STATE TRACKER
# ==============================================================================
class PoolStateTracker:
    def __init__(self, allowed_pools):
        self.allowed_pools = allowed_pools
        self.reset()
    def reset(self):
        self.pool_states = {p: 'DEFINED' for p in self.allowed_pools}
    def can_trade(self, pool_id):
        return pool_id in self.allowed_pools and self.pool_states.get(pool_id) == 'DEFINED'
    def mark_traded(self, pool_id):
        self.pool_states[pool_id] = 'TRADED'

# ==============================================================================
# FEATURE ENGINEERING (From Phase 14)
# ==============================================================================
def engineer_pools(df_bars):
    df = df_bars.copy()
    df['time'] = pd.to_datetime(df.index)
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    mask_asia = (df['hour'] >= 0) & (df['hour'] < 3)
    mask_london = (df['hour'] >= 3) & ((df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30)))
    mask_overnight = (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30))
    mask_lunch = ((df['hour'] == 11) & (df['minute'] >= 30)) | (df['hour'] == 12) | ((df['hour'] == 13) & (df['minute'] < 30))
    asia_hl = df[mask_asia].groupby('date').agg(ASIA_H=('high', 'max'), ASIA_L=('low', 'min'))
    london_hl = df[mask_london].groupby('date').agg(LON_H=('high', 'max'), LON_L=('low', 'min'))
    overnight_hl = df[mask_overnight].groupby('date').agg(ONH=('high', 'max'), ONL=('low', 'min'))
    lunch_hl = df[mask_lunch].groupby('date').agg(LUNCH_H=('high', 'max'), LUNCH_L=('low', 'min'))
    df = df.merge(asia_hl, on='date', how='left').merge(london_hl, on='date', how='left')
    df = df.merge(overnight_hl, on='date', how='left').merge(lunch_hl, on='date', how='left')
    return df

# ==============================================================================
# SIGNAL DETECTION (With Reclaim Mode)
# ==============================================================================
def detect_signals(df_bars, tracker, current_date, reclaim_mode='wick'):
    """
    reclaim_mode: 'wick' (low < level is enough) or 'close' (close must be past level)
    """
    signals = []
    mask_trade = ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) | ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    trade_bars = df_bars[mask_trade & (df_bars['date'] == current_date)]
    
    # Retained Pools: LUNCH_L, LUNCH_H, ASIA_L, LON_L, ONL
    POOLS = [
        ('LUNCH_H', 'LUNCH_L'),
        ('ASIA_H', 'ASIA_L'),  # Only ASIA_L is in allowed, ASIA_H not allowed
        ('LON_H', 'LON_L'),    # Only LON_L is in allowed
        ('ONH', 'ONL'),        # Only ONL is in allowed
    ]
    
    for idx, row in trade_bars.iterrows():
        for high_pool, low_pool in POOLS:
            h_col, l_col = high_pool, low_pool
            h_level = row.get(h_col)
            l_level = row.get(l_col)
            if pd.isna(h_level) or pd.isna(l_level): continue
            
            # Long: Sweep Low & Reclaim
            if tracker.can_trade(low_pool):
                sweep_occurred = row['low'] < l_level
                if reclaim_mode == 'wick':
                    reclaim_occurred = row['close'] > l_level
                else:  # 'close' mode: previous bar closed above, this bar opened below, closed above
                    reclaim_occurred = row['close'] > l_level and row['open'] < l_level
                
                if sweep_occurred and reclaim_occurred:
                    sl = row['low'] - 2.0
                    tp = h_level
                    signals.append((row['time'], low_pool, 1, sl, tp))
                    tracker.mark_traded(low_pool)
                    
            # Short: Sweep High & Reclaim (Only LUNCH_H allowed)
            if tracker.can_trade(high_pool):
                sweep_occurred = row['high'] > h_level
                if reclaim_mode == 'wick':
                    reclaim_occurred = row['close'] < h_level
                else:
                    reclaim_occurred = row['close'] < h_level and row['open'] > h_level
                    
                if sweep_occurred and reclaim_occurred:
                    sl = row['high'] + 2.0
                    tp = l_level
                    signals.append((row['time'], high_pool, -1, sl, tp))
                    tracker.mark_traded(high_pool)
    return signals

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")

def run_test_a():
    # Allowed pools per user directive
    ALLOWED_POOLS = ['LUNCH_L', 'LUNCH_H', 'ASIA_L', 'LON_L', 'ONL']
    
    results = []
    
    for mode in ['wick', 'close']:
        print(f"\n=== TESTING RECLAIM MODE: {mode.upper()} ===")
        all_trades = []
        
        for m in range(1, 13):
            m_str = f"{m:02d}"
            try:
                df_bars = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_1m.parquet")
                df_ticks = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_ticks.parquet")
                df_ticks.index = pd.to_datetime(df_ticks.index)
            except:
                continue
            df_bars = engineer_pools(df_bars)
            tester = GoldenBacktester(df_ticks)
            tracker = PoolStateTracker(ALLOWED_POOLS)
            dates = df_bars['date'].unique()
            
            for d in dates:
                tracker.reset()
                signals = detect_signals(df_bars, tracker, d, reclaim_mode=mode)
                for sig_time, pool_id, direction, sl, tp in signals:
                    fill_p, fill_t = tester.find_fill(sig_time, direction)
                    if fill_p is None: continue
                    pnl, reason, _ = tester.backtest_trade(fill_t, direction, sl, tp)
                    all_trades.append({'pool': pool_id, 'pnl': pnl, 'mode': mode})
        
        if all_trades:
            df = pd.DataFrame(all_trades)
            # Per-pool breakdown
            for pool in ALLOWED_POOLS:
                pdf = df[df['pool'] == pool]
                if pdf.empty: continue
                total_pnl = pdf['pnl'].sum()
                count = len(pdf)
                expectancy = total_pnl / count if count > 0 else 0
                results.append({'Mode': mode, 'Pool': pool, 'PnL': total_pnl, 'Trades': count, 'Expectancy': expectancy})
                print(f"  {pool}: PnL={total_pnl:.2f}, Trades={count}, Exp={expectancy:.2f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("output/Phase15_TestA_Results.csv", index=False)
    print("\n[DONE] Results saved to Phase15_TestA_Results.csv")
    
    # Summary: Which mode is better per pool?
    print("\n--- MODE COMPARISON (Expectancy) ---")
    pivot = results_df.pivot(index='Pool', columns='Mode', values='Expectancy')
    print(pivot)
    pivot.to_csv("output/Phase15_TestA_Comparison.csv")

if __name__ == "__main__":
    run_test_a()
