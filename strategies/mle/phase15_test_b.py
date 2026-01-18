"""
PHASE 15: TEST B - DOWNSIDE-ONLY PORTFOLIO
Tests: ASIA_L, LON_L, LUNCH_L, ONL (No Highs).
Uses: Close Reclaim Only (from Test A).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ==============================================================================
# INLINED BACKTESTER
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
        return self.tick_asks[idx] if direction == 1 else self.tick_bids[idx], self.tick_times[idx]

    def backtest_trade(self, entry_time, direction, sl_price, tp_price, slippage_pts=0.25, spread_pts=0.25):
        idx = self.tick_times.searchsorted(entry_time)
        if idx >= len(self.tick_times): return 0.0, "NO_DATA", entry_time
        penalty = (spread_pts / 2.0) + slippage_pts
        real_entry = self.tick_asks[idx] + penalty if direction == 1 else self.tick_bids[idx] - penalty
        if direction == 1 and sl_price >= real_entry: sl_price = real_entry - 20
        if direction == -1 and sl_price <= real_entry: sl_price = real_entry + 20
        end_time = entry_time + np.timedelta64(4, 'h')
        end_idx = self.tick_times.searchsorted(end_time)
        prices = self.tick_bids[idx:end_idx] if direction == 1 else self.tick_asks[idx:end_idx]
        times = self.tick_times[idx:end_idx]
        for i in range(len(prices)):
            p, t = prices[i], times[i]
            if (direction == 1 and p <= sl_price) or (direction == -1 and p >= sl_price):
                exit_p = p - penalty if direction == 1 else p + penalty
                return (exit_p - real_entry) if direction == 1 else (real_entry - exit_p), "SL", t
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
# FEATURE ENGINEERING
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
# SIGNAL DETECTION (Close Reclaim, Downside Only)
# ==============================================================================
def detect_signals_downside(df_bars, tracker, current_date):
    signals = []
    mask_trade = ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) | ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    trade_bars = df_bars[mask_trade & (df_bars['date'] == current_date)]
    
    # Downside pools only: ASIA_L, LON_L, LUNCH_L, ONL
    POOLS = [
        ('ASIA_H', 'ASIA_L'),
        ('LON_H', 'LON_L'),
        ('LUNCH_H', 'LUNCH_L'),
        ('ONH', 'ONL'),
    ]
    
    for idx, row in trade_bars.iterrows():
        for high_pool, low_pool in POOLS:
            h_level = row.get(high_pool)
            l_level = row.get(low_pool)
            if pd.isna(h_level) or pd.isna(l_level): continue
            
            # Long only: Sweep Low & Close Reclaim
            if tracker.can_trade(low_pool):
                sweep = row['low'] < l_level
                reclaim = row['close'] > l_level and row['open'] < l_level
                if sweep and reclaim:
                    sl = row['low'] - 2.0
                    tp = h_level
                    signals.append((row['time'], low_pool, 1, sl, tp))
                    tracker.mark_traded(low_pool)
    return signals

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")

def run_test_b():
    ALLOWED_POOLS = ['ASIA_L', 'LON_L', 'LUNCH_L', 'ONL']  # Downside only
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
            signals = detect_signals_downside(df_bars, tracker, d)
            for sig_time, pool_id, direction, sl, tp in signals:
                fill_p, fill_t = tester.find_fill(sig_time, direction)
                if fill_p is None: continue
                pnl, reason, _ = tester.backtest_trade(fill_t, direction, sl, tp)
                all_trades.append({'month': m, 'date': d, 'pool': pool_id, 'pnl': pnl})

    if not all_trades:
        print("Zero trades!")
        return
        
    res = pd.DataFrame(all_trades)
    res['cum_pnl'] = res['pnl'].cumsum()
    
    # Monthly Summary
    summary = []
    for m in range(1, 13):
        m_df = res[res['month'] == m]
        if m_df.empty:
            summary.append({'Month': m, 'PnL': 0, 'Trades': 0, 'PF': 0, 'Trades/Day': 0})
            continue
        days = m_df['date'].nunique()
        gp = m_df[m_df['pnl'] > 0]['pnl'].sum()
        gl = abs(m_df[m_df['pnl'] < 0]['pnl'].sum())
        pf = gp/gl if gl > 0 else 9.9
        summary.append({'Month': m, 'PnL': m_df['pnl'].sum(), 'Trades': len(m_df), 'PF': pf, 'Trades/Day': len(m_df)/days if days > 0 else 0})
    
    summary_df = pd.DataFrame(summary)
    print("\n--- TEST B: DOWNSIDE-ONLY PORTFOLIO ---")
    print(summary_df)
    print(f"\nAnnual PnL: {res['pnl'].sum():.2f}")
    print(f"Total Trades: {len(res)}")
    print(f"Avg Exp: {res['pnl'].sum() / len(res):.2f}")
    
    summary_df.to_csv("output/Phase15_TestB_Downside_Monthly.csv", index=False)
    res.to_csv("output/Phase15_TestB_All_Trades.csv", index=False)

if __name__ == "__main__":
    run_test_b()
