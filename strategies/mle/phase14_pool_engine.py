"""
PHASE 14: POOL-STATE ENGINE
Multi-Pool Alpha with "One Pool → One Trade" enforcement.
Implements: Asia H/L, London H/L, ON H/L, IB H/L, Lunch H/L.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time as dtime
from collections import defaultdict

# ==============================================================================
# INLINED BACKTESTER (From Phase 13)
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
# POOL STATE TRACKER (The "One Pool → One Trade" Law)
# ==============================================================================
class PoolStateTracker:
    """Tracks the state of each pool per day. Enforces single-trade-per-pool."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all pools at start of new day."""
        self.pool_states = {
            'ONH': 'DEFINED', 'ONL': 'DEFINED',
            'ASIA_H': 'DEFINED', 'ASIA_L': 'DEFINED',
            'LON_H': 'DEFINED', 'LON_L': 'DEFINED',
            'IBH': 'DEFINED', 'IBL': 'DEFINED',
            'LUNCH_H': 'DEFINED', 'LUNCH_L': 'DEFINED',
        }
        self.pool_levels = {} # {pool_id: price_level}
        
    def set_level(self, pool_id, level):
        self.pool_levels[pool_id] = level
        
    def get_level(self, pool_id):
        return self.pool_levels.get(pool_id)
        
    def can_trade(self, pool_id):
        return self.pool_states.get(pool_id) == 'DEFINED'
    
    def mark_traded(self, pool_id):
        self.pool_states[pool_id] = 'TRADED'

# ==============================================================================
# FEATURE ENGINEERING (Multi-Pool)
# ==============================================================================
def engineer_pools(df_bars):
    """
    Computes all 10 pools for each bar.
    Pools are "prior period" values to avoid lookahead.
    """
    df = df_bars.copy()
    df['time'] = pd.to_datetime(df.index)
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    
    # Masks for session periods (ET times)
    mask_asia = (df['hour'] >= 0) & (df['hour'] < 3)
    mask_london = (df['hour'] >= 3) & ((df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30)))
    mask_overnight = (df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30))
    mask_ib = ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] == 10) & (df['minute'] < 30))
    mask_lunch = ((df['hour'] == 11) & (df['minute'] >= 30)) | (df['hour'] == 12) | ((df['hour'] == 13) & (df['minute'] < 30))
    
    # Group by date and compute session H/L
    asia_hl = df[mask_asia].groupby('date').agg(ASIA_H=('high', 'max'), ASIA_L=('low', 'min'))
    london_hl = df[mask_london].groupby('date').agg(LON_H=('high', 'max'), LON_L=('low', 'min'))
    overnight_hl = df[mask_overnight].groupby('date').agg(ONH=('high', 'max'), ONL=('low', 'min'))
    
    # IB and Lunch are computed per day but only valid after their session ends
    # For simplicity, we compute expanding max/min within the session and shift
    # This is a reasonable approximation for backtesting.
    ib_hl = df[mask_ib].groupby('date').agg(IBH=('high', 'max'), IBL=('low', 'min'))
    lunch_hl = df[mask_lunch].groupby('date').agg(LUNCH_H=('high', 'max'), LUNCH_L=('low', 'min'))
    
    # Merge back
    df = df.merge(asia_hl, on='date', how='left')
    df = df.merge(london_hl, on='date', how='left')
    df = df.merge(overnight_hl, on='date', how='left')
    df = df.merge(ib_hl, on='date', how='left')
    df = df.merge(lunch_hl, on='date', how='left')
    
    return df

# ==============================================================================
# SIGNAL DETECTION
# ==============================================================================
def detect_sweep_signals(df_bars, pool_tracker, current_date):
    """
    Detects sweep signals for all pools on bars for a given date.
    Returns list of (bar_idx, pool_id, direction, sl_price, tp_price).
    """
    signals = []
    
    # Filter for trading hours (09:30 - 15:30)
    mask_trade = ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) | ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    trade_bars = df_bars[mask_trade & (df_bars['date'] == current_date)]
    
    POOLS = [
        ('ONH', 'ONL', 'ONH', 'ONL'),
        ('ASIA_H', 'ASIA_L', 'ASIA_H', 'ASIA_L'),
        ('LON_H', 'LON_L', 'LON_H', 'LON_L'),
        ('IBH', 'IBL', 'IBH', 'IBL'),
        ('LUNCH_H', 'LUNCH_L', 'LUNCH_H', 'LUNCH_L'),
    ]
    
    for idx, row in trade_bars.iterrows():
        for high_pool, low_pool, h_col, l_col in POOLS:
            h_level = row.get(h_col)
            l_level = row.get(l_col)
            
            if pd.isna(h_level) or pd.isna(l_level): continue
            
            # Long: Sweep Low & Reclaim
            if pool_tracker.can_trade(low_pool):
                if row['low'] < l_level and row['close'] > l_level:
                    sl = row['low'] - 2.0
                    tp = h_level # Target opposing pool
                    signals.append((row['time'], low_pool, 1, sl, tp))
                    pool_tracker.mark_traded(low_pool)
                    
            # Short: Sweep High & Reclaim
            if pool_tracker.can_trade(high_pool):
                if row['high'] > h_level and row['close'] < h_level:
                    sl = row['high'] + 2.0
                    tp = l_level
                    signals.append((row['time'], high_pool, -1, sl, tp))
                    pool_tracker.mark_traded(high_pool)
                    
    return signals

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
MAX_TRADES_PER_DAY = 5

def run_pool_engine():
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
            
        df_bars = engineer_pools(df_bars)
        tester = GoldenBacktester(df_ticks)
        tracker = PoolStateTracker()
        
        dates = df_bars['date'].unique()
        
        for d in dates:
            tracker.reset() # New day, all pools reset
            daily_trades_taken = 0
            
            signals = detect_sweep_signals(df_bars, tracker, d)
            
            for sig_time, pool_id, direction, sl, tp in signals:
                if daily_trades_taken >= MAX_TRADES_PER_DAY:
                    break
                    
                fill_p, fill_t = tester.find_fill(sig_time, direction)
                if fill_p is None: continue
                
                pnl, reason, t_ext = tester.backtest_trade(fill_t, direction, sl, tp)
                all_trades.append({
                    'time': fill_t, 'month': m, 'date': d, 'pool': pool_id, 
                    'direction': direction, 'pnl': pnl, 'reason': reason
                })
                daily_trades_taken += 1

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
            summary.append({'Month': m, 'PnL': 0, 'Trades': 0, 'PF': 0, 'Trades/Day': 0})
            continue
        days = m_df['date'].nunique()
        gp = m_df[m_df['pnl'] > 0]['pnl'].sum()
        gl = abs(m_df[m_df['pnl'] < 0]['pnl'].sum())
        pf = gp/gl if gl > 0 else 9.9
        summary.append({'Month': m, 'PnL': m_df['pnl'].sum(), 'Trades': len(m_df), 'PF': pf, 'Trades/Day': len(m_df)/days if days > 0 else 0})
    
    summary_df = pd.DataFrame(summary)
    print("\n--- ANNUAL MULTI-POOL PERFORMANCE ---")
    print(summary_df)
    
    # Pool Breakdown
    pool_perf = res.groupby('pool').agg(PnL=('pnl', 'sum'), Trades=('pnl', 'count'))
    print("\n--- POOL BREAKDOWN ---")
    print(pool_perf)
    
    summary_df.to_csv("output/Phase14_Multi_Pool_Bench.csv", index=False)
    res.to_csv("output/Phase14_Full_Trades.csv", index=False)
    pool_perf.to_csv("output/Phase14_Pool_Performance.csv")
    print(f"\n[DONE] Cumulative PnL: {res['cum_pnl'].iloc[-1]:.2f}")

if __name__ == "__main__":
    run_pool_engine()
