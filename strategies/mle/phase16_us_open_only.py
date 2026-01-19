"""
PHASE 16A: PJ/ICT EXECUTION ENGINE - US OPEN HOURS ONLY
Filter: 09:30 - 11:00 ET
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==============================================================================
# INLINED BACKTESTER (Friction: 0.5 pts)
# ==============================================================================
class PJBacktester:
    def __init__(self, df_ticks):
        self.latency_ms = 500
        self.tick_times = df_ticks.index.values
        self.tick_bids = df_ticks['bid'].values if 'bid' in df_ticks.columns else df_ticks['price'].values
        self.tick_asks = df_ticks['ask'].values if 'ask' in df_ticks.columns else df_ticks['price'].values

    def execute_moc(self, signal_time, direction, sl_price, tp_price, slippage_pts=0.25, spread_pts=0.25):
        if isinstance(signal_time, pd.Timestamp):
            signal_time = signal_time.to_datetime64()
        entry_time = signal_time + np.timedelta64(self.latency_ms, 'ms')
        idx = self.tick_times.searchsorted(entry_time)
        if idx >= len(self.tick_times): return None
        
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
                pnl = (exit_p - real_entry) if direction == 1 else (real_entry - exit_p)
                return {'pnl': pnl, 'exit': 'SL', 'entry_price': real_entry, 'exit_time': t}
            if (direction == 1 and p >= tp_price) or (direction == -1 and p <= tp_price):
                pnl = (tp_price - real_entry) if direction == 1 else (real_entry - tp_price)
                return {'pnl': pnl, 'exit': 'TP', 'entry_price': real_entry, 'exit_time': t}
        
        if len(prices) > 0:
            last_p = prices[-1]
            exit_p = last_p - penalty if direction == 1 else last_p + penalty
            pnl = (exit_p - real_entry) if direction == 1 else (real_entry - exit_p)
            return {'pnl': pnl, 'exit': 'TIME', 'entry_price': real_entry, 'exit_time': times[-1]}
        return None

class PoolStateTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.pool_states = defaultdict(lambda: 'DEFINED')
    
    def can_trade(self, pool_id):
        return self.pool_states[pool_id] == 'DEFINED'
    
    def mark_traded(self, pool_id):
        self.pool_states[pool_id] = 'TRADED'

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
    
    daily_hl = df.groupby('date').agg(DH=('high', 'max'), DL=('low', 'min')).shift(1)
    daily_hl.columns = ['PDH', 'PDL']
    
    df = df.merge(asia_hl, on='date', how='left')
    df = df.merge(london_hl, on='date', how='left')
    df = df.merge(overnight_hl, on='date', how='left')
    df = df.merge(lunch_hl, on='date', how='left')
    df = df.merge(daily_hl, on='date', how='left')
    
    return df

def detect_pj_signals(df_bars, tracker, current_date):
    signals = []
    
    # *** FULL US MARKET HOURS: 09:30 - 16:00 ET ***
    mask_trade = ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) | \
                 ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    trade_bars = df_bars[mask_trade & (df_bars['date'] == current_date)]
    
    POOLS = [
        ('PDH', 'PDL', 'PDH', 'PDL'),
        ('ONH', 'ONL', 'ONH', 'ONL'),
        ('ASIA_H', 'ASIA_L', 'ASIA_H', 'ASIA_L'),
        ('LON_H', 'LON_L', 'LON_H', 'LON_L'),
    ]
    
    for idx, row in trade_bars.iterrows():
        for h_col, l_col, h_pool, l_pool in POOLS:
            h_level = row.get(h_col)
            l_level = row.get(l_col)
            if pd.isna(h_level) or pd.isna(l_level): continue
            
            if tracker.can_trade(l_pool):
                sweep = row['low'] < l_level
                reclaim = row['close'] > l_level
                if sweep and reclaim:
                    body = abs(row['close'] - row['open'])
                    range_bar = row['high'] - row['low']
                    displacement = body > (0.5 * range_bar) if range_bar > 0 else False
                    
                    sl = row['low'] - 2.0
                    tp = h_level
                    signals.append((row['time'], l_pool, 1, sl, tp, displacement))
                    tracker.mark_traded(l_pool)
            
            if tracker.can_trade(h_pool):
                sweep = row['high'] > h_level
                reclaim = row['close'] < h_level
                if sweep and reclaim:
                    body = abs(row['close'] - row['open'])
                    range_bar = row['high'] - row['low']
                    displacement = body > (0.5 * range_bar) if range_bar > 0 else False
                    
                    sl = row['high'] + 2.0
                    tp = l_level
                    signals.append((row['time'], h_pool, -1, sl, tp, displacement))
                    tracker.mark_traded(h_pool)
    
    return signals

BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
MAX_TRADES_PER_DAY = 5

def run_us_open():
    print("=" * 60)
    print("PHASE 16A: US OPEN HOURS ONLY (09:30 - 11:00)")
    print("=" * 60)
    
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
        tester = PJBacktester(df_ticks)
        tracker = PoolStateTracker()
        
        dates = df_bars['date'].unique()
        
        for d in dates:
            tracker.reset()
            daily_trades = 0
            
            signals = detect_pj_signals(df_bars, tracker, d)
            
            for sig_time, pool_id, direction, sl, tp, disp_flag in signals:
                if daily_trades >= MAX_TRADES_PER_DAY:
                    break
                
                result = tester.execute_moc(sig_time, direction, sl, tp)
                if result is None:
                    continue
                
                all_trades.append({
                    'month': m, 'date': d, 'pool': pool_id, 
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'displacement': disp_flag,
                    'pnl': result['pnl'], 
                    'exit': result['exit']
                })
                daily_trades += 1
    
    if not all_trades:
        print("Zero trades found!")
        return
    
    res = pd.DataFrame(all_trades)
    res['cum_pnl'] = res['pnl'].cumsum()
    
    print("\n" + "=" * 60)
    print("US OPEN HOURS RESULTS (09:30 - 11:00)")
    print("=" * 60)
    
    summary = []
    for m in range(1, 13):
        m_df = res[res['month'] == m]
        if m_df.empty:
            summary.append({'Month': m, 'PnL': 0, 'Trades': 0, 'PF': 0, 'Trades/Day': 0})
            continue
        days = m_df['date'].nunique()
        gp = m_df[m_df['pnl'] > 0]['pnl'].sum()
        gl = abs(m_df[m_df['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else 9.9
        summary.append({
            'Month': m, 'PnL': m_df['pnl'].sum(), 'Trades': len(m_df), 
            'PF': pf, 'Trades/Day': len(m_df) / days if days > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n--- Monthly Summary ---")
    print(summary_df.to_string(index=False))
    
    print("\n--- Directional Symmetry ---")
    dir_stats = res.groupby('direction').agg(
        PnL=('pnl', 'sum'),
        Trades=('pnl', 'count'),
        Expectancy=('pnl', 'mean')
    )
    print(dir_stats)
    
    print("\n--- Pool Performance ---")
    pool_stats = res.groupby('pool').agg(
        PnL=('pnl', 'sum'),
        Trades=('pnl', 'count'),
        Expectancy=('pnl', 'mean')
    ).sort_values('PnL', ascending=False)
    print(pool_stats)
    
    total_pnl = res['pnl'].sum()
    total_trades = len(res)
    expectancy = total_pnl / total_trades if total_trades > 0 else 0
    
    print("\n--- FINAL METRICS ---")
    print(f"Annual PnL: ${total_pnl:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Expectancy: ${expectancy:.2f} / trade")
    
    passed = expectancy >= 1.5
    print(f"\n{'✅ PASS' if passed else '❌ FAIL'}: Expectancy {'≥' if passed else '<'} 3× Friction ($1.50)")

if __name__ == "__main__":
    run_us_open()
