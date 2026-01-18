"""
PHASE 16B: PJ/ICT DELAYED ENTRY TEST
Tests T+1 and T+2 entries vs MOC baseline.

Purpose: Determine whether waiting reduces slippage/improves confirmation
         or destroys edge via missed impulse.

Everything IDENTICAL to 16A except entry timing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==============================================================================
# INLINED BACKTESTER (Friction: 0.5 pts)
# ==============================================================================
class PJBacktester:
    def __init__(self, df_ticks, df_bars):
        self.latency_ms = 500
        self.tick_times = df_ticks.index.values
        self.tick_bids = df_ticks['bid'].values if 'bid' in df_ticks.columns else df_ticks['price'].values
        self.tick_asks = df_ticks['ask'].values if 'ask' in df_ticks.columns else df_ticks['price'].values
        self.df_bars = df_bars  # For delayed entry bar lookup

    def execute_trade(self, entry_time, direction, sl_price, tp_price, slippage_pts=0.25, spread_pts=0.25):
        """Execute trade at given entry_time."""
        if isinstance(entry_time, pd.Timestamp):
            entry_time = entry_time.to_datetime64()
        actual_entry = entry_time + np.timedelta64(self.latency_ms, 'ms')
        idx = self.tick_times.searchsorted(actual_entry)
        if idx >= len(self.tick_times): return None
        
        penalty = (spread_pts / 2.0) + slippage_pts
        real_entry = self.tick_asks[idx] + penalty if direction == 1 else self.tick_bids[idx] - penalty
        
        # Safety: ensure SL is valid
        if direction == 1 and sl_price >= real_entry: sl_price = real_entry - 20
        if direction == -1 and sl_price <= real_entry: sl_price = real_entry + 20
        
        # Trade duration: max 4 hours
        end_time = actual_entry + np.timedelta64(4, 'h')
        end_idx = self.tick_times.searchsorted(end_time)
        
        prices = self.tick_bids[idx:end_idx] if direction == 1 else self.tick_asks[idx:end_idx]
        times = self.tick_times[idx:end_idx]
        
        for i in range(len(prices)):
            p, t = prices[i], times[i]
            if (direction == 1 and p <= sl_price) or (direction == -1 and p >= sl_price):
                exit_p = p - penalty if direction == 1 else p + penalty
                pnl = (exit_p - real_entry) if direction == 1 else (real_entry - exit_p)
                return {'pnl': pnl, 'exit': 'SL', 'entry_price': real_entry}
            if (direction == 1 and p >= tp_price) or (direction == -1 and p <= tp_price):
                pnl = (tp_price - real_entry) if direction == 1 else (real_entry - tp_price)
                return {'pnl': pnl, 'exit': 'TP', 'entry_price': real_entry}
        
        if len(prices) > 0:
            last_p = prices[-1]
            exit_p = last_p - penalty if direction == 1 else last_p + penalty
            pnl = (exit_p - real_entry) if direction == 1 else (real_entry - exit_p)
            return {'pnl': pnl, 'exit': 'TIME', 'entry_price': real_entry}
        return None

    def get_entry_time_delayed(self, signal_bar_time, delay_bars):
        """Get entry time for T+N delayed entry (open of N bars later)."""
        try:
            # Find the position of the signal bar in the bars dataframe
            bar_positions = self.df_bars.index.get_indexer([signal_bar_time])
            if len(bar_positions) > 0 and bar_positions[0] >= 0:
                bar_idx = bar_positions[0]
                if bar_idx + delay_bars < len(self.df_bars):
                    return self.df_bars.index[bar_idx + delay_bars]
        except:
            pass
        return None

# ==============================================================================
# POOL STATE TRACKER
# ==============================================================================
class PoolStateTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.pool_states = defaultdict(lambda: 'DEFINED')
    def can_trade(self, pool_id):
        return self.pool_states[pool_id] == 'DEFINED'
    def mark_traded(self, pool_id):
        self.pool_states[pool_id] = 'TRADED'

# ==============================================================================
# FEATURE ENGINEERING (Same as 16A)
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
    daily_hl = df.groupby('date').agg(DH=('high', 'max'), DL=('low', 'min')).shift(1)
    daily_hl.columns = ['PDH', 'PDL']
    
    df = df.merge(asia_hl, on='date', how='left').merge(london_hl, on='date', how='left')
    df = df.merge(overnight_hl, on='date', how='left').merge(lunch_hl, on='date', how='left')
    df = df.merge(daily_hl, on='date', how='left')
    return df

# ==============================================================================
# SIGNAL DETECTION (Same as 16A)
# ==============================================================================
def detect_pj_signals(df_bars, tracker, current_date):
    signals = []
    mask_trade = ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) | \
                 ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    trade_bars = df_bars[mask_trade & (df_bars['date'] == current_date)]
    
    POOLS = [
        ('PDH', 'PDL', 'PDH', 'PDL'),
        ('ONH', 'ONL', 'ONH', 'ONL'),
        ('ASIA_H', 'ASIA_L', 'ASIA_H', 'ASIA_L'),
        ('LON_H', 'LON_L', 'LON_H', 'LON_L'),
        ('LUNCH_H', 'LUNCH_L', 'LUNCH_H', 'LUNCH_L'),
    ]
    
    for idx, row in trade_bars.iterrows():
        for h_col, l_col, h_pool, l_pool in POOLS:
            h_level = row.get(h_col)
            l_level = row.get(l_col)
            if pd.isna(h_level) or pd.isna(l_level): continue
            
            # LONG
            if tracker.can_trade(l_pool):
                sweep = row['low'] < l_level
                reclaim = row['close'] > l_level
                if sweep and reclaim:
                    sl = row['low'] - 2.0
                    tp = h_level
                    signals.append((idx, row['time'], l_pool, 1, sl, tp))
                    tracker.mark_traded(l_pool)
            
            # SHORT
            if tracker.can_trade(h_pool):
                sweep = row['high'] > h_level
                reclaim = row['close'] < h_level
                if sweep and reclaim:
                    sl = row['high'] + 2.0
                    tp = l_level
                    signals.append((idx, row['time'], h_pool, -1, sl, tp))
                    tracker.mark_traded(h_pool)
    
    return signals

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
MAX_TRADES_PER_DAY = 5

def run_phase_16b():
    print("=" * 60)
    print("PHASE 16B: PJ/ICT DELAYED ENTRY TEST (T+1, T+2)")
    print("=" * 60)
    
    results = {'MOC': [], 'T+1': [], 'T+2': []}
    
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
        tester = PJBacktester(df_ticks, df_bars)
        
        dates = df_bars['date'].unique()
        
        for d in dates:
            # Fresh tracker per day, per variant (so signals are identical)
            tracker = PoolStateTracker()
            signals = detect_pj_signals(df_bars, tracker, d)
            
            daily_trades = 0
            for bar_idx, sig_time, pool_id, direction, sl, tp in signals:
                if daily_trades >= MAX_TRADES_PER_DAY:
                    break
                
                # MOC (same as 16A)
                result_moc = tester.execute_trade(sig_time, direction, sl, tp)
                if result_moc:
                    results['MOC'].append({
                        'month': m, 'pool': pool_id, 'direction': direction,
                        'pnl': result_moc['pnl'], 'exit': result_moc['exit']
                    })
                
                # Find bar position for delayed entries
                try:
                    # bar_idx is the pandas index (timestamp) of the signal bar
                    # Find its integer position in the original dataframe
                    bar_pos = df_bars.reset_index().index[df_bars.index == bar_idx].tolist()
                    if bar_pos:
                        pos = bar_pos[0]
                        
                        # T+1 Entry
                        if pos + 1 < len(df_bars):
                            t1_time = df_bars.iloc[pos + 1]['time']
                            result_t1 = tester.execute_trade(t1_time, direction, sl, tp)
                            if result_t1:
                                results['T+1'].append({
                                    'month': m, 'pool': pool_id, 'direction': direction,
                                    'pnl': result_t1['pnl'], 'exit': result_t1['exit']
                                })
                        
                        # T+2 Entry
                        if pos + 2 < len(df_bars):
                            t2_time = df_bars.iloc[pos + 2]['time']
                            result_t2 = tester.execute_trade(t2_time, direction, sl, tp)
                            if result_t2:
                                results['T+2'].append({
                                    'month': m, 'pool': pool_id, 'direction': direction,
                                    'pnl': result_t2['pnl'], 'exit': result_t2['exit']
                                })
                except Exception as e:
                    pass  # Skip on any lookup error
                
                daily_trades += 1
    
    # === ANALYSIS ===
    print("\n" + "=" * 60)
    print("PHASE 16B RESULTS: MOC vs T+1 vs T+2")
    print("=" * 60)
    
    comparison = []
    for variant in ['MOC', 'T+1', 'T+2']:
        df = pd.DataFrame(results[variant])
        if df.empty:
            continue
        total_pnl = df['pnl'].sum()
        total_trades = len(df)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        edge_cost = expectancy / 0.5 if expectancy > 0 else 0
        
        comparison.append({
            'Variant': variant,
            'PnL': total_pnl,
            'Trades': total_trades,
            'Expectancy': expectancy,
            'Edge/Cost': edge_cost
        })
        
        # Save individual results
        df.to_csv(f"output/Phase16B_{variant}_Trades.csv", index=False)
    
    comp_df = pd.DataFrame(comparison)
    print("\n--- ENTRY TIMING COMPARISON ---")
    print(comp_df.to_string(index=False))
    
    # Trade count decay (missed trades)
    moc_trades = comp_df[comp_df['Variant'] == 'MOC']['Trades'].values[0]
    for _, row in comp_df.iterrows():
        if row['Variant'] != 'MOC':
            decay = (moc_trades - row['Trades']) / moc_trades * 100
            print(f"{row['Variant']} Trade Decay: {decay:.1f}% missed")
    
    # Expectancy delta
    moc_exp = comp_df[comp_df['Variant'] == 'MOC']['Expectancy'].values[0]
    print(f"\n--- EXPECTANCY DELTA ---")
    for _, row in comp_df.iterrows():
        if row['Variant'] != 'MOC':
            delta = row['Expectancy'] - moc_exp
            pct = delta / moc_exp * 100 if moc_exp > 0 else 0
            print(f"{row['Variant']} vs MOC: ${delta:+.2f} ({pct:+.1f}%)")
    
    # === VERDICT ===
    print("\n--- VERDICT ---")
    best = comp_df.loc[comp_df['Expectancy'].idxmax()]
    print(f"Best Entry: {best['Variant']} (Exp: ${best['Expectancy']:.2f})")
    
    if best['Variant'] == 'MOC':
        print("→ Immediacy is OPTIMAL. Edge is front-loaded.")
        print("→ MOC is the canonical entry. Waiting degrades performance.")
    else:
        print(f"→ Delayed entry ({best['Variant']}) improves confirmation.")
        print("→ Worth testing 16C (Breaker Retest).")
    
    comp_df.to_csv("output/Phase16B_Comparison.csv", index=False)
    print("\n[DONE] Results saved to output/Phase16B_*.csv")

if __name__ == "__main__":
    run_phase_16b()
