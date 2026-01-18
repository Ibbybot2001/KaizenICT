"""
PHASE 16A: OUT-OF-SAMPLE VALIDATION
Proper 70/30 Train/Test Split

Training Set: Jan-Aug (8 months = 67%)
Test Set: Sep-Dec (4 months = 33%)

The model is locked based on Training. Test set reveals true edge.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]  # Jan-Aug
TEST_MONTHS = [9, 10, 11, 12]  # Sep-Dec

# ==============================================================================
# INLINED BACKTESTER
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
                return {'pnl': pnl, 'exit': 'SL'}
            if (direction == 1 and p >= tp_price) or (direction == -1 and p <= tp_price):
                pnl = (tp_price - real_entry) if direction == 1 else (real_entry - tp_price)
                return {'pnl': pnl, 'exit': 'TP'}
        
        if len(prices) > 0:
            last_p = prices[-1]
            exit_p = last_p - penalty if direction == 1 else last_p + penalty
            pnl = (exit_p - real_entry) if direction == 1 else (real_entry - exit_p)
            return {'pnl': pnl, 'exit': 'TIME'}
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
    daily_hl = df.groupby('date').agg(DH=('high', 'max'), DL=('low', 'min')).shift(1)
    daily_hl.columns = ['PDH', 'PDL']
    
    df = df.merge(asia_hl, on='date', how='left').merge(london_hl, on='date', how='left')
    df = df.merge(overnight_hl, on='date', how='left').merge(lunch_hl, on='date', how='left')
    df = df.merge(daily_hl, on='date', how='left')
    return df

# ==============================================================================
# SIGNAL DETECTION
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
                    signals.append((row['time'], l_pool, 1, sl, tp))
                    tracker.mark_traded(l_pool)
            
            # SHORT
            if tracker.can_trade(h_pool):
                sweep = row['high'] > h_level
                reclaim = row['close'] < h_level
                if sweep and reclaim:
                    sl = row['high'] + 2.0
                    tp = l_level
                    signals.append((row['time'], h_pool, -1, sl, tp))
                    tracker.mark_traded(h_pool)
    
    return signals

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
MAX_TRADES_PER_DAY = 5

def run_validation():
    print("=" * 60)
    print("PHASE 16A: OUT-OF-SAMPLE VALIDATION (70/30 SPLIT)")
    print("=" * 60)
    print(f"Training: Months {TRAIN_MONTHS}")
    print(f"Testing:  Months {TEST_MONTHS}")
    print("=" * 60)
    
    train_trades = []
    test_trades = []
    
    for m in range(1, 13):
        m_str = f"{m:02d}"
        try:
            df_bars = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_1m.parquet")
            df_ticks = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_ticks.parquet")
            df_ticks.index = pd.to_datetime(df_ticks.index)
        except Exception as e:
            continue
        
        df_bars = engineer_pools(df_bars)
        tester = PJBacktester(df_ticks)
        tracker = PoolStateTracker()
        
        dates = df_bars['date'].unique()
        
        for d in dates:
            tracker.reset()
            daily_trades = 0
            
            signals = detect_pj_signals(df_bars, tracker, d)
            
            for sig_time, pool_id, direction, sl, tp in signals:
                if daily_trades >= MAX_TRADES_PER_DAY:
                    break
                
                result = tester.execute_moc(sig_time, direction, sl, tp)
                if result is None:
                    continue
                
                trade_data = {
                    'month': m, 'pool': pool_id, 
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'pnl': result['pnl'], 'exit': result['exit']
                }
                
                if m in TRAIN_MONTHS:
                    train_trades.append(trade_data)
                else:
                    test_trades.append(trade_data)
                
                daily_trades += 1
    
    # === ANALYSIS ===
    def calc_stats(trades, name):
        df = pd.DataFrame(trades)
        if df.empty:
            print(f"\n{name}: No trades")
            return None
        
        total_pnl = df['pnl'].sum()
        total_trades = len(df)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        
        gp = df[df['pnl'] > 0]['pnl'].sum()
        gl = abs(df[df['pnl'] < 0]['pnl'].sum())
        pf = gp / gl if gl > 0 else 9.9
        
        win_rate = len(df[df['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
        
        return {
            'Set': name,
            'PnL': total_pnl,
            'Trades': total_trades,
            'Expectancy': expectancy,
            'PF': pf,
            'Win%': win_rate,
            'Edge/Cost': expectancy / 0.5 if expectancy > 0 else 0
        }
    
    train_stats = calc_stats(train_trades, 'TRAINING (Jan-Aug)')
    test_stats = calc_stats(test_trades, 'TESTING (Sep-Dec)')
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    results = []
    if train_stats: results.append(train_stats)
    if test_stats: results.append(test_stats)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # === VERDICT ===
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if test_stats:
        test_pf = test_stats['PF']
        test_exp = test_stats['Expectancy']
        
        if test_pf >= 1.5 and test_exp >= 1.5:
            print("✅ PASS: Out-of-sample edge is REAL")
            print(f"   Test PF: {test_pf:.2f} (≥1.5)")
            print(f"   Test Exp: ${test_exp:.2f} (≥3x friction)")
        elif test_pf >= 1.3:
            print("⚠️ MARGINAL: Edge exists but weaker than training")
            print(f"   Test PF: {test_pf:.2f}")
        else:
            print("❌ FAIL: Edge does not survive out-of-sample")
            print(f"   Test PF: {test_pf:.2f}")
    
    results_df.to_csv("output/Phase16A_Validation_Split.csv", index=False)
    print("\n[DONE] Results saved to output/Phase16A_Validation_Split.csv")

if __name__ == "__main__":
    run_validation()
