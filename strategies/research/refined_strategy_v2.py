"""
Refined V-Shape Recovery Strategy with Scaling
===============================================
- SL placed beyond tagged liquidity (15-40 pt range)
- 50% out at TP1 (structure), 50% at TP2 (target)
- SL → BE after TP1 hit
- Conviction candle required for BOTH longs and shorts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import os

# Configuration
DATA_PATH = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
OUTPUT_DIR = r"C:\Users\CEO\ICT reinforcement\output\charts\refined_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy Parameters
SWING_LOOKBACK = 20
MIN_SWEEP_DEPTH = 5.0
RECOVERY_WINDOW = 15
RECOVERY_THRESHOLD = 0.50    # V-shape if recovery >50%
MIN_STOP = 15.0              # Minimum stop distance
MAX_STOP = 40.0              # Maximum stop distance
TP1_STRUCTURE = True         # Take 50% at structure
TP2_MULT = 1.5               # Remaining 50% at this R multiple


def load_data():
    """Load price data"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    return df


def find_swing_lows(df, lookback=20):
    """Find swing lows"""
    swing_lows = []
    for i in range(lookback, len(df) - lookback):
        window = df.iloc[i - lookback:i + lookback + 1]
        center_low = df.iloc[i]['low']
        if center_low == window['low'].min():
            swing_lows.append({'time': df.index[i], 'price': center_low, 'idx': i})
    return swing_lows


def find_nearest_structure(df, entry_idx, direction, lookback=30):
    """
    Find nearest structure for TP1
    - For longs: nearest swing high before entry
    - For shorts: nearest swing low before entry
    """
    search_start = max(0, entry_idx - lookback)
    pre_entry = df.iloc[search_start:entry_idx]
    
    if len(pre_entry) < 5:
        return None
    
    if direction == 'LONG':
        # Find swing highs
        for i in range(len(pre_entry) - 3, 2, -1):
            if (pre_entry.iloc[i]['high'] > pre_entry.iloc[i-1]['high'] and
                pre_entry.iloc[i]['high'] > pre_entry.iloc[i+1]['high']):
                return pre_entry.iloc[i]['high']
        return pre_entry['high'].max()
    else:
        # Find swing lows
        for i in range(len(pre_entry) - 3, 2, -1):
            if (pre_entry.iloc[i]['low'] < pre_entry.iloc[i-1]['low'] and
                pre_entry.iloc[i]['low'] < pre_entry.iloc[i+1]['low']):
                return pre_entry.iloc[i]['low']
        return pre_entry['low'].min()


def find_liquidity_stop(df, entry_idx, entry_price, direction, sweep_low=None, sweep_high=None):
    """
    Place stop beyond tagged liquidity, within 15-40 pt range
    """
    if direction == 'LONG':
        # For longs, stop below the sweep low (the tagged liquidity)
        if sweep_low:
            raw_stop = sweep_low - 2.0  # Just below tagged liquidity
        else:
            raw_stop = entry_price - 20.0
        
        distance = entry_price - raw_stop
        
        # Clamp to 15-40 range
        if distance < MIN_STOP:
            raw_stop = entry_price - MIN_STOP
        elif distance > MAX_STOP:
            raw_stop = entry_price - MAX_STOP
            
        return raw_stop
    else:
        # For shorts, stop above the recovery high (tagged liquidity)
        if sweep_high:
            raw_stop = sweep_high + 2.0
        else:
            raw_stop = entry_price + 20.0
        
        distance = raw_stop - entry_price
        
        if distance < MIN_STOP:
            raw_stop = entry_price + MIN_STOP
        elif distance > MAX_STOP:
            raw_stop = entry_price + MAX_STOP
            
        return raw_stop


def check_conviction_candle(df, setup_idx, direction):
    """
    Check for conviction candle after setup
    - LONG: need bullish candle (close > open) after setup
    - SHORT: need bearish candle (close < open) after setup
    """
    if setup_idx + 1 >= len(df):
        return False, None
    
    next_bar = df.iloc[setup_idx + 1]
    
    if direction == 'LONG':
        # Need bullish conviction
        if next_bar['close'] > next_bar['open']:
            return True, setup_idx + 1
    else:
        # Need bearish conviction
        if next_bar['close'] < next_bar['open']:
            return True, setup_idx + 1
    
    return False, None


def calculate_scaled_outcome(df, entry_idx, direction, entry_price, stop_price, tp1_price, tp2_price):
    """
    Calculate outcome with 50/50 scaling:
    - 50% out at TP1 (structure)
    - SL → BE after TP1
    - Remaining 50% targets TP2
    """
    max_bars = 80
    position_1 = 0.5  # First half
    position_2 = 0.5  # Second half
    
    tp1_hit = False
    total_r = 0
    exit_bar = 0
    result_detail = ""
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        bars_elapsed = i - entry_idx
        
        if direction == 'LONG':
            # Check stop first
            if not tp1_hit and bar['low'] <= stop_price:
                # Full stop loss
                total_r = -1.0  # Full 1R loss
                exit_bar = bars_elapsed
                result_detail = "FULL_STOP"
                break
            
            if tp1_hit and bar['low'] <= entry_price:
                # BE stop on second half
                # Already have +0.5R from TP1, now flat on second half
                exit_bar = bars_elapsed
                result_detail = "TP1_THEN_BE"
                break
            
            # Check TP1
            if not tp1_hit and bar['high'] >= tp1_price:
                tp1_hit = True
                r1 = (tp1_price - entry_price) / (entry_price - stop_price)
                total_r = position_1 * r1  # First 50% profit
            
            # Check TP2
            if tp1_hit and bar['high'] >= tp2_price:
                r2 = (tp2_price - entry_price) / (entry_price - stop_price)
                total_r += position_2 * r2  # Add second 50%
                exit_bar = bars_elapsed
                result_detail = "FULL_TARGET"
                break
                
        else:  # SHORT
            # Check stop first
            if not tp1_hit and bar['high'] >= stop_price:
                total_r = -1.0
                exit_bar = bars_elapsed
                result_detail = "FULL_STOP"
                break
            
            if tp1_hit and bar['high'] >= entry_price:
                exit_bar = bars_elapsed
                result_detail = "TP1_THEN_BE"
                break
            
            # Check TP1
            if not tp1_hit and bar['low'] <= tp1_price:
                tp1_hit = True
                r1 = (entry_price - tp1_price) / (stop_price - entry_price)
                total_r = position_1 * r1
            
            # Check TP2
            if tp1_hit and bar['low'] <= tp2_price:
                r2 = (entry_price - tp2_price) / (stop_price - entry_price)
                total_r += position_2 * r2
                exit_bar = bars_elapsed
                result_detail = "FULL_TARGET"
                break
    
    else:
        # Time exit
        exit_bar = max_bars
        final_price = df.iloc[min(entry_idx + max_bars - 1, len(df) - 1)]['close']
        if direction == 'LONG':
            if tp1_hit:
                r2 = (final_price - entry_price) / (entry_price - stop_price)
                total_r += position_2 * max(r2, 0)  # Can't go below BE
            else:
                total_r = (final_price - entry_price) / (entry_price - stop_price)
        else:
            if tp1_hit:
                r2 = (entry_price - final_price) / (stop_price - entry_price)
                total_r += position_2 * max(r2, 0)
            else:
                total_r = (entry_price - final_price) / (stop_price - entry_price)
        result_detail = "TIME_EXIT"
    
    outcome = 'WIN' if total_r > 0 else 'LOSS'
    
    return {
        'outcome': outcome,
        'total_r': total_r,
        'tp1_hit': tp1_hit,
        'exit_bar': exit_bar,
        'detail': result_detail
    }


def detect_setup(df, swing_low_idx, swing_low_price):
    """Detect V-shape or continuation setup with conviction filter"""
    
    search_start = swing_low_idx + 1
    search_end = min(swing_low_idx + 100, len(df))
    
    for i in range(search_start, search_end):
        bar = df.iloc[i]
        
        # Check for sweep
        if bar['low'] < swing_low_price - MIN_SWEEP_DEPTH:
            sweep_low = bar['low']
            sweep_depth = swing_low_price - sweep_low
            sweep_idx = i
            sweep_time = df.index[i]
            
            # Wait for close back above swing low
            for j in range(i + 1, min(i + 30, len(df))):
                if df.iloc[j]['close'] > swing_low_price:
                    # Sweep complete, measure recovery
                    recovery_start = j
                    
                    # Find drop range
                    pre_sweep = df.iloc[max(0, i-20):i+1]
                    recent_high = pre_sweep['high'].max()
                    drop_size = recent_high - sweep_low
                    
                    if drop_size < 10:
                        break
                    
                    # Measure recovery
                    recovery_end = min(j + RECOVERY_WINDOW, len(df))
                    recovery_window = df.iloc[j:recovery_end]
                    
                    if len(recovery_window) < 5:
                        break
                    
                    recovery_high = recovery_window['high'].max()
                    recovery_amount = recovery_high - sweep_low
                    recovery_ratio = recovery_amount / drop_size if drop_size > 0 else 0
                    
                    # Determine direction
                    if recovery_ratio >= RECOVERY_THRESHOLD:
                        direction = 'LONG'
                        setup_idx = j
                    else:
                        direction = 'SHORT'
                        setup_idx = j + len(recovery_window) - 1  # End of weak recovery
                    
                    # Check conviction candle
                    has_conviction, entry_idx = check_conviction_candle(df, setup_idx, direction)
                    
                    if not has_conviction:
                        break
                    
                    entry_bar = df.iloc[entry_idx]
                    entry_price = entry_bar['close']
                    entry_time = df.index[entry_idx]
                    
                    # Calculate stops based on tagged liquidity
                    if direction == 'LONG':
                        stop_price = find_liquidity_stop(df, entry_idx, entry_price, 'LONG', 
                                                         sweep_low=sweep_low)
                        tp1_price = find_nearest_structure(df, entry_idx, 'LONG')
                        if tp1_price is None or tp1_price <= entry_price:
                            tp1_price = entry_price + (entry_price - stop_price) * 0.8
                        stop_distance = entry_price - stop_price
                        tp2_price = entry_price + stop_distance * TP2_MULT
                    else:
                        recovery_high_for_stop = recovery_window['high'].max()
                        stop_price = find_liquidity_stop(df, entry_idx, entry_price, 'SHORT',
                                                         sweep_high=recovery_high_for_stop)
                        tp1_price = find_nearest_structure(df, entry_idx, 'SHORT')
                        if tp1_price is None or tp1_price >= entry_price:
                            tp1_price = entry_price - (stop_price - entry_price) * 0.8
                        stop_distance = stop_price - entry_price
                        tp2_price = entry_price - stop_distance * TP2_MULT
                    
                    # Calculate outcome
                    result = calculate_scaled_outcome(df, entry_idx, direction, 
                                                      entry_price, stop_price, tp1_price, tp2_price)
                    
                    return {
                        'swing_low_price': swing_low_price,
                        'swing_low_idx': swing_low_idx,
                        'sweep_time': sweep_time,
                        'sweep_low': sweep_low,
                        'sweep_depth': sweep_depth,
                        'recovery_ratio': recovery_ratio,
                        'drop_size': drop_size,
                        'direction': direction,
                        'entry_idx': entry_idx,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'stop_distance': abs(entry_price - stop_price),
                        'tp1_price': tp1_price,
                        'tp2_price': tp2_price,
                        'outcome': result['outcome'],
                        'total_r': result['total_r'],
                        'tp1_hit': result['tp1_hit'],
                        'exit_bar': result['exit_bar'],
                        'detail': result['detail']
                    }
                    
            break
    
    return None


def run_backtest(df, max_trades=300):
    """Run backtest"""
    print("\nFinding swing lows...")
    swing_lows = find_swing_lows(df, SWING_LOOKBACK)
    print(f"Found {len(swing_lows)} swing lows")
    
    trades = []
    last_trade_idx = 0
    
    print("\nScanning for setups...")
    for sl in swing_lows:
        if sl['idx'] < last_trade_idx + 60:
            continue
            
        result = detect_setup(df, sl['idx'], sl['price'])
        
        if result:
            trades.append(result)
            last_trade_idx = result['entry_idx']
            
            if len(trades) % 50 == 0:
                print(f"  Found {len(trades)} trades...")
            
            if len(trades) >= max_trades:
                break
    
    return trades


def plot_trade(df, trade, idx, output_dir):
    """Create trade chart"""
    bars_before = 50
    bars_after = 80
    
    start_idx = max(0, trade['swing_low_idx'] - bars_before)
    end_idx = min(len(df), trade['entry_idx'] + bars_after)
    
    window = df.iloc[start_idx:end_idx].copy()
    
    if len(window) < 30:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    
    bg_color = '#1a1a2e'
    bull_color = '#26a69a'
    bear_color = '#ef5350'
    text_color = '#e0e0e0'
    
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot candles
    for i, (t, row) in enumerate(window.iterrows()):
        color = bull_color if row['close'] >= row['open'] else bear_color
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        body = min(row['open'], row['close'])
        height = max(abs(row['close'] - row['open']), 0.25)
        rect = plt.Rectangle((i - 0.35, body), 0.7, height, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    
    time_to_x = {t: i for i, t in enumerate(window.index)}
    
    # Swing low
    ax.axhline(y=trade['swing_low_price'], color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(2, trade['swing_low_price'] + 2, f"SWING LOW: {trade['swing_low_price']:.2f}", 
            color='#FFD700', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#FFD700', alpha=0.9))
    
    # Entry
    entry_x = time_to_x.get(trade['entry_time'], 40)
    entry_color = '#00FF7F' if trade['direction'] == 'LONG' else '#FF6B6B'
    ax.annotate(f"ENTRY ({trade['direction']})", xy=(entry_x, trade['entry_price']), 
                xytext=(entry_x + 3, trade['entry_price'] + 5),
                fontsize=11, fontweight='bold', color=entry_color, ha='left',
                arrowprops=dict(arrowstyle='->', color=entry_color, lw=2))
    ax.axhline(y=trade['entry_price'], color=entry_color, linestyle='-', linewidth=1.5, alpha=0.6)
    
    # Stop
    ax.axhline(y=trade['stop_price'], color='#FF4444', linestyle='--', linewidth=2)
    ax.text(len(window) - 10, trade['stop_price'], 
            f"STOP: {trade['stop_price']:.2f} ({trade['stop_distance']:.1f} pts)", 
            color='#FF4444', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#FF4444', alpha=0.9))
    
    # TP1 (structure)
    ax.axhline(y=trade['tp1_price'], color='#00BFFF', linestyle='-.', linewidth=2)
    ax.text(len(window) - 10, trade['tp1_price'], f"TP1 (50%): {trade['tp1_price']:.2f}", 
            color='#00BFFF', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#00BFFF', alpha=0.9))
    
    # TP2 (target)
    ax.axhline(y=trade['tp2_price'], color='#00FF7F', linestyle='--', linewidth=2)
    ax.text(len(window) - 10, trade['tp2_price'], f"TP2 (50%): {trade['tp2_price']:.2f}", 
            color='#00FF7F', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#00FF7F', alpha=0.9))
    
    # Title
    result_color = '#00FF7F' if trade['outcome'] == 'WIN' else '#FF4444'
    title = (f"{trade['direction']} | Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}\n"
             f"Recovery: {trade['recovery_ratio']*100:.0f}% | Stop: {trade['stop_distance']:.1f} pts | "
             f"Result: {trade['detail']} ({trade['total_r']:+.2f}R)")
    ax.set_title(title, color=result_color, fontsize=13, fontweight='bold', pad=20)
    
    # Style
    ax.set_xlim(-1, len(window))
    ax.set_ylim(window['low'].min() - 10, window['high'].max() + 10)
    ax.tick_params(colors=text_color)
    ax.grid(True, color='#2a2a4a', alpha=0.3)
    
    # Analysis box
    analysis = (
        f"TRADE ANALYSIS\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Direction: {trade['direction']}\n"
        f"Recovery: {trade['recovery_ratio']*100:.0f}%\n"
        f"Stop: {trade['stop_distance']:.1f} pts\n"
        f"TP1 Hit: {'Yes' if trade['tp1_hit'] else 'No'}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Result: {trade['detail']}\n"
        f"Total P&L: {trade['total_r']:+.2f}R\n"
        f"Bars: {trade['exit_bar']}"
    )
    ax.text(0.98, 0.02, analysis, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#4a4a6a'),
            color=text_color, family='monospace')
    
    plt.tight_layout()
    
    tag = 'win' if trade['outcome'] == 'WIN' else 'loss'
    filename = f"{tag}_{trade['direction']}_{idx}_{trade['entry_time'].strftime('%Y%m%d_%H%M')}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    
    return path


def main():
    df = load_data()
    print(f"Loaded {len(df)} bars")
    
    trades = run_backtest(df, max_trades=300)
    
    print(f"\n{'='*60}")
    print(f"REFINED STRATEGY BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {len(trades)}")
    
    if not trades:
        print("No trades found!")
        return
    
    trade_df = pd.DataFrame(trades)
    
    # Summary by direction
    for direction in ['LONG', 'SHORT']:
        subset = trade_df[trade_df['direction'] == direction]
        if len(subset) == 0:
            continue
            
        wins = (subset['outcome'] == 'WIN').sum()
        total_r = subset['total_r'].sum()
        win_rate = wins / len(subset) * 100
        tp1_rate = subset['tp1_hit'].sum() / len(subset) * 100
        avg_stop = subset['stop_distance'].mean()
        
        print(f"\n{direction}:")
        print(f"  Trades: {len(subset)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  TP1 Hit Rate: {tp1_rate:.1f}%")
        print(f"  Avg Stop: {avg_stop:.1f} pts")
        print(f"  Total R: {total_r:+.2f}")
        
        # Detail breakdown
        for detail in subset['detail'].unique():
            count = (subset['detail'] == detail).sum()
            print(f"    - {detail}: {count}")
    
    # Overall
    total_r = trade_df['total_r'].sum()
    win_rate = (trade_df['outcome'] == 'WIN').sum() / len(trade_df) * 100
    tp1_rate = trade_df['tp1_hit'].sum() / len(trade_df) * 100
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {len(trades)} trades | Win Rate: {win_rate:.1f}% | TP1 Rate: {tp1_rate:.1f}%")
    print(f"TOTAL P&L: {total_r:+.2f}R")
    print(f"{'='*60}")
    
    # Generate charts - 3 wins, 3 losses per direction
    print("\nGenerating example charts...")
    
    for direction in ['LONG', 'SHORT']:
        subset = trade_df[trade_df['direction'] == direction]
        
        wins = subset[subset['outcome'] == 'WIN'].head(3)
        losses = subset[subset['outcome'] == 'LOSS'].head(3)
        
        for idx, (_, trade) in enumerate(wins.iterrows(), 1):
            path = plot_trade(df, trade, f"{direction}_win_{idx}", OUTPUT_DIR)
            if path:
                print(f"  Saved: {path}")
        
        for idx, (_, trade) in enumerate(losses.iterrows(), 1):
            path = plot_trade(df, trade, f"{direction}_loss_{idx}", OUTPUT_DIR)
            if path:
                print(f"  Saved: {path}")
    
    print(f"\nCharts saved to: {OUTPUT_DIR}")
    
    return trade_df


if __name__ == "__main__":
    main()
