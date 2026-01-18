"""
Turtle Soup Strategy v3
=======================
Key Rules:
1. TREND FILTER (Option C): 
   - LONGS: Recovery must close ABOVE nearest swing high resistance
   - SHORTS: Breakdown must close BELOW nearest swing low support
   
2. TURTLE SOUP ENTRY:
   - SHORTS: Enter on bearish candle close BELOW order block
   - LONGS: Enter on bullish candle close ABOVE order block
   
3. TP1 at PRIOR MAJOR swing level (not just nearest structure)

4. 50/50 scaling, SL to BE after TP1, stops beyond tagged liquidity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration
DATA_PATH = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
OUTPUT_DIR = r"C:\Users\CEO\ICT reinforcement\output\charts\turtle_soup_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy Parameters
SWING_LOOKBACK = 20
MIN_SWEEP_DEPTH = 5.0
RECOVERY_WINDOW = 15
RECOVERY_THRESHOLD = 0.50
MIN_STOP = 15.0
MAX_STOP = 40.0
TP2_MULT = 1.5


def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    return df


def find_swing_points(df, idx, lookback=50):
    """Find swing highs and lows before entry point"""
    start = max(0, idx - lookback)
    window = df.iloc[start:idx]
    
    if len(window) < 10:
        return [], []
    
    swing_highs = []
    swing_lows = []
    
    for i in range(3, len(window) - 3):
        # Swing high
        if (window.iloc[i]['high'] > window.iloc[i-1]['high'] and
            window.iloc[i]['high'] > window.iloc[i-2]['high'] and
            window.iloc[i]['high'] > window.iloc[i+1]['high'] and
            window.iloc[i]['high'] > window.iloc[i+2]['high']):
            swing_highs.append({
                'price': window.iloc[i]['high'],
                'idx': start + i,
                'time': window.index[i]
            })
        
        # Swing low
        if (window.iloc[i]['low'] < window.iloc[i-1]['low'] and
            window.iloc[i]['low'] < window.iloc[i-2]['low'] and
            window.iloc[i]['low'] < window.iloc[i+1]['low'] and
            window.iloc[i]['low'] < window.iloc[i+2]['low']):
            swing_lows.append({
                'price': window.iloc[i]['low'],
                'idx': start + i,
                'time': window.index[i]
            })
    
    return swing_highs, swing_lows


def find_order_block(df, sweep_idx, direction, lookback=15):
    """
    Find the order block before the sweep
    - For LONG: Last bearish candle before the drop (bullish OB)
    - For SHORT: Last bullish candle before the rally (bearish OB)
    """
    start = max(0, sweep_idx - lookback)
    window = df.iloc[start:sweep_idx]
    
    if direction == 'LONG':
        # Find last significant bearish candle (the bullish OB to reclaim)
        for i in range(len(window) - 1, -1, -1):
            bar = window.iloc[i]
            if bar['close'] < bar['open']:  # Bearish
                body_size = bar['open'] - bar['close']
                if body_size > 2.0:  # Significant
                    return {
                        'high': bar['high'],
                        'low': bar['low'],
                        'idx': start + i
                    }
    else:
        # Find last significant bullish candle (the bearish OB to break)
        for i in range(len(window) - 1, -1, -1):
            bar = window.iloc[i]
            if bar['close'] > bar['open']:  # Bullish
                body_size = bar['close'] - bar['open']
                if body_size > 2.0:
                    return {
                        'high': bar['high'],
                        'low': bar['low'],
                        'idx': start + i
                    }
    
    return None


def check_trend_filter(df, entry_idx, entry_price, direction, swing_highs, swing_lows):
    """
    Option C: Recovery must close above/below key resistance/support
    - LONGS: Entry price > nearest swing high resistance
    - SHORTS: Entry price < nearest swing low support
    """
    if direction == 'LONG':
        # Find nearest swing high before entry
        recent_highs = [sh for sh in swing_highs if sh['idx'] < entry_idx]
        if not recent_highs:
            return True  # No resistance, allow
        
        nearest_resistance = max(recent_highs, key=lambda x: x['idx'])
        
        # Entry must close ABOVE this resistance
        if entry_price > nearest_resistance['price']:
            return True
        return False
        
    else:  # SHORT
        # Find nearest swing low before entry
        recent_lows = [sl for sl in swing_lows if sl['idx'] < entry_idx]
        if not recent_lows:
            return True
        
        nearest_support = max(recent_lows, key=lambda x: x['idx'])
        
        # Entry must close BELOW this support
        if entry_price < nearest_support['price']:
            return True
        return False


def check_turtle_soup_entry(df, setup_idx, direction, order_block):
    """
    Turtle soup entry: Wait for close below/above order block
    """
    if order_block is None:
        return False, None
    
    # Look for confirmation candle after setup
    for i in range(setup_idx + 1, min(setup_idx + 10, len(df))):
        bar = df.iloc[i]
        
        if direction == 'LONG':
            # Need bullish close above OB high
            if bar['close'] > bar['open'] and bar['close'] > order_block['high']:
                return True, i
        else:
            # Need bearish close below OB low
            if bar['close'] < bar['open'] and bar['close'] < order_block['low']:
                return True, i
    
    return False, None


def find_major_swing_for_tp1(swing_points, entry_price, direction):
    """
    Find prior MAJOR swing level for TP1
    Use the second swing, not just the nearest one
    """
    if direction == 'LONG':
        # For long, TP1 at prior swing high
        valid_highs = [sh for sh in swing_points if sh['price'] > entry_price]
        valid_highs.sort(key=lambda x: x['price'])
        
        if len(valid_highs) >= 2:
            return valid_highs[1]['price']  # Second swing high
        elif len(valid_highs) == 1:
            return valid_highs[0]['price']
        return None
    else:
        # For short, TP1 at prior swing low
        valid_lows = [sl for sl in swing_points if sl['price'] < entry_price]
        valid_lows.sort(key=lambda x: x['price'], reverse=True)
        
        if len(valid_lows) >= 2:
            return valid_lows[1]['price']  # Second swing low
        elif len(valid_lows) == 1:
            return valid_lows[0]['price']
        return None


def calculate_outcome(df, entry_idx, direction, entry_price, stop_price, tp1_price, tp2_price):
    """50/50 scaling with SL to BE after TP1"""
    max_bars = 80
    tp1_hit = False
    total_r = 0
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        bars_elapsed = i - entry_idx
        
        if direction == 'LONG':
            if not tp1_hit and bar['low'] <= stop_price:
                return {'outcome': 'LOSS', 'total_r': -1.0, 'tp1_hit': False, 
                        'exit_bar': bars_elapsed, 'detail': 'FULL_STOP'}
            
            if tp1_hit and bar['low'] <= entry_price:
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars_elapsed, 'detail': 'TP1_THEN_BE'}
            
            if not tp1_hit and bar['high'] >= tp1_price:
                tp1_hit = True
                r1 = (tp1_price - entry_price) / (entry_price - stop_price)
                total_r = 0.5 * r1
            
            if tp1_hit and bar['high'] >= tp2_price:
                r2 = (tp2_price - entry_price) / (entry_price - stop_price)
                total_r += 0.5 * r2
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars_elapsed, 'detail': 'FULL_TARGET'}
        else:
            if not tp1_hit and bar['high'] >= stop_price:
                return {'outcome': 'LOSS', 'total_r': -1.0, 'tp1_hit': False,
                        'exit_bar': bars_elapsed, 'detail': 'FULL_STOP'}
            
            if tp1_hit and bar['high'] >= entry_price:
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars_elapsed, 'detail': 'TP1_THEN_BE'}
            
            if not tp1_hit and bar['low'] <= tp1_price:
                tp1_hit = True
                r1 = (entry_price - tp1_price) / (stop_price - entry_price)
                total_r = 0.5 * r1
            
            if tp1_hit and bar['low'] <= tp2_price:
                r2 = (entry_price - tp2_price) / (stop_price - entry_price)
                total_r += 0.5 * r2
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars_elapsed, 'detail': 'FULL_TARGET'}
    
    return {'outcome': 'LOSS' if total_r <= 0 else 'WIN', 'total_r': total_r,
            'tp1_hit': tp1_hit, 'exit_bar': max_bars, 'detail': 'TIME_EXIT'}


def detect_setup(df, swing_low_idx, swing_low_price):
    """Full setup detection with all filters"""
    
    for i in range(swing_low_idx + 1, min(swing_low_idx + 100, len(df))):
        bar = df.iloc[i]
        
        if bar['low'] < swing_low_price - MIN_SWEEP_DEPTH:
            sweep_low = bar['low']
            sweep_depth = swing_low_price - sweep_low
            sweep_idx = i
            
            # Wait for close back above swing low
            for j in range(i + 1, min(i + 30, len(df))):
                if df.iloc[j]['close'] > swing_low_price:
                    # Measure recovery
                    pre_sweep = df.iloc[max(0, i-20):i+1]
                    recent_high = pre_sweep['high'].max()
                    drop_size = recent_high - sweep_low
                    
                    if drop_size < 10:
                        break
                    
                    recovery_end = min(j + RECOVERY_WINDOW, len(df))
                    recovery_window = df.iloc[j:recovery_end]
                    
                    if len(recovery_window) < 5:
                        break
                    
                    recovery_high = recovery_window['high'].max()
                    recovery_ratio = (recovery_high - sweep_low) / drop_size
                    
                    # Determine direction
                    if recovery_ratio >= RECOVERY_THRESHOLD:
                        direction = 'LONG'
                    else:
                        direction = 'SHORT'
                    
                    # Get swing points for filtering
                    swing_highs, swing_lows = find_swing_points(df, j, lookback=60)
                    
                    # Find order block
                    order_block = find_order_block(df, sweep_idx, direction)
                    
                    # Check turtle soup entry
                    setup_idx = j if direction == 'LONG' else j + len(recovery_window) - 1
                    has_entry, entry_idx = check_turtle_soup_entry(df, setup_idx, direction, order_block)
                    
                    if not has_entry:
                        break
                    
                    entry_bar = df.iloc[entry_idx]
                    entry_price = entry_bar['close']
                    entry_time = df.index[entry_idx]
                    
                    # Check trend filter (Option C)
                    if not check_trend_filter(df, entry_idx, entry_price, direction, swing_highs, swing_lows):
                        break
                    
                    # Calculate stop
                    if direction == 'LONG':
                        raw_stop = sweep_low - 2.0
                        stop_distance = entry_price - raw_stop
                        stop_distance = max(MIN_STOP, min(MAX_STOP, stop_distance))
                        stop_price = entry_price - stop_distance
                        
                        tp1_price = find_major_swing_for_tp1(swing_highs, entry_price, 'LONG')
                        if tp1_price is None:
                            tp1_price = entry_price + stop_distance * 0.8
                        tp2_price = entry_price + stop_distance * TP2_MULT
                    else:
                        raw_stop = recovery_high + 2.0
                        stop_distance = raw_stop - entry_price
                        stop_distance = max(MIN_STOP, min(MAX_STOP, stop_distance))
                        stop_price = entry_price + stop_distance
                        
                        tp1_price = find_major_swing_for_tp1(swing_lows, entry_price, 'SHORT')
                        if tp1_price is None:
                            tp1_price = entry_price - stop_distance * 0.8
                        tp2_price = entry_price - stop_distance * TP2_MULT
                    
                    result = calculate_outcome(df, entry_idx, direction, 
                                               entry_price, stop_price, tp1_price, tp2_price)
                    
                    return {
                        'swing_low_price': swing_low_price,
                        'swing_low_idx': swing_low_idx,
                        'sweep_low': sweep_low,
                        'sweep_depth': sweep_depth,
                        'recovery_ratio': recovery_ratio,
                        'direction': direction,
                        'entry_idx': entry_idx,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'stop_distance': stop_distance,
                        'tp1_price': tp1_price,
                        'tp2_price': tp2_price,
                        'order_block': order_block,
                        **result
                    }
                    
            break
    
    return None


def run_backtest(df, max_trades=300):
    print("\nFinding swing lows...")
    swing_lows = []
    for i in range(SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
        window = df.iloc[i - SWING_LOOKBACK:i + SWING_LOOKBACK + 1]
        if df.iloc[i]['low'] == window['low'].min():
            swing_lows.append({'time': df.index[i], 'price': df.iloc[i]['low'], 'idx': i})
    
    print(f"Found {len(swing_lows)} swing lows")
    
    trades = []
    last_idx = 0
    
    print("\nScanning for setups...")
    for sl in swing_lows:
        if sl['idx'] < last_idx + 60:
            continue
        
        result = detect_setup(df, sl['idx'], sl['price'])
        
        if result:
            trades.append(result)
            last_idx = result['entry_idx']
            
            if len(trades) % 25 == 0:
                print(f"  Found {len(trades)} trades...")
            
            if len(trades) >= max_trades:
                break
    
    return trades


def plot_trade(df, trade, idx, output_dir):
    bars_before = 60
    bars_after = 80
    
    start_idx = max(0, trade['swing_low_idx'] - bars_before)
    end_idx = min(len(df), trade['entry_idx'] + bars_after)
    window = df.iloc[start_idx:end_idx].copy()
    
    if len(window) < 40:
        return None
    
    fig, ax = plt.subplots(figsize=(18, 11))
    
    bg = '#1a1a2e'
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    
    for i, (t, row) in enumerate(window.iterrows()):
        c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([i, i], [row['low'], row['high']], color=c, lw=1)
        b = min(row['open'], row['close'])
        h = max(abs(row['close'] - row['open']), 0.25)
        ax.add_patch(plt.Rectangle((i-0.35, b), 0.7, h, fc=c, ec=c))
    
    time_to_x = {t: i for i, t in enumerate(window.index)}
    
    # Levels
    ax.axhline(trade['swing_low_price'], color='#FFD700', ls='--', lw=2, alpha=0.8)
    ax.axhline(trade['entry_price'], color='#00FF7F' if trade['direction']=='LONG' else '#FF6B6B', lw=1.5, alpha=0.6)
    ax.axhline(trade['stop_price'], color='#FF4444', ls='--', lw=2)
    ax.axhline(trade['tp1_price'], color='#00BFFF', ls='-.', lw=2)
    ax.axhline(trade['tp2_price'], color='#00FF7F', ls='--', lw=2)
    
    # Order block
    if trade['order_block']:
        ob = trade['order_block']
        ob_x = time_to_x.get(df.index[ob['idx']], 0) if ob['idx'] < len(df) else 0
        ax.add_patch(plt.Rectangle((ob_x, ob['low']), 10, ob['high']-ob['low'],
                                    fc='#9C27B0', alpha=0.3, ec='#9C27B0'))
    
    # Entry marker
    entry_x = time_to_x.get(trade['entry_time'], 50)
    ax.annotate(f"ENTRY ({trade['direction']})", xy=(entry_x, trade['entry_price']),
                xytext=(entry_x+3, trade['entry_price']+5), fontsize=11, fontweight='bold',
                color='#00FF7F' if trade['direction']=='LONG' else '#FF6B6B',
                arrowprops=dict(arrowstyle='->', lw=2, color='#00FF7F' if trade['direction']=='LONG' else '#FF6B6B'))
    
    # Labels
    ax.text(len(window)-8, trade['stop_price'], f"STOP: {trade['stop_distance']:.1f}pts", 
            color='#FF4444', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', fc=bg, ec='#FF4444'))
    ax.text(len(window)-8, trade['tp1_price'], f"TP1 (50%)", color='#00BFFF', fontsize=9, fontweight='bold')
    ax.text(len(window)-8, trade['tp2_price'], f"TP2 (50%)", color='#00FF7F', fontsize=9, fontweight='bold')
    
    result_color = '#00FF7F' if trade['outcome'] == 'WIN' else '#FF4444'
    title = (f"{trade['direction']} | {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}\n"
             f"Recovery: {trade['recovery_ratio']*100:.0f}% | "
             f"Result: {trade['detail']} ({trade['total_r']:+.2f}R)")
    ax.set_title(title, color=result_color, fontsize=13, fontweight='bold', pad=20)
    
    ax.set_xlim(-1, len(window))
    ax.set_ylim(window['low'].min()-10, window['high'].max()+10)
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, color='#2a2a4a', alpha=0.3)
    
    plt.tight_layout()
    
    tag = 'win' if trade['outcome'] == 'WIN' else 'loss'
    path = os.path.join(output_dir, f"{tag}_{trade['direction']}_{idx}_{trade['entry_time'].strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(path, dpi=150, facecolor=bg, bbox_inches='tight')
    plt.close()
    
    return path


def main():
    df = load_data()
    print(f"Loaded {len(df)} bars")
    
    trades = run_backtest(df, max_trades=300)
    
    print(f"\n{'='*60}")
    print(f"TURTLE SOUP STRATEGY v3 RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {len(trades)}")
    
    if not trades:
        print("No trades found!")
        return
    
    trade_df = pd.DataFrame(trades)
    
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
        
        for detail in subset['detail'].unique():
            count = (subset['detail'] == detail).sum()
            print(f"    - {detail}: {count}")
    
    total_r = trade_df['total_r'].sum()
    win_rate = (trade_df['outcome'] == 'WIN').sum() / len(trade_df) * 100
    tp1_rate = trade_df['tp1_hit'].sum() / len(trade_df) * 100
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {len(trades)} trades | Win Rate: {win_rate:.1f}% | TP1 Rate: {tp1_rate:.1f}%")
    print(f"TOTAL P&L: {total_r:+.2f}R")
    print(f"{'='*60}")
    
    # Charts
    print("\nGenerating charts...")
    for direction in ['LONG', 'SHORT']:
        subset = trade_df[trade_df['direction'] == direction]
        for idx, (_, trade) in enumerate(subset[subset['outcome']=='WIN'].head(3).iterrows(), 1):
            path = plot_trade(df, trade, f"{direction}_win_{idx}", OUTPUT_DIR)
            if path: print(f"  {path}")
        for idx, (_, trade) in enumerate(subset[subset['outcome']=='LOSS'].head(3).iterrows(), 1):
            path = plot_trade(df, trade, f"{direction}_loss_{idx}", OUTPUT_DIR)
            if path: print(f"  {path}")
    
    print(f"\nCharts: {OUTPUT_DIR}")
    return trade_df


if __name__ == "__main__":
    main()
