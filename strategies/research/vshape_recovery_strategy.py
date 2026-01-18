"""
V-Shape Recovery Strategy
=========================
Wait for sweep to COMPLETE, then:
- V-Shape Recovery → LONG (counter-trend fade)
- Failed Recovery / Continuation Below Breaker → SHORT

Rules:
1. Sweep Complete: Price goes below swing low, spends 3+ bars below, then closes back above
2. V-Shape Recovery: Recovery retraces >60% of the drop within 10 bars
3. Breaker Block: Last bullish candle before the drop
4. Continuation Short: Bounce fails, closes below breaker block
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import os

# Configuration
DATA_PATH = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
OUTPUT_DIR = r"C:\Users\CEO\ICT reinforcement\output\charts\vshape_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy Parameters
SWING_LOOKBACK = 20          # Bars to look for swing low
MIN_SWEEP_DEPTH = 5.0        # Minimum pts below swing low
BARS_BELOW_MIN = 3           # Min bars spent below swing low
RECOVERY_WINDOW = 15         # Bars to measure recovery
RECOVERY_THRESHOLD = 0.50    # Recovery must retrace 50% of drop
STOP_LOSS_PTS = 12.0         # Fixed stop
TARGET_MULT = 1.5            # R:R target
BREAKER_LOOKBACK = 10        # Bars to find breaker block


def load_data():
    """Load price data"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    return df


def find_swing_lows(df, lookback=20):
    """Find all swing lows in the data"""
    swing_lows = []
    
    for i in range(lookback, len(df) - lookback):
        window = df.iloc[i - lookback:i + lookback + 1]
        center_low = df.iloc[i]['low']
        
        # Is this the lowest point in the window?
        if center_low == window['low'].min():
            swing_lows.append({
                'time': df.index[i],
                'price': center_low,
                'idx': i
            })
    
    return swing_lows


def detect_sweep_and_recovery(df, swing_low_idx, swing_low_price):
    """
    Detect if a sweep occurred and measure the recovery quality.
    
    Returns:
        dict with sweep details, recovery type, and trade direction
    """
    # Look for sweep after the swing low forms
    search_start = swing_low_idx + 1
    search_end = min(swing_low_idx + 100, len(df))
    
    for i in range(search_start, search_end):
        bar = df.iloc[i]
        
        # Check for sweep (price goes below swing low)
        if bar['low'] < swing_low_price - MIN_SWEEP_DEPTH:
            sweep_low = bar['low']
            sweep_depth = swing_low_price - sweep_low
            sweep_idx = i
            sweep_time = df.index[i]
            
            # Now wait for sweep to complete - price must close back above swing low
            for j in range(i + 1, min(i + 30, len(df))):
                if df.iloc[j]['close'] > swing_low_price:
                    # Sweep is complete! Now measure recovery
                    recovery_start = j
                    
                    # Find the drop range (from recent high to sweep low)
                    pre_sweep = df.iloc[max(0, i-20):i+1]
                    recent_high = pre_sweep['high'].max()
                    drop_size = recent_high - sweep_low
                    
                    if drop_size < 10:  # Skip tiny moves
                        break
                    
                    # Find breaker block (last bullish candle before the drop)
                    breaker_price = None
                    for k in range(i-1, max(0, i-BREAKER_LOOKBACK), -1):
                        if df.iloc[k]['close'] > df.iloc[k]['open']:  # Bullish candle
                            breaker_price = df.iloc[k]['low']  # Bottom of breaker
                            breaker_idx = k
                            break
                    
                    # Measure recovery over next N bars
                    recovery_end = min(j + RECOVERY_WINDOW, len(df))
                    recovery_window = df.iloc[j:recovery_end]
                    
                    if len(recovery_window) < 5:
                        break
                    
                    recovery_high = recovery_window['high'].max()
                    recovery_amount = recovery_high - sweep_low
                    recovery_ratio = recovery_amount / drop_size if drop_size > 0 else 0
                    
                    # Determine trade type
                    entry_price = df.iloc[j]['close']
                    
                    if recovery_ratio >= RECOVERY_THRESHOLD:
                        # V-Shape recovery - LONG
                        trade_type = "V_SHAPE_LONG"
                        stop_price = entry_price - STOP_LOSS_PTS
                        target_price = entry_price + (STOP_LOSS_PTS * TARGET_MULT)
                    else:
                        # Weak recovery - check for continuation short
                        if breaker_price and recovery_window['close'].iloc[-1] < breaker_price:
                            trade_type = "CONTINUATION_SHORT"
                            entry_price = recovery_window['close'].iloc[-1]
                            stop_price = entry_price + STOP_LOSS_PTS
                            target_price = entry_price - (STOP_LOSS_PTS * TARGET_MULT)
                        else:
                            trade_type = "NO_TRADE"
                            return None
                    
                    # Calculate outcome
                    outcome = calculate_outcome(df, j, trade_type, entry_price, stop_price, target_price)
                    
                    return {
                        'swing_low_price': swing_low_price,
                        'swing_low_idx': swing_low_idx,
                        'sweep_time': sweep_time,
                        'sweep_low': sweep_low,
                        'sweep_depth': sweep_depth,
                        'recovery_start_idx': j,
                        'entry_time': df.index[j],
                        'entry_price': entry_price,
                        'recovery_ratio': recovery_ratio,
                        'drop_size': drop_size,
                        'breaker_price': breaker_price,
                        'trade_type': trade_type,
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'outcome': outcome['result'],
                        'pnl_r': outcome['pnl_r'],
                        'bars_to_exit': outcome['bars']
                    }
                    
            break  # Only process first sweep
    
    return None


def calculate_outcome(df, entry_idx, trade_type, entry, stop, target):
    """Calculate trade outcome"""
    max_bars = 60
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        
        if trade_type == "V_SHAPE_LONG":
            if bar['low'] <= stop:
                return {'result': 'LOSS', 'pnl_r': -1.0, 'bars': i - entry_idx}
            if bar['high'] >= target:
                return {'result': 'WIN', 'pnl_r': TARGET_MULT, 'bars': i - entry_idx}
        
        elif trade_type == "CONTINUATION_SHORT":
            if bar['high'] >= stop:
                return {'result': 'LOSS', 'pnl_r': -1.0, 'bars': i - entry_idx}
            if bar['low'] <= target:
                return {'result': 'WIN', 'pnl_r': TARGET_MULT, 'bars': i - entry_idx}
    
    # Time exit
    final_price = df.iloc[min(entry_idx + max_bars - 1, len(df) - 1)]['close']
    if trade_type == "V_SHAPE_LONG":
        pnl = (final_price - entry) / STOP_LOSS_PTS
    else:
        pnl = (entry - final_price) / STOP_LOSS_PTS
    
    return {'result': 'TIME_EXIT', 'pnl_r': pnl, 'bars': max_bars}


def run_backtest(df, max_trades=500):
    """Run backtest and collect trades"""
    print("\nFinding swing lows...")
    swing_lows = find_swing_lows(df, SWING_LOOKBACK)
    print(f"Found {len(swing_lows)} swing lows")
    
    trades = []
    last_trade_idx = 0
    
    print("\nScanning for setups...")
    for sl in swing_lows:
        # Don't overlap trades
        if sl['idx'] < last_trade_idx + 50:
            continue
            
        result = detect_sweep_and_recovery(df, sl['idx'], sl['price'])
        
        if result:
            trades.append(result)
            last_trade_idx = result['recovery_start_idx']
            
            if len(trades) >= max_trades:
                break
    
    return trades


def plot_trade(df, trade, idx, output_dir):
    """Create TradingView-style chart for a trade"""
    # Get window around trade
    bars_before = 50
    bars_after = 70
    
    start_idx = max(0, trade['swing_low_idx'] - bars_before)
    end_idx = min(len(df), trade['recovery_start_idx'] + bars_after)
    
    window = df.iloc[start_idx:end_idx].copy()
    
    if len(window) < 30:
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    
    # Colors
    bg_color = '#1a1a2e'
    bull_color = '#26a69a'
    bear_color = '#ef5350'
    text_color = '#e0e0e0'
    
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot candlesticks
    for i, (t, row) in enumerate(window.iterrows()):
        color = bull_color if row['close'] >= row['open'] else bear_color
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        body_bottom = min(row['open'], row['close'])
        body_height = max(abs(row['close'] - row['open']), 0.25)
        rect = plt.Rectangle((i - 0.35, body_bottom), 0.7, body_height, 
                              facecolor=color, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
    
    # Map times to x positions
    time_to_x = {t: i for i, t in enumerate(window.index)}
    
    # Draw swing low level
    ax.axhline(y=trade['swing_low_price'], color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(2, trade['swing_low_price'] + 2, f"SWING LOW: {trade['swing_low_price']:.2f}", 
            color='#FFD700', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#FFD700', alpha=0.9))
    
    # Mark sweep area
    sweep_x = time_to_x.get(trade['sweep_time'], 30)
    ax.annotate('SWEEP', xy=(sweep_x, trade['sweep_low']), xytext=(sweep_x, trade['sweep_low'] - 8),
                fontsize=12, fontweight='bold', color='#FF6B6B', ha='center',
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))
    
    # Draw sweep depth
    ax.plot([sweep_x, sweep_x], [trade['swing_low_price'], trade['sweep_low']], 
            color='#FF6B6B', linewidth=2, linestyle=':')
    ax.text(sweep_x + 1, (trade['swing_low_price'] + trade['sweep_low']) / 2, 
            f"-{trade['sweep_depth']:.1f} pts", color='#FF6B6B', fontsize=10, fontweight='bold')
    
    # Mark entry
    entry_x = time_to_x.get(trade['entry_time'], 40)
    entry_color = '#00FF7F' if trade['trade_type'] == 'V_SHAPE_LONG' else '#FF4444'
    direction = 'LONG' if trade['trade_type'] == 'V_SHAPE_LONG' else 'SHORT'
    
    ax.annotate(f'ENTRY ({direction})', xy=(entry_x, trade['entry_price']), 
                xytext=(entry_x + 3, trade['entry_price'] + 5),
                fontsize=12, fontweight='bold', color=entry_color, ha='left',
                arrowprops=dict(arrowstyle='->', color=entry_color, lw=2))
    
    # Entry line
    ax.axhline(y=trade['entry_price'], color=entry_color, linestyle='-', linewidth=1.5, alpha=0.6)
    
    # Stop loss line
    ax.axhline(y=trade['stop_price'], color='#FF4444', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(len(window) - 8, trade['stop_price'], f"STOP: {trade['stop_price']:.2f}", 
            color='#FF4444', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#FF4444', alpha=0.9))
    
    # Target line
    ax.axhline(y=trade['target_price'], color='#00FF7F', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(len(window) - 8, trade['target_price'], f"TARGET: {trade['target_price']:.2f}", 
            color='#00FF7F', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=bg_color, edgecolor='#00FF7F', alpha=0.9))
    
    # Breaker block
    if trade['breaker_price']:
        ax.axhline(y=trade['breaker_price'], color='#9C27B0', linestyle='-.', linewidth=1.5, alpha=0.7)
        ax.text(2, trade['breaker_price'] - 2, f"BREAKER: {trade['breaker_price']:.2f}", 
                color='#9C27B0', fontsize=10, fontweight='bold')
    
    # Title
    result_color = '#00FF7F' if trade['outcome'] == 'WIN' else '#FF4444'
    title = (f"{trade['trade_type'].replace('_', ' ')} | Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}\n"
             f"Sweep: {trade['sweep_depth']:.1f} pts | Recovery: {trade['recovery_ratio']*100:.0f}% | "
             f"Result: {trade['outcome']} ({trade['pnl_r']:+.2f}R)")
    
    ax.set_title(title, color=result_color, fontsize=14, fontweight='bold', pad=20)
    
    # Style
    ax.set_xlim(-1, len(window))
    ax.set_ylim(window['low'].min() - 10, window['high'].max() + 10)
    ax.set_xlabel('Bars', color=text_color, fontsize=11)
    ax.set_ylabel('Price', color=text_color, fontsize=11)
    ax.tick_params(colors=text_color)
    ax.grid(True, color='#2a2a4a', alpha=0.3)
    
    # X-axis time labels
    x_ticks = range(0, len(window), 10)
    x_labels = [window.index[i].strftime('%H:%M') if i < len(window) else '' for i in x_ticks]
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Analysis box
    analysis = (
        f"TRADE ANALYSIS\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Type: {trade['trade_type']}\n"
        f"Sweep Depth: {trade['sweep_depth']:.1f} pts\n"
        f"Recovery: {trade['recovery_ratio']*100:.0f}% of drop\n"
        f"Entry: {trade['entry_price']:.2f}\n"
        f"Stop: {trade['stop_price']:.2f}\n"
        f"Target: {trade['target_price']:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Result: {trade['outcome']}\n"
        f"P&L: {trade['pnl_r']:+.2f}R\n"
        f"Bars: {trade['bars_to_exit']}"
    )
    
    props = dict(boxstyle='round', facecolor=bg_color, edgecolor='#4a4a6a', alpha=0.95)
    ax.text(0.98, 0.02, analysis, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, color=text_color, family='monospace')
    
    plt.tight_layout()
    
    # Save
    result_tag = 'win' if trade['outcome'] == 'WIN' else 'loss'
    filename = f"{result_tag}_{idx}_{trade['entry_time'].strftime('%Y%m%d_%H%M')}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, facecolor=bg_color, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    # Load data
    df = load_data()
    print(f"Loaded {len(df)} bars")
    
    # Run backtest
    trades = run_backtest(df, max_trades=200)
    
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {len(trades)}")
    
    if not trades:
        print("No trades found!")
        return
    
    # Analyze results
    trade_df = pd.DataFrame(trades)
    
    # By trade type
    for trade_type in trade_df['trade_type'].unique():
        subset = trade_df[trade_df['trade_type'] == trade_type]
        wins = (subset['outcome'] == 'WIN').sum()
        losses = (subset['outcome'] == 'LOSS').sum()
        total_r = subset['pnl_r'].sum()
        win_rate = wins / len(subset) * 100 if len(subset) > 0 else 0
        
        print(f"\n{trade_type}:")
        print(f"  Trades: {len(subset)}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total R: {total_r:+.2f}")
    
    # Overall
    total_r = trade_df['pnl_r'].sum()
    win_rate = (trade_df['outcome'] == 'WIN').sum() / len(trade_df) * 100
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {len(trades)} trades | Win Rate: {win_rate:.1f}% | Total: {total_r:+.2f}R")
    print(f"{'='*60}")
    
    # Generate example charts - 3 wins, 3 losses for each type
    print("\nGenerating example charts...")
    
    for trade_type in trade_df['trade_type'].unique():
        subset = trade_df[trade_df['trade_type'] == trade_type]
        
        wins = subset[subset['outcome'] == 'WIN'].head(3)
        losses = subset[subset['outcome'] == 'LOSS'].head(3)
        
        for idx, (_, trade) in enumerate(wins.iterrows(), 1):
            path = plot_trade(df, trade, f"{trade_type}_win_{idx}", OUTPUT_DIR)
            if path:
                print(f"  Saved: {path}")
        
        for idx, (_, trade) in enumerate(losses.iterrows(), 1):
            path = plot_trade(df, trade, f"{trade_type}_loss_{idx}", OUTPUT_DIR)
            if path:
                print(f"  Saved: {path}")
    
    print(f"\nCharts saved to: {OUTPUT_DIR}")
    
    return trade_df


if __name__ == "__main__":
    main()
