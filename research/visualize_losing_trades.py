"""
Visualize Losing Trades - TradingView-style Charts
Shows the swing low, sweep, entry, and outcome for manual review
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration
DATA_PATH = r"c:\Users\CEO\ICT reinforcement\ict_backtest\ml_lab\data\kaizen_1m_data_ibkr_2yr.csv"
OUTCOMES_PATH = r"c:\Users\CEO\ICT reinforcement\ict_backtest\ml_lab\liquidity_discovery\phase3_outcome_matrix.csv"
OUTPUT_DIR = r"c:\Users\CEO\ICT reinforcement\ict_backtest\ml_lab\liquidity_discovery\trade_charts"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load price data and trade outcomes"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    # Convert to naive timestamps (remove timezone)
    df['time'] = df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    
    outcomes = pd.read_csv(OUTCOMES_PATH)
    outcomes['timestamp'] = pd.to_datetime(outcomes['timestamp'])
    
    return df, outcomes

def get_losing_trades(outcomes, min_sweep_size=5.0, event_types=['EQL', 'PDL'], reaction_type='DEEP_RETRACE', n_trades=5):
    """
    Filter for losing trades that match the strategy logic:
    - Sweep of EQL or PDL (downside sweep)
    - Deep retrace reaction (shows promise)
    - But still lost on fixed 1.0R target
    """
    mask = (
        (outcomes['event_type'].isin(event_types)) &
        (outcomes['sweep_size_pts'] >= min_sweep_size) &
        (outcomes['reaction_20'] == reaction_type) &
        (outcomes['fixed_1.0r_result'] == 'LOSS')
    )
    
    losing = outcomes[mask].copy()
    
    # Sort by sweep size to get most "convincing" looking sweeps
    losing = losing.sort_values('sweep_size_pts', ascending=False)
    
    return losing.head(n_trades)

def find_swing_low(df, entry_time, lookback_bars=50):
    """Find the swing low that was swept before entry"""
    # Get data before entry
    mask = df.index < entry_time
    pre_entry = df[mask].tail(lookback_bars)
    
    if len(pre_entry) < 10:
        return None, None
    
    # Find the lowest point
    swing_low_idx = pre_entry['low'].argmin()
    swing_low_time = pre_entry.index[swing_low_idx]
    swing_low_price = pre_entry['low'].iloc[swing_low_idx]
    
    return swing_low_time, swing_low_price

def plot_trade_chart(df, entry_time, sweep_size, event_type, outcome_r, chart_idx, output_dir):
    """
    Create TradingView-style candlestick chart showing:
    - Price action before and after entry
    - Swing low level
    - Sweep below swing low
    - Entry point
    - Stop loss and outcome
    """
    # Get data window: 60 bars before, 40 bars after
    bars_before = 60
    bars_after = 40
    
    # Find entry in dataframe - search for nearest match
    try:
        # First try exact match
        if entry_time in df.index:
            entry_pos = df.index.get_loc(entry_time)
        else:
            # Find nearest timestamp
            time_diffs = abs(df.index - entry_time)
            entry_pos = time_diffs.argmin()
            actual_time = df.index[entry_pos]
            if time_diffs.min().total_seconds() > 300:  # More than 5 min off
                print(f"Could not find close match for {entry_time} (nearest: {actual_time})")
                return None
    except Exception as e:
        print(f"Error finding entry time {entry_time}: {e}")
        return None
    
    start_pos = max(0, entry_pos - bars_before)
    end_pos = min(len(df), entry_pos + bars_after)
    
    window = df.iloc[start_pos:end_pos].copy()
    
    if len(window) < 20:
        print(f"Not enough data for {entry_time}")
        return None
    
    # Find swing low in the pre-entry window
    pre_entry = window[window.index < entry_time]
    if len(pre_entry) < 5:
        return None
    
    # Find swing low (lowest point in lookback)
    lookback = pre_entry.tail(30)
    swing_low_idx = lookback['low'].argmin()
    swing_low_time = lookback.index[swing_low_idx]
    swing_low_price = lookback['low'].iloc[swing_low_idx]
    
    # Get entry price (close at entry bar)
    entry_bar = window[window.index >= entry_time].iloc[0] if len(window[window.index >= entry_time]) > 0 else None
    if entry_bar is None:
        return None
    entry_price = entry_bar['close']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Colors
    bg_color = '#1a1a2e'
    grid_color = '#2a2a4a'
    bull_color = '#26a69a'
    bear_color = '#ef5350'
    text_color = '#e0e0e0'
    
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Plot candlesticks
    x_positions = range(len(window))
    for i, (idx, row) in enumerate(window.iterrows()):
        color = bull_color if row['close'] >= row['open'] else bear_color
        
        # Wick
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        
        # Body
        body_bottom = min(row['open'], row['close'])
        body_height = abs(row['close'] - row['open'])
        if body_height < 0.25:
            body_height = 0.25
        
        rect = plt.Rectangle((i - 0.35, body_bottom), 0.7, body_height, 
                              facecolor=color, edgecolor=color, linewidth=1)
        ax.add_patch(rect)
    
    # Find x position for key times
    time_to_x = {t: i for i, t in enumerate(window.index)}
    
    # Draw swing low horizontal line
    swing_low_x = time_to_x.get(swing_low_time, 0)
    ax.axhline(y=swing_low_price, color='#FFD700', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(1, swing_low_price + 1, f'SWING LOW: {swing_low_price:.2f}', 
            color='#FFD700', fontsize=11, fontweight='bold', 
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#FFD700', alpha=0.9))
    
    # Mark the sweep point (where price went below swing low)
    sweep_bars = window[(window['low'] < swing_low_price) & (window.index <= entry_time)]
    if len(sweep_bars) > 0:
        sweep_time = sweep_bars.index[0]
        sweep_x = time_to_x.get(sweep_time, 0)
        sweep_low = sweep_bars.iloc[0]['low']
        
        ax.annotate('SWEEP', xy=(sweep_x, sweep_low), xytext=(sweep_x, sweep_low - 5),
                    fontsize=12, fontweight='bold', color='#FF6B6B',
                    ha='center', va='top',
                    arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))
        
        # Show sweep depth
        ax.plot([sweep_x, sweep_x], [swing_low_price, sweep_low], 
                color='#FF6B6B', linewidth=2, linestyle=':')
        ax.text(sweep_x + 0.5, (swing_low_price + sweep_low) / 2, 
                f'-{sweep_size:.1f} pts', color='#FF6B6B', fontsize=10, fontweight='bold')
    
    # Mark entry point
    entry_x = time_to_x.get(entry_time, bars_before)
    if entry_x not in time_to_x.values():
        # Find nearest
        for t, x in time_to_x.items():
            if abs((t - entry_time).total_seconds()) < 120:
                entry_x = x
                break
    
    ax.annotate('ENTRY', xy=(entry_x, entry_price), xytext=(entry_x + 3, entry_price + 3),
                fontsize=12, fontweight='bold', color='#00FF7F',
                ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#00FF7F', lw=2))
    
    # Draw entry horizontal line
    ax.axhline(y=entry_price, color='#00FF7F', linestyle='-', linewidth=1.5, alpha=0.6)
    
    # Calculate and show stop loss (1R below entry based on sweep)
    # Assuming stop below the sweep low
    stop_price = entry_price - 10  # 10 point stop as per strategy
    ax.axhline(y=stop_price, color='#FF4444', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(len(window) - 5, stop_price - 1, f'STOP: {stop_price:.2f}', 
            color='#FF4444', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#FF4444', alpha=0.9))
    
    # Title with trade details
    title_text = (f"LOSING TRADE #{chart_idx}\n"
                  f"{event_type} Sweep | Entry: {entry_time.strftime('%Y-%m-%d %H:%M')} EST\n"
                  f"Sweep Size: {sweep_size:.1f} pts | Result: {outcome_r:.2f}R LOSS")
    
    ax.set_title(title_text, color=text_color, fontsize=14, fontweight='bold', pad=20)
    
    # Style axes
    ax.set_xlim(-1, len(window))
    ax.set_ylim(window['low'].min() - 5, window['high'].max() + 5)
    ax.set_xlabel('Bars', color=text_color, fontsize=11)
    ax.set_ylabel('Price', color=text_color, fontsize=11)
    ax.tick_params(colors=text_color)
    ax.grid(True, color=grid_color, alpha=0.3, linestyle='-')
    
    # Add time labels on x-axis (every 10 bars)
    x_ticks = range(0, len(window), 10)
    x_labels = [window.index[i].strftime('%H:%M') if i < len(window) else '' for i in x_ticks]
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#FFD700', edgecolor='#FFD700', label='Swing Low Level'),
        mpatches.Patch(facecolor='#FF6B6B', edgecolor='#FF6B6B', label='Sweep Below'),
        mpatches.Patch(facecolor='#00FF7F', edgecolor='#00FF7F', label='Entry'),
        mpatches.Patch(facecolor='#FF4444', edgecolor='#FF4444', label='Stop Loss'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    
    # Add analysis box
    analysis_text = (
        f"TRADE ANALYSIS\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Event Type: {event_type}\n"
        f"Sweep Size: {sweep_size:.1f} pts\n"
        f"Entry Price: {entry_price:.2f}\n"
        f"Stop Price: {stop_price:.2f}\n"
        f"Result: LOSS ({outcome_r:.2f}R)\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Question: Would you\n"
        f"have taken this trade?"
    )
    
    props = dict(boxstyle='round', facecolor=bg_color, edgecolor='#4a4a6a', alpha=0.95)
    ax.text(0.98, 0.02, analysis_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, color=text_color, family='monospace')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, f"losing_trade_{chart_idx}_{entry_time.strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(output_path, dpi=150, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    """Generate losing trade visualizations"""
    # Load data
    df, outcomes = load_data()
    
    print(f"\nLoaded {len(df)} bars of price data")
    print(f"Loaded {len(outcomes)} trade events")
    
    # Get losing trades that look like they should have worked
    # (Deep retrace after significant sweep)
    print("\nFinding losing trades with DEEP_RETRACE after sweep...")
    losing_trades = get_losing_trades(
        outcomes, 
        min_sweep_size=10.0,  # Significant sweep
        event_types=['EQL', 'PDL'],  # Downside sweeps only
        reaction_type='DEEP_RETRACE',  # Showed promise
        n_trades=5
    )
    
    print(f"\nFound {len(losing_trades)} qualifying losing trades")
    print("\nGenerating charts...")
    
    generated = []
    for idx, (_, trade) in enumerate(losing_trades.iterrows(), 1):
        entry_time = trade['timestamp']
        sweep_size = trade['sweep_size_pts']
        event_type = trade['event_type']
        outcome_r = trade['fixed_1.0r_r']
        
        print(f"\n--- Trade {idx} ---")
        print(f"Time: {entry_time}")
        print(f"Type: {event_type}")
        print(f"Sweep: {sweep_size:.1f} pts")
        print(f"Reaction: {trade['reaction_20']}")
        print(f"Result: {outcome_r:.2f}R")
        
        result = plot_trade_chart(df, entry_time, sweep_size, event_type, outcome_r, idx, OUTPUT_DIR)
        if result:
            generated.append(result)
    
    print(f"\n\n{'='*50}")
    print(f"Generated {len(generated)} trade charts")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*50}")
    
    return generated

if __name__ == "__main__":
    main()
