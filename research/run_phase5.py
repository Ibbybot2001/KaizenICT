"""
Phase 5 Runner: OOS Validation

FROZEN RULES (DO NOT MODIFY):
- Reaction: DEEP_RETRACE
- Sweep Size: Macro Q4 (largest quartile)
- Direction: FADE
- Event Type: PDL (primary), EQH (secondary test)
- Time Window: 15:00-16:00
- Exit: time_20
- No management, no TP optimization

Split: Chronological (Year 1 = Train, Year 2 = Test)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

from research.labels import MultiPathLabeler

DATA_PATH = "data/kaizen_1m_data_ibkr_2yr.csv"
EVENTS_PATH = "output/phases/phase1_event_catalog.csv"
OUTPUT_PATH = "reports/phase5_oos_results.md"

# FROZEN RULES
REACTION_FILTER = 'DEEP_RETRACE'
SWEEP_SIZE_FILTER = 'macro_q4'  # Top quartile
TIME_WINDOW = (15.0, 16.0)  # 15:00-16:00
EXIT_SCHEMA = 'time_20'
FIXED_SL = 10.0

def load_data():
    """Load price data."""
    print(f"Loading price data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df = df.sort_index()
    df.index = df.index.tz_convert(None)
    df = df.between_time('09:30', '16:00')
    return df

def load_events():
    """Load event catalog."""
    print(f"Loading events from {EVENTS_PATH}...")
    events = pd.read_csv(EVENTS_PATH)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    return events

def compute_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR."""
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def classify_reaction(data: pd.DataFrame, bar_idx: int, horizon: int = 20) -> str:
    """Classify reaction at horizon."""
    if bar_idx + horizon >= len(data):
        return 'UNKNOWN'
    
    entry_close = data.iloc[bar_idx]['close']
    window = data.iloc[bar_idx + 1:bar_idx + horizon + 1]
    
    if len(window) == 0:
        return 'UNKNOWN'
    
    # Get sweep direction from event (we'll infer from entry)
    # For now, compute max extension/retrace
    max_up = window['high'].max() - entry_close
    max_down = entry_close - window['low'].min()
    
    atr = compute_atr(data).iloc[bar_idx]
    if atr <= 0:
        return 'UNKNOWN'
    
    # Simple classification based on retrace
    if max_down / atr >= 1.5:
        return 'DEEP_RETRACE'
    elif max_down / atr >= 0.5:
        return 'SHALLOW_PULLBACK'
    elif max_up / atr >= 2.0:
        return 'CONTINUATION'
    else:
        return 'CHOP'

def simulate_trade(data: pd.DataFrame, bar_idx: int, event_type: str) -> dict:
    """Simulate a single trade with frozen rules."""
    entry_price = data.iloc[bar_idx]['close']
    
    # Direction: FADE (inverse of sweep)
    if event_type in ['EQL', 'PDL']:
        direction = 'LONG'  # Fade downside sweep
    else:
        direction = 'SHORT'  # Fade upside sweep
    
    # Exit after 20 bars
    exit_idx = min(bar_idx + 20, len(data) - 1)
    exit_price = data.iloc[exit_idx]['close']
    
    # Calculate R
    if direction == 'LONG':
        r_mult = (exit_price - entry_price) / FIXED_SL
    else:
        r_mult = (entry_price - exit_price) / FIXED_SL
    
    # Friction
    r_mult -= 0.05
    
    return {
        'r_multiple': r_mult,
        'direction': direction,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'bars_held': 20
    }

def main():
    print("=" * 60)
    print("PHASE 5: OOS VALIDATION")
    print("=" * 60)
    print("\nFROZEN RULES:")
    print("  - Reaction: DEEP_RETRACE")
    print("  - Sweep Size: Macro Q4")
    print("  - Time Window: 15:00-16:00")
    print("  - Exit: time_20")
    print("  - No optimization allowed")
    
    # Load data
    data = load_data()
    events = load_events()
    
    print(f"\nPrice bars: {len(data)}")
    print(f"Total events: {len(events)}")
    
    # Build timestamp lookup
    data_reset = data.reset_index()
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(data_reset['time'])}
    
    # Compute ATR for the dataset
    print("Computing ATR...")
    atr = compute_atr(data)
    
    # Apply FROZEN filters to events
    print("\nApplying frozen filters...")
    
    # Filter 1: Time window (15:00-16:00)
    events['hour_decimal'] = pd.to_datetime(events['timestamp']).dt.hour + pd.to_datetime(events['timestamp']).dt.minute / 60
    events = events[(events['hour_decimal'] >= TIME_WINDOW[0]) & (events['hour_decimal'] < TIME_WINDOW[1])]
    print(f"After time filter: {len(events)}")
    
    # Filter 2: Macro sweeps only (>= 3 pts)
    events = events[events['is_micro_sweep'] == False]
    print(f"After macro filter: {len(events)}")
    
    # Filter 3: Q4 (top quartile by sweep size)
    if len(events) > 0:
        q4_threshold = events['sweep_size_pts'].quantile(0.75)
        events = events[events['sweep_size_pts'] >= q4_threshold]
    print(f"After Q4 filter: {len(events)}")
    
    # Determine split point (50/50 chronological by timestamp)
    events = events.sort_values('timestamp')
    
    # Use median timestamp for 50/50 split
    midpoint = events['timestamp'].median()
    
    print(f"\nChronological Split (50/50):")
    print(f"  Midpoint: {midpoint}")
    
    train_events = events[events['timestamp'] < midpoint]
    test_events = events[events['timestamp'] >= midpoint]
    
    print(f"  Train: {train_events['timestamp'].min()} to {train_events['timestamp'].max()}")
    print(f"  Test: {test_events['timestamp'].min()} to {test_events['timestamp'].max()}")
    print(f"  Train events: {len(train_events)}")
    print(f"  Test events: {len(test_events)}")
    
    # Simulate on BOTH sets
    results = {}
    
    for name, event_set in [('TRAIN (Year 1)', train_events), ('TEST (Year 2 - OOS)', test_events)]:
        print(f"\n--- {name} ---")
        
        trades = []
        
        for _, event in event_set.iterrows():
            ts = event['timestamp']
            
            if ts not in timestamp_to_idx:
                continue
            
            bar_idx = timestamp_to_idx[ts]
            
            if bar_idx + 20 >= len(data):
                continue
            
            # Check reaction type (classify first)
            reaction = classify_reaction(data, bar_idx, horizon=20)
            
            if reaction != REACTION_FILTER:
                continue
            
            # Simulate trade
            trade = simulate_trade(data, bar_idx, event['event_type'])
            trade['timestamp'] = ts
            trade['event_type'] = event['event_type']
            trades.append(trade)
        
        if len(trades) == 0:
            print("No trades found!")
            results[name] = {'count': 0}
            continue
        
        trades_df = pd.DataFrame(trades)
        
        # Compute metrics
        mean_r = trades_df['r_multiple'].mean()
        median_r = trades_df['r_multiple'].median()
        total_r = trades_df['r_multiple'].sum()
        count = len(trades_df)
        win_rate = (trades_df['r_multiple'] > 0).mean() * 100
        
        # Rolling drawdown
        cumulative = trades_df['r_multiple'].cumsum()
        rolling_max = cumulative.cummax()
        drawdown = cumulative - rolling_max
        max_dd = drawdown.min()
        
        # Monthly breakdown
        trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
        monthly = trades_df.groupby('month')['r_multiple'].sum()
        worst_month = monthly.min()
        flat_months = (monthly == 0).sum() / len(monthly) * 100 if len(monthly) > 0 else 0
        
        print(f"Trades: {count}")
        print(f"Mean R: {mean_r:.4f}")
        print(f"Median R: {median_r:.4f}")
        print(f"Total R: {total_r:.2f}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Max DD: {max_dd:.2f} R")
        print(f"Worst Month: {worst_month:.2f} R")
        
        results[name] = {
            'count': count,
            'mean_r': mean_r,
            'median_r': median_r,
            'total_r': total_r,
            'win_rate': win_rate,
            'max_dd': max_dd,
            'worst_month': worst_month
        }
    
    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING OOS REPORT...")
    
    report = """# Phase 5: OOS Validation Results

## FROZEN RULES (Not Optimized)
- Reaction: DEEP_RETRACE
- Sweep Size: Macro Q4 (Top Quartile)
- Time Window: 15:00-16:00
- Exit: time_20
- Direction: FADE

---

"""
    
    for name, metrics in results.items():
        report += f"## {name}\n\n"
        if metrics['count'] == 0:
            report += "No trades found.\n\n"
        else:
            report += f"| Metric | Value |\n"
            report += f"|--------|-------|\n"
            report += f"| Trades | {metrics['count']} |\n"
            report += f"| Mean R | {metrics['mean_r']:.4f} |\n"
            report += f"| Median R | {metrics['median_r']:.4f} |\n"
            report += f"| Total R | {metrics['total_r']:.2f} |\n"
            report += f"| Win Rate | {metrics['win_rate']:.1f}% |\n"
            report += f"| Max DD | {metrics['max_dd']:.2f} R |\n"
            report += f"| Worst Month | {metrics['worst_month']:.2f} R |\n\n"
    
    # Verdict
    train_r = results.get('TRAIN (Year 1)', {}).get('mean_r', 0)
    test_r = results.get('TEST (Year 2 - OOS)', {}).get('mean_r', 0)
    
    report += "---\n\n## VERDICT\n\n"
    
    if test_r > 0:
        report += f"**OOS PASSED**: Test set shows positive expectancy ({test_r:.4f} R)\n\n"
        if test_r >= train_r * 0.5:
            report += "Edge **retained** at least 50% of in-sample performance.\n"
        else:
            report += "Edge **degraded** but still positive. Exercise caution.\n"
    else:
        report += f"**OOS FAILED**: Test set shows negative expectancy ({test_r:.4f} R)\n\n"
        report += "Edge did not survive OOS. Do not trade this.\n"
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(report)
    
    print(f"Saved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
