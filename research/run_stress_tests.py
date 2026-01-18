"""
Stress Test Runner
Final validation before live execution

Tests (NOT optimization):
1. Slippage: +1, +2 pts
2. Entry Delay: 1 bar
3. Exit Variants: time_15, time_20, time_25
4. Throttle: 1 trade/day max

If edge dies under any of these → flag for review
If edge survives (even weakened) → green light
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_PATH = "ml_lab/data/kaizen_1m_data_ibkr_2yr.csv"
EVENTS_PATH = "ml_lab/liquidity_discovery/phase1_event_catalog.csv"
OUTPUT_PATH = "ml_lab/liquidity_discovery/stress_test_results.md"

# Base parameters (from validated strategy)
BASE_SL = 10.0
BASE_EXIT_BARS = 20
BASE_FRICTION = 0.05

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    df = df.sort_index()
    df.index = df.index.tz_convert(None)
    df = df.between_time('09:30', '16:00')
    return df

def load_events():
    events = pd.read_csv(EVENTS_PATH)
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    return events

def apply_base_filters(events):
    """Apply frozen filters: 15:00-16:00, Macro, Q4"""
    events['hour_decimal'] = events['timestamp'].dt.hour + events['timestamp'].dt.minute / 60
    events = events[(events['hour_decimal'] >= 15.0) & (events['hour_decimal'] < 16.0)]
    events = events[events['is_micro_sweep'] == False]
    q4 = events['sweep_size_pts'].quantile(0.75)
    events = events[events['sweep_size_pts'] >= q4]
    return events

def simulate_trade(data, bar_idx, event_type, exit_bars=20, entry_delay=0, slippage_pts=0):
    """Simulate a single trade with stress parameters."""
    entry_idx = bar_idx + entry_delay
    if entry_idx + exit_bars >= len(data):
        return None
    
    entry_price = data.iloc[entry_idx]['close']
    
    # Apply slippage (worse entry)
    if event_type in ['EQL', 'PDL']:
        direction = 'LONG'
        entry_price += slippage_pts  # Worse for longs
    else:
        direction = 'SHORT'
        entry_price -= slippage_pts  # Worse for shorts
    
    # Exit after N bars
    exit_idx = entry_idx + exit_bars
    exit_price = data.iloc[exit_idx]['close']
    
    # Calculate R
    if direction == 'LONG':
        r_mult = (exit_price - entry_price) / BASE_SL
    else:
        r_mult = (entry_price - exit_price) / BASE_SL
    
    r_mult -= BASE_FRICTION
    
    return r_mult

def run_stress_test(data, events, timestamp_to_idx, 
                   exit_bars=20, entry_delay=0, slippage_pts=0, 
                   throttle_per_day=None):
    """Run a single stress test configuration."""
    trades = []
    trades_by_date = {}
    
    for _, event in events.iterrows():
        ts = event['timestamp']
        
        if ts not in timestamp_to_idx:
            continue
        
        bar_idx = timestamp_to_idx[ts]
        
        # Throttle check
        if throttle_per_day is not None:
            date = ts.date()
            if date not in trades_by_date:
                trades_by_date[date] = 0
            if trades_by_date[date] >= throttle_per_day:
                continue
            trades_by_date[date] += 1
        
        r = simulate_trade(data, bar_idx, event['event_type'], 
                          exit_bars=exit_bars, 
                          entry_delay=entry_delay,
                          slippage_pts=slippage_pts)
        
        if r is not None:
            trades.append(r)
    
    if len(trades) == 0:
        return {'count': 0, 'mean_r': 0, 'total_r': 0, 'win_rate': 0}
    
    trades = np.array(trades)
    return {
        'count': len(trades),
        'mean_r': trades.mean(),
        'total_r': trades.sum(),
        'win_rate': (trades > 0).mean() * 100
    }

def main():
    print("=" * 60)
    print("STRESS TEST SUITE")
    print("=" * 60)
    
    # Load data
    data = load_data()
    events = load_events()
    
    # Build timestamp lookup
    data_reset = data.reset_index()
    timestamp_to_idx = {ts: idx for idx, ts in enumerate(data_reset['time'])}
    
    # Apply base filters
    events = apply_base_filters(events)
    print(f"Filtered events: {len(events)}")
    
    # Use OOS portion only (second half)
    midpoint = events['timestamp'].median()
    oos_events = events[events['timestamp'] >= midpoint]
    print(f"OOS events for stress testing: {len(oos_events)}")
    
    # Define stress tests
    tests = [
        {'name': 'BASELINE', 'exit_bars': 20, 'entry_delay': 0, 'slippage_pts': 0, 'throttle': None},
        {'name': '+1 pt Slippage', 'exit_bars': 20, 'entry_delay': 0, 'slippage_pts': 1, 'throttle': None},
        {'name': '+2 pts Slippage', 'exit_bars': 20, 'entry_delay': 0, 'slippage_pts': 2, 'throttle': None},
        {'name': '1-Bar Entry Delay', 'exit_bars': 20, 'entry_delay': 1, 'slippage_pts': 0, 'throttle': None},
        {'name': 'Exit: 15 bars', 'exit_bars': 15, 'entry_delay': 0, 'slippage_pts': 0, 'throttle': None},
        {'name': 'Exit: 25 bars', 'exit_bars': 25, 'entry_delay': 0, 'slippage_pts': 0, 'throttle': None},
        {'name': '1 Trade/Day Throttle', 'exit_bars': 20, 'entry_delay': 0, 'slippage_pts': 0, 'throttle': 1},
        {'name': 'WORST CASE (+2 slip, delay, throttle)', 'exit_bars': 20, 'entry_delay': 1, 'slippage_pts': 2, 'throttle': 1},
    ]
    
    results = []
    
    for test in tests:
        print(f"\nRunning: {test['name']}...")
        result = run_stress_test(
            data, oos_events, timestamp_to_idx,
            exit_bars=test['exit_bars'],
            entry_delay=test['entry_delay'],
            slippage_pts=test['slippage_pts'],
            throttle_per_day=test['throttle']
        )
        result['name'] = test['name']
        results.append(result)
        print(f"  Trades: {result['count']}, Mean R: {result['mean_r']:.4f}, WR: {result['win_rate']:.1f}%")
    
    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING STRESS TEST REPORT...")
    
    report = """# Stress Test Results

## Purpose
Validate that the edge survives real-world execution conditions.
These are NOT optimizations — just fragility checks.

---

## Results

| Test | Trades | Mean R | Total R | Win Rate | Status |
|------|--------|--------|---------|----------|--------|
"""
    
    baseline_r = results[0]['mean_r']
    
    for r in results:
        status = "PASS" if r['mean_r'] > 0 else "FAIL"
        if r['mean_r'] > 0 and r['mean_r'] < baseline_r * 0.5:
            status = "DEGRADED"
        
        report += f"| {r['name']} | {r['count']} | {r['mean_r']:.4f} | {r['total_r']:.2f} | {r['win_rate']:.1f}% | {status} |\n"
    
    report += """
---

## Interpretation

- **PASS**: Edge survives this condition
- **DEGRADED**: Edge weakened but still positive
- **FAIL**: Edge destroyed - do NOT trade under this condition

---

## Verdict

"""
    
    all_pass = all(r['mean_r'] > 0 for r in results)
    if all_pass:
        report += "**ALL TESTS PASSED** — Edge is robust. Green light for live execution.\n"
    else:
        failed = [r['name'] for r in results if r['mean_r'] <= 0]
        report += f"**SOME TESTS FAILED**: {', '.join(failed)}\n"
        report += "Review failed conditions before live trading.\n"
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(report)
    
    print(f"Saved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("STRESS TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
