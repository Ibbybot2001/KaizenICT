"""
Phase 4 Runner: Edge Isolation

Analyzes DEEP_RETRACE events (60% of sweeps, +0.53R baseline) to find
rejection conditions that amplify expectancy.

Fixed Exit: time_20 (not optimized)
Target: 90-97% rejection with preserved/amplified expectancy
"""

import pandas as pd
import numpy as np
import os

from research.edge_analysis import EdgeIsolator

OUTCOMES_PATH = "output/phases/phase3_outcome_matrix.csv"
OUTPUT_PATH = "reports/phase4_edge_report.md"

def main():
    print("=" * 60)
    print("PHASE 4: EDGE ISOLATION")
    print("=" * 60)
    
    # Load outcome matrix
    print(f"Loading outcomes from {OUTCOMES_PATH}...")
    outcomes = pd.read_csv(OUTCOMES_PATH)
    print(f"Total events: {len(outcomes)}")
    
    # Initialize isolator
    isolator = EdgeIsolator(outcomes)
    
    # Baseline
    baseline = isolator.baseline_stats()
    print("\n--- BASELINE (All DEEP_RETRACE, time_20) ---")
    print(f"Count: {baseline['count']}")
    print(f"Mean R: {baseline['mean_r']:.4f}")
    print(f"Win Rate: {baseline['win_rate']*100:.1f}%")
    print(f"Total R: {baseline['total_r']:.2f}")
    
    # Hour window analysis
    print("\n--- BY TIME OF DAY ---")
    hour_df = isolator.analyze_by_hour_window()
    print(hour_df.to_string(index=False))
    
    # Event type analysis
    print("\n--- BY EVENT TYPE ---")
    type_df = isolator.analyze_by_event_type()
    print(type_df.to_string(index=False))
    
    # Sweep size analysis
    print("\n--- BY SWEEP SIZE ---")
    size_df = isolator.analyze_by_sweep_size()
    print(size_df.to_string(index=False))
    
    # Volatility analysis
    print("\n--- BY VOLATILITY (ATR) ---")
    vol_df = isolator.analyze_by_volatility()
    print(vol_df.to_string(index=False))
    
    # Find best filters
    print("\n--- BEST SINGLE FILTERS ---")
    best_filters = isolator.find_best_filter_combination()
    for f in best_filters:
        print(f"{f['filter']}: {f['value']} -> {f['mean_r']:.4f} R ({f['count']} trades)")
    
    # Test combined filters (incrementally)
    print("\n--- FILTER STACKING (Progressive Rejection) ---")
    
    # Stack 1: Best event type only
    if best_filters:
        stack1 = [best_filters[1]]  # Event type
        filtered1 = isolator.apply_filter_stack(stack1)
        if len(filtered1) > 0:
            rejection1 = 1 - len(filtered1) / baseline['count']
            mean_r1 = filtered1['time_20_r'].mean()
            print(f"Stack 1 (Event Type): {len(filtered1)} trades ({rejection1*100:.1f}% rejected) -> {mean_r1:.4f} R")
    
    # Stack 2: Event type + Hour window
    if len(best_filters) >= 2:
        stack2 = [best_filters[1], best_filters[0]]  # Event type + Hour
        filtered2 = isolator.apply_filter_stack(stack2)
        if len(filtered2) > 0:
            rejection2 = 1 - len(filtered2) / baseline['count']
            mean_r2 = filtered2['time_20_r'].mean()
            print(f"Stack 2 (+Hour): {len(filtered2)} trades ({rejection2*100:.1f}% rejected) -> {mean_r2:.4f} R")
    
    # Stack 3: All three
    if len(best_filters) >= 3:
        stack3 = [best_filters[1], best_filters[0], best_filters[2]]
        filtered3 = isolator.apply_filter_stack(stack3)
        if len(filtered3) > 0:
            rejection3 = 1 - len(filtered3) / baseline['count']
            mean_r3 = filtered3['time_20_r'].mean()
            print(f"Stack 3 (+Size): {len(filtered3)} trades ({rejection3*100:.1f}% rejected) -> {mean_r3:.4f} R")
    
    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING EDGE REPORT...")
    
    report = f"""# Phase 4: Edge Isolation Report

## Baseline (DEEP_RETRACE, time_20 exit)
- **Events**: {baseline['count']}
- **Mean R**: {baseline['mean_r']:.4f}
- **Win Rate**: {baseline['win_rate']*100:.1f}%

---

## Analysis by Time of Day
{hour_df.to_string(index=False)}

## Analysis by Event Type
{type_df.to_string(index=False)}

## Analysis by Sweep Size
{size_df.to_string(index=False)}

## Analysis by Volatility
{vol_df.to_string(index=False)}

---

## Best Single Filters
"""
    for f in best_filters:
        report += f"- **{f['filter']}**: {f['value']} -> {f['mean_r']:.4f} R ({f['count']} trades)\n"
    
    report += """
---

## Next Steps
1. Cross-validate on OOS data
2. Test filter interactions
3. Define final rejection thresholds
"""
    
    with open(OUTPUT_PATH, 'w') as f:
        f.write(report)
    
    print(f"Saved to {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
