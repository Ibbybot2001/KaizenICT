"""
Run Phase 3B Conditional Slice Evaluation

Tests whether concepts beat naive vol INSIDE the states where they apply.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from ml_lab.ml.conditional_slicer import ConditionalSlicer, SliceEvaluator
from ml_lab.ml.label_generator import LabelType


def load_data():
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    print(f"Loaded {len(df)} bars")
    return df


def main():
    print("=" * 60)
    print("PHASE 3B: CONDITIONAL SLICE EVALUATION")
    print("=" * 60)
    print()
    print("Question: Do concepts beat naive vol INSIDE their states?")
    print()
    
    data = load_data()
    
    # Use subset for speed (can expand later)
    subset_size = 30000  # Reduced for faster computation
    subset = data.iloc[:subset_size]
    print(f"Using {len(subset)} bars for slice evaluation")
    
    # Create slicer with conservative thresholds
    print("\nCreating conditional slicer...")
    slicer = ConditionalSlicer(
        data=subset,
        zone_proximity_pts=15.0,  # Within 15 points of zone
        post_touch_bars=5,  # 5 bars after displacement
        displacement_threshold=2.0,  # 2 std displacement
        compression_threshold=0.3,  # Bottom 30% range
    )
    
    # Compute gating flags
    flags = slicer.compute_gating_flags()
    
    # Print flag distributions
    print("\n--- SLICE SIZES ---")
    for col in flags.columns:
        pct = flags[col].mean()
        n = flags[col].sum()
        print(f"  {col}: {n} bars ({pct:.1%})")
    
    # Evaluate slices
    print("\n--- SLICE EVALUATION ---")
    evaluator = SliceEvaluator(
        data=subset,
        slicer=slicer,
        label_type=LabelType.FORWARD_RETURN_SIGN,
        horizon=10,
    )
    
    results = evaluator.evaluate_all_slices()
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in results:
        if 'error' in r:
            print(f"{r['slice']}: ERROR - {r['error']}")
        else:
            winner = "CONCEPT" if r['concept_beats_naive'] else "naive"
            delta = r['delta_auc']
            print(f"{r['slice']:20s} | n={r['n_samples']:5d} | "
                  f"concept={r['concept_auc_mean']:.3f} vs naive={r['naive_auc_mean']:.3f} | "
                  f"Δ={delta:+.3f} | {winner}")
    
    # Generate report
    report = evaluator.generate_report()
    report_path = project_root / 'ml_lab' / 'reports' / 'phase3b_slices.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
