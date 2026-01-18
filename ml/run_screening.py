"""
Run Phase 3A Concept Screening on REAL MNQ Data

Tests both:
1. forward_return_sign (direction)
2. path_expansion (movement amplitude)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from ml.concept_screener import ConceptScreener
from ml.label_generator import LabelType


def load_mnq_data():
    """Load real MNQ 1-minute data from canonical path."""
    path = project_root / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    print(f"Loading data from: {path}")
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    
    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def run_screening_direction(data, subset_size=50000):
    """Run screening with forward_return_sign label."""
    print("\n" + "=" * 60)
    print("SCREENING: FORWARD_RETURN_SIGN (direction prediction)")
    print("=" * 60)
    
    # Use subset for speed
    subset = data.iloc[:subset_size] if len(data) > subset_size else data
    print(f"Using {len(subset)} bars")
    
    screener = ConceptScreener(
        data=subset,
        label_type=LabelType.FORWARD_RETURN_SIGN,
        horizon=10,
        min_accuracy_threshold=0.52
    )
    
    results = screener.screen_all_families()
    
    print("\nRESULTS (Direction):")
    print("-" * 40)
    for r in sorted(results, key=lambda x: x.test_accuracy_mean, reverse=True):
        status = "PASS" if r.passed_threshold else "FAIL"
        print(f"{r.primitive_family:15s} | {r.model_type:8s} | "
              f"Test: {r.test_accuracy_mean:.1%} +/- {r.test_accuracy_std:.1%} | {status}")
    
    return screener, results


def run_screening_expansion(data, subset_size=50000):
    """Run screening with path_expansion label."""
    print("\n" + "=" * 60)
    print("SCREENING: PATH_EXPANSION (movement amplitude)")
    print("=" * 60)
    
    subset = data.iloc[:subset_size] if len(data) > subset_size else data
    print(f"Using {len(subset)} bars")
    
    screener = ConceptScreener(
        data=subset,
        label_type=LabelType.PATH_EXPANSION,
        horizon=20,  # 20 bars to move 15 points
        min_accuracy_threshold=0.52
    )
    
    results = screener.screen_all_families()
    
    print("\nRESULTS (Expansion):")
    print("-" * 40)
    for r in sorted(results, key=lambda x: x.test_accuracy_mean, reverse=True):
        status = "PASS" if r.passed_threshold else "FAIL"
        print(f"{r.primitive_family:15s} | {r.model_type:8s} | "
              f"Test: {r.test_accuracy_mean:.1%} +/- {r.test_accuracy_std:.1%} | {status}")
    
    return screener, results


def main():
    print("=" * 60)
    print("PHASE 3A: CONCEPT VIABILITY SCREENING - REAL MNQ DATA")
    print("=" * 60)
    
    # Load real data
    data = load_mnq_data()
    
    # Run both screenings
    screener_dir, results_dir = run_screening_direction(data)
    screener_exp, results_exp = run_screening_expansion(data)
    
    # Combined report
    print("\n" + "=" * 60)
    print("COMBINED TRUTH TABLE")
    print("=" * 60)
    
    # Aggregate results by family
    dir_pass = {r.primitive_family for r in results_dir if r.passed_threshold}
    exp_pass = {r.primitive_family for r in results_exp if r.passed_threshold}
    
    all_families = set(r.primitive_family for r in results_dir)
    
    print("\n| Family | Direction | Expansion | Status |")
    print("|--------|-----------|-----------|--------|")
    for family in sorted(all_families):
        d = "PASS" if family in dir_pass else "FAIL"
        e = "PASS" if family in exp_pass else "FAIL"
        status = "-> 3B" if (family in dir_pass or family in exp_pass) else "BLOCKED"
        print(f"| {family:12s} | {d:9s} | {e:9s} | {status} |")
    
    # Save reports
    report_dir = project_root / 'reports'
    
    # Direction report
    with open(report_dir / 'phase3a_direction.md', 'w', encoding='utf-8') as f:
        f.write(screener_dir.generate_report())
    
    # Expansion report  
    with open(report_dir / 'phase3a_expansion.md', 'w', encoding='utf-8') as f:
        f.write(screener_exp.generate_report())
    
    # Combined JSON
    import json
    combined = {
        'direction': [r.to_dict() for r in results_dir],
        'expansion': [r.to_dict() for r in results_exp],
    }
    with open(project_root / 'output' / 'results' / 'phase3a_real_data.json', 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\nReports saved to: {report_dir}")
    print("Phase 3A complete.")


if __name__ == '__main__':
    main()
