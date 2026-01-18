"""
Run Phase 3B: Interaction Discovery on real MNQ data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from ml_lab.ml.interaction_runner import InteractionRunner
from ml_lab.ml.label_generator import LabelType


def load_data():
    """Load real MNQ data."""
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    print(f"Loaded {len(df)} bars")
    return df


def main():
    print("=" * 60)
    print("PHASE 3B: INTERACTION DISCOVERY")
    print("=" * 60)
    
    data = load_data()
    
    # Use 50K bars for speed
    subset = data.iloc[:50000]
    print(f"Using {len(subset)} bars for experiments")
    
    # Run experiments
    runner = InteractionRunner(
        data=subset,
        label_type=LabelType.FORWARD_RETURN_SIGN,
        horizon=10,
        auc_threshold=0.54,
        max_auc_gap=0.03
    )
    
    results = runner.run_all_experiments()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    promising = runner.get_promising_models()
    print(f"\nPromising models: {len(promising)}")
    for m in promising:
        print(f"  - {m.model_id}: AUC={m.test_auc_mean:.3f}, gap={m.auc_gap:.3f}")
    
    # Save report
    report = runner.generate_report()
    report_path = project_root / 'ml_lab' / 'reports' / 'phase3b_results.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save JSON
    json_path = project_root / 'ml_lab' / 'runs' / 'phase3b_results.json'
    runner.save_results(json_path)
    print(f"JSON saved to: {json_path}")
    
    return runner


if __name__ == '__main__':
    runner = main()
