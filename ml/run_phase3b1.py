"""
Run Phase 3B.1: Representation Fix with Zone-Relative Features

This fixes the representation problem identified in Phase 3B:
- Zones as coordinate systems, not flags
- Distance features (signed)
- Tighter slices (3-5 pts instead of 15)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from ml.zone_relative import ZoneRelativeBuilder
from primitives.displacement import DisplacementDetector
from primitives.compression import CompressionDetector
from ml.label_generator import LabelGenerator, LabelType
from ml.walk_forward import PurgedWalkForward
from ml.interaction_runner import compute_auc


def load_data():
    path = project_root / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    return df


def build_all_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build combined feature matrix with zone-relative features."""
    print("Building zone-relative features...")
    zone_builder = ZoneRelativeBuilder()
    zone_features = zone_builder.build_features(data)
    
    print("Adding displacement/compression features...")
    disp_detector = DisplacementDetector(threshold_zscore=2.0)
    comp_detector = CompressionDetector()
    
    # Add displacement/compression
    zone_features['disp_zscore'] = 0.0
    zone_features['is_displacement'] = False
    zone_features['comp_score'] = 0.0
    zone_features['is_compressed'] = False
    
    for bar_idx in range(len(data)):
        disp = disp_detector.compute_at(bar_idx, data)
        if disp:
            zone_features.iloc[bar_idx, zone_features.columns.get_loc('disp_zscore')] = disp.range_zscore
            zone_features.iloc[bar_idx, zone_features.columns.get_loc('is_displacement')] = disp.is_displacement
        
        comp = comp_detector.compute_at(bar_idx, data)
        if comp:
            zone_features.iloc[bar_idx, zone_features.columns.get_loc('comp_score')] = comp.compression_score
            zone_features.iloc[bar_idx, zone_features.columns.get_loc('is_compressed')] = comp.is_compressed
    
    return zone_features


def build_naive_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build naive volatility features for comparison."""
    naive = pd.DataFrame(index=data.index)
    naive['range'] = data['high'] - data['low']
    naive['return'] = data['close'].pct_change()
    naive['rolling_range_5'] = naive['range'].rolling(5).mean()
    naive['rolling_range_20'] = naive['range'].rolling(20).mean()
    return naive.fillna(0)


def evaluate_model(X, y, cv):
    """Evaluate logistic regression model."""
    fold_aucs = []
    
    for train_idx, test_idx in cv.split(len(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Standardize
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0) + 1e-8
        X_train_s = (X_train - train_mean) / train_std
        X_test_s = (X_test - train_mean) / train_std
        
        # Fit logistic
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        bias = 0.0
        
        for _ in range(100):
            z = X_train_s @ weights + bias
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            error = predictions - y_train
            weights = weights - 0.1 * ((X_train_s.T @ error) / len(y_train) + weights)
            bias = bias - 0.1 * np.mean(error)
        
        test_proba = 1 / (1 + np.exp(-np.clip(X_test_s @ weights + bias, -500, 500)))
        fold_aucs.append(compute_auc(y_test, test_proba))
    
    return np.mean(fold_aucs), np.std(fold_aucs)


def main():
    print("=" * 60)
    print("PHASE 3B.1: REPRESENTATION FIX")
    print("=" * 60)
    print()
    print("Using zone-relative features (distance, age, inside)")
    print("Tighter slices: 3-5 pts instead of 15")
    print()
    
    data = load_data()
    
    # Use 25K bars for speed
    subset = data.iloc[:25000]
    print(f"Using {len(subset)} bars")
    
    # Build features
    features = build_all_features(subset)
    naive_features = build_naive_features(subset)
    
    # Generate labels
    label_gen = LabelGenerator()
    labels_df = label_gen.generate_labels(subset, LabelType.FORWARD_RETURN_SIGN, horizon=10)
    
    # Align
    common_idx = features.index.intersection(labels_df.index)
    features = features.loc[common_idx]
    naive_features = naive_features.loc[common_idx]
    y = labels_df.loc[common_idx]['label'].values
    
    # Fill NaNs
    features = features.fillna(0)
    
    # Cross-validation setup
    cv = PurgedWalkForward(n_splits=3, purge_bars=30)
    
    # Define tighter slices
    slices = {
        'all': np.ones(len(features), dtype=bool),
        'near_zone_3pt': np.abs(features['dist_to_nearest_zone']) <= 3,
        'near_zone_5pt': np.abs(features['dist_to_nearest_zone']) <= 5,
        'near_fvg_3pt': np.abs(features['dist_to_nearest_fvg']) <= 3,
        'inside_zone': features['inside_zone'].astype(bool),
        'approaching_zone': (features['approaching'].astype(bool)) & (np.abs(features['dist_to_nearest_zone']) <= 5),
        'post_displacement': features['is_displacement'].astype(bool),
        'compressed': features['is_compressed'].astype(bool),
        'high_context': (np.abs(features['dist_to_nearest_zone']) <= 5) & 
                        (features['is_displacement'] | features['is_compressed']),
    }
    
    # Evaluate each slice
    print("\n--- SLICE SIZES ---")
    for name, mask in slices.items():
        pct = mask.sum() / len(mask)
        print(f"  {name}: {mask.sum()} bars ({pct:.1%})")
    
    print("\n--- EVALUATION ---")
    print()
    print(f"{'Slice':<25} | {'N':>6} | {'Concept AUC':>11} | {'Naive AUC':>10} | {'Δ':>7} | {'Winner':>8}")
    print("-" * 80)
    
    results = []
    
    for name, mask in slices.items():
        slice_idx = np.where(mask)[0]
        
        if len(slice_idx) < 500:
            print(f"{name:<25} | {len(slice_idx):>6} | {'SKIP':>11} | {'':>10} | {'':>7} | {'too few':>8}")
            continue
        
        # Prepare data
        X_concept = features.iloc[slice_idx].select_dtypes(include=[np.number]).values
        X_naive = naive_features.iloc[slice_idx].values
        y_slice = y[slice_idx]
        
        # Evaluate concept model
        concept_auc, concept_std = evaluate_model(X_concept, y_slice, cv)
        
        # Evaluate naive model
        naive_auc, naive_std = evaluate_model(X_naive, y_slice, cv)
        
        delta = concept_auc - naive_auc
        winner = "CONCEPT" if delta > 0 else "naive"
        
        print(f"{name:<25} | {len(slice_idx):>6} | {concept_auc:>11.3f} | {naive_auc:>10.3f} | {delta:>+7.3f} | {winner:>8}")
        
        results.append({
            'slice': name,
            'n': len(slice_idx),
            'concept_auc': concept_auc,
            'naive_auc': naive_auc,
            'delta': delta,
            'winner': winner,
        })
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    concept_wins = sum(1 for r in results if r['winner'] == 'CONCEPT')
    print(f"\nConcept wins: {concept_wins}/{len(results)} slices")
    
    if concept_wins > 0:
        print("\nSlices where concepts beat naive vol:")
        for r in results:
            if r['winner'] == 'CONCEPT':
                print(f"  {r['slice']}: Δ = {r['delta']:+.3f}")
    
    # Save report
    report_path = project_root / 'reports' / 'phase3b1_results.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3B.1: Representation Fix Results\n\n")
        f.write("## Zone-Relative Features\n")
        f.write("- dist_to_nearest_zone (signed)\n")
        f.write("- dist_to_nearest_fvg (signed)\n")
        f.write("- dist_to_nearest_swing (signed)\n")
        f.write("- zone_age, inside_zone, approaching\n\n")
        f.write("## Results\n\n")
        f.write(f"| Slice | N | Concept AUC | Naive AUC | Δ | Winner |\n")
        f.write("|-------|---|-------------|-----------|---|--------|\n")
        for r in results:
            f.write(f"| {r['slice']} | {r['n']} | {r['concept_auc']:.3f} | {r['naive_auc']:.3f} | {r['delta']:+.3f} | {r['winner']} |\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
