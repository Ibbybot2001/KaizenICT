"""
Phase 3B.2: Ablation Grid

Question: Which primitives are doing real work, which are passengers?

Ablate one family at a time within winning slices:
- Remove displacement
- Remove compression  
- Remove zones
- Remove liquidity

Judgment criteria:
- ΔAUC >= 0.02: ESSENTIAL
- ΔAUC 0.01-0.02: SUPPORTING
- ΔAUC < 0.01: DECORATIVE

Red flag: if NO ablation hurts performance, model is noise.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from ml_lab.ml.zone_relative import ZoneRelativeBuilder
from ml_lab.primitives.displacement import DisplacementDetector
from ml_lab.primitives.compression import CompressionDetector
from ml_lab.ml.label_generator import LabelGenerator, LabelType
from ml_lab.ml.walk_forward import PurgedWalkForward
from ml_lab.ml.interaction_runner import compute_auc


def load_data():
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    return df


def build_features_with_ablation(data: pd.DataFrame, 
                                  exclude_family: str = None) -> pd.DataFrame:
    """
    Build feature matrix, optionally excluding a family.
    
    Args:
        exclude_family: 'displacement', 'compression', 'zones', or None for full
    """
    print(f"Building features (excluding: {exclude_family or 'none'})...")
    
    zone_builder = ZoneRelativeBuilder()
    features = zone_builder.build_features(data)
    
    # Add displacement features
    if exclude_family != 'displacement':
        disp_detector = DisplacementDetector(threshold_zscore=2.0)
        features['disp_zscore'] = 0.0
        features['is_displacement'] = False
        for bar_idx in range(len(data)):
            disp = disp_detector.compute_at(bar_idx, data)
            if disp:
                features.iloc[bar_idx, features.columns.get_loc('disp_zscore')] = disp.range_zscore
                features.iloc[bar_idx, features.columns.get_loc('is_displacement')] = disp.is_displacement
    
    # Add compression features
    if exclude_family != 'compression':
        comp_detector = CompressionDetector()
        features['comp_score'] = 0.0
        features['is_compressed'] = False
        for bar_idx in range(len(data)):
            comp = comp_detector.compute_at(bar_idx, data)
            if comp:
                features.iloc[bar_idx, features.columns.get_loc('comp_score')] = comp.compression_score
                features.iloc[bar_idx, features.columns.get_loc('is_compressed')] = comp.is_compressed
    
    # Zone features are in zone_builder output
    # If excluding zones, set zone features to neutral
    if exclude_family == 'zones':
        zone_cols = ['dist_to_nearest_fvg', 'dist_to_nearest_swing', 'dist_to_nearest_zone',
                     'nearest_zone_age', 'inside_zone', 'approaching', 'n_fvg_active', 'n_swing_active']
        for col in zone_cols:
            if col in features.columns:
                features[col] = 0 if col not in ['inside_zone', 'approaching'] else False
    
    return features


def evaluate_model(X, y, cv):
    """Evaluate logistic regression model."""
    fold_aucs = []
    
    for train_idx, test_idx in cv.split(len(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        if len(X_test) < 50:
            continue
        
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
    
    return np.mean(fold_aucs) if fold_aucs else 0.5


def main():
    print("=" * 60)
    print("PHASE 3B.2: ABLATION GRID")
    print("=" * 60)
    print()
    print("Question: Which primitives are essential vs decorative?")
    print()
    
    data = load_data()
    subset = data.iloc[:20000]  # Smaller for speed (repeated builds)
    print(f"Using {len(subset)} bars")
    
    # Generate labels
    label_gen = LabelGenerator()
    labels_df = label_gen.generate_labels(subset, LabelType.FORWARD_RETURN_SIGN, horizon=10)
    
    # Build FULL feature set first (for slice definitions)
    full_features = build_features_with_ablation(subset, exclude_family=None)
    
    common_idx = full_features.index.intersection(labels_df.index)
    full_features = full_features.loc[common_idx]
    y_all = labels_df.loc[common_idx]['label'].values
    
    # Define winning slices
    slices = {
        'post_displacement': full_features['is_displacement'].astype(bool).values if 'is_displacement' in full_features.columns else np.zeros(len(full_features), dtype=bool),
        'compressed': full_features['is_compressed'].astype(bool).values if 'is_compressed' in full_features.columns else np.zeros(len(full_features), dtype=bool),
        'high_context': (np.abs(full_features['dist_to_nearest_zone']) <= 5).values & (
            (full_features.get('is_displacement', pd.Series(False, index=full_features.index)).astype(bool)) | 
            (full_features.get('is_compressed', pd.Series(False, index=full_features.index)).astype(bool))
        ).values,
    }
    
    cv = PurgedWalkForward(n_splits=3, purge_bars=30)
    
    # Run ablation for each family
    families_to_ablate = [None, 'displacement', 'compression', 'zones']
    
    results = []
    
    for slice_name, mask in slices.items():
        slice_idx = np.where(mask)[0]
        if len(slice_idx) < 300:
            print(f"\n{slice_name}: SKIP (only {len(slice_idx)} samples)")
            continue
        
        print(f"\n{'=' * 40}")
        print(f"SLICE: {slice_name} (N={len(slice_idx)})")
        print(f"{'=' * 40}")
        
        baseline_auc = None
        
        for exclude in families_to_ablate:
            # Rebuild features with exclusion
            ablated_features = build_features_with_ablation(subset, exclude_family=exclude)
            ablated_features = ablated_features.loc[common_idx].fillna(0)
            
            # Get slice data
            X_slice = ablated_features.iloc[slice_idx].select_dtypes(include=[np.number]).values
            y_slice = y_all[slice_idx]
            
            auc = evaluate_model(X_slice, y_slice, cv)
            
            label = f"FULL" if exclude is None else f"NO_{exclude.upper()}"
            
            if exclude is None:
                baseline_auc = auc
                delta = 0.0
                impact = "baseline"
            else:
                delta = auc - baseline_auc
                if delta <= -0.02:
                    impact = "ESSENTIAL"
                elif delta <= -0.01:
                    impact = "SUPPORTING"
                else:
                    impact = "decorative"
            
            print(f"  {label:20s} | AUC={auc:.3f} | Δ={delta:+.3f} | {impact}")
            
            results.append({
                'slice': slice_name,
                'ablation': label,
                'auc': auc,
                'delta': delta,
                'impact': impact,
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    
    essential_primitives = set()
    supporting_primitives = set()
    
    for r in results:
        if r['impact'] == 'ESSENTIAL':
            essential_primitives.add(r['ablation'].replace('NO_', '').lower())
        elif r['impact'] == 'SUPPORTING':
            supporting_primitives.add(r['ablation'].replace('NO_', '').lower())
    
    print(f"\nESSENTIAL primitives (Δ <= -0.02): {essential_primitives or 'none'}")
    print(f"SUPPORTING primitives (Δ <= -0.01): {supporting_primitives or 'none'}")
    
    # Red flag check
    has_essential = any(r['impact'] == 'ESSENTIAL' for r in results if r['ablation'] != 'FULL')
    if not has_essential:
        print("\n⚠️ RED FLAG: No primitive is essential! Model may be learning noise.")
    else:
        print("\n✅ At least one primitive is essential in at least one slice.")
    
    # Save report
    report_path = project_root / 'ml_lab' / 'reports' / 'phase3b2_ablation.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3B.2: Ablation Grid Results\n\n")
        f.write("## Judgment Criteria\n")
        f.write("- ΔAUC <= -0.02: **ESSENTIAL**\n")
        f.write("- ΔAUC -0.01 to -0.02: **SUPPORTING**\n")
        f.write("- ΔAUC > -0.01: decorative\n\n")
        f.write("## Results by Slice\n\n")
        
        current_slice = None
        for r in results:
            if r['slice'] != current_slice:
                current_slice = r['slice']
                f.write(f"\n### {current_slice}\n\n")
                f.write("| Ablation | AUC | Δ | Impact |\n")
                f.write("|----------|-----|---|--------|\n")
            f.write(f"| {r['ablation']} | {r['auc']:.3f} | {r['delta']:+.3f} | {r['impact']} |\n")
        
        f.write("\n## Summary\n\n")
        f.write(f"- Essential: {essential_primitives or 'none'}\n")
        f.write(f"- Supporting: {supporting_primitives or 'none'}\n")
    
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
