"""
Phase 3B.3: Zone Role Reassignment & Regime Modeling

Goal: Move zones from "predictors" to "state selectors".
Test models inside specific regimes with appropriate feature sets.

Regimes:
1. Pre-Displacement Near Zone
   - State: NOT displaced AND near zone (<= 5 pts)
   - Question: Do zones + compression predict breakout?
   
2. Post-Displacement
   - State: Displacement active
   - Question: Does momentum continue? (Zones explicitly excluded)
   
3. Compressed & Approaching
   - State: Compressed AND Approaching Zone AND Dist <= 10
   - Question: Do zones help predict expansion direction?

Feature Sets:
- BASE: Displacement + Compression features
- ZONES: Zone-relative features (dist, inside, age)
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


def build_feature_sets(data: pd.DataFrame):
    """Build separated feature sets for flexible combination."""
    print("Building feature sets...")
    
    # Zone features (Fixed: capped at 20.0)
    zone_builder = ZoneRelativeBuilder()
    zone_feats = zone_builder.build_features(data)
    
    # Core Primitive features
    prim_feats = pd.DataFrame(index=data.index)
    
    disp_detector = DisplacementDetector(threshold_zscore=2.0)
    comp_detector = CompressionDetector()
    
    prim_feats['disp_zscore'] = 0.0
    prim_feats['is_displacement'] = False
    prim_feats['comp_score'] = 0.0
    prim_feats['is_compressed'] = False
    
    print("Computing primitives...")
    for bar_idx in range(len(data)):
        if bar_idx % 10000 == 0:
            print(f"  {bar_idx}/{len(data)}")
            
        disp = disp_detector.compute_at(bar_idx, data)
        if disp:
            prim_feats.iloc[bar_idx, prim_feats.columns.get_loc('disp_zscore')] = disp.range_zscore
            prim_feats.iloc[bar_idx, prim_feats.columns.get_loc('is_displacement')] = disp.is_displacement
        
        comp = comp_detector.compute_at(bar_idx, data)
        if comp:
            prim_feats.iloc[bar_idx, prim_feats.columns.get_loc('comp_score')] = comp.compression_score
            prim_feats.iloc[bar_idx, prim_feats.columns.get_loc('is_compressed')] = comp.is_compressed
            
    return prim_feats, zone_feats


def evaluate_model(X, y, cv):
    """Evaluate logistic regression model."""
    if len(X) < 50:
        return 0.5
        
    fold_aucs = []
    
    for train_idx, test_idx in cv.split(len(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Eliminate constant columns
        std = np.std(X_train, axis=0)
        valid_cols = std > 1e-9
        if not np.any(valid_cols):
            fold_aucs.append(0.5)
            continue
            
        X_train = X_train[:, valid_cols]
        X_test = X_test[:, valid_cols]
        
        # Standardize
        train_mean = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        X_train_s = (X_train - train_mean) / train_std
        X_test_s = (X_test - train_mean) / train_std
        
        # Fit logistic
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        bias = 0.0
        
        # Simple SGD
        for _ in range(100):
            z = X_train_s @ weights + bias
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            error = predictions - y_train
            weights = weights - 0.1 * ((X_train_s.T @ error) / len(y_train) + 0.01 * weights) # L2 reg
            bias = bias - 0.1 * np.mean(error)
        
        test_proba = 1 / (1 + np.exp(-np.clip(X_test_s @ weights + bias, -500, 500)))
        fold_aucs.append(compute_auc(y_test, test_proba))
    
    return np.mean(fold_aucs) if fold_aucs else 0.5


def main():
    print("=" * 60)
    print("PHASE 3B.3: REGIME MODELING")
    print("=" * 60)
    
    data = load_data()
    subset = data.iloc[:25000] # Use 25k for development
    print(f"Using {len(subset)} bars")
    
    # 1. Build Features
    prim_feats, zone_feats = build_feature_sets(subset)
    
    # 2. Generate Labels
    label_gen = LabelGenerator()
    labels = label_gen.generate_labels(subset, LabelType.FORWARD_RETURN_SIGN, horizon=10)
    
    # Align everything
    idx = prim_feats.index.intersection(zone_feats.index).intersection(labels.index)
    prim_feats = prim_feats.loc[idx]
    zone_feats = zone_feats.loc[idx]
    y = labels.loc[idx]['label'].values
    
    # 3. Define Regimes
    print("\nDefining Regimes...")
    
    # Regime 1: Post-Displacement
    # Logic: Event already happened. Signal is pure momentum/continuation.
    mask_post_disp = prim_feats['is_displacement'].astype(bool).values
    
    # Regime 2: Pre-Displacement Near Zone
    # Logic: No event yet, but structure is close. Waiting for reaction.
    mask_pre_disp_near = (
        (~prim_feats['is_displacement'].astype(bool)) & 
        (np.abs(zone_feats['dist_to_nearest_zone']) <= 5.0)
    ).values
    
    # Regime 3: Compressed & Approaching
    # Logic: Coiled energy moving toward structure.
    mask_comp_appr = (
        (prim_feats['is_compressed'].astype(bool)) &
        (zone_feats['approaching'].astype(bool)) &
        (np.abs(zone_feats['dist_to_nearest_zone']) <= 10.0)
    ).values
    
    regimes = {
        'Post-Displacement': mask_post_disp,
        'Pre-Disp Near Zone': mask_pre_disp_near,
        'Comp & Approaching': mask_comp_appr
    }
    
    # 4. Evaluate Models per Regime
    cv = PurgedWalkForward(n_splits=3, purge_bars=30)
    
    results = []
    
    print("\nEvaluating Regimes...")
    print(f"{'Regime':<20} | {'N':>6} | {'Model':<20} | {'AUC':>6}")
    print("-" * 60)
    
    for name, mask in regimes.items():
        regime_idx = np.where(mask)[0]
        
        if len(regime_idx) < 300:
            print(f"{name:<20} | {len(regime_idx):>6} | SKIP (too few)")
            continue
            
        y_slice = y[regime_idx]
        
        # Model A: Primitives Only (Disp + Comp)
        X_prim = prim_feats.iloc[regime_idx].select_dtypes(include=[np.number]).values
        auc_prim = evaluate_model(X_prim, y_slice, cv)
        print(f"{name:<20} | {len(regime_idx):>6} | {'Primitives Only':<20} | {auc_prim:.3f}")
        
        # Model B: Zones Only
        X_zone = zone_feats.iloc[regime_idx].select_dtypes(include=[np.number]).values
        auc_zone = evaluate_model(X_zone, y_slice, cv)
        print(f"{name:<20} | {len(regime_idx):>6} | {'Zones Only':<20} | {auc_zone:.3f}")
        
        # Model C: Full (Prim + Zones)
        X_full = np.hstack([X_prim, X_zone])
        auc_full = evaluate_model(X_full, y_slice, cv)
        print(f"{name:<20} | {len(regime_idx):>6} | {'Full Combined':<20} | {auc_full:.3f}")
        
        results.append({
            'regime': name,
            'auc_prim': auc_prim,
            'auc_zone': auc_zone,
            'auc_full': auc_full
        })
        
    # 5. Report
    report_path = project_root / 'ml_lab' / 'reports' / 'phase3b3_regimes.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3B.3: Regime Modeling Results\n\n")
        f.write("## Approach\n")
        f.write("- **Representation Fix**: Distances capped at 20.0 (no '999')\n")
        f.write("- **Regimes**: Slices defined by state (Post-Disp, Pre-Disp Near, Compressed Approach)\n")
        f.write("- **Models**: Tested Primitives vs Zones vs Full within each regime\n\n")
        f.write("## Results\n\n")
        f.write("| Regime | N | Primitives AUC | Zones AUC | Full AUC | Best |\n")
        f.write("|--------|---|----------------|-----------|----------|------|\n")
        
        for r in results:
            best = max(r['auc_prim'], r['auc_zone'], r['auc_full'])
            if best == r['auc_prim']: winner = "Primitives"
            elif best == r['auc_zone']: winner = "Zones"
            else: winner = "Full"
            
            f.write(f"| {r['regime']} | - | {r['auc_prim']:.3f} | {r['auc_zone']:.3f} | {r['auc_full']:.3f} | {winner} |\n")
            
    print(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
