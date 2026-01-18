"""
Phase 3B.4: Friction Tests

Stress-test the winning regime models against execution friction.

Models:
1. Post-Displacement (Zones Only)
2. Comp & Approaching (Zones Only)

Variants:
- Baseline: Standard execution
- Delay: Features at t predict label from t+1 (1 bar latency)
- Slippage: Label requires return > 0.5 pts (2 ticks) to be positive
- Combined: Delay + Slippage

Pass Criteria:
- AUC Degradation <= 0.03
- AUC remains > 0.52 (better than naive)
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


def build_regime_masks(data, prim_feats, zone_feats):
    """Define regime masks."""
    # Post-Displacement
    mask_post_disp = prim_feats['is_displacement'].astype(bool).values
    
    # Comp & Approaching
    mask_comp_appr = (
        (prim_feats['is_compressed'].astype(bool)) &
        (zone_feats['approaching'].astype(bool)) &
        (np.abs(zone_feats['dist_to_nearest_zone']) <= 10.0)
    ).values
    
    return {
        'Post-Displacement': mask_post_disp,
        'Comp & Approaching': mask_comp_appr
    }


def evaluate_with_friction(X, y_raw_returns, cv, delay=False, slippage_threshold=0.0):
    """
    Evaluate model with friction injection.
    
    Args:
        X: Feature matrix
        y_raw_returns: Raw future returns (float)
        delay: If True, shift features back by 1 (or labels forward)
        slippage_threshold: Return must exceed this to be 'Positive'
    """
    
    # Apply Delay
    if delay:
        # Features at t use label from t+1
        # So we align X[0:-1] with y[1:]
        X_adj = X[:-1]
        y_ret_adj = y_raw_returns[1:]
    else:
        X_adj = X
        y_ret_adj = y_raw_returns
        
    # Apply Slippage to Label
    # Label is 1 if return > threshold, else 0
    y_target = (y_ret_adj > slippage_threshold).astype(int)
    
    fold_aucs = []
    
    for train_idx, test_idx in cv.split(len(X_adj)):
        X_train, y_train = X_adj[train_idx], y_target[train_idx]
        X_test, y_test = X_adj[test_idx], y_target[test_idx]
        
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
            weights = weights - 0.1 * ((X_train_s.T @ error) / len(y_train) + 0.01 * weights)
            bias = bias - 0.1 * np.mean(error)
        
        test_proba = 1 / (1 + np.exp(-np.clip(X_test_s @ weights + bias, -500, 500)))
        fold_aucs.append(compute_auc(y_test, test_proba))
    
    return np.mean(fold_aucs) if fold_aucs else 0.5


def main():
    print("=" * 60)
    print("PHASE 3B.4: FRICTION TESTS")
    print("=" * 60)
    
    data = load_data()
    subset = data.iloc[:25000] # Use 25k to match 3B.3 baseline
    print(f"Using {len(subset)} bars")
    
    # Build Features
    print("Building features...")
    zone_builder = ZoneRelativeBuilder()
    zone_feats = zone_builder.build_features(subset)
    
    disp_detector = DisplacementDetector(threshold_zscore=2.0)
    comp_detector = CompressionDetector()
    
    prim_feats = pd.DataFrame(index=subset.index)
    prim_feats['is_displacement'] = False
    prim_feats['is_compressed'] = False
    
    for i in range(len(subset)):
        d = disp_detector.compute_at(i, subset)
        if d: prim_feats.iloc[i, 0] = d.is_displacement
        c = comp_detector.compute_at(i, subset)
        if c: prim_feats.iloc[i, 1] = c.is_compressed
        
    # Get Raw Returns for Label Generation
    # We need raw returns to apply slippage threshold
    horizon = 10
    future_returns = subset['close'].shift(-horizon) - subset['close']
    # Drop NaNs at end
    valid_idx = future_returns.dropna().index
    
    # Align all
    idx = zone_feats.index.intersection(prim_feats.index).intersection(valid_idx)
    
    zone_feats = zone_feats.loc[idx]
    prim_feats = prim_feats.loc[idx]
    future_returns = future_returns.loc[idx].values
    
    # Define Regimes
    regimes = build_regime_masks(subset.loc[idx], prim_feats, zone_feats)
    
    results = []
    
    cv = PurgedWalkForward(n_splits=3, purge_bars=30)
    
    print("\nRunning Friction Tests...")
    print(f"{'Regime':<20} | {'Variant':<15} | {'AUC':<6} | {'ΔBase':<6} | {'Status'}")
    print("-" * 75)
    
    for regime_name, mask in regimes.items():
        regime_idx = np.where(mask)[0]
        
        if len(regime_idx) < 300:
            print(f"{regime_name}: SKIP (N={len(regime_idx)})")
            continue
            
        print(f"\n>> {regime_name} (N={len(regime_idx)})")
        
        # Use Zones Only Features per Phase 3B.3 result
        X_regime = zone_feats.iloc[regime_idx].select_dtypes(include=[np.number]).values
        y_regime = future_returns[regime_idx]
        
        # 1. Baseline
        auc_base = evaluate_with_friction(X_regime, y_regime, cv, delay=False, slippage_threshold=0.0)
        print(f"{'':<20} | {'Baseline':<15} | {auc_base:.3f} | {'-':<6} | Reference")
        results.append({'regime': regime_name, 'variant': 'Baseline', 'auc': auc_base, 'delta': 0})
        
        # 2. Delay
        auc_delay = evaluate_with_friction(X_regime, y_regime, cv, delay=True, slippage_threshold=0.0)
        delta = auc_delay - auc_base
        status = "PASS" if abs(delta) <= 0.03 else "FAIL"
        print(f"{'':<20} | {'Delay (+1 bar)':<15} | {auc_delay:.3f} | {delta:+.3f} | {status}")
        results.append({'regime': regime_name, 'variant': 'Delay', 'auc': auc_delay, 'delta': delta})

        # 3. Slippage
        auc_slip = evaluate_with_friction(X_regime, y_regime, cv, delay=False, slippage_threshold=0.5)
        delta = auc_slip - auc_base
        status = "PASS" if abs(delta) <= 0.03 else "FAIL"
        print(f"{'':<20} | {'Slippage (>0.5)':<15} | {auc_slip:.3f} | {delta:+.3f} | {status}")
        results.append({'regime': regime_name, 'variant': 'Slippage', 'auc': auc_slip, 'delta': delta})

        # 4. Combined
        auc_comb = evaluate_with_friction(X_regime, y_regime, cv, delay=True, slippage_threshold=0.5)
        delta = auc_comb - auc_base
        status = "PASS" if abs(delta) <= 0.03 else "FAIL"
        print(f"{'':<20} | {'Combined':<15} | {auc_comb:.3f} | {delta:+.3f} | {status}")
        results.append({'regime': regime_name, 'variant': 'Combined', 'auc': auc_comb, 'delta': delta})

    # Save Report
    report_path = project_root / 'ml_lab' / 'reports' / 'phase3b4_friction.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3B.4: Friction Test Results\n\n")
        f.write("## Protocol\n")
        f.write("- Delay: Features[t] -> Label[t+1]\n")
        f.write("- Slippage: Label requires Return > 0.5 pts\n")
        f.write("- Pass Criteria: Degradation <= 0.03\n\n")
        f.write("## Results\n\n")
        f.write("| Regime | Variant | AUC | Δ | Status |\n")
        f.write("|--------|---------|-----|---|--------|\n")
        for r in results:
            status = "PASS" if abs(r['delta']) <= 0.03 else "FAIL" # Re-eval for report
            if r['variant'] == 'Baseline': status = "-"
            f.write(f"| {r['regime']} | {r['variant']} | {r['auc']:.3f} | {r['delta']:+.3f} | {status} |\n")
            
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
