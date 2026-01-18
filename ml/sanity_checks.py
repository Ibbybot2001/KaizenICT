"""
Phase 3B Sanity Checks - per protocol

1. Label balance per fold
2. Feature leakage scan
3. Sparse feature audit
4. Null/inf audit
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from ml_lab.ml.feature_builder import FeatureBuilder, FeatureConfig, PrimitiveFamily
from ml_lab.ml.label_generator import LabelGenerator, LabelType
from ml_lab.ml.walk_forward import PurgedWalkForward


def load_data():
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    return df.iloc[:50000]


def run_sanity_checks():
    print("=" * 60)
    print("PHASE 3B SANITY CHECKS")
    print("=" * 60)
    
    data = load_data()
    print(f"Data: {len(data)} bars")
    
    # Build features
    config = FeatureConfig(enabled_families={
        PrimitiveFamily.DISPLACEMENT,
        PrimitiveFamily.COMPRESSION,
        PrimitiveFamily.ZONES,
        PrimitiveFamily.LIQUIDITY
    })
    builder = FeatureBuilder(config)
    features_df = builder.build_feature_matrix(data)
    
    # Generate labels
    label_gen = LabelGenerator()
    labels_df = label_gen.generate_labels(data, LabelType.FORWARD_RETURN_SIGN, horizon=10)
    
    # Align
    common_idx = features_df.index.intersection(labels_df.index)
    X = features_df.loc[common_idx].drop(columns=['bar_idx'], errors='ignore').fillna(0)
    y = labels_df.loc[common_idx]['label']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label shape: {len(y)}")
    
    # CHECK 1: Label balance per fold
    print("\n--- CHECK 1: Label Balance per Fold ---")
    cv = PurgedWalkForward(n_splits=5, purge_bars=30)
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(len(X))):
        train_pos = y.iloc[train_idx].mean()
        test_pos = y.iloc[test_idx].mean()
        print(f"Fold {fold_id}: train={train_pos:.1%} pos, test={test_pos:.1%} pos")
    
    # CHECK 2: Sparse feature audit
    print("\n--- CHECK 2: Sparse Feature Audit ---")
    sparsity = (X == 0).mean()
    sparse_features = sparsity[sparsity > 0.99].sort_values(ascending=False)
    if len(sparse_features) > 0:
        print("WARNING: Features with >99% zeros:")
        for feat, sparse_pct in sparse_features.items():
            print(f"  {feat}: {sparse_pct:.1%} zeros")
    else:
        print("OK: No extremely sparse features")
    
    # Non-zero stats for zone/liquidity
    zone_liq_cols = [c for c in X.columns if 'zone' in c or 'liq' in c]
    print("\nZone/Liquidity feature non-zero rates:")
    for col in zone_liq_cols:
        nonzero_rate = (X[col] != 0).mean()
        print(f"  {col}: {nonzero_rate:.1%} non-zero")
    
    # CHECK 3: Null/Inf audit
    print("\n--- CHECK 3: Null/Inf Audit ---")
    null_count = X.isna().sum().sum()
    inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
    print(f"Null values: {null_count}")
    print(f"Inf values: {inf_count}")
    
    # CHECK 4: Class balance overall
    print("\n--- CHECK 4: Overall Class Balance ---")
    print(f"Positive class rate: {y.mean():.1%}")
    
    # CHECK 5: Feature correlation with label
    print("\n--- CHECK 5: Feature-Label Correlation (top 10) ---")
    correlations = {}
    for col in X.columns:
        correlations[col] = np.corrcoef(X[col].values, y.values)[0, 1]
    
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for feat, corr in sorted_corr:
        print(f"  {feat}: {corr:.4f}")
    
    print("\n" + "=" * 60)
    print("SANITY CHECKS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    run_sanity_checks()
