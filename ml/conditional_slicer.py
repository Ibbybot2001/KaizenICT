"""
Conditional Slice Evaluation - Phase 3B Step 4

The sample space was wrong. Concepts don't predict globally.
They predict in CONDITIONAL STATES where they apply.

Gating Flags (filters, not features):
- near_zone: price within X points of any active zone
- post_touch: within Y bars of zone touch
- post_displacement: after displacement event
- compressed_regime: during compression

These define WHERE to evaluate, not WHAT to predict.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from pathlib import Path

from ..primitives.zones import ZoneDetector, ZoneType
from ..primitives.displacement import DisplacementDetector
from ..primitives.compression import CompressionDetector
from ..ml.feature_builder import FeatureBuilder, FeatureConfig, PrimitiveFamily
from ..ml.label_generator import LabelGenerator, LabelType
from ..ml.walk_forward import PurgedWalkForward


@dataclass
class SliceStats:
    """Statistics for a conditional slice."""
    slice_name: str
    total_rows: int
    slice_rows: int
    slice_pct: float
    label_pos_rate: float


class ConditionalSlicer:
    """
    Creates conditional sample slices where concepts should apply.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 zone_proximity_pts: float = 15.0,
                 post_touch_bars: int = 5,
                 displacement_threshold: float = 2.0,
                 compression_threshold: float = 0.3):
        self.data = data
        self.zone_proximity_pts = zone_proximity_pts
        self.post_touch_bars = post_touch_bars
        self.displacement_threshold = displacement_threshold
        self.compression_threshold = compression_threshold
        
        # Detectors
        self.zone_detector = ZoneDetector(swing_left=5, swing_right=5)
        self.disp_detector = DisplacementDetector(threshold_zscore=displacement_threshold)
        self.comp_detector = CompressionDetector()
        
        # Pre-computed flags
        self._flags = None
    
    def compute_gating_flags(self) -> pd.DataFrame:
        """
        Compute all gating flags for the dataset.
        
        Returns DataFrame with boolean flags for each condition.
        """
        n = len(self.data)
        flags = pd.DataFrame(index=self.data.index)
        
        # Initialize all flags to False
        flags['near_zone'] = False
        flags['post_displacement'] = False
        flags['compressed'] = False
        flags['near_fvg'] = False
        flags['near_swing'] = False
        
        # Track active zones and their positions
        active_zones = []  # List of (zone_type, upper, lower, created_at)
        
        print("Computing gating flags...")
        
        for bar_idx in range(n):
            if bar_idx % 10000 == 0:
                print(f"  Processing bar {bar_idx}/{n}")
            
            current_close = self.data.iloc[bar_idx]['close']
            
            # Find new zones at this bar
            new_zones = self.zone_detector.compute_all_zones_at(bar_idx, self.data)
            for z in new_zones:
                active_zones.append({
                    'type': z.zone_type,
                    'upper': z.upper,
                    'lower': z.lower,
                    'created': z.created_at,
                })
            
            # Check proximity to any active zone
            near_any = False
            near_fvg = False
            near_swing = False
            
            for zone in active_zones:
                # Zone was created before this bar (past-only)
                if zone['created'] > bar_idx:
                    continue
                
                # Calculate distance to zone
                dist_to_upper = abs(current_close - zone['upper'])
                dist_to_lower = abs(current_close - zone['lower'])
                min_dist = min(dist_to_upper, dist_to_lower)
                
                if min_dist <= self.zone_proximity_pts:
                    near_any = True
                    if zone['type'] in [ZoneType.FVG_BULL, ZoneType.FVG_BEAR]:
                        near_fvg = True
                    elif zone['type'] in [ZoneType.SWING_HIGH, ZoneType.SWING_LOW]:
                        near_swing = True
            
            flags.iloc[bar_idx, flags.columns.get_loc('near_zone')] = near_any
            flags.iloc[bar_idx, flags.columns.get_loc('near_fvg')] = near_fvg
            flags.iloc[bar_idx, flags.columns.get_loc('near_swing')] = near_swing
            
            # Check displacement
            disp = self.disp_detector.compute_at(bar_idx, self.data)
            if disp and disp.is_displacement:
                flags.iloc[bar_idx, flags.columns.get_loc('post_displacement')] = True
                # Also mark next N bars as post-displacement
                for future_bar in range(bar_idx + 1, min(bar_idx + self.post_touch_bars + 1, n)):
                    flags.iloc[future_bar, flags.columns.get_loc('post_displacement')] = True
            
            # Check compression
            comp = self.comp_detector.compute_at(bar_idx, self.data)
            if comp and comp.is_compressed:
                flags.iloc[bar_idx, flags.columns.get_loc('compressed')] = True
        
        # Clean up zone tracking (remove very old zones)
        # This is for memory, not logic
        
        self._flags = flags
        return flags
    
    def get_slice_mask(self, slice_name: str) -> pd.Series:
        """Get boolean mask for a named slice."""
        if self._flags is None:
            self.compute_gating_flags()
        
        if slice_name == 'all':
            return pd.Series([True] * len(self._flags), index=self._flags.index)
        elif slice_name == 'near_zone':
            return self._flags['near_zone']
        elif slice_name == 'near_fvg':
            return self._flags['near_fvg']
        elif slice_name == 'near_swing':
            return self._flags['near_swing']
        elif slice_name == 'post_displacement':
            return self._flags['post_displacement']
        elif slice_name == 'compressed':
            return self._flags['compressed']
        elif slice_name == 'high_context':
            # Intersection: near_zone AND (post_displacement OR compressed)
            return self._flags['near_zone'] & (
                self._flags['post_displacement'] | self._flags['compressed']
            )
        else:
            raise ValueError(f"Unknown slice: {slice_name}")
    
    def get_slice_stats(self, slice_name: str, labels: pd.Series) -> SliceStats:
        """Get statistics for a slice."""
        mask = self.get_slice_mask(slice_name)
        common = mask.index.intersection(labels.index)
        mask = mask.loc[common]
        labels = labels.loc[common]
        
        slice_rows = mask.sum()
        slice_labels = labels[mask]
        
        return SliceStats(
            slice_name=slice_name,
            total_rows=len(labels),
            slice_rows=int(slice_rows),
            slice_pct=float(slice_rows / len(labels)) if len(labels) > 0 else 0,
            label_pos_rate=float(slice_labels.mean()) if len(slice_labels) > 0 else 0,
        )


class SliceEvaluator:
    """
    Evaluates models on conditional slices.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 slicer: ConditionalSlicer,
                 label_type: LabelType = LabelType.FORWARD_RETURN_SIGN,
                 horizon: int = 10,
                 auc_threshold: float = 0.54):
        self.data = data
        self.slicer = slicer
        self.label_type = label_type
        self.horizon = horizon
        self.auc_threshold = auc_threshold
        
        self.label_gen = LabelGenerator()
        self.results = []
    
    def evaluate_slice(self, slice_name: str, 
                       model_type: str = 'logistic') -> Dict:
        """Evaluate a model on a conditional slice."""
        from ..ml.interaction_runner import compute_auc
        
        # Get slice mask
        mask = self.slicer.get_slice_mask(slice_name)
        
        # Build features on FULL data
        config = FeatureConfig(enabled_families={
            PrimitiveFamily.DISPLACEMENT,
            PrimitiveFamily.COMPRESSION,
            PrimitiveFamily.ZONES,
            PrimitiveFamily.LIQUIDITY
        })
        builder = FeatureBuilder(config)
        features_df = builder.build_feature_matrix(self.data)
        
        # Generate labels on FULL data
        labels_df = self.label_gen.generate_labels(
            self.data, self.label_type, self.horizon
        )
        
        # Align and apply slice mask
        common_idx = features_df.index.intersection(labels_df.index).intersection(mask.index)
        slice_mask = mask.loc[common_idx]
        
        # Filter to slice
        slice_idx = common_idx[slice_mask]
        
        if len(slice_idx) < 500:
            return {
                'slice': slice_name,
                'error': f'Insufficient samples: {len(slice_idx)}',
                'n_samples': len(slice_idx),
            }
        
        X = features_df.loc[slice_idx].drop(columns=['bar_idx'], errors='ignore').fillna(0)
        y = labels_df.loc[slice_idx]['label'].values
        
        # Also get naive vol features for C1 comparison
        naive_df = pd.DataFrame(index=self.data.index)
        naive_df['range'] = self.data['high'] - self.data['low']
        naive_df['return'] = self.data['close'].pct_change()
        naive_df['rolling_range_5'] = naive_df['range'].rolling(5).mean()
        naive_df['rolling_range_20'] = naive_df['range'].rolling(20).mean()
        naive_df = naive_df.fillna(0)
        X_naive = naive_df.loc[slice_idx].values
        
        # Cross-validate
        cv = PurgedWalkForward(n_splits=3, purge_bars=max(self.horizon + 20, 30))
        
        concept_aucs = []
        naive_aucs = []
        
        for train_idx, test_idx in cv.split(len(X)):
            X_train, y_train = X.iloc[train_idx].values, y[train_idx]
            X_test, y_test = X.iloc[test_idx].values, y[test_idx]
            
            X_naive_train = X_naive[train_idx]
            X_naive_test = X_naive[test_idx]
            
            # Standardize
            train_mean = np.mean(X_train, axis=0)
            train_std = np.std(X_train, axis=0) + 1e-8
            X_train_s = (X_train - train_mean) / train_std
            X_test_s = (X_test - train_mean) / train_std
            
            naive_mean = np.mean(X_naive_train, axis=0)
            naive_std = np.std(X_naive_train, axis=0) + 1e-8
            X_naive_train_s = (X_naive_train - naive_mean) / naive_std
            X_naive_test_s = (X_naive_test - naive_mean) / naive_std
            
            # Fit concept model
            _, concept_proba = self._fit_logistic(X_train_s, y_train, X_test_s)
            concept_auc = compute_auc(y_test, concept_proba)
            concept_aucs.append(concept_auc)
            
            # Fit naive vol model
            _, naive_proba = self._fit_logistic(X_naive_train_s, y_train, X_naive_test_s)
            naive_auc = compute_auc(y_test, naive_proba)
            naive_aucs.append(naive_auc)
        
        result = {
            'slice': slice_name,
            'n_samples': len(slice_idx),
            'pct_of_total': len(slice_idx) / len(common_idx),
            'concept_auc_mean': np.mean(concept_aucs),
            'concept_auc_std': np.std(concept_aucs),
            'naive_auc_mean': np.mean(naive_aucs),
            'naive_auc_std': np.std(naive_aucs),
            'concept_beats_naive': np.mean(concept_aucs) > np.mean(naive_aucs),
            'delta_auc': np.mean(concept_aucs) - np.mean(naive_aucs),
        }
        
        self.results.append(result)
        return result
    
    def _fit_logistic(self, X_train, y_train, X_test):
        """Simple logistic regression."""
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        bias = 0.0
        
        reg_lambda = 1.0
        lr = 0.1
        n_iter = 100
        
        for _ in range(n_iter):
            z = X_train @ weights + bias
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
            error = predictions - y_train
            grad_w = (X_train.T @ error) / len(y_train) + reg_lambda * weights
            grad_b = np.mean(error)
            
            weights = weights - lr * grad_w
            bias = bias - lr * grad_b
        
        train_proba = 1 / (1 + np.exp(-np.clip(X_train @ weights + bias, -500, 500)))
        test_proba = 1 / (1 + np.exp(-np.clip(X_test @ weights + bias, -500, 500)))
        
        return train_proba, test_proba
    
    def evaluate_all_slices(self) -> List[Dict]:
        """Evaluate all predefined slices."""
        slices = [
            'all',  # Global baseline
            'near_zone',
            'near_fvg',
            'near_swing',
            'post_displacement',
            'compressed',
            'high_context',  # near_zone AND (displacement OR compressed)
        ]
        
        for slice_name in slices:
            print(f"\nEvaluating slice: {slice_name}")
            try:
                result = self.evaluate_slice(slice_name)
                print(f"  n={result.get('n_samples', 0)}, "
                      f"concept_AUC={result.get('concept_auc_mean', 0):.3f}, "
                      f"naive_AUC={result.get('naive_auc_mean', 0):.3f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                self.results.append({'slice': slice_name, 'error': str(e)})
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate slice evaluation report."""
        lines = [
            "# Phase 3B Conditional Slice Evaluation",
            "",
            "**Question**: Do concepts beat naive vol INSIDE the states where they apply?",
            "",
            "## Results",
            "",
            "| Slice | N | % Data | Concept AUC | Naive AUC | Δ AUC | Winner |",
            "|-------|---|--------|-------------|-----------|-------|--------|",
        ]
        
        for r in self.results:
            if 'error' in r:
                lines.append(f"| {r['slice']} | - | - | ERROR | - | - | - |")
            else:
                winner = "CONCEPT" if r['concept_beats_naive'] else "NAIVE"
                lines.append(
                    f"| {r['slice']} | {r['n_samples']} | {r['pct_of_total']:.1%} | "
                    f"{r['concept_auc_mean']:.3f} | {r['naive_auc_mean']:.3f} | "
                    f"{r['delta_auc']:+.3f} | {winner} |"
                )
        
        return "\n".join(lines)
