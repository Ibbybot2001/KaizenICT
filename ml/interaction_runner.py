"""
Phase 3B: Interaction Discovery

Constrained models with mandatory ablation, friction, and stability checks.

PROMOTION CRITERIA (per user specification):
- Direction: AUC ≥ 0.54 AND accuracy ≥ baseline + 1%
- Train-test gap: (train_AUC - test_AUC) ≤ 0.03
- Fold stability: no single fold carrying results

CONTROLS:
- C0: Do-nothing classifier (most common class)
- C1: Naive volatility proxy (no concept features)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import json

from ml.feature_builder import FeatureBuilder, FeatureConfig, PrimitiveFamily
from ml.label_generator import LabelGenerator, LabelType
from ml.walk_forward import PurgedWalkForward
from engine.constants import MIN_SL_POINTS


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    model_id: str
    model_type: str
    features_used: List[str]
    label_type: str
    
    # Per-fold metrics
    fold_accuracies: List[float] = field(default_factory=list)
    fold_aucs: List[float] = field(default_factory=list)
    fold_train_aucs: List[float] = field(default_factory=list)
    
    # Aggregates
    test_accuracy_mean: float = 0.0
    test_accuracy_std: float = 0.0
    test_auc_mean: float = 0.0
    test_auc_std: float = 0.0
    train_auc_mean: float = 0.0
    auc_gap: float = 0.0  # train - test
    
    # Promotion check
    is_promising: bool = False
    promotion_reason: str = ""
    
    n_samples: int = 0
    n_folds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'features': self.features_used,
            'label': self.label_type,
            'test_acc_mean': float(self.test_accuracy_mean),
            'test_acc_std': float(self.test_accuracy_std),
            'test_auc_mean': float(self.test_auc_mean),
            'test_auc_std': float(self.test_auc_std),
            'train_auc_mean': float(self.train_auc_mean),
            'auc_gap': float(self.auc_gap),
            'is_promising': self.is_promising,
            'promotion_reason': self.promotion_reason,
            'n_samples': self.n_samples,
            'n_folds': self.n_folds,
            'fold_aucs': [float(x) for x in self.fold_aucs],
        }


def compute_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute AUC without sklearn dependency."""
    # Sort by predicted probability descending
    order = np.argsort(-y_pred_proba)
    y_sorted = y_true[order]
    
    # Count positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Calculate AUC using the rank-based formula
    pos_ranks = np.where(y_sorted == 1)[0]
    auc = (np.sum(n_pos + n_neg - pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    
    return float(np.clip(auc, 0, 1))


class InteractionRunner:
    """
    Phase 3B experiment runner.
    
    Runs baseline, control, and interaction models with proper evaluation.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 label_type: LabelType = LabelType.FORWARD_RETURN_SIGN,
                 horizon: int = 10,
                 auc_threshold: float = 0.54,
                 max_auc_gap: float = 0.03):
        self.data = data
        self.label_type = label_type
        self.horizon = horizon
        self.auc_threshold = auc_threshold
        self.max_auc_gap = max_auc_gap
        
        self.label_generator = LabelGenerator()
        self.results: List[ExperimentResult] = []
        self.baseline_accuracy: float = 0.5
    
    def _build_features(self, enabled_families: Set[PrimitiveFamily],
                        include_interactions: bool = False) -> pd.DataFrame:
        """Build feature matrix for specified families."""
        config = FeatureConfig(enabled_families=enabled_families)
        builder = FeatureBuilder(config)
        return builder.build_feature_matrix(self.data)
    
    def _generate_labels(self) -> pd.DataFrame:
        """Generate labels."""
        return self.label_generator.generate_labels(
            self.data,
            label_type=self.label_type,
            horizon=self.horizon
        )
    
    def run_control_c0(self) -> ExperimentResult:
        """C0: Do-nothing classifier - always predicts most common class."""
        labels_df = self._generate_labels()
        
        # Most common class
        most_common = labels_df['label'].mode().iloc[0]
        baseline_acc = (labels_df['label'] == most_common).mean()
        
        self.baseline_accuracy = baseline_acc
        
        result = ExperimentResult(
            model_id='C0_do_nothing',
            model_type='control',
            features_used=[],
            label_type=self.label_type.value,
            test_accuracy_mean=baseline_acc,
            test_auc_mean=0.5,  # Random AUC
            n_samples=len(labels_df),
            n_folds=1,
            promotion_reason="CONTROL: baseline reference"
        )
        
        self.results.append(result)
        return result
    
    def run_control_c1(self) -> ExperimentResult:
        """C1: Naive volatility proxy - no concept features."""
        # Build naive features: rolling range, return
        naive_df = pd.DataFrame(index=self.data.index)
        naive_df['range'] = self.data['high'] - self.data['low']
        naive_df['return'] = self.data['close'].pct_change()
        naive_df['rolling_range_5'] = naive_df['range'].rolling(5).mean()
        naive_df['rolling_range_20'] = naive_df['range'].rolling(20).mean()
        naive_df['abs_return'] = naive_df['return'].abs()
        naive_df = naive_df.fillna(0)
        
        labels_df = self._generate_labels()
        
        # Align
        common_idx = naive_df.index.intersection(labels_df.index)
        X = naive_df.loc[common_idx].values
        y = labels_df.loc[common_idx]['label'].values
        
        return self._run_model(
            model_id='C1_naive_vol',
            model_type='control',
            X=X,
            y=y,
            feature_names=['range', 'return', 'rolling_range_5', 'rolling_range_20', 'abs_return'],
            timestamps=common_idx
        )
    
    def run_baseline(self, model_id: str, families: Set[PrimitiveFamily]) -> ExperimentResult:
        """Run a baseline model with specified primitive families."""
        features_df = self._build_features(families)
        labels_df = self._generate_labels()
        
        common_idx = features_df.index.intersection(labels_df.index)
        X = features_df.loc[common_idx].drop(columns=['bar_idx'], errors='ignore')
        y = labels_df.loc[common_idx]['label']
        
        X = X.fillna(0)
        feature_names = list(X.columns)
        
        return self._run_model(
            model_id=model_id,
            model_type='baseline',
            X=X.values,
            y=y.values,
            feature_names=feature_names,
            timestamps=common_idx
        )
    
    def run_interaction(self, model_id: str, 
                        families: Set[PrimitiveFamily],
                        interaction_pairs: List[Tuple[str, str]] = None) -> ExperimentResult:
        """Run interaction model with explicit interaction terms."""
        features_df = self._build_features(families)
        labels_df = self._generate_labels()
        
        common_idx = features_df.index.intersection(labels_df.index)
        X = features_df.loc[common_idx].drop(columns=['bar_idx'], errors='ignore')
        y = labels_df.loc[common_idx]['label']
        
        X = X.fillna(0)
        
        # Add interaction terms
        if interaction_pairs:
            for f1, f2 in interaction_pairs:
                if f1 in X.columns and f2 in X.columns:
                    X[f'{f1}_x_{f2}'] = X[f1] * X[f2]
        
        feature_names = list(X.columns)
        
        return self._run_model(
            model_id=model_id,
            model_type='interaction',
            X=X.values,
            y=y.values,
            feature_names=feature_names,
            timestamps=common_idx
        )
    
    def _run_model(self, model_id: str, model_type: str,
                   X: np.ndarray, y: np.ndarray,
                   feature_names: List[str],
                   timestamps: pd.DatetimeIndex) -> ExperimentResult:
        """Run cross-validated model and compute metrics."""
        cv = PurgedWalkForward(n_splits=5, purge_bars=max(self.horizon + 20, 30))
        
        fold_accs = []
        fold_aucs = []
        fold_train_aucs = []
        
        for train_idx, test_idx in cv.split(len(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Standardize
            train_mean = np.mean(X_train, axis=0)
            train_std = np.std(X_train, axis=0) + 1e-8
            X_train_s = (X_train - train_mean) / train_std
            X_test_s = (X_test - train_mean) / train_std
            
            # Fit logistic regression with L2
            train_proba, test_proba = self._fit_logistic(X_train_s, y_train, X_test_s)
            
            # Compute metrics
            test_pred = (test_proba > 0.5).astype(float)
            test_acc = np.mean(test_pred == y_test)
            test_auc = compute_auc(y_test, test_proba)
            train_auc = compute_auc(y_train, train_proba)
            
            fold_accs.append(test_acc)
            fold_aucs.append(test_auc)
            fold_train_aucs.append(train_auc)
        
        # Aggregate
        test_acc_mean = np.mean(fold_accs)
        test_acc_std = np.std(fold_accs)
        test_auc_mean = np.mean(fold_aucs)
        test_auc_std = np.std(fold_aucs)
        train_auc_mean = np.mean(fold_train_aucs)
        auc_gap = train_auc_mean - test_auc_mean
        
        # Check promotion criteria
        is_promising = False
        reason = ""
        
        if test_auc_mean >= self.auc_threshold:
            if auc_gap <= self.max_auc_gap:
                if test_acc_mean >= self.baseline_accuracy + 0.01:
                    # Check fold stability (no single fold > mean + 2*std)
                    fold_stable = all(
                        abs(auc - test_auc_mean) < 2 * test_auc_std + 0.02
                        for auc in fold_aucs
                    )
                    if fold_stable:
                        is_promising = True
                        reason = f"AUC={test_auc_mean:.3f}, gap={auc_gap:.3f}, stable"
                    else:
                        reason = f"FAIL: fold instability"
                else:
                    reason = f"FAIL: acc {test_acc_mean:.3f} < baseline+1%"
            else:
                reason = f"FAIL: gap {auc_gap:.3f} > {self.max_auc_gap}"
        else:
            reason = f"FAIL: AUC {test_auc_mean:.3f} < {self.auc_threshold}"
        
        result = ExperimentResult(
            model_id=model_id,
            model_type=model_type,
            features_used=feature_names,
            label_type=self.label_type.value,
            fold_accuracies=fold_accs,
            fold_aucs=fold_aucs,
            fold_train_aucs=fold_train_aucs,
            test_accuracy_mean=test_acc_mean,
            test_accuracy_std=test_acc_std,
            test_auc_mean=test_auc_mean,
            test_auc_std=test_auc_std,
            train_auc_mean=train_auc_mean,
            auc_gap=auc_gap,
            is_promising=is_promising,
            promotion_reason=reason,
            n_samples=len(X),
            n_folds=len(fold_accs)
        )
        
        self.results.append(result)
        return result
    
    def _fit_logistic(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit logistic regression, return train and test probabilities."""
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
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run complete Phase 3B experiment grid."""
        print("Running Phase 3B experiments...")
        
        # Controls
        print("\n--- CONTROLS ---")
        c0 = self.run_control_c0()
        print(f"C0 do-nothing: acc={c0.test_accuracy_mean:.3f}")
        
        c1 = self.run_control_c1()
        print(f"C1 naive vol:  acc={c1.test_accuracy_mean:.3f}, AUC={c1.test_auc_mean:.3f}")
        
        # Baselines (single family)
        print("\n--- BASELINES ---")
        for family in [PrimitiveFamily.DISPLACEMENT, PrimitiveFamily.COMPRESSION,
                       PrimitiveFamily.ZONES, PrimitiveFamily.LIQUIDITY]:
            result = self.run_baseline(f'B_{family.value}', {family})
            status = "PROMISING" if result.is_promising else ""
            print(f"B_{family.value}: acc={result.test_accuracy_mean:.3f}, "
                  f"AUC={result.test_auc_mean:.3f} {status}")
        
        # Interaction models
        print("\n--- INTERACTIONS ---")
        
        # I1: Expansion × Zone
        i1 = self.run_interaction(
            'I1_exp_zone',
            {PrimitiveFamily.DISPLACEMENT, PrimitiveFamily.ZONES},
            [('disp_range_zscore', 'zone_new_fvg_bull'),
             ('disp_range_zscore', 'zone_new_fvg_bear')]
        )
        print(f"I1 exp×zone: acc={i1.test_accuracy_mean:.3f}, AUC={i1.test_auc_mean:.3f}")
        
        # I2: Expansion × Liquidity
        i2 = self.run_interaction(
            'I2_exp_liq',
            {PrimitiveFamily.DISPLACEMENT, PrimitiveFamily.LIQUIDITY},
            [('disp_range_zscore', 'liq_new_equal_highs'),
             ('disp_range_zscore', 'liq_new_equal_lows')]
        )
        print(f"I2 exp×liq:  acc={i2.test_accuracy_mean:.3f}, AUC={i2.test_auc_mean:.3f}")
        
        # I3: Compression × Zone
        i3 = self.run_interaction(
            'I3_comp_zone',
            {PrimitiveFamily.COMPRESSION, PrimitiveFamily.ZONES},
            [('comp_score', 'zone_new_fvg_bull'),
             ('comp_score', 'zone_new_swing_high')]
        )
        print(f"I3 comp×zone: acc={i3.test_accuracy_mean:.3f}, AUC={i3.test_auc_mean:.3f}")
        
        # I4: Full model (all families)
        i4 = self.run_interaction(
            'I4_full',
            set(PrimitiveFamily) - {PrimitiveFamily.OVERLAP, PrimitiveFamily.SPEED, 
                                    PrimitiveFamily.ROLE_REVERSAL},
            []  # Let GBM find interactions
        )
        print(f"I4 full:     acc={i4.test_accuracy_mean:.3f}, AUC={i4.test_auc_mean:.3f}")
        
        return self.results
    
    def get_promising_models(self) -> List[ExperimentResult]:
        """Get models that passed promotion criteria."""
        return [r for r in self.results if r.is_promising]
    
    def generate_report(self) -> str:
        """Generate Phase 3B report."""
        lines = [
            "# Phase 3B Interaction Discovery Report",
            "",
            f"**Label**: {self.label_type.value}",
            f"**Horizon**: {self.horizon} bars",
            f"**AUC Threshold**: {self.auc_threshold}",
            f"**Max AUC Gap**: {self.max_auc_gap}",
            f"**Baseline Accuracy**: {self.baseline_accuracy:.1%}",
            "",
            "## Results",
            "",
            "| Model | Type | Acc | AUC | Gap | Fold Std | Status |",
            "|-------|------|-----|-----|-----|----------|--------|",
        ]
        
        for r in sorted(self.results, key=lambda x: x.test_auc_mean, reverse=True):
            status = "✅ PROMISING" if r.is_promising else r.promotion_reason[:20]
            lines.append(
                f"| {r.model_id} | {r.model_type} | {r.test_accuracy_mean:.1%} | "
                f"{r.test_auc_mean:.3f} | {r.auc_gap:.3f} | {r.test_auc_std:.3f} | {status} |"
            )
        
        lines.extend([
            "",
            "## Promising Models for Ablation",
            "",
        ])
        
        promising = self.get_promising_models()
        if promising:
            for m in promising:
                lines.append(f"- **{m.model_id}**: {m.promotion_reason}")
        else:
            lines.append("*No models met promotion criteria.*")
        
        return "\n".join(lines)
    
    def save_results(self, path: Path):
        """Save results as JSON."""
        results_dict = [r.to_dict() for r in self.results]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
