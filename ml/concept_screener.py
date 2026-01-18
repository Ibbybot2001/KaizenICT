"""
Concept Screener - Phase 3A

Simple models to test if primitives contain predictive info AT ALL.

PRINCIPLE: If it's real, simple models feel it first.

Models:
- Logistic Regression (with regularization)
- Shallow Decision Trees

Each primitive family tested IN ISOLATION.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json

from ml.feature_builder import FeatureBuilder, FeatureConfig, PrimitiveFamily, create_ablation_configs
from ml.label_generator import LabelGenerator, LabelType
from ml.walk_forward import PurgedWalkForward, create_default_cv
from engine.constants import MIN_SL_POINTS


@dataclass
class ScreeningResult:
    """Result from a single screening experiment."""
    primitive_family: str
    model_type: str
    label_type: str
    
    # Performance across folds
    train_accuracy_mean: float
    train_accuracy_std: float
    test_accuracy_mean: float
    test_accuracy_std: float
    
    # For classification
    test_precision: Optional[float] = None
    test_recall: Optional[float] = None
    test_f1: Optional[float] = None
    
    # For regression/profit
    test_mean_return: Optional[float] = None
    test_sharpe: Optional[float] = None
    
    # Feature importance (if available)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Validity
    n_samples: int = 0
    n_folds: int = 0
    passed_threshold: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primitive_family': self.primitive_family,
            'model_type': self.model_type,
            'label_type': self.label_type,
            'train_acc_mean': float(self.train_accuracy_mean),
            'train_acc_std': float(self.train_accuracy_std),
            'test_acc_mean': float(self.test_accuracy_mean),
            'test_acc_std': float(self.test_accuracy_std),
            'n_samples': int(self.n_samples),
            'n_folds': int(self.n_folds),
            'passed': bool(self.passed_threshold),
        }


class ConceptScreener:
    """
    Phase 3A: Test if primitives contain predictive info.
    
    RULE: One primitive family at a time.
    Simple models with heavy regularization.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 label_type: LabelType = LabelType.FORWARD_RETURN_SIGN,
                 horizon: int = 10,
                 sl_points: float = MIN_SL_POINTS,
                 tp_points: float = 20.0,
                 min_accuracy_threshold: float = 0.52):  # Just above random
        """
        Args:
            data: OHLCV dataframe
            label_type: Type of label for prediction
            horizon: Forward horizon for labels
            sl_points: SL distance for TP/SL labels
            tp_points: TP distance for TP/SL labels
            min_accuracy_threshold: Minimum accuracy to pass screening
        """
        self.data = data
        self.label_type = label_type
        self.horizon = horizon
        self.sl_points = sl_points
        self.tp_points = tp_points
        self.min_accuracy_threshold = min_accuracy_threshold
        
        self.label_generator = LabelGenerator()
        self.results: List[ScreeningResult] = []
    
    def screen_family(self, family: PrimitiveFamily, 
                      model_type: str = 'logistic') -> ScreeningResult:
        """
        Screen a single primitive family.
        
        Args:
            family: Primitive family to test
            model_type: 'logistic' or 'tree'
            
        Returns:
            ScreeningResult with performance metrics
        """
        # Build features with ONLY this family
        config = FeatureConfig(enabled_families={family})
        builder = FeatureBuilder(config)
        
        # Generate features
        features_df = builder.build_feature_matrix(self.data)
        
        # Generate labels
        labels_df = self.label_generator.generate_labels(
            self.data,
            label_type=self.label_type,
            horizon=self.horizon,
            sl_points=self.sl_points,
            tp_points=self.tp_points
        )
        
        # Align features and labels
        common_idx = features_df.index.intersection(labels_df.index)
        if len(common_idx) < 100:
            return ScreeningResult(
                primitive_family=family.value,
                model_type=model_type,
                label_type=self.label_type.value,
                train_accuracy_mean=0.0,
                train_accuracy_std=0.0,
                test_accuracy_mean=0.0,
                test_accuracy_std=0.0,
                n_samples=len(common_idx),
                n_folds=0,
                passed_threshold=False
            )
        
        X = features_df.loc[common_idx].drop(columns=['bar_idx'], errors='ignore')
        y = labels_df.loc[common_idx]['label']
        
        # Clean data
        X = X.fillna(0)
        feature_names = list(X.columns)
        
        # Cross-validate
        cv = create_default_cv(len(X))
        
        train_accs = []
        test_accs = []
        importances = {name: [] for name in feature_names}
        
        for train_idx, test_idx in cv.split(len(X)):
            X_train = X.iloc[train_idx].values
            y_train = y.iloc[train_idx].values
            X_test = X.iloc[test_idx].values
            y_test = y.iloc[test_idx].values
            
            # Standardize using TRAIN data only
            train_mean = np.mean(X_train, axis=0)
            train_std = np.std(X_train, axis=0) + 1e-8
            X_train_scaled = (X_train - train_mean) / train_std
            X_test_scaled = (X_test - train_mean) / train_std
            
            # Fit model
            if model_type == 'logistic':
                train_acc, test_acc, coefs = self._fit_logistic(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                # Record importance as absolute coefficient
                for i, name in enumerate(feature_names):
                    if i < len(coefs):
                        importances[name].append(abs(coefs[i]))
            else:
                train_acc, test_acc, feat_imp = self._fit_tree(
                    X_train_scaled, y_train, X_test_scaled, y_test
                )
                for i, name in enumerate(feature_names):
                    if i < len(feat_imp):
                        importances[name].append(feat_imp[i])
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        
        # Aggregate importances
        avg_importance = {
            name: np.mean(vals) if vals else 0.0
            for name, vals in importances.items()
        }
        
        test_acc_mean = np.mean(test_accs) if test_accs else 0.0
        
        return ScreeningResult(
            primitive_family=family.value,
            model_type=model_type,
            label_type=self.label_type.value,
            train_accuracy_mean=np.mean(train_accs) if train_accs else 0.0,
            train_accuracy_std=np.std(train_accs) if train_accs else 0.0,
            test_accuracy_mean=test_acc_mean,
            test_accuracy_std=np.std(test_accs) if test_accs else 0.0,
            feature_importance=avg_importance,
            n_samples=len(X),
            n_folds=len(train_accs),
            passed_threshold=test_acc_mean >= self.min_accuracy_threshold
        )
    
    def _fit_logistic(self, X_train, y_train, X_test, y_test) -> Tuple[float, float, np.ndarray]:
        """Fit logistic regression with L2 regularization."""
        # Simple implementation without sklearn dependency
        # Using numpy-based logistic regression with L2
        
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        bias = 0.0
        
        # L2 regularization strength (heavy per spec)
        reg_lambda = 1.0
        lr = 0.1
        n_iter = 100
        
        for _ in range(n_iter):
            # Forward pass
            z = X_train @ weights + bias
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            
            # Gradient with L2 regularization
            error = predictions - y_train
            grad_w = (X_train.T @ error) / len(y_train) + reg_lambda * weights
            grad_b = np.mean(error)
            
            # Update
            weights = weights - lr * grad_w
            bias = bias - lr * grad_b
        
        # Accuracy
        train_pred = (1 / (1 + np.exp(-np.clip(X_train @ weights + bias, -500, 500)))) > 0.5
        test_pred = (1 / (1 + np.exp(-np.clip(X_test @ weights + bias, -500, 500)))) > 0.5
        
        train_acc = np.mean(train_pred == y_train)
        test_acc = np.mean(test_pred == y_test)
        
        return train_acc, test_acc, weights
    
    def _fit_tree(self, X_train, y_train, X_test, y_test) -> Tuple[float, float, np.ndarray]:
        """Fit shallow decision tree (depth=3)."""
        # Simple decision stump implementation
        # Returns feature importance based on split quality
        
        n_features = X_train.shape[1]
        best_feature = 0
        best_threshold = 0
        best_accuracy = 0.5
        
        for feat_idx in range(n_features):
            # Try median as threshold
            threshold = np.median(X_train[:, feat_idx])
            pred = (X_train[:, feat_idx] > threshold).astype(float)
            acc = np.mean(pred == y_train)
            
            # Also try inverted
            acc_inv = np.mean((1 - pred) == y_train)
            if acc_inv > acc:
                acc = acc_inv
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_feature = feat_idx
                best_threshold = threshold
        
        # Test accuracy
        test_pred = (X_test[:, best_feature] > best_threshold).astype(float)
        test_acc = max(
            np.mean(test_pred == y_test),
            np.mean((1 - test_pred) == y_test)
        )
        
        # Feature importance (1 for selected feature, 0 otherwise)
        importance = np.zeros(n_features)
        importance[best_feature] = 1.0
        
        return best_accuracy, test_acc, importance
    
    def screen_all_families(self) -> List[ScreeningResult]:
        """
        Screen all primitive families.
        
        Returns:
            List of ScreeningResult for each family
        """
        self.results = []
        
        for family in PrimitiveFamily:
            # Skip overlap, speed, role_reversal for now (need zone context)
            if family in {PrimitiveFamily.OVERLAP, PrimitiveFamily.SPEED, 
                         PrimitiveFamily.ROLE_REVERSAL}:
                continue
            
            # Test with logistic regression
            result_log = self.screen_family(family, 'logistic')
            self.results.append(result_log)
            
            # Test with tree
            result_tree = self.screen_family(family, 'tree')
            self.results.append(result_tree)
        
        return self.results
    
    def get_passing_concepts(self) -> List[str]:
        """Get primitive families that passed screening."""
        passing = set()
        for result in self.results:
            if result.passed_threshold:
                passing.add(result.primitive_family)
        return list(passing)
    
    def generate_report(self) -> str:
        """Generate screening report."""
        lines = [
            "# Phase 3A Concept Screening Report",
            "",
            f"**Label Type**: {self.label_type.value}",
            f"**Horizon**: {self.horizon} bars",
            f"**Threshold**: {self.min_accuracy_threshold:.1%}",
            "",
            "## Results by Primitive Family",
            "",
            "| Family | Model | Test Acc | Train Acc | Passed |",
            "|--------|-------|----------|-----------|--------|",
        ]
        
        for result in sorted(self.results, key=lambda x: x.test_accuracy_mean, reverse=True):
            passed = "✅" if result.passed_threshold else "❌"
            lines.append(
                f"| {result.primitive_family} | {result.model_type} | "
                f"{result.test_accuracy_mean:.1%} | {result.train_accuracy_mean:.1%} | {passed} |"
            )
        
        lines.extend([
            "",
            "## Passing Concepts",
            "",
        ])
        
        passing = self.get_passing_concepts()
        if passing:
            for concept in passing:
                lines.append(f"- {concept}")
        else:
            lines.append("*No concepts passed the screening threshold.*")
        
        return "\n".join(lines)
    
    def save_results(self, path: Path):
        """Save results as JSON."""
        results_dict = [r.to_dict() for r in self.results]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2)
