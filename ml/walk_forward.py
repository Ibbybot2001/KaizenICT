"""
Purged Walk-Forward Validation - Phase 3A

Time-series cross-validation with purge gap to prevent leakage.

CRITICAL RULES:
1. Splits are time-ordered (no shuffling)
2. Purge gap between train and test prevents overlap
3. Scalers fit on training segment ONLY
"""

from dataclasses import dataclass
from typing import Iterator, Tuple, List, Optional
import pandas as pd
import numpy as np


@dataclass
class SplitInfo:
    """Information about a single train/test split."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_bars: int
    
    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start
    
    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class PurgedWalkForward:
    """
    Time-series cross-validation with purge gap.
    
    Prevents leakage from:
    - Overlapping samples
    - Lookahead in label generation
    - Feature engineering artifacts
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 train_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 train_pct: float = 0.7,
                 purge_bars: int = 20,
                 embargo_bars: int = 0):
        """
        Args:
            n_splits: Number of folds
            train_size: Fixed training window size (bars). If None, uses train_pct.
            test_size: Fixed test window size. If None, auto-calculated.
            train_pct: Percentage of each fold for training (if train_size=None)
            purge_bars: Gap between train end and test start
            embargo_bars: Gap after test (for embargo period, optional)
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.train_pct = train_pct
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
    
    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.
        
        Args:
            n_samples: Total number of samples
            
        Yields:
            (train_indices, test_indices) for each fold
        """
        for split_info in self.get_split_info(n_samples):
            train_idx = np.arange(split_info.train_start, split_info.train_end)
            test_idx = np.arange(split_info.test_start, split_info.test_end)
            yield train_idx, test_idx
    
    def get_split_info(self, n_samples: int) -> List[SplitInfo]:
        """
        Get detailed information about each split.
        
        Returns:
            List of SplitInfo objects
        """
        splits = []
        
        # Calculate fold size
        fold_size = n_samples // self.n_splits
        
        for fold_id in range(self.n_splits):
            fold_start = fold_id * fold_size
            fold_end = min((fold_id + 1) * fold_size, n_samples)
            
            if self.train_size is not None:
                # Fixed train size
                train_start = fold_start
                train_end = min(fold_start + self.train_size, fold_end - self.purge_bars)
            else:
                # Percentage-based
                available = fold_end - fold_start - self.purge_bars
                train_len = int(available * self.train_pct)
                train_start = fold_start
                train_end = fold_start + train_len
            
            # Test starts after purge gap
            test_start = train_end + self.purge_bars
            
            if self.test_size is not None:
                test_end = min(test_start + self.test_size, fold_end)
            else:
                test_end = fold_end
            
            # Skip invalid splits
            if test_start >= test_end or train_start >= train_end:
                continue
            
            splits.append(SplitInfo(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_bars=self.purge_bars
            ))
        
        return splits
    
    def split_df(self, df: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split a DataFrame into train/test folds.
        
        Yields:
            (train_df, test_df) for each fold
        """
        for train_idx, test_idx in self.split(len(df)):
            yield df.iloc[train_idx], df.iloc[test_idx]


class ExpandingWalkForward:
    """
    Expanding window walk-forward validation.
    
    Training window grows with each fold.
    """
    
    def __init__(self,
                 n_splits: int = 5,
                 initial_train_size: int = 1000,
                 test_size: int = 200,
                 purge_bars: int = 20):
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.purge_bars = purge_bars
    
    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding train/test indices."""
        # Calculate step size to fit n_splits
        remaining = n_samples - self.initial_train_size - self.purge_bars - self.test_size
        step_size = remaining // self.n_splits if self.n_splits > 1 else remaining
        
        for fold_id in range(self.n_splits):
            train_end = self.initial_train_size + (fold_id * step_size)
            test_start = train_end + self.purge_bars
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_start >= n_samples or test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx


def create_default_cv(n_samples: int) -> PurgedWalkForward:
    """
    Create default cross-validator appropriate for sample size.
    """
    if n_samples < 10000:
        return PurgedWalkForward(n_splits=3, purge_bars=10)
    elif n_samples < 50000:
        return PurgedWalkForward(n_splits=5, purge_bars=20)
    else:
        return PurgedWalkForward(n_splits=5, purge_bars=50)
