"""
DISPLACEMENT - Impulse Detection

What it captures:
Was there a *meaningful initiative move* relative to recent behavior?

Outputs:
- range_zscore: Z-score of current bar range vs rolling distribution
- body_zscore: Z-score of current bar body vs rolling distribution
- is_displacement: Boolean flag if exceeds threshold

CRITICAL RULES:
1. Rolling window uses ONLY bars [t-lookback, t-1] (past bars only)
2. Current bar (t) is compared against this distribution
3. No "largest move in next X bars" logic

A displacement is a FACT about current bar relative to recent history.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class DisplacementResult:
    """Result of displacement computation at a single bar."""
    bar_idx: int
    range_value: float
    body_value: float
    range_zscore: float
    body_zscore: float
    rolling_range_mean: float
    rolling_range_std: float
    rolling_body_mean: float
    rolling_body_std: float
    is_range_displacement: bool
    is_body_displacement: bool
    
    @property
    def is_displacement(self) -> bool:
        """True if either range or body is displaced."""
        return self.is_range_displacement or self.is_body_displacement
    
    def to_dict(self) -> dict:
        return {
            'bar_idx': self.bar_idx,
            'range': self.range_value,
            'body': self.body_value,
            'range_zscore': self.range_zscore,
            'body_zscore': self.body_zscore,
            'is_displacement': self.is_displacement,
        }


class DisplacementDetector:
    """
    Detects displacement (unusual initiative moves) using past-only logic.
    
    TIMING: At bar t, we compute statistics using bars [t-lookback : t-1]
    and compare bar t against this distribution.
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 threshold_zscore: float = 2.0,
                 min_periods: int = 10):
        """
        Args:
            lookback: Number of past bars for rolling statistics
            threshold_zscore: Z-score threshold for displacement flag
            min_periods: Minimum bars required before computing z-score
        """
        self.lookback = lookback
        self.threshold_zscore = threshold_zscore
        self.min_periods = min_periods
    
    def compute_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[DisplacementResult]:
        """
        Compute displacement metrics at bar_idx using only past data.
        
        STRICT PAST-ONLY:
        - Rolling stats use bars [bar_idx - lookback : bar_idx - 1]
        - Current bar (bar_idx) is compared against this distribution
        
        Args:
            bar_idx: Current bar index
            data: Full OHLCV dataframe
            
        Returns:
            DisplacementResult if computable, None if insufficient history
        """
        # Need at least min_periods of history BEFORE current bar
        if bar_idx < self.min_periods:
            return None
        
        # Current bar values
        current_bar = data.iloc[bar_idx]
        current_range = current_bar['high'] - current_bar['low']
        current_body = abs(current_bar['close'] - current_bar['open'])
        
        # Historical window: STRICTLY PAST (excludes current bar)
        hist_start = max(0, bar_idx - self.lookback)
        hist_end = bar_idx  # Exclusive - does NOT include current bar
        
        if hist_end - hist_start < self.min_periods:
            return None
        
        hist_data = data.iloc[hist_start:hist_end]
        
        # Compute historical statistics
        hist_ranges = hist_data['high'] - hist_data['low']
        hist_bodies = abs(hist_data['close'] - hist_data['open'])
        
        range_mean = hist_ranges.mean()
        range_std = hist_ranges.std()
        body_mean = hist_bodies.mean()
        body_std = hist_bodies.std()
        
        # Handle zero std (flat markets)
        if range_std == 0 or np.isnan(range_std):
            range_zscore = 0.0
        else:
            range_zscore = (current_range - range_mean) / range_std
        
        if body_std == 0 or np.isnan(body_std):
            body_zscore = 0.0
        else:
            body_zscore = (current_body - body_mean) / body_std
        
        return DisplacementResult(
            bar_idx=bar_idx,
            range_value=current_range,
            body_value=current_body,
            range_zscore=range_zscore,
            body_zscore=body_zscore,
            rolling_range_mean=range_mean,
            rolling_range_std=range_std,
            rolling_body_mean=body_mean,
            rolling_body_std=body_std,
            is_range_displacement=abs(range_zscore) >= self.threshold_zscore,
            is_body_displacement=abs(body_zscore) >= self.threshold_zscore,
        )
    
    def compute_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute displacement metrics for all bars in dataset.
        
        Returns:
            DataFrame with columns: range_zscore, body_zscore, is_displacement
        """
        results = []
        
        for bar_idx in range(len(data)):
            result = self.compute_at(bar_idx, data)
            if result:
                results.append({
                    'bar_idx': bar_idx,
                    'timestamp': data.index[bar_idx],
                    'range_zscore': result.range_zscore,
                    'body_zscore': result.body_zscore,
                    'is_range_displacement': result.is_range_displacement,
                    'is_body_displacement': result.is_body_displacement,
                    'is_displacement': result.is_displacement,
                })
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results).set_index('timestamp')


def compute_displacement_at(bar_idx: int, data: pd.DataFrame,
                           lookback: int = 20, 
                           threshold: float = 2.0) -> Optional[DisplacementResult]:
    """
    Convenience function: compute displacement at bar_idx.
    
    This is the primary interface for ML feature building.
    """
    detector = DisplacementDetector(lookback=lookback, threshold_zscore=threshold)
    return detector.compute_at(bar_idx, data)


def compute_range_zscore_at(bar_idx: int, data: pd.DataFrame,
                           lookback: int = 20) -> float:
    """
    Get just the range z-score at bar_idx.
    
    Returns 0.0 if insufficient history.
    """
    result = compute_displacement_at(bar_idx, data, lookback)
    return result.range_zscore if result else 0.0


def compute_body_zscore_at(bar_idx: int, data: pd.DataFrame,
                          lookback: int = 20) -> float:
    """
    Get just the body z-score at bar_idx.
    
    Returns 0.0 if insufficient history.
    """
    result = compute_displacement_at(bar_idx, data, lookback)
    return result.body_zscore if result else 0.0
