"""
COMPRESSION - Range Contraction Detection

What it captures:
Is price COILING instead of expanding?

Allowed measures:
- Rolling range contraction
- HL/LH slope flattening
- Volatility percentile vs recent past

Outputs:
- range_percentile: Current range vs historical distribution
- compression_score: Composite compression measure

CRITICAL RULES:
1. Compression is DESCRIPTIVE, not predictive
2. Must NOT assume breakout direction
3. No lookahead via window alignment

A compression is a FACT about volatility contraction.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class CompressionResult:
    """Result of compression computation at a single bar."""
    bar_idx: int
    
    # Current bar metrics
    current_range: float
    current_body: float
    
    # Percentile metrics (0-100, lower = more compressed)
    range_percentile: float
    body_percentile: float
    
    # Rolling metrics
    rolling_range_min: float
    rolling_range_max: float
    rolling_range_mean: float
    
    # Compression score (0-1, higher = more compressed)
    compression_score: float
    
    # Range contraction detection
    is_compressed: bool  # Below threshold percentile
    
    def to_dict(self) -> dict:
        return {
            'bar_idx': self.bar_idx,
            'range_percentile': self.range_percentile,
            'compression_score': self.compression_score,
            'is_compressed': self.is_compressed,
        }


class CompressionDetector:
    """
    Detects compression (range contraction) using past-only logic.
    
    TIMING: At bar t, statistics use bars [t-lookback : t-1].
    Current bar (t) is compared against this distribution.
    """
    
    def __init__(self, 
                 lookback: int = 20,
                 compression_threshold: float = 25.0,
                 min_periods: int = 10):
        """
        Args:
            lookback: Number of past bars for percentile calculation
            compression_threshold: Percentile threshold for 'is_compressed'
            min_periods: Minimum bars required
        """
        self.lookback = lookback
        self.compression_threshold = compression_threshold
        self.min_periods = min_periods
    
    def compute_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[CompressionResult]:
        """
        Compute compression metrics at bar_idx.
        
        STRICT PAST-ONLY: Rolling stats use bars [bar_idx-lookback : bar_idx-1].
        """
        if bar_idx < self.min_periods:
            return None
        
        # Current bar
        current_bar = data.iloc[bar_idx]
        current_range = current_bar['high'] - current_bar['low']
        current_body = abs(current_bar['close'] - current_bar['open'])
        
        # Historical window - STRICTLY PAST
        hist_start = max(0, bar_idx - self.lookback)
        hist_end = bar_idx  # Exclusive
        
        if hist_end - hist_start < self.min_periods:
            return None
        
        hist_data = data.iloc[hist_start:hist_end]
        hist_ranges = (hist_data['high'] - hist_data['low']).values
        hist_bodies = abs(hist_data['close'] - hist_data['open']).values
        
        # Compute percentiles
        range_percentile = self._percentile_rank(current_range, hist_ranges)
        body_percentile = self._percentile_rank(current_body, hist_bodies)
        
        # Rolling stats
        rolling_range_min = np.min(hist_ranges)
        rolling_range_max = np.max(hist_ranges)
        rolling_range_mean = np.mean(hist_ranges)
        
        # Compression score: inverse of range percentile, normalized to 0-1
        compression_score = 1.0 - (range_percentile / 100.0)
        
        is_compressed = range_percentile <= self.compression_threshold
        
        return CompressionResult(
            bar_idx=bar_idx,
            current_range=current_range,
            current_body=current_body,
            range_percentile=range_percentile,
            body_percentile=body_percentile,
            rolling_range_min=rolling_range_min,
            rolling_range_max=rolling_range_max,
            rolling_range_mean=rolling_range_mean,
            compression_score=compression_score,
            is_compressed=is_compressed,
        )
    
    def _percentile_rank(self, value: float, distribution: np.ndarray) -> float:
        """
        Compute percentile rank of value within distribution.
        
        Returns 0-100 percentile.
        """
        if len(distribution) == 0:
            return 50.0
        
        count_below = np.sum(distribution < value)
        count_equal = np.sum(distribution == value)
        
        # Percentile = (count below + 0.5 * count equal) / total * 100
        percentile = (count_below + 0.5 * count_equal) / len(distribution) * 100
        return percentile
    
    def compute_consecutive_compression(self, bar_idx: int, data: pd.DataFrame,
                                        lookback: int = 5) -> int:
        """
        Count consecutive bars of compression.
        
        Useful for detecting sustained compression periods.
        """
        count = 0
        for i in range(bar_idx, max(0, bar_idx - lookback) - 1, -1):
            result = self.compute_at(i, data)
            if result and result.is_compressed:
                count += 1
            else:
                break
        return count


def compute_compression_at(bar_idx: int, data: pd.DataFrame,
                          lookback: int = 20) -> Optional[CompressionResult]:
    """
    Convenience function: Compute compression at bar_idx.
    """
    detector = CompressionDetector(lookback=lookback)
    return detector.compute_at(bar_idx, data)


def compute_range_percentile_at(bar_idx: int, data: pd.DataFrame,
                               lookback: int = 20) -> float:
    """
    Get just the range percentile at bar_idx.
    
    Returns 50.0 if insufficient history.
    """
    result = compute_compression_at(bar_idx, data, lookback)
    return result.range_percentile if result else 50.0
