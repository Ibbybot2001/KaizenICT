"""
OVERLAP / ACCEPTANCE - Zone Interaction Measurement

What it captures:
Is price ACCEPTED at a level or merely probing it?

Core measurements:
- bars_in_zone: Count of consecutive bars with price in zone
- body_overlap_ratio: Fraction of bar body overlapping with zone
- wick_penetration_ratio: Fraction of wick that penetrated zone
- consecutive_closes_inside: Count of consecutive closes inside zone

CRITICAL RULES:
1. Overlap is counted FORWARD in time only (after zone creation)
2. No resetting based on future rejection
3. Wick ≠ acceptance; body overlap + time = acceptance

Zone is an INPUT (from zones.py), not defined here.
"""

from dataclasses import dataclass
from typing import Optional, List
import pandas as pd
import numpy as np


@dataclass
class OverlapResult:
    """Result of overlap computation at a single bar for one zone."""
    bar_idx: int
    zone_upper: float
    zone_lower: float
    
    # Raw measurements
    bar_high: float
    bar_low: float
    bar_open: float
    bar_close: float
    
    # Computed metrics
    body_overlap_ratio: float  # [0, 1] - fraction of body inside zone
    wick_in_zone: bool  # Did wick touch zone?
    body_in_zone: bool  # Is any part of body inside zone?
    close_in_zone: bool  # Is close inside zone?
    
    # Zone interaction type
    touches_zone: bool  # Did bar H/L touch zone at all?
    
    def to_dict(self) -> dict:
        return {
            'bar_idx': self.bar_idx,
            'body_overlap_ratio': self.body_overlap_ratio,
            'wick_in_zone': self.wick_in_zone,
            'body_in_zone': self.body_in_zone,
            'close_in_zone': self.close_in_zone,
            'touches_zone': self.touches_zone,
        }


@dataclass
class AcceptanceState:
    """Tracks acceptance state for a zone over time."""
    zone_id: str
    zone_upper: float
    zone_lower: float
    first_bar_after_creation: int
    
    # Running counters (updated bar by bar)
    bars_in_zone: int = 0
    consecutive_closes_inside: int = 0
    max_consecutive_closes: int = 0
    total_body_overlap: float = 0.0  # Sum of body overlap ratios
    touch_count: int = 0
    
    def update(self, overlap: OverlapResult) -> None:
        """Update acceptance state with new bar's overlap result."""
        if overlap.touches_zone:
            self.touch_count += 1
        
        if overlap.body_in_zone:
            self.bars_in_zone += 1
            self.total_body_overlap += overlap.body_overlap_ratio
        
        if overlap.close_in_zone:
            self.consecutive_closes_inside += 1
            self.max_consecutive_closes = max(
                self.max_consecutive_closes, 
                self.consecutive_closes_inside
            )
        else:
            self.consecutive_closes_inside = 0  # Reset on close outside
    
    @property
    def avg_body_overlap(self) -> float:
        """Average body overlap ratio for bars that were in zone."""
        if self.bars_in_zone == 0:
            return 0.0
        return self.total_body_overlap / self.bars_in_zone
    
    @property
    def acceptance_score(self) -> float:
        """
        Composite acceptance score.
        Higher = more acceptance (body overlap + time).
        """
        # Weight consecutive closes heavily (acceptance)
        # Also consider total bars and overlap quality
        return (
            self.max_consecutive_closes * 2.0 +
            self.bars_in_zone * 0.5 +
            self.avg_body_overlap * 5.0
        )


class OverlapCalculator:
    """
    Calculates overlap metrics between price bars and zones.
    
    TIMING: Overlap is computed for bars AFTER zone creation only.
    No retroactive overlap calculation.
    """
    
    def compute_overlap_at(self, bar_idx: int, data: pd.DataFrame,
                           zone_upper: float, zone_lower: float) -> OverlapResult:
        """
        Compute overlap between bar at bar_idx and a zone.
        
        Args:
            bar_idx: Current bar index
            data: OHLCV dataframe
            zone_upper: Upper boundary of zone
            zone_lower: Lower boundary of zone
            
        Returns:
            OverlapResult with all overlap metrics
        """
        bar = data.iloc[bar_idx]
        bar_high = bar['high']
        bar_low = bar['low']
        bar_open = bar['open']
        bar_close = bar['close']
        
        # Body boundaries
        body_top = max(bar_open, bar_close)
        body_bottom = min(bar_open, bar_close)
        body_size = body_top - body_bottom
        
        # Zone touch check
        touches_zone = (bar_low <= zone_upper) and (bar_high >= zone_lower)
        
        # Wick in zone (any part of bar H/L overlaps zone, but not body)
        wick_touches = touches_zone and not (
            body_bottom <= zone_upper and body_top >= zone_lower
        )
        
        # Body in zone check
        body_overlaps_zone = (body_bottom <= zone_upper) and (body_top >= zone_lower)
        
        # Calculate body overlap ratio
        if body_size == 0 or not body_overlaps_zone:
            body_overlap_ratio = 0.0
        else:
            # Overlap region
            overlap_top = min(body_top, zone_upper)
            overlap_bottom = max(body_bottom, zone_lower)
            overlap_size = max(0, overlap_top - overlap_bottom)
            body_overlap_ratio = overlap_size / body_size
        
        # Close in zone
        close_in_zone = zone_lower <= bar_close <= zone_upper
        
        return OverlapResult(
            bar_idx=bar_idx,
            zone_upper=zone_upper,
            zone_lower=zone_lower,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_open=bar_open,
            bar_close=bar_close,
            body_overlap_ratio=body_overlap_ratio,
            wick_in_zone=wick_touches,
            body_in_zone=body_overlaps_zone,
            close_in_zone=close_in_zone,
            touches_zone=touches_zone,
        )
    
    def compute_bars_in_zone(self, data: pd.DataFrame, 
                             zone_upper: float, zone_lower: float,
                             start_bar: int, end_bar: int) -> int:
        """
        Count bars with body overlapping zone in range [start_bar, end_bar].
        
        Args:
            data: OHLCV data
            zone_upper, zone_lower: Zone boundaries
            start_bar: First bar to check (inclusive)
            end_bar: Last bar to check (inclusive)
            
        Returns:
            Count of bars with body in zone
        """
        count = 0
        for i in range(start_bar, min(end_bar + 1, len(data))):
            result = self.compute_overlap_at(i, data, zone_upper, zone_lower)
            if result.body_in_zone:
                count += 1
        return count
    
    def compute_consecutive_closes(self, data: pd.DataFrame,
                                   zone_upper: float, zone_lower: float,
                                   start_bar: int, end_bar: int) -> int:
        """
        Count maximum consecutive closes inside zone.
        
        Args:
            start_bar: First bar to check
            end_bar: Last bar to check
            
        Returns:
            Maximum consecutive closes inside zone
        """
        max_consec = 0
        current_consec = 0
        
        for i in range(start_bar, min(end_bar + 1, len(data))):
            result = self.compute_overlap_at(i, data, zone_upper, zone_lower)
            if result.close_in_zone:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0
        
        return max_consec


def compute_body_overlap_at(bar_idx: int, data: pd.DataFrame,
                           zone_upper: float, zone_lower: float) -> float:
    """
    Convenience function: Get body overlap ratio at bar_idx.
    
    Returns 0.0 if no overlap.
    """
    calc = OverlapCalculator()
    result = calc.compute_overlap_at(bar_idx, data, zone_upper, zone_lower)
    return result.body_overlap_ratio


def compute_bars_in_zone_since(data: pd.DataFrame,
                               zone_upper: float, zone_lower: float,
                               zone_created_at: int,
                               current_bar: int) -> int:
    """
    Count bars in zone since zone creation up to current bar.
    
    Args:
        data: OHLCV data
        zone_upper, zone_lower: Zone boundaries
        zone_created_at: Bar index when zone was created
        current_bar: Current bar index
        
    Returns:
        Count of bars with body in zone
    """
    calc = OverlapCalculator()
    return calc.compute_bars_in_zone(
        data, zone_upper, zone_lower,
        start_bar=zone_created_at + 1,  # Start AFTER creation
        end_bar=current_bar
    )
