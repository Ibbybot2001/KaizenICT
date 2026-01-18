"""
ROLE REVERSAL - Level Function Change Detection

What it captures:
Did a level CHANGE FUNCTION (support → resistance or vice versa)?

KEY RULE: NO FUTURE CONFIRMATION PIVOTS
Uses past-only swing logic with proper confirmation delay.

Outputs:
- approach_direction: Direction price approached level
- rejection_strength: How strongly price rejected
- retest_failure_flag: Did retest fail to hold?

A role reversal is a FACT about how price interacted with a level.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class ApproachDirection(str, Enum):
    FROM_ABOVE = 'FROM_ABOVE'
    FROM_BELOW = 'FROM_BELOW'


@dataclass
class RoleReversalResult:
    """Result of role reversal analysis for a level."""
    level_price: float
    bar_idx: int
    
    # Approach analysis
    approach_direction: Optional[ApproachDirection]
    
    # Rejection metrics
    rejection_occurred: bool
    rejection_strength: float  # How far price moved away after touch
    rejection_bars: int  # How many bars the rejection held
    
    # Retest analysis
    retest_occurred: bool
    retest_bar_idx: Optional[int]
    retest_held: bool  # Did level hold on retest?
    
    # Role status
    role_reversed: bool  # Did support become resistance or vice versa?
    
    def to_dict(self) -> dict:
        return {
            'level_price': self.level_price,
            'bar_idx': self.bar_idx,
            'approach_direction': self.approach_direction.value if self.approach_direction else None,
            'rejection_strength': self.rejection_strength,
            'rejection_occurred': self.rejection_occurred,
            'retest_held': self.retest_held,
            'role_reversed': self.role_reversed,
        }


class RoleReversalDetector:
    """
    Detects role reversal at price levels using past-only logic.
    
    TIMING: All analysis uses only bars up to current_bar_idx.
    No future confirmation allowed.
    """
    
    def __init__(self, 
                 rejection_min_points: float = 5.0,
                 rejection_lookback: int = 5,
                 retest_tolerance: float = 2.0):
        """
        Args:
            rejection_min_points: Minimum points for valid rejection
            rejection_lookback: Bars to look back for approach direction
            retest_tolerance: Points tolerance for level retest
        """
        self.rejection_min_points = rejection_min_points
        self.rejection_lookback = rejection_lookback
        self.retest_tolerance = retest_tolerance
    
    def detect_approach_direction(self, bar_idx: int, data: pd.DataFrame,
                                   level: float) -> Optional[ApproachDirection]:
        """
        Determine how price approached a level.
        
        Uses ONLY past bars to determine approach direction.
        """
        if bar_idx < self.rejection_lookback:
            return None
        
        # Look at bars before touch
        lookback_start = max(0, bar_idx - self.rejection_lookback)
        recent_closes = data.iloc[lookback_start:bar_idx]['close'].values
        
        if len(recent_closes) == 0:
            return None
        
        avg_close = np.mean(recent_closes)
        
        if avg_close > level:
            return ApproachDirection.FROM_ABOVE
        elif avg_close < level:
            return ApproachDirection.FROM_BELOW
        else:
            return None
    
    def compute_rejection_strength(self, touch_bar_idx: int, current_bar_idx: int,
                                   data: pd.DataFrame, level: float,
                                   approach: ApproachDirection) -> Tuple[float, int]:
        """
        Compute how strongly price rejected from level.
        
        Returns:
            (max_distance, bars_held) - max distance from level and duration
        """
        max_distance = 0.0
        bars_held = 0
        
        for i in range(touch_bar_idx + 1, current_bar_idx + 1):
            bar = data.iloc[i]
            
            if approach == ApproachDirection.FROM_BELOW:
                # Approached from below, expecting rejection down
                distance = level - bar['low']
            else:
                # Approached from above, expecting rejection up
                distance = bar['high'] - level
            
            if distance > 0:
                bars_held += 1
                max_distance = max(max_distance, distance)
            else:
                break  # Price came back to level
        
        return max_distance, bars_held
    
    def detect_retest(self, touch_bar_idx: int, current_bar_idx: int,
                      data: pd.DataFrame, level: float,
                      approach: ApproachDirection) -> Tuple[bool, Optional[int], bool]:
        """
        Check if there was a retest of the level.
        
        Returns:
            (retest_occurred, retest_bar_idx, retest_held)
        """
        rejection_ended = False
        retest_bar = None
        
        # First, find where rejection ended
        for i in range(touch_bar_idx + 1, current_bar_idx + 1):
            bar = data.iloc[i]
            
            # Check if price returned to level
            if approach == ApproachDirection.FROM_BELOW:
                # Was rejected down, now coming back up
                if bar['high'] >= level - self.retest_tolerance:
                    retest_bar = i
                    break
            else:
                # Was rejected up, now coming back down
                if bar['low'] <= level + self.retest_tolerance:
                    retest_bar = i
                    break
        
        if retest_bar is None:
            return False, None, False
        
        # Check if retest held (price rejected again)
        retest_held = False
        for i in range(retest_bar + 1, current_bar_idx + 1):
            bar = data.iloc[i]
            
            if approach == ApproachDirection.FROM_BELOW:
                # Retesting from below now (role reversed to resistance)
                if bar['close'] < level - self.rejection_min_points:
                    retest_held = True
                    break
            else:
                # Retesting from above now (role reversed to support)
                if bar['close'] > level + self.rejection_min_points:
                    retest_held = True
                    break
        
        return True, retest_bar, retest_held
    
    def analyze_at(self, touch_bar_idx: int, current_bar_idx: int,
                   data: pd.DataFrame, level: float) -> Optional[RoleReversalResult]:
        """
        Analyze role reversal at a level.
        
        Args:
            touch_bar_idx: Bar where level was touched
            current_bar_idx: Current bar index
            data: OHLCV data
            level: Price level being analyzed
            
        Returns:
            RoleReversalResult with analysis
        """
        if current_bar_idx <= touch_bar_idx:
            return None
        
        # Determine approach direction
        approach = self.detect_approach_direction(touch_bar_idx, data, level)
        if approach is None:
            return RoleReversalResult(
                level_price=level,
                bar_idx=current_bar_idx,
                approach_direction=None,
                rejection_occurred=False,
                rejection_strength=0.0,
                rejection_bars=0,
                retest_occurred=False,
                retest_bar_idx=None,
                retest_held=False,
                role_reversed=False,
            )
        
        # Compute rejection
        rejection_strength, rejection_bars = self.compute_rejection_strength(
            touch_bar_idx, current_bar_idx, data, level, approach
        )
        rejection_occurred = rejection_strength >= self.rejection_min_points
        
        # Detect retest
        retest_occurred, retest_bar_idx, retest_held = self.detect_retest(
            touch_bar_idx, current_bar_idx, data, level, approach
        )
        
        # Role reversal = approached from one side, rejected, retested from other side
        role_reversed = rejection_occurred and retest_occurred and retest_held
        
        return RoleReversalResult(
            level_price=level,
            bar_idx=current_bar_idx,
            approach_direction=approach,
            rejection_occurred=rejection_occurred,
            rejection_strength=rejection_strength,
            rejection_bars=rejection_bars,
            retest_occurred=retest_occurred,
            retest_bar_idx=retest_bar_idx,
            retest_held=retest_held,
            role_reversed=role_reversed,
        )


def compute_role_reversal_at(touch_bar_idx: int, current_bar_idx: int,
                            data: pd.DataFrame, level: float) -> Optional[RoleReversalResult]:
    """
    Convenience function: Analyze role reversal at a level.
    """
    detector = RoleReversalDetector()
    return detector.analyze_at(touch_bar_idx, current_bar_idx, data, level)
