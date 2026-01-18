"""
SPEED - Urgency Constraint Measurement

What it captures:
Did price MOVE AWAY FAST ENOUGH after a touch?

Definition:
Measure distance from zone boundary within Y bars after first touch.

Outputs:
- max_excursion_after_touch: Maximum distance reached
- bars_since_touch: Bars elapsed since first touch
- speed_ratio: excursion / bars (velocity proxy)

CRITICAL RULES:
1. Touch must be defined MECHANICALLY (H/L crosses boundary)
2. Speed window starts at FIRST TOUCH BAR CLOSE
3. Metrics freeze once window expires
4. No "best excursion in hindsight"

Speed is a FACT about post-touch behavior, not a prediction.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class SpeedResult:
    """Result of speed computation for a zone touch."""
    touch_bar_idx: int
    current_bar_idx: int
    touch_price: float
    zone_upper: float
    zone_lower: float
    direction: str  # 'UP' or 'DOWN' (direction price moved from zone)
    
    # Distance metrics
    max_excursion: float  # Maximum distance from zone after touch
    current_distance: float  # Current distance from zone
    
    # Time metrics
    bars_since_touch: int
    
    # Speed metric
    speed_ratio: float  # max_excursion / bars_since_touch
    
    # Window status
    window_active: bool  # Is speed window still open?
    
    def to_dict(self) -> dict:
        return {
            'touch_bar_idx': self.touch_bar_idx,
            'current_bar_idx': self.current_bar_idx,
            'direction': self.direction,
            'max_excursion': self.max_excursion,
            'bars_since_touch': self.bars_since_touch,
            'speed_ratio': self.speed_ratio,
            'window_active': self.window_active,
        }


class SpeedTracker:
    """
    Tracks speed (urgency) of price movement after zone touch.
    
    TIMING: Speed is measured starting from the bar AFTER touch.
    Window expires after max_window_bars.
    """
    
    def __init__(self, max_window_bars: int = 10, min_excursion: float = 0.0):
        """
        Args:
            max_window_bars: Maximum bars to track after touch
            min_excursion: Minimum excursion to consider valid
        """
        self.max_window_bars = max_window_bars
        self.min_excursion = min_excursion
    
    def detect_touch(self, bar_idx: int, data: pd.DataFrame,
                     zone_upper: float, zone_lower: float) -> Tuple[bool, Optional[str]]:
        """
        Check if bar at bar_idx touches the zone.
        
        Returns:
            (touched, direction) where direction is 'UP' or 'DOWN'
            indicating which side price came from.
        """
        bar = data.iloc[bar_idx]
        bar_high = bar['high']
        bar_low = bar['low']
        bar_close = bar['close']
        
        # Touch = H/L penetrates zone boundary
        touches_upper = bar_high >= zone_upper and bar_low <= zone_upper
        touches_lower = bar_low <= zone_lower and bar_high >= zone_lower
        
        if touches_upper and bar_close < zone_upper:
            # Touched from below, rejected down
            return True, 'DOWN'
        elif touches_lower and bar_close > zone_lower:
            # Touched from above, rejected up
            return True, 'UP'
        elif touches_upper or touches_lower:
            # Generic touch
            if bar_close > zone_upper:
                return True, 'UP'
            elif bar_close < zone_lower:
                return True, 'DOWN'
            else:
                return True, None  # Close inside zone
        
        return False, None
    
    def compute_speed_at(self, touch_bar_idx: int, current_bar_idx: int,
                         data: pd.DataFrame, zone_upper: float, zone_lower: float,
                         direction: str) -> Optional[SpeedResult]:
        """
        Compute speed metrics from touch_bar to current_bar.
        
        STRICT PAST-ONLY: Only uses bars up to current_bar_idx.
        
        Args:
            touch_bar_idx: Bar index where touch occurred
            current_bar_idx: Current bar index
            data: OHLCV data
            zone_upper, zone_lower: Zone boundaries
            direction: Expected move direction ('UP' or 'DOWN')
            
        Returns:
            SpeedResult with speed metrics
        """
        if current_bar_idx <= touch_bar_idx:
            return None
        
        bars_since_touch = current_bar_idx - touch_bar_idx
        window_active = bars_since_touch <= self.max_window_bars
        
        # Compute excursion for each bar after touch
        max_excursion = 0.0
        
        end_bar = current_bar_idx + 1
        if not window_active:
            # Window expired - freeze at window end
            end_bar = min(touch_bar_idx + self.max_window_bars + 1, len(data))
        
        for i in range(touch_bar_idx + 1, min(end_bar, len(data))):
            bar = data.iloc[i]
            
            if direction == 'UP':
                # Measure distance above zone
                excursion = bar['high'] - zone_upper
            else:
                # Measure distance below zone
                excursion = zone_lower - bar['low']
            
            max_excursion = max(max_excursion, excursion)
        
        # Current distance
        current_bar = data.iloc[current_bar_idx]
        if direction == 'UP':
            current_distance = current_bar['close'] - zone_upper
        else:
            current_distance = zone_lower - current_bar['close']
        
        # Speed ratio (handle division by zero)
        if bars_since_touch > 0:
            speed_ratio = max_excursion / bars_since_touch
        else:
            speed_ratio = 0.0
        
        return SpeedResult(
            touch_bar_idx=touch_bar_idx,
            current_bar_idx=current_bar_idx,
            touch_price=data.iloc[touch_bar_idx]['close'],
            zone_upper=zone_upper,
            zone_lower=zone_lower,
            direction=direction,
            max_excursion=max_excursion,
            current_distance=current_distance,
            bars_since_touch=bars_since_touch,
            speed_ratio=speed_ratio,
            window_active=window_active,
        )
    
    def meets_speed_requirement(self, speed_result: SpeedResult,
                               required_points: float,
                               max_bars: int) -> bool:
        """
        Check if speed requirement is met.
        
        Args:
            speed_result: Result from compute_speed_at
            required_points: Required excursion in points
            max_bars: Maximum bars allowed to reach requirement
            
        Returns:
            True if price moved required_points within max_bars
        """
        if speed_result.bars_since_touch > max_bars:
            # Check if requirement was met within the window
            # This requires looking at frozen max_excursion
            return speed_result.max_excursion >= required_points
        
        return speed_result.max_excursion >= required_points


def compute_speed_after_touch(data: pd.DataFrame,
                             touch_bar_idx: int,
                             current_bar_idx: int,
                             zone_upper: float,
                             zone_lower: float,
                             direction: str,
                             max_window: int = 10) -> Optional[SpeedResult]:
    """
    Convenience function: Compute speed metrics after a zone touch.
    """
    tracker = SpeedTracker(max_window_bars=max_window)
    return tracker.compute_speed_at(
        touch_bar_idx, current_bar_idx, data,
        zone_upper, zone_lower, direction
    )
