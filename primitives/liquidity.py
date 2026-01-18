"""
LIQUIDITY - Equal Highs/Lows Detection

What it captures:
Obvious resting orders (equal highs/lows within tolerance).

Definition:
Two or more highs/lows within tolerance ε over lookback N.

Outputs:
- liquidity_level: Price level where liquidity sits
- touch_count: Number of touches at that level
- age: Bars since level was established

CRITICAL RULES:
1. Liquidity level ONLY exists after SECOND touch
2. No labeling "this was liquidity" retroactively
3. Detection occurs only after second occurrence

A liquidity level is a FACT about price clustering, not a target.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class LiquidityType(str, Enum):
    EQUAL_HIGHS = 'EQUAL_HIGHS'  # Buy-side liquidity above
    EQUAL_LOWS = 'EQUAL_LOWS'  # Sell-side liquidity below


@dataclass
class LiquidityLevel:
    """A detected liquidity level."""
    level_price: float
    liquidity_type: LiquidityType
    touch_count: int
    first_touch_bar: int
    last_touch_bar: int
    created_at: int  # Bar index when level BECAME KNOWABLE (2nd touch)
    created_time: Optional[pd.Timestamp] = None
    
    # Touch history
    touch_bars: List[int] = None  # Bar indices of each touch
    
    # Status
    swept: bool = False
    swept_at: Optional[int] = None
    
    @property
    def age_bars(self) -> int:
        """Bars since level was created (2nd touch)."""
        return self.last_touch_bar - self.created_at if self.created_at else 0
    
    def to_dict(self) -> dict:
        return {
            'level_price': self.level_price,
            'type': self.liquidity_type.value,
            'touch_count': self.touch_count,
            'created_at': self.created_at,
            'swept': self.swept,
        }


class LiquidityDetector:
    """
    Detects liquidity levels (equal highs/lows) using past-only logic.
    
    TIMING: Liquidity level is created on SECOND touch.
    No retroactive level creation.
    """
    
    def __init__(self, 
                 tolerance: float = 1.0,
                 lookback: int = 50,
                 min_touches: int = 2):
        """
        Args:
            tolerance: Points tolerance for "equal" price
            lookback: Bars to look back for touches
            min_touches: Minimum touches to form level (must be >= 2)
        """
        self.tolerance = tolerance
        self.lookback = lookback
        self.min_touches = max(2, min_touches)  # At least 2
    
    def find_equal_highs_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[LiquidityLevel]:
        """
        Find equal highs forming liquidity at bar_idx.
        
        TIMING: Level only exists if current bar forms 2nd+ touch.
        
        Returns:
            LiquidityLevel if detected at this bar, None otherwise
        """
        if bar_idx < 1:
            return None
        
        current_high = data.iloc[bar_idx]['high']
        
        # Look back for matching highs
        lookback_start = max(0, bar_idx - self.lookback)
        
        touch_bars = [bar_idx]  # Current bar is a touch
        
        for i in range(lookback_start, bar_idx):
            past_high = data.iloc[i]['high']
            if abs(past_high - current_high) <= self.tolerance:
                touch_bars.append(i)
        
        # Sort by bar index
        touch_bars.sort()
        
        # Need at least min_touches
        if len(touch_bars) < self.min_touches:
            return None
        
        # CRITICAL: Level only becomes knowable when we hit the min_touches-th touch
        # Check if current bar IS the touch that makes it a level
        if touch_bars.index(bar_idx) < self.min_touches - 1:
            # We already had enough touches before - but did we detect it then?
            # This touch doesn't CREATE the level
            return None
        
        # Current bar is the touch that creates/reinforces the level
        if len(touch_bars) == self.min_touches and touch_bars[-1] == bar_idx:
            # This is the creating touch!
            avg_level = sum(data.iloc[i]['high'] for i in touch_bars) / len(touch_bars)
            
            return LiquidityLevel(
                level_price=avg_level,
                liquidity_type=LiquidityType.EQUAL_HIGHS,
                touch_count=len(touch_bars),
                first_touch_bar=touch_bars[0],
                last_touch_bar=touch_bars[-1],
                created_at=bar_idx,  # Created NOW
                created_time=data.iloc[bar_idx].name,
                touch_bars=touch_bars,
            )
        
        return None
    
    def find_equal_lows_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[LiquidityLevel]:
        """
        Find equal lows forming liquidity at bar_idx.
        
        Same timing rules as find_equal_highs_at.
        """
        if bar_idx < 1:
            return None
        
        current_low = data.iloc[bar_idx]['low']
        
        lookback_start = max(0, bar_idx - self.lookback)
        
        touch_bars = [bar_idx]
        
        for i in range(lookback_start, bar_idx):
            past_low = data.iloc[i]['low']
            if abs(past_low - current_low) <= self.tolerance:
                touch_bars.append(i)
        
        touch_bars.sort()
        
        if len(touch_bars) < self.min_touches:
            return None
        
        # Check if this is the creating touch
        if len(touch_bars) == self.min_touches and touch_bars[-1] == bar_idx:
            avg_level = sum(data.iloc[i]['low'] for i in touch_bars) / len(touch_bars)
            
            return LiquidityLevel(
                level_price=avg_level,
                liquidity_type=LiquidityType.EQUAL_LOWS,
                touch_count=len(touch_bars),
                first_touch_bar=touch_bars[0],
                last_touch_bar=touch_bars[-1],
                created_at=bar_idx,
                created_time=data.iloc[bar_idx].name,
                touch_bars=touch_bars,
            )
        
        return None
    
    def compute_all_at(self, bar_idx: int, data: pd.DataFrame) -> List[LiquidityLevel]:
        """
        Find all liquidity levels created at bar_idx.
        
        Returns:
            List of LiquidityLevel objects created at this bar
        """
        levels = []
        
        eq_high = self.find_equal_highs_at(bar_idx, data)
        if eq_high:
            levels.append(eq_high)
        
        eq_low = self.find_equal_lows_at(bar_idx, data)
        if eq_low:
            levels.append(eq_low)
        
        return levels
    
    def build_liquidity_history(self, data: pd.DataFrame) -> List[LiquidityLevel]:
        """
        Build complete liquidity level history for the dataset.
        
        Returns:
            List of all liquidity levels with their creation times
        """
        all_levels = []
        
        for bar_idx in range(len(data)):
            levels = self.compute_all_at(bar_idx, data)
            all_levels.extend(levels)
        
        return all_levels


def compute_liquidity_at(bar_idx: int, data: pd.DataFrame,
                        tolerance: float = 1.0) -> List[LiquidityLevel]:
    """
    Convenience function: Find all liquidity levels created at bar_idx.
    """
    detector = LiquidityDetector(tolerance=tolerance)
    return detector.compute_all_at(bar_idx, data)
