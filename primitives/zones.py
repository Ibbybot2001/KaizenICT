"""
ZONES - Structural Area Detection

Zone types:
- FVG (Fair Value Gap) inefficiencies
- Prior swing highs/lows

CRITICAL RULES:
1. Zone can ONLY be created when defining candles are fully closed
2. Zone timestamps reflect CREATION TIME, not origin time
3. Zones persist forward unless explicitly invalidated

A zone is a FACT about past structure, not a trade signal.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class ZoneType(str, Enum):
    """Types of structural zones."""
    FVG_BULL = 'FVG_BULL'  # Bullish Fair Value Gap
    FVG_BEAR = 'FVG_BEAR'  # Bearish Fair Value Gap
    SWING_HIGH = 'SWING_HIGH'  # Prior swing high
    SWING_LOW = 'SWING_LOW'  # Prior swing low


@dataclass
class Zone:
    """
    A structural zone with proper timestamps.
    
    Attributes:
        upper: Upper boundary of zone
        lower: Lower boundary of zone
        zone_type: Type of zone (FVG, Swing, etc.)
        created_at: Bar index when zone BECAME KNOWABLE (critical!)
        origin_bar: Bar index of the event that created the zone
        created_time: Timestamp when zone became knowable
        mitigated: Whether zone has been filled/mitigated
        mitigated_at: Bar index when mitigated
    """
    upper: float
    lower: float
    zone_type: ZoneType
    created_at: int  # Bar index when zone BECAME KNOWABLE
    origin_bar: int  # Bar index of the origin event
    created_time: Optional[pd.Timestamp] = None
    mitigated: bool = False
    mitigated_at: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def midpoint(self) -> float:
        """Zone midpoint."""
        return (self.upper + self.lower) / 2
    
    @property
    def size(self) -> float:
        """Zone size in points."""
        return self.upper - self.lower
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within zone boundaries."""
        return self.lower <= price <= self.upper
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'upper': self.upper,
            'lower': self.lower,
            'zone_type': self.zone_type.value,
            'created_at': self.created_at,
            'origin_bar': self.origin_bar,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'mitigated': self.mitigated,
            'mitigated_at': self.mitigated_at,
            'size': self.size,
        }


class ZoneDetector:
    """
    Detects structural zones with strict past-only logic.
    
    CRITICAL: All zone detection uses ONLY closed candles.
    Zone created_at timestamp reflects when the zone BECAME KNOWABLE.
    """
    
    def __init__(self, 
                 swing_left: int = 5,
                 swing_right: int = 5,
                 min_fvg_size: float = 0.0):
        """
        Args:
            swing_left: Left bars for swing detection
            swing_right: Right bars for swing confirmation (creates delay!)
            min_fvg_size: Minimum FVG size in points (0 = no minimum)
        """
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.min_fvg_size = min_fvg_size
    
    def compute_fvg_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[Zone]:
        """
        Detect FVG at bar_idx using only past data.
        
        FVG is created by 3 candles: [bar_idx-2, bar_idx-1, bar_idx]
        The gap is between candle 0's high/low and candle 2's low/high.
        
        TIMING: FVG is knowable at bar_idx close (when candle 2 closes).
        
        Args:
            bar_idx: Current bar index
            data: Full OHLCV dataframe (but we only access [:bar_idx+1])
            
        Returns:
            Zone if FVG detected, None otherwise
        """
        # Need at least 3 bars
        if bar_idx < 2:
            return None
        
        # STRICT: Only access closed candles up to bar_idx
        candle_0 = data.iloc[bar_idx - 2]
        candle_1 = data.iloc[bar_idx - 1]  # Middle candle (gap candle)
        candle_2 = data.iloc[bar_idx]
        
        # Bullish FVG: Candle 2's low > Candle 0's high
        # Gap is between Candle 0 high (bottom) and Candle 2 low (top)
        if candle_2['low'] > candle_0['high']:
            gap_size = candle_2['low'] - candle_0['high']
            if gap_size >= self.min_fvg_size:
                return Zone(
                    upper=candle_2['low'],
                    lower=candle_0['high'],
                    zone_type=ZoneType.FVG_BULL,
                    created_at=bar_idx,  # Knowable NOW
                    origin_bar=bar_idx,  # FVG created by this bar closing
                    created_time=candle_2.name,
                    metadata={'gap_size': gap_size, 'middle_bar': bar_idx - 1}
                )
        
        # Bearish FVG: Candle 2's high < Candle 0's low
        # Gap is between Candle 2 high (bottom) and Candle 0 low (top)
        if candle_2['high'] < candle_0['low']:
            gap_size = candle_0['low'] - candle_2['high']
            if gap_size >= self.min_fvg_size:
                return Zone(
                    upper=candle_0['low'],
                    lower=candle_2['high'],
                    zone_type=ZoneType.FVG_BEAR,
                    created_at=bar_idx,  # Knowable NOW
                    origin_bar=bar_idx,
                    created_time=candle_2.name,
                    metadata={'gap_size': gap_size, 'middle_bar': bar_idx - 1}
                )
        
        return None
    
    def compute_swing_high_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[Zone]:
        """
        Detect confirmed swing high at bar_idx.
        
        CRITICAL TIMING:
        - Swing peak is at bar [bar_idx - swing_right]
        - Confirmation requires swing_right bars to the right
        - Therefore, swing is KNOWABLE at bar_idx (not before!)
        
        Args:
            bar_idx: Current bar index
            data: Full OHLCV dataframe
            
        Returns:
            Zone if swing high confirmed, None otherwise
        """
        # Need enough history: left + right + 1
        min_bars = self.swing_left + self.swing_right + 1
        if bar_idx < min_bars - 1:
            return None
        
        # The potential peak bar
        peak_idx = bar_idx - self.swing_right
        
        # STRICT: Only access bars up to bar_idx
        peak_high = data.iloc[peak_idx]['high']
        
        # Check left side: peak must be higher than all left neighbors
        left_start = peak_idx - self.swing_left
        if left_start < 0:
            return None
        
        left_highs = data.iloc[left_start:peak_idx]['high'].values
        if not (peak_high > np.max(left_highs)):
            return None
        
        # Check right side: peak must be higher than all right neighbors
        right_end = peak_idx + self.swing_right + 1
        right_highs = data.iloc[peak_idx + 1:right_end]['high'].values
        
        if len(right_highs) < self.swing_right:
            return None
        
        if not (peak_high > np.max(right_highs)):
            return None
        
        # Swing high confirmed!
        # It becomes KNOWABLE at bar_idx (when right confirmation completes)
        return Zone(
            upper=peak_high,
            lower=peak_high,  # Point zone (can expand to range if desired)
            zone_type=ZoneType.SWING_HIGH,
            created_at=bar_idx,  # Confirmation time!
            origin_bar=peak_idx,  # Actual peak location
            created_time=data.iloc[bar_idx].name,
            metadata={
                'peak_bar': peak_idx,
                'peak_time': data.iloc[peak_idx].name.isoformat(),
                'confirmation_delay': self.swing_right,
            }
        )
    
    def compute_swing_low_at(self, bar_idx: int, data: pd.DataFrame) -> Optional[Zone]:
        """
        Detect confirmed swing low at bar_idx.
        
        Same timing rules as swing_high_at.
        """
        min_bars = self.swing_left + self.swing_right + 1
        if bar_idx < min_bars - 1:
            return None
        
        peak_idx = bar_idx - self.swing_right
        
        peak_low = data.iloc[peak_idx]['low']
        
        # Check left side
        left_start = peak_idx - self.swing_left
        if left_start < 0:
            return None
        
        left_lows = data.iloc[left_start:peak_idx]['low'].values
        if not (peak_low < np.min(left_lows)):
            return None
        
        # Check right side
        right_end = peak_idx + self.swing_right + 1
        right_lows = data.iloc[peak_idx + 1:right_end]['low'].values
        
        if len(right_lows) < self.swing_right:
            return None
        
        if not (peak_low < np.min(right_lows)):
            return None
        
        return Zone(
            upper=peak_low,
            lower=peak_low,
            zone_type=ZoneType.SWING_LOW,
            created_at=bar_idx,
            origin_bar=peak_idx,
            created_time=data.iloc[bar_idx].name,
            metadata={
                'peak_bar': peak_idx,
                'peak_time': data.iloc[peak_idx].name.isoformat(),
                'confirmation_delay': self.swing_right,
            }
        )
    
    def compute_all_zones_at(self, bar_idx: int, data: pd.DataFrame) -> List[Zone]:
        """
        Compute all zones that become knowable at bar_idx.
        
        Returns:
            List of newly created zones at this bar
        """
        zones = []
        
        # Check for FVG
        fvg = self.compute_fvg_at(bar_idx, data)
        if fvg:
            zones.append(fvg)
        
        # Check for swing high
        swing_h = self.compute_swing_high_at(bar_idx, data)
        if swing_h:
            zones.append(swing_h)
        
        # Check for swing low
        swing_l = self.compute_swing_low_at(bar_idx, data)
        if swing_l:
            zones.append(swing_l)
        
        return zones
    
    def build_zone_history(self, data: pd.DataFrame) -> Dict[int, List[Zone]]:
        """
        Build complete zone history for the dataset.
        
        Returns:
            Dict mapping bar_idx -> list of zones created at that bar
        """
        history = {}
        
        for bar_idx in range(len(data)):
            zones = self.compute_all_zones_at(bar_idx, data)
            if zones:
                history[bar_idx] = zones
        
        return history
    
    def get_active_zones_at(self, bar_idx: int, data: pd.DataFrame,
                           zone_history: Dict[int, List[Zone]]) -> List[Zone]:
        """
        Get all zones that are active (created and not mitigated) at bar_idx.
        
        Args:
            bar_idx: Current bar index
            data: OHLCV data
            zone_history: Pre-built zone history from build_zone_history()
            
        Returns:
            List of active zones
        """
        active = []
        
        for created_idx, zones in zone_history.items():
            if created_idx > bar_idx:
                continue  # Zone not yet created
            
            for zone in zones:
                if zone.mitigated and zone.mitigated_at is not None:
                    if zone.mitigated_at <= bar_idx:
                        continue  # Already mitigated
                active.append(zone)
        
        return active


def compute_zones_at(bar_idx: int, data: pd.DataFrame, 
                     detector: Optional[ZoneDetector] = None) -> List[Zone]:
    """
    Convenience function: compute all zones knowable at bar_idx.
    
    This is the primary interface for ML feature building.
    """
    if detector is None:
        detector = ZoneDetector()
    return detector.compute_all_zones_at(bar_idx, data)
