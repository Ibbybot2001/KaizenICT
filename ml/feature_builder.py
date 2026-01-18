"""
Feature Builder - Phase 3A

Builds feature vectors from primitives for ML consumption.

CRITICAL RULES:
1. All features computed via primitives (already verified past-only)
2. Each primitive family can be toggled for ablation
3. Features are orthogonal - one family per concept

This is the bridge between primitives and ML models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
import pandas as pd
import numpy as np

# Import all primitives
from primitives.zones import ZoneDetector, ZoneType
from primitives.displacement import DisplacementDetector
from primitives.overlap import OverlapCalculator
from primitives.speed import SpeedTracker
from primitives.compression import CompressionDetector
from primitives.liquidity import LiquidityDetector
from primitives.role_reversal import RoleReversalDetector


class PrimitiveFamily(str, Enum):
    """Primitive families for ablation testing."""
    ZONES = 'zones'
    DISPLACEMENT = 'displacement'
    OVERLAP = 'overlap'
    SPEED = 'speed'
    COMPRESSION = 'compression'
    LIQUIDITY = 'liquidity'
    ROLE_REVERSAL = 'role_reversal'


@dataclass
class FeatureConfig:
    """Configuration for feature building."""
    # Enabled families (for ablation, disable some)
    enabled_families: Set[PrimitiveFamily] = field(
        default_factory=lambda: set(PrimitiveFamily)
    )
    
    # Displacement
    displacement_lookback: int = 20
    displacement_threshold: float = 2.0
    
    # Zones
    swing_left: int = 5
    swing_right: int = 5
    min_fvg_size: float = 2.0
    
    # Compression
    compression_lookback: int = 20
    
    # Speed
    speed_window: int = 10
    
    # Liquidity
    liquidity_tolerance: float = 1.0
    liquidity_lookback: int = 50
    
    # Include absolute features alongside relative (per user feedback)
    include_absolute: bool = True


class FeatureBuilder:
    """
    Builds feature vectors from primitives.
    
    ABLATION SUPPORT: Each primitive family can be disabled via config.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Initialize detectors
        self.displacement_detector = DisplacementDetector(
            lookback=self.config.displacement_lookback,
            threshold_zscore=self.config.displacement_threshold
        )
        self.zone_detector = ZoneDetector(
            swing_left=self.config.swing_left,
            swing_right=self.config.swing_right,
            min_fvg_size=self.config.min_fvg_size
        )
        self.compression_detector = CompressionDetector(
            lookback=self.config.compression_lookback
        )
        self.overlap_calculator = OverlapCalculator()
        self.speed_tracker = SpeedTracker(
            max_window_bars=self.config.speed_window
        )
        self.liquidity_detector = LiquidityDetector(
            tolerance=self.config.liquidity_tolerance,
            lookback=self.config.liquidity_lookback
        )
    
    def build_features_at(self, bar_idx: int, data: pd.DataFrame,
                          zone_history: Optional[Dict] = None) -> Dict[str, float]:
        """
        Build feature vector at bar_idx using enabled primitives.
        
        Args:
            bar_idx: Current bar index
            data: OHLCV dataframe
            zone_history: Pre-built zone history (optional, for efficiency)
            
        Returns:
            Dictionary of feature_name -> value
        """
        features = {}
        
        # DISPLACEMENT FEATURES
        if PrimitiveFamily.DISPLACEMENT in self.config.enabled_families:
            disp = self.displacement_detector.compute_at(bar_idx, data)
            if disp:
                features['disp_range_zscore'] = disp.range_zscore
                features['disp_body_zscore'] = disp.body_zscore
                features['disp_is_displacement'] = float(disp.is_displacement)
                
                # Absolute features (per user feedback on z-score flattening)
                if self.config.include_absolute:
                    features['disp_range_abs'] = disp.range_value
                    features['disp_body_abs'] = disp.body_value
        
        # COMPRESSION FEATURES
        if PrimitiveFamily.COMPRESSION in self.config.enabled_families:
            comp = self.compression_detector.compute_at(bar_idx, data)
            if comp:
                features['comp_range_percentile'] = comp.range_percentile
                features['comp_score'] = comp.compression_score
                features['comp_is_compressed'] = float(comp.is_compressed)
                
                if self.config.include_absolute:
                    features['comp_range_abs'] = comp.current_range
        
        # ZONE FEATURES (count of active zones by type)
        if PrimitiveFamily.ZONES in self.config.enabled_families:
            zones = self.zone_detector.compute_all_zones_at(bar_idx, data)
            
            features['zone_new_fvg_bull'] = sum(
                1 for z in zones if z.zone_type == ZoneType.FVG_BULL
            )
            features['zone_new_fvg_bear'] = sum(
                1 for z in zones if z.zone_type == ZoneType.FVG_BEAR
            )
            features['zone_new_swing_high'] = sum(
                1 for z in zones if z.zone_type == ZoneType.SWING_HIGH
            )
            features['zone_new_swing_low'] = sum(
                1 for z in zones if z.zone_type == ZoneType.SWING_LOW
            )
        
        # LIQUIDITY FEATURES
        if PrimitiveFamily.LIQUIDITY in self.config.enabled_families:
            liq_levels = self.liquidity_detector.compute_all_at(bar_idx, data)
            features['liq_new_equal_highs'] = sum(
                1 for l in liq_levels if l.liquidity_type.value == 'EQUAL_HIGHS'
            )
            features['liq_new_equal_lows'] = sum(
                1 for l in liq_levels if l.liquidity_type.value == 'EQUAL_LOWS'
            )
        
        # Fill missing with 0
        for key in features:
            if features[key] is None or np.isnan(features[key]):
                features[key] = 0.0
        
        return features
    
    def build_feature_matrix(self, data: pd.DataFrame,
                             start_bar: int = 0,
                             end_bar: Optional[int] = None) -> pd.DataFrame:
        """
        Build feature matrix for a range of bars.
        
        Returns:
            DataFrame with bar index and all features
        """
        if end_bar is None:
            end_bar = len(data)
        
        records = []
        for bar_idx in range(start_bar, end_bar):
            features = self.build_features_at(bar_idx, data)
            features['bar_idx'] = bar_idx
            features['timestamp'] = data.index[bar_idx]
            records.append(features)
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records).set_index('timestamp')
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        names = []
        
        if PrimitiveFamily.DISPLACEMENT in self.config.enabled_families:
            names.extend([
                'disp_range_zscore', 'disp_body_zscore', 'disp_is_displacement'
            ])
            if self.config.include_absolute:
                names.extend(['disp_range_abs', 'disp_body_abs'])
        
        if PrimitiveFamily.COMPRESSION in self.config.enabled_families:
            names.extend([
                'comp_range_percentile', 'comp_score', 'comp_is_compressed'
            ])
            if self.config.include_absolute:
                names.append('comp_range_abs')
        
        if PrimitiveFamily.ZONES in self.config.enabled_families:
            names.extend([
                'zone_new_fvg_bull', 'zone_new_fvg_bear',
                'zone_new_swing_high', 'zone_new_swing_low'
            ])
        
        if PrimitiveFamily.LIQUIDITY in self.config.enabled_families:
            names.extend(['liq_new_equal_highs', 'liq_new_equal_lows'])
        
        return names


def create_ablation_configs() -> Dict[str, FeatureConfig]:
    """
    Create feature configs for ablation testing.
    
    Returns configs with each primitive family disabled.
    """
    configs = {}
    
    # Full model (all families)
    configs['full'] = FeatureConfig()
    
    # Ablation: remove each family
    for family in PrimitiveFamily:
        ablated = set(PrimitiveFamily) - {family}
        configs[f'no_{family.value}'] = FeatureConfig(enabled_families=ablated)
    
    # Single family models (for Phase 3A screening)
    for family in PrimitiveFamily:
        configs[f'only_{family.value}'] = FeatureConfig(
            enabled_families={family}
        )
    
    return configs
