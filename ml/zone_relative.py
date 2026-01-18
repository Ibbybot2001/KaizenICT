"""
Zone-Relative Features - Phase 3B.1 Representation Fix

Zones are COORDINATE SYSTEMS, not predictors.
Features need to capture WHERE price is relative to zones,
not just WHETHER zones exist.

Features:
- dist_to_nearest_fvg (signed)
- dist_to_nearest_swing
- dist_to_nearest_liquidity
- zone_age (bars since creation)
- bars_since_touch
- inside_zone (bool)
- approaching_zone (bool)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np

from primitives.zones import ZoneDetector, ZoneType, Zone


@dataclass
class ZoneRelativeFeatures:
    """Relational features for a single bar relative to zones."""
    # Distance features (signed: negative = below, positive = above)
    dist_to_nearest_fvg: float
    dist_to_nearest_swing: float
    dist_to_nearest_zone: float  # Any zone
    
    # Zone type of nearest
    nearest_zone_type: str
    
    # Temporal features
    nearest_zone_age: int  # Bars since zone was created
    
    # State features
    inside_zone: bool
    approaching: bool  # Moving toward zone (vs away)
    
    # Zone counts
    n_fvg_active: int
    n_swing_active: int


class ZoneRelativeBuilder:
    """
    Builds zone-relative features for each bar.
    
    Tracks active zones and computes distance/state relative to them.
    """
    
    def __init__(self,
                 swing_left: int = 5,
                 swing_right: int = 5,
                 max_zone_age: int = 500):  # Forget zones older than this
        self.swing_left = swing_left
        self.swing_right = swing_right
        self.max_zone_age = max_zone_age
        self.zone_detector = ZoneDetector(swing_left, swing_right)
    
    def build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build zone-relative features for all bars.
        
        Returns DataFrame with zone-relative features.
        """
        n = len(data)
        
        # Initialize feature columns
        features = pd.DataFrame(index=data.index)
        features['dist_to_nearest_fvg'] = np.nan
        features['dist_to_nearest_swing'] = np.nan
        features['dist_to_nearest_zone'] = np.nan
        features['nearest_zone_age'] = np.nan
        features['inside_zone'] = False
        features['approaching'] = False
        features['n_fvg_active'] = 0
        features['n_swing_active'] = 0
        
        # Track active zones
        active_zones: List[Dict] = []
        
        print("Building zone-relative features...")
        
        for bar_idx in range(n):
            if bar_idx % 10000 == 0:
                print(f"  Processing bar {bar_idx}/{n}")
            
            current_close = data.iloc[bar_idx]['close']
            prev_close = data.iloc[bar_idx - 1]['close'] if bar_idx > 0 else current_close
            
            # Find new zones at this bar
            new_zones = self.zone_detector.compute_all_zones_at(bar_idx, data)
            for z in new_zones:
                active_zones.append({
                    'type': z.zone_type,
                    'upper': z.upper,
                    'lower': z.lower,
                    'mid': (z.upper + z.lower) / 2,
                    'created': z.created_at,
                })
            
            # Prune old zones
            active_zones = [
                z for z in active_zones 
                if bar_idx - z['created'] <= self.max_zone_age
            ]
            
            if not active_zones:
                continue
            
            # Compute distances to each zone type
            fvg_zones = [z for z in active_zones 
                         if z['type'] in [ZoneType.FVG_BULL, ZoneType.FVG_BEAR]]
            swing_zones = [z for z in active_zones 
                           if z['type'] in [ZoneType.SWING_HIGH, ZoneType.SWING_LOW]]
            
            # Find nearest FVG
            if fvg_zones:
                dists = []
                for z in fvg_zones:
                    # Distance is signed: positive = above zone, negative = below
                    if current_close > z['upper']:
                        d = current_close - z['upper']
                    elif current_close < z['lower']:
                        d = current_close - z['lower']  # Negative
                    else:
                        d = 0  # Inside zone
                    dists.append((abs(d), d, z))
                
                _, nearest_dist, nearest_fvg = min(dists, key=lambda x: x[0])
                features.iloc[bar_idx, features.columns.get_loc('dist_to_nearest_fvg')] = nearest_dist
            
            # Find nearest swing
            if swing_zones:
                dists = []
                for z in swing_zones:
                    if z['type'] == ZoneType.SWING_HIGH:
                        d = current_close - z['upper']  # Positive if below swing high
                    else:  # SWING_LOW
                        d = current_close - z['lower']
                    dists.append((abs(d), d, z))
                
                _, nearest_dist, nearest_swing = min(dists, key=lambda x: x[0])
                features.iloc[bar_idx, features.columns.get_loc('dist_to_nearest_swing')] = nearest_dist
            
            # Find nearest zone of any type
            all_dists = []
            for z in active_zones:
                if current_close > z['upper']:
                    d = current_close - z['upper']
                elif current_close < z['lower']:
                    d = current_close - z['lower']
                else:
                    d = 0
                all_dists.append((abs(d), d, z))
            
            abs_dist, signed_dist, nearest_zone = min(all_dists, key=lambda x: x[0])
            features.iloc[bar_idx, features.columns.get_loc('dist_to_nearest_zone')] = signed_dist
            features.iloc[bar_idx, features.columns.get_loc('nearest_zone_age')] = bar_idx - nearest_zone['created']
            
            # Inside zone check
            inside = nearest_zone['lower'] <= current_close <= nearest_zone['upper']
            features.iloc[bar_idx, features.columns.get_loc('inside_zone')] = inside
            
            # Approaching check (moving toward zone vs away)
            prev_dist = abs(prev_close - nearest_zone['mid'])
            curr_dist = abs(current_close - nearest_zone['mid'])
            approaching = curr_dist < prev_dist
            features.iloc[bar_idx, features.columns.get_loc('approaching')] = approaching
            
            # Zone counts
            features.iloc[bar_idx, features.columns.get_loc('n_fvg_active')] = len(fvg_zones)
            features.iloc[bar_idx, features.columns.get_loc('n_swing_active')] = len(swing_zones)
        
        # Fill NaN with capped values (no zone nearby)
        # 20.0 pts is approx 2-3x ATR, representing "far" without skewing linear models
        features['dist_to_nearest_fvg'] = features['dist_to_nearest_fvg'].fillna(20.0)
        features['dist_to_nearest_swing'] = features['dist_to_nearest_swing'].fillna(20.0)
        features['dist_to_nearest_zone'] = features['dist_to_nearest_zone'].fillna(20.0)
        features['nearest_zone_age'] = features['nearest_zone_age'].fillna(self.max_zone_age)
        
        # Clip distances to range [-20, 20] to ensure stability
        cols_to_clip = ['dist_to_nearest_fvg', 'dist_to_nearest_swing', 'dist_to_nearest_zone']
        for col in cols_to_clip:
            features[col] = features[col].clip(-20.0, 20.0)
        
        return features


def build_zone_relative_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to build zone-relative features."""
    builder = ZoneRelativeBuilder()
    return builder.build_features(data)
