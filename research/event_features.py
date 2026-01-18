"""
Event Feature Builder
Phase 1: Measure each liquidity event

Features:
- Sweep Size (absolute, ATR-normalized)
- Sweep Speed (bars from touch to breach)
- Volatility Context (rolling ATR at event)
- Time of Day (EST)
- Sweep Classification (Micro < 3 pts, Macro >= 3 pts)
"""

import pandas as pd
import numpy as np
from typing import Optional

class EventFeatureBuilder:
    """
    Enriches liquidity events with contextual features.
    """
    
    def __init__(self, atr_period: int = 14):
        self.atr_period = atr_period
        
    def compute_atr(self, data: pd.DataFrame) -> pd.Series:
        """Compute Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        return atr
    
    def enrich_events(self, events: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add contextual features to each event.
        """
        if events.empty:
            return events
            
        events = events.copy()
        
        # Compute ATR for the entire dataset
        atr = self.compute_atr(data)
        
        # Add features
        features = []
        
        for _, event in events.iterrows():
            bar_idx = event['bar_idx']
            
            # ATR at event
            atr_at_event = atr.iloc[bar_idx] if bar_idx < len(atr) else np.nan
            
            # Sweep size normalized by ATR
            sweep_atr = event['sweep_size_pts'] / atr_at_event if atr_at_event > 0 else 0
            
            # Time of day (hours since midnight EST)
            event_time = event['timestamp']
            hour_decimal = event_time.hour + event_time.minute / 60
            
            # Sweep classification
            is_micro = event['sweep_size_pts'] < 3.0
            
            # Day of week
            day_of_week = event_time.dayofweek
            
            features.append({
                'atr_at_event': atr_at_event,
                'sweep_atr_normalized': sweep_atr,
                'hour_decimal': hour_decimal,
                'is_micro_sweep': is_micro,
                'day_of_week': day_of_week
            })
            
        feature_df = pd.DataFrame(features)
        
        # Merge
        events = pd.concat([events.reset_index(drop=True), feature_df], axis=1)
        
        return events
