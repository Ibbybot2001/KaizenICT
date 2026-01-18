"""
Reaction Classifier
Phase 2: Classify post-event behavior

Reaction Types (Mutually Exclusive):
A. Immediate Continuation - Price extends in sweep direction, minimal pullback
B. Shallow Pullback - Partial retrace then continuation
C. Deep Retrace - Large retrace of sweep
D. Chop / Overlap - No meaningful expansion, ranging behavior
E. Volatility Expansion/Collapse - ATR change after event

NO TRADE ASSUMPTIONS. Pure observation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ReactionMeasurement:
    """Measurements at a single time horizon."""
    horizon_bars: int
    
    # Price movement
    max_extension_pts: float  # Max move in sweep direction
    max_retrace_pts: float    # Max move against sweep direction
    net_move_pts: float       # Net price change
    
    # Volatility
    atr_before: float
    atr_after: float
    atr_change_pct: float
    
    # Classification
    reaction_type: str  # 'CONTINUATION', 'SHALLOW_PULLBACK', 'DEEP_RETRACE', 'CHOP'
    vol_behavior: str   # 'EXPANSION', 'COLLAPSE', 'STABLE'

class ReactionClassifier:
    """
    Classifies post-event price behavior.
    
    Rules (Conceptual - Tuned by Data):
    - CONTINUATION: Extension > 2*ATR, Retrace < 0.5*ATR
    - SHALLOW_PULLBACK: Extension > ATR, Retrace between 0.5-1.5 ATR
    - DEEP_RETRACE: Retrace > 1.5*ATR
    - CHOP: Neither extension nor retrace significant
    """
    
    def __init__(self, 
                 atr_period: int = 14,
                 continuation_ext_mult: float = 2.0,
                 continuation_ret_mult: float = 0.5,
                 shallow_ext_mult: float = 1.0,
                 deep_retrace_mult: float = 1.5,
                 vol_change_threshold: float = 0.2):
        
        self.atr_period = atr_period
        self.continuation_ext_mult = continuation_ext_mult
        self.continuation_ret_mult = continuation_ret_mult
        self.shallow_ext_mult = shallow_ext_mult
        self.deep_retrace_mult = deep_retrace_mult
        self.vol_change_threshold = vol_change_threshold
    
    def compute_atr(self, data: pd.DataFrame) -> pd.Series:
        """Compute ATR."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()
    
    def classify_reaction(self, 
                         extension: float, 
                         retrace: float, 
                         atr: float) -> str:
        """Classify reaction type based on price moves relative to ATR."""
        if atr <= 0:
            return 'UNKNOWN'
        
        ext_ratio = extension / atr
        ret_ratio = retrace / atr
        
        # Continuation: Big extension, small retrace
        if ext_ratio >= self.continuation_ext_mult and ret_ratio <= self.continuation_ret_mult:
            return 'CONTINUATION'
        
        # Deep Retrace: Large retrace (regardless of extension)
        if ret_ratio >= self.deep_retrace_mult:
            return 'DEEP_RETRACE'
        
        # Shallow Pullback: Meaningful extension with moderate retrace
        if ext_ratio >= self.shallow_ext_mult and ret_ratio > self.continuation_ret_mult:
            return 'SHALLOW_PULLBACK'
        
        # Chop: Neither significant extension nor retrace
        return 'CHOP'
    
    def classify_volatility(self, atr_before: float, atr_after: float) -> str:
        """Classify volatility behavior."""
        if atr_before <= 0:
            return 'UNKNOWN'
        
        change_pct = (atr_after - atr_before) / atr_before
        
        if change_pct > self.vol_change_threshold:
            return 'EXPANSION'
        elif change_pct < -self.vol_change_threshold:
            return 'COLLAPSE'
        else:
            return 'STABLE'
    
    def measure_reaction(self,
                        data: pd.DataFrame,
                        event_idx: int,
                        sweep_direction: str,  # 'UP' (sweep above) or 'DOWN' (sweep below)
                        horizon: int,
                        atr: pd.Series) -> ReactionMeasurement:
        """
        Measure reaction at a specific horizon.
        
        sweep_direction:
        - 'UP': Event was a sweep of highs (EQH/PDH). 
                Extension = further highs. Retrace = lower lows.
        - 'DOWN': Event was a sweep of lows (EQL/PDL).
                Extension = further lows. Retrace = higher highs.
        """
        if event_idx + horizon >= len(data):
            return None
        
        # Get window
        start_idx = event_idx + 1
        end_idx = event_idx + horizon + 1
        window = data.iloc[start_idx:end_idx]
        
        if len(window) == 0:
            return None
        
        event_close = data.iloc[event_idx]['close']
        
        # Calculate extension and retrace based on sweep direction
        if sweep_direction == 'UP':
            # Sweep was upward (e.g., EQH/PDH)
            # Extension = higher highs (continuation of sweep)
            # Retrace = lower lows (reversal)
            max_extension_pts = window['high'].max() - event_close
            max_retrace_pts = event_close - window['low'].min()
        else:
            # Sweep was downward (e.g., EQL/PDL)
            # Extension = lower lows (continuation of sweep)
            # Retrace = higher highs (reversal)
            max_extension_pts = event_close - window['low'].min()
            max_retrace_pts = window['high'].max() - event_close
        
        # Ensure non-negative
        max_extension_pts = max(0, max_extension_pts)
        max_retrace_pts = max(0, max_retrace_pts)
        
        # Net move
        net_move = window.iloc[-1]['close'] - event_close
        if sweep_direction == 'DOWN':
            net_move = -net_move  # Normalize so positive = continued in sweep direction
        
        # ATR before/after
        atr_before = atr.iloc[event_idx] if event_idx < len(atr) else 0
        atr_after = atr.iloc[end_idx - 1] if end_idx - 1 < len(atr) else atr_before
        atr_change_pct = (atr_after - atr_before) / atr_before if atr_before > 0 else 0
        
        # Classify
        reaction_type = self.classify_reaction(max_extension_pts, max_retrace_pts, atr_before)
        vol_behavior = self.classify_volatility(atr_before, atr_after)
        
        return ReactionMeasurement(
            horizon_bars=horizon,
            max_extension_pts=max_extension_pts,
            max_retrace_pts=max_retrace_pts,
            net_move_pts=net_move,
            atr_before=atr_before,
            atr_after=atr_after,
            atr_change_pct=atr_change_pct,
            reaction_type=reaction_type,
            vol_behavior=vol_behavior
        )
