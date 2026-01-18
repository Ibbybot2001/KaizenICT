"""
Label Generator - Phase 3A

Generates labels for supervised learning.

CRITICAL RULES:
1. Labels use FUTURE data (that's the target)
2. Labels must NEVER leak into features
3. Label availability time explicitly tracked

Labels are TARGETS, features are FACTS.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

from engine.constants import MIN_SL_POINTS


class LabelType(str, Enum):
    """Types of labels for ML."""
    FORWARD_RETURN = 'forward_return'
    FORWARD_RETURN_SIGN = 'forward_return_sign'
    HIT_TP_BEFORE_SL = 'hit_tp_before_sl'
    PATH_EXPANSION = 'path_expansion'  # Did price move X points before Y bars?


@dataclass
class LabelResult:
    """Result of label computation."""
    bar_idx: int
    label_type: LabelType
    value: float
    horizon_bars: int
    valid: bool  # Whether label could be computed (enough future data)


class LabelGenerator:
    """
    Generates labels for supervised learning.
    
    TIMING: Labels use future bars (that's intentional - they're targets).
    Features must be computed separately and NOT include label info.
    """
    
    def forward_return(self, bar_idx: int, data: pd.DataFrame,
                       horizon: int) -> Optional[LabelResult]:
        """
        Compute forward return over horizon bars.
        
        Label = (close[bar_idx + horizon] - close[bar_idx]) / close[bar_idx]
        
        Args:
            bar_idx: Current bar index
            data: OHLCV dataframe
            horizon: Number of bars forward
            
        Returns:
            LabelResult or None if insufficient future data
        """
        future_idx = bar_idx + horizon
        if future_idx >= len(data):
            return None
        
        current_close = data.iloc[bar_idx]['close']
        future_close = data.iloc[future_idx]['close']
        
        return_pct = (future_close - current_close) / current_close
        
        return LabelResult(
            bar_idx=bar_idx,
            label_type=LabelType.FORWARD_RETURN,
            value=return_pct,
            horizon_bars=horizon,
            valid=True
        )
    
    def forward_return_sign(self, bar_idx: int, data: pd.DataFrame,
                            horizon: int) -> Optional[LabelResult]:
        """
        Compute forward return sign (1 = up, 0 = down).
        
        Binary classification target.
        """
        ret = self.forward_return(bar_idx, data, horizon)
        if ret is None:
            return None
        
        return LabelResult(
            bar_idx=bar_idx,
            label_type=LabelType.FORWARD_RETURN_SIGN,
            value=1.0 if ret.value > 0 else 0.0,
            horizon_bars=horizon,
            valid=True
        )
    
    def hit_tp_before_sl(self, bar_idx: int, data: pd.DataFrame,
                         direction: str, sl_points: float, tp_points: float,
                         max_horizon: int = 100) -> Optional[LabelResult]:
        """
        Compute whether TP is hit before SL.
        
        This simulates a trade outcome without actually executing.
        
        Args:
            bar_idx: Entry bar index
            data: OHLCV dataframe
            direction: 'LONG' or 'SHORT'
            sl_points: Stop loss distance in points
            tp_points: Take profit distance in points
            max_horizon: Maximum bars to look ahead
            
        Returns:
            LabelResult with value = 1 if TP hit first, 0 if SL hit first,
            None if neither hit within horizon or insufficient data
        """
        # Enforce minimum SL
        if sl_points < MIN_SL_POINTS:
            sl_points = MIN_SL_POINTS
        
        entry_price = data.iloc[bar_idx]['close']
        
        if direction == 'LONG':
            sl_price = entry_price - sl_points
            tp_price = entry_price + tp_points
        else:
            sl_price = entry_price + sl_points
            tp_price = entry_price - tp_points
        
        # Scan future bars
        end_idx = min(bar_idx + max_horizon + 1, len(data))
        
        for i in range(bar_idx + 1, end_idx):
            bar = data.iloc[i]
            
            if direction == 'LONG':
                # Check SL first (conservative)
                if bar['low'] <= sl_price:
                    return LabelResult(
                        bar_idx=bar_idx,
                        label_type=LabelType.HIT_TP_BEFORE_SL,
                        value=0.0,  # SL hit
                        horizon_bars=i - bar_idx,
                        valid=True
                    )
                if bar['high'] >= tp_price:
                    return LabelResult(
                        bar_idx=bar_idx,
                        label_type=LabelType.HIT_TP_BEFORE_SL,
                        value=1.0,  # TP hit
                        horizon_bars=i - bar_idx,
                        valid=True
                    )
            else:
                # SHORT
                if bar['high'] >= sl_price:
                    return LabelResult(
                        bar_idx=bar_idx,
                        label_type=LabelType.HIT_TP_BEFORE_SL,
                        value=0.0,
                        horizon_bars=i - bar_idx,
                        valid=True
                    )
                if bar['low'] <= tp_price:
                    return LabelResult(
                        bar_idx=bar_idx,
                        label_type=LabelType.HIT_TP_BEFORE_SL,
                        value=1.0,
                        horizon_bars=i - bar_idx,
                        valid=True
                    )
        
        # Neither hit within horizon
        return None
    
    def path_expansion(self, bar_idx: int, data: pd.DataFrame,
                       expansion_points: float, max_horizon: int = 30) -> Optional[LabelResult]:
        """
        Did price move >= X points in EITHER direction before Y bars?
        
        Tests path expansion, not direction. Aligned with displacement,
        liquidity, and role reversal concepts.
        
        Args:
            bar_idx: Current bar index
            data: OHLCV dataframe
            expansion_points: Required movement in points
            max_horizon: Maximum bars to check
            
        Returns:
            LabelResult with value=1 if expansion occurred, 0 otherwise
        """
        if bar_idx + max_horizon >= len(data):
            return None
        
        entry_price = data.iloc[bar_idx]['close']
        
        for i in range(bar_idx + 1, bar_idx + max_horizon + 1):
            bar = data.iloc[i]
            
            # Check if price moved X points in either direction
            up_move = bar['high'] - entry_price
            down_move = entry_price - bar['low']
            
            if up_move >= expansion_points or down_move >= expansion_points:
                return LabelResult(
                    bar_idx=bar_idx,
                    label_type=LabelType.PATH_EXPANSION,
                    value=1.0,  # Expansion occurred
                    horizon_bars=i - bar_idx,
                    valid=True
                )
        
        # No expansion within horizon
        return LabelResult(
            bar_idx=bar_idx,
            label_type=LabelType.PATH_EXPANSION,
            value=0.0,  # No expansion
            horizon_bars=max_horizon,
            valid=True
        )
    
    def generate_labels(self, data: pd.DataFrame,
                        label_type: LabelType,
                        horizon: int = 10,
                        direction: str = 'LONG',
                        sl_points: float = 10.0,
                        tp_points: float = 20.0,
                        expansion_points: float = 15.0) -> pd.DataFrame:
        """
        Generate labels for all applicable bars.
        
        Args:
            data: OHLCV dataframe
            label_type: Type of label to generate
            horizon: Bars forward (for return-based labels)
            direction: Trade direction (for TP/SL labels)
            sl_points, tp_points: SL/TP distances
            expansion_points: Points for PATH_EXPANSION label
            
        Returns:
            DataFrame with bar_idx, label, valid columns
        """
        records = []
        
        for bar_idx in range(len(data) - horizon - 1):
            if label_type == LabelType.FORWARD_RETURN:
                result = self.forward_return(bar_idx, data, horizon)
            elif label_type == LabelType.FORWARD_RETURN_SIGN:
                result = self.forward_return_sign(bar_idx, data, horizon)
            elif label_type == LabelType.HIT_TP_BEFORE_SL:
                result = self.hit_tp_before_sl(
                    bar_idx, data, direction, sl_points, tp_points
                )
            elif label_type == LabelType.PATH_EXPANSION:
                result = self.path_expansion(bar_idx, data, expansion_points, horizon)
            else:
                result = None
            
            if result and result.valid:
                records.append({
                    'bar_idx': bar_idx,
                    'timestamp': data.index[bar_idx],
                    'label': result.value,
                    'horizon': result.horizon_bars,
                })
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records).set_index('timestamp')


def compute_forward_return(bar_idx: int, data: pd.DataFrame,
                           horizon: int) -> Optional[float]:
    """Convenience: Get forward return as float."""
    gen = LabelGenerator()
    result = gen.forward_return(bar_idx, data, horizon)
    return result.value if result else None


def compute_tp_sl_outcome(bar_idx: int, data: pd.DataFrame,
                          direction: str, sl_points: float, 
                          tp_points: float) -> Optional[int]:
    """Convenience: Get TP/SL outcome (1=TP, 0=SL, None=neither)."""
    gen = LabelGenerator()
    result = gen.hit_tp_before_sl(bar_idx, data, direction, sl_points, tp_points)
    return int(result.value) if result else None
