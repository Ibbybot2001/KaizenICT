"""
Multi-Path Outcome Labeler
Phase 3: Label each event under multiple exit schemas

Exit Schemas:
- fixed_1r: TP = 1R, SL = 1R
- fixed_1.5r: TP = 1.5R, SL = 1R
- fixed_2r: TP = 2R, SL = 1R
- partial_1_2: 50% @ 1R, 50% @ 2R
- time_10: Exit after 10 bars (no SL/TP)
- time_20: Exit after 20 bars
- time_60: Exit after 60 bars

NO FILTERING. Label everything broadly.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Fixed SL in points (from research)
FIXED_SL_PTS = 10.0
FRICTION_R = 0.05  # 5% of R friction

@dataclass
class TradeOutcome:
    """Result of simulating a trade under a specific exit schema."""
    schema: str
    result: str  # 'WIN', 'LOSS', 'TIME_EXIT'
    r_multiple: float  # Net R after friction
    bars_to_exit: int
    exit_price: float

class MultiPathLabeler:
    """
    Simulates trade outcomes under multiple exit schemas.
    
    Key Assumption:
    - Trade direction is determined by sweep type:
      - EQH/PDH (sweep above): Trade SHORT (fade the sweep)
      - EQL/PDL (sweep below): Trade LONG (fade the sweep)
    
    This tests the "reversal" hypothesis from Phase 2 (61% deep retrace).
    """
    
    def __init__(self, sl_pts: float = FIXED_SL_PTS, friction_r: float = FRICTION_R):
        self.sl_pts = sl_pts
        self.friction_r = friction_r
    
    def get_trade_direction(self, event_type: str) -> str:
        """
        Determine trade direction based on sweep type.
        For Phase 3: We test FADING the sweep (reversal trades).
        """
        if event_type in ['EQH', 'PDH']:
            return 'SHORT'  # Fade sweep above
        else:
            return 'LONG'   # Fade sweep below
    
    def simulate_fixed_exit(self,
                           data: pd.DataFrame,
                           entry_idx: int,
                           direction: str,
                           tp_r: float,
                           max_bars: int = 100) -> TradeOutcome:
        """
        Simulate fixed SL/TP exit.
        
        Args:
            tp_r: Take profit in R-multiples (e.g., 1.0, 1.5, 2.0)
        """
        entry_price = data.iloc[entry_idx]['close']
        
        # Calculate SL/TP prices
        if direction == 'LONG':
            sl_price = entry_price - self.sl_pts
            tp_price = entry_price + (self.sl_pts * tp_r)
        else:  # SHORT
            sl_price = entry_price + self.sl_pts
            tp_price = entry_price - (self.sl_pts * tp_r)
        
        # Walk forward
        for i in range(1, min(max_bars, len(data) - entry_idx)):
            bar_idx = entry_idx + i
            bar = data.iloc[bar_idx]
            
            if direction == 'LONG':
                # Check SL first (conservative)
                if bar['low'] <= sl_price:
                    return TradeOutcome(
                        schema=f'fixed_{tp_r}r',
                        result='LOSS',
                        r_multiple=-1.0 - self.friction_r,
                        bars_to_exit=i,
                        exit_price=sl_price
                    )
                # Check TP
                if bar['high'] >= tp_price:
                    return TradeOutcome(
                        schema=f'fixed_{tp_r}r',
                        result='WIN',
                        r_multiple=tp_r - self.friction_r,
                        bars_to_exit=i,
                        exit_price=tp_price
                    )
            else:  # SHORT
                # Check SL first
                if bar['high'] >= sl_price:
                    return TradeOutcome(
                        schema=f'fixed_{tp_r}r',
                        result='LOSS',
                        r_multiple=-1.0 - self.friction_r,
                        bars_to_exit=i,
                        exit_price=sl_price
                    )
                # Check TP
                if bar['low'] <= tp_price:
                    return TradeOutcome(
                        schema=f'fixed_{tp_r}r',
                        result='WIN',
                        r_multiple=tp_r - self.friction_r,
                        bars_to_exit=i,
                        exit_price=tp_price
                    )
        
        # Time expired without hit - exit at market
        exit_price = data.iloc[min(entry_idx + max_bars, len(data) - 1)]['close']
        if direction == 'LONG':
            r_mult = (exit_price - entry_price) / self.sl_pts
        else:
            r_mult = (entry_price - exit_price) / self.sl_pts
        
        return TradeOutcome(
            schema=f'fixed_{tp_r}r',
            result='TIME_EXIT',
            r_multiple=r_mult - self.friction_r,
            bars_to_exit=max_bars,
            exit_price=exit_price
        )
    
    def simulate_partial_exit(self,
                             data: pd.DataFrame,
                             entry_idx: int,
                             direction: str,
                             max_bars: int = 100) -> TradeOutcome:
        """
        Simulate partial exit: 50% at 1R, 50% at 2R.
        """
        entry_price = data.iloc[entry_idx]['close']
        
        if direction == 'LONG':
            sl_price = entry_price - self.sl_pts
            tp1_price = entry_price + self.sl_pts      # 1R
            tp2_price = entry_price + (2 * self.sl_pts) # 2R
        else:
            sl_price = entry_price + self.sl_pts
            tp1_price = entry_price - self.sl_pts
            tp2_price = entry_price - (2 * self.sl_pts)
        
        hit_tp1 = False
        total_r = 0.0
        exit_bar = 0
        
        for i in range(1, min(max_bars, len(data) - entry_idx)):
            bar_idx = entry_idx + i
            bar = data.iloc[bar_idx]
            
            if direction == 'LONG':
                # Check SL
                if bar['low'] <= sl_price:
                    if hit_tp1:
                        # Already took 50% at 1R, lose 50% at -1R
                        total_r = 0.5 * 1.0 + 0.5 * (-1.0)
                    else:
                        total_r = -1.0
                    return TradeOutcome(
                        schema='partial_1_2',
                        result='LOSS' if total_r < 0 else 'WIN',
                        r_multiple=total_r - self.friction_r,
                        bars_to_exit=i,
                        exit_price=sl_price
                    )
                
                # Check TP1
                if not hit_tp1 and bar['high'] >= tp1_price:
                    hit_tp1 = True
                    # Move SL to breakeven for remaining 50%
                    sl_price = entry_price
                
                # Check TP2
                if hit_tp1 and bar['high'] >= tp2_price:
                    total_r = 0.5 * 1.0 + 0.5 * 2.0  # = 1.5R
                    return TradeOutcome(
                        schema='partial_1_2',
                        result='WIN',
                        r_multiple=total_r - self.friction_r,
                        bars_to_exit=i,
                        exit_price=tp2_price
                    )
            else:  # SHORT
                if bar['high'] >= sl_price:
                    if hit_tp1:
                        total_r = 0.5 * 1.0 + 0.5 * (-1.0)
                    else:
                        total_r = -1.0
                    return TradeOutcome(
                        schema='partial_1_2',
                        result='LOSS' if total_r < 0 else 'WIN',
                        r_multiple=total_r - self.friction_r,
                        bars_to_exit=i,
                        exit_price=sl_price
                    )
                
                if not hit_tp1 and bar['low'] <= tp1_price:
                    hit_tp1 = True
                    sl_price = entry_price
                
                if hit_tp1 and bar['low'] <= tp2_price:
                    total_r = 0.5 * 1.0 + 0.5 * 2.0
                    return TradeOutcome(
                        schema='partial_1_2',
                        result='WIN',
                        r_multiple=total_r - self.friction_r,
                        bars_to_exit=i,
                        exit_price=tp2_price
                    )
        
        # Time expired
        exit_price = data.iloc[min(entry_idx + max_bars, len(data) - 1)]['close']
        if direction == 'LONG':
            r_mult = (exit_price - entry_price) / self.sl_pts
        else:
            r_mult = (entry_price - exit_price) / self.sl_pts
        
        if hit_tp1:
            total_r = 0.5 * 1.0 + 0.5 * r_mult
        else:
            total_r = r_mult
        
        return TradeOutcome(
            schema='partial_1_2',
            result='TIME_EXIT',
            r_multiple=total_r - self.friction_r,
            bars_to_exit=max_bars,
            exit_price=exit_price
        )
    
    def simulate_time_exit(self,
                          data: pd.DataFrame,
                          entry_idx: int,
                          direction: str,
                          exit_bars: int) -> TradeOutcome:
        """
        Simulate pure time-based exit (no SL/TP).
        """
        entry_price = data.iloc[entry_idx]['close']
        exit_idx = min(entry_idx + exit_bars, len(data) - 1)
        exit_price = data.iloc[exit_idx]['close']
        
        if direction == 'LONG':
            r_mult = (exit_price - entry_price) / self.sl_pts
        else:
            r_mult = (entry_price - exit_price) / self.sl_pts
        
        result = 'WIN' if r_mult > 0 else 'LOSS'
        
        return TradeOutcome(
            schema=f'time_{exit_bars}',
            result=result,
            r_multiple=r_mult - self.friction_r,
            bars_to_exit=exit_bars,
            exit_price=exit_price
        )
    
    def label_event(self, 
                   data: pd.DataFrame, 
                   event_idx: int, 
                   event_type: str) -> Dict[str, TradeOutcome]:
        """
        Label a single event under all exit schemas.
        Returns dict of schema -> outcome.
        """
        direction = self.get_trade_direction(event_type)
        
        outcomes = {}
        
        # Fixed exits
        for tp_r in [1.0, 1.5, 2.0]:
            outcome = self.simulate_fixed_exit(data, event_idx, direction, tp_r)
            outcomes[f'fixed_{tp_r}r'] = outcome
        
        # Partial exit
        outcomes['partial_1_2'] = self.simulate_partial_exit(data, event_idx, direction)
        
        # Time exits
        for bars in [10, 20, 60]:
            outcomes[f'time_{bars}'] = self.simulate_time_exit(data, event_idx, direction, bars)
        
        return outcomes
