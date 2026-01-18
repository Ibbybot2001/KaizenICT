"""
Liquidity Event Detector
Phase 1: Enumerate all forms of liquidity interaction

Event Types:
1. Equal Highs/Lows (EQH/EQL) - Price touches same level twice
2. Prior Session H/L (PDH/PDL) - Previous day's high/low
3. Range Extremes - Daily/Weekly range boundaries
4. Sweep Classification - Micro (< 3 pts) vs Macro (> 3 pts)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time

@dataclass
class LiquidityEvent:
    """A single liquidity event."""
    timestamp: pd.Timestamp
    event_type: str  # 'EQH', 'EQL', 'PDH', 'PDL', 'RANGE_HIGH', 'RANGE_LOW'
    level_price: float
    sweep_price: float  # Price that breached the level
    sweep_size_pts: float  # How far past the level
    bar_idx: int
    
class LiquidityEventDetector:
    """
    Detects liquidity events in price data.
    
    All events are detected with NO LOOKAHEAD:
    - Equal H/L requires second touch to confirm the first
    - PDH/PDL only knowable after session close
    - Range extremes computed from prior data only
    """
    
    def __init__(self, 
                 equal_hl_tolerance: float = 1.0,  # Points tolerance for "equal"
                 min_bars_between: int = 5,  # Min bars between touches
                 session_start: time = time(9, 30),
                 session_end: time = time(16, 0),
                 earliest_entry: time = time(9, 45)):
        
        self.equal_hl_tolerance = equal_hl_tolerance
        self.min_bars_between = min_bars_between
        self.session_start = session_start
        self.session_end = session_end
        self.earliest_entry = earliest_entry
        
    def detect_equal_highs(self, data: pd.DataFrame, lookback: int = 50) -> List[LiquidityEvent]:
        """
        Detect Equal Highs (EQH) - Second touch of a prior high.
        Event is recorded when the second touch SWEEPS (breaches) the level.
        """
        events = []
        highs = data['high'].values
        
        for i in range(lookback, len(data)):
            current_high = highs[i]
            
            # Look back for prior highs within tolerance
            for j in range(i - self.min_bars_between, max(0, i - lookback), -1):
                prior_high = highs[j]
                
                # Check if current high sweeps the prior high
                if current_high >= prior_high - self.equal_hl_tolerance:
                    # This is a sweep
                    sweep_size = current_high - prior_high
                    
                    if sweep_size >= 0:  # Only count actual sweeps
                        events.append(LiquidityEvent(
                            timestamp=data.index[i],
                            event_type='EQH',
                            level_price=prior_high,
                            sweep_price=current_high,
                            sweep_size_pts=sweep_size,
                            bar_idx=i
                        ))
                        break  # One event per bar
                        
        return events
    
    def detect_equal_lows(self, data: pd.DataFrame, lookback: int = 50) -> List[LiquidityEvent]:
        """
        Detect Equal Lows (EQL) - Second touch of a prior low.
        Event is recorded when the second touch SWEEPS (breaches) the level.
        """
        events = []
        lows = data['low'].values
        
        for i in range(lookback, len(data)):
            current_low = lows[i]
            
            for j in range(i - self.min_bars_between, max(0, i - lookback), -1):
                prior_low = lows[j]
                
                if current_low <= prior_low + self.equal_hl_tolerance:
                    sweep_size = prior_low - current_low
                    
                    if sweep_size >= 0:
                        events.append(LiquidityEvent(
                            timestamp=data.index[i],
                            event_type='EQL',
                            level_price=prior_low,
                            sweep_price=current_low,
                            sweep_size_pts=sweep_size,
                            bar_idx=i
                        ))
                        break
                        
        return events
    
    def detect_pdh_pdl(self, data: pd.DataFrame) -> List[LiquidityEvent]:
        """
        Detect Prior Day High/Low (PDH/PDL) sweeps.
        PDH/PDL is only knowable AFTER the prior day closes.
        """
        events = []
        
        # Group by date
        data = data.copy()
        data['date'] = data.index.date
        
        dates = data['date'].unique()
        
        for i in range(1, len(dates)):
            prior_date = dates[i - 1]
            current_date = dates[i]
            
            prior_day = data[data['date'] == prior_date]
            current_day = data[data['date'] == current_date]
            
            if len(prior_day) == 0 or len(current_day) == 0:
                continue
                
            pdh = prior_day['high'].max()
            pdl = prior_day['low'].min()
            
            # Check each bar of current day for PDH sweep
            for idx, row in current_day.iterrows():
                # Filter by time (must be after earliest_entry)
                bar_time = idx.time()
                if bar_time < self.earliest_entry:
                    continue
                    
                # PDH Sweep
                if row['high'] >= pdh:
                    sweep_size = row['high'] - pdh
                    events.append(LiquidityEvent(
                        timestamp=idx,
                        event_type='PDH',
                        level_price=pdh,
                        sweep_price=row['high'],
                        sweep_size_pts=max(0, sweep_size),
                        bar_idx=data.index.get_loc(idx)
                    ))
                    
                # PDL Sweep
                if row['low'] <= pdl:
                    sweep_size = pdl - row['low']
                    events.append(LiquidityEvent(
                        timestamp=idx,
                        event_type='PDL',
                        level_price=pdl,
                        sweep_price=row['low'],
                        sweep_size_pts=max(0, sweep_size),
                        bar_idx=data.index.get_loc(idx)
                    ))
                    
        return events
    
    def detect_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all liquidity events and return as DataFrame.
        """
        all_events = []
        
        print("  Detecting Equal Highs...")
        all_events.extend(self.detect_equal_highs(data))
        
        print("  Detecting Equal Lows...")
        all_events.extend(self.detect_equal_lows(data))
        
        print("  Detecting PDH/PDL...")
        all_events.extend(self.detect_pdh_pdl(data))
        
        # Convert to DataFrame
        if not all_events:
            return pd.DataFrame()
            
        df = pd.DataFrame([
            {
                'timestamp': e.timestamp,
                'event_type': e.event_type,
                'level_price': e.level_price,
                'sweep_price': e.sweep_price,
                'sweep_size_pts': e.sweep_size_pts,
                'bar_idx': e.bar_idx
            }
            for e in all_events
        ])
        
        return df.sort_values('timestamp').reset_index(drop=True)
