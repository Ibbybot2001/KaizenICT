"""
MLE Signal Engine (Pandas Port)
Handles Market Structure (Swings), Displacement, and FVG logic.
Ported from 'code strategy.txt' (originally Polars) to Pandas.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum

class Trend(Enum):
    UP = 1
    DOWN = -1
    NEUTRAL = 0

@dataclass
class LiquidityPool:
    id: str
    price: float
    type: str # 'High', 'Low'
    created_at: int # Index or Timestamp
    strength: int # Number of touches/swings
    
@dataclass
class FVG:
    top: float
    bottom: float
    is_bullish: bool
    created_at: int
    idx: int

class SignalEngine:
    def __init__(self, df: pd.DataFrame):
        """
        df: Pandas DataFrame with M1 data (must have: open, high, low, close)
        """
        self.df = df.copy()
        # Ensure column names are lower case
        self.df.columns = [c.lower() for c in self.df.columns]
        
    def precompute_market_structure(self) -> pd.DataFrame:
        """
        Adds vector-calculated columns for Swings, Displacement, and basic FVG candidates.
        """
        df = self.df
        
        # 1. Body Size & Displacement
        df['body_size'] = abs(df['close'] - df['open'])
        df['range_size'] = df['high'] - df['low']
        
        # Rolling Median Body (20)
        # Shift 1 to avoid lookahead (compare current body to PREVIOUS 20 median)
        df['median_body_20'] = df['body_size'].rolling(window=20).median().shift(1)
        
        # Logic: Body >= 1.25 * Median
        df['is_displacement'] = df['body_size'] >= (df['median_body_20'] * 1.25)
        
        # 2. Swing Highs / Lows (2 bar lookback/forward)
        # Verified at T (using T-2, T-1, T+1, T+2 checks centered at T)
        # Wait, the polars code used shift logic to find if T is a swing point.
        # T is swing high if High[T] > T-1, T-2, T+1, T+2.
        # This requires future data (T+1, T+2). 
        # In a LIVE/Backtest loop, we only know this at T+2.
        # So we will mark 'is_swing_high' at the bar defined as the swing.
        # The Orchestrator must read this with a lag (look at T-2).
        
        h = df['high']
        l = df['low']
        
        # We need to shift 'future' bars back to current time to compare
        # Or more intuitively:
        # Swing High at i: H[i] > H[i-1], H[i-2], H[i+1], H[i+2]
        
        # Using centered rolling window or shifts
        # Prev 2
        prev1 = h.shift(1)
        prev2 = h.shift(2)
        # Next 2 (requires shift(-1))
        next1 = h.shift(-1)
        next2 = h.shift(-2)
        
        is_sh = (h > prev1) & (h > prev2) & (h > next1) & (h > next2)
        
        # Lows
        l_prev1 = l.shift(1)
        l_prev2 = l.shift(2)
        l_next1 = l.shift(-1)
        l_next2 = l.shift(-2)
        
        is_sl = (l < l_prev1) & (l < l_prev2) & (l < l_next1) & (l < l_next2)
        
        df['is_swing_high'] = is_sh.fillna(False)
        df['is_swing_low'] = is_sl.fillna(False)
        
        # 3. FVG Identification
        # Defined at completion of bar T
        # Bull: L[T] > H[T-2]
        # Bear: H[T] < L[T-2]
        
        prev2_h = h.shift(2)
        prev2_l = l.shift(2)
        
        is_fvg_bull = df['low'] > prev2_h
        is_fvg_bear = df['high'] < prev2_l
        
        df['fvg_bull'] = is_fvg_bull
        df['fvg_bear'] = is_fvg_bear
        
        df['fvg_bull_bottom'] = prev2_h
        df['fvg_bull_top'] = df['low']
        
        df['fvg_bear_top'] = prev2_l
        df['fvg_bear_bottom'] = df['high']
        
        # Candle Morphology for filter (Body %)
        # Ratio = (Close - Open) / (High - Low)  (Signed? Math doc says Close-Open / High-Low)
        # Usually abs(body) / range.
        # Math doc: "ratio of (Close - Open) / (High - Low)" -> implied signed?
        # "IF Candle Body ($m$) > 0.50 (Strong Displacement)" -> usually means Absolute body strength.
        # Let's use abs.
        with np.errstate(divide='ignore', invalid='ignore'):
            df['morphology'] = df['body_size'] / df['range_size']
        df['morphology'] = df['morphology'].fillna(0.0)
        
        return df

class LiquidityManager:
    """
    Manages active Liquidity Pools statefully.
    """
    def __init__(self):
        self.pools: List[LiquidityPool] = []
        self.pool_counter = 0
        
    def add_swing(self, price: float, type: str, index: int):
        self.pool_counter += 1
        pool = LiquidityPool(
            id=f"{type}_{self.pool_counter}",
            price=price,
            type=type,
            created_at=index,
            strength=1
        )
        self.pools.append(pool)
        
    def check_sweeps(self, high: float, low: float) -> List[LiquidityPool]:
        """
        Returns list of pools swept by the current bar.
        Only keeps UN-swept pools.
        """
        swept = []
        keep = []
        
        for pool in self.pools:
            is_swept = False
            if pool.type == 'High':
                if high > pool.price:
                    is_swept = True
            elif pool.type == 'Low':
                if low < pool.price:
                    is_swept = True
            
            if is_swept:
                swept.append(pool)
            else:
                keep.append(pool)
        
        self.pools = keep
        return swept
