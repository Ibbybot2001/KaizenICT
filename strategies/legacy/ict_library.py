import pandas as pd
import numpy as np
from collections import deque

def calculate_swings(df, left=3, right=3):
    """
    Identify Swing Highs and Swing Lows.
    RETURNS: Two Series (swing_highs, swing_lows) with Price at the CONFIRMED index.
    
    CRITICAL: A swing at index `i` is only confirmed at index `i + right`.
    To avoid lookahead bias, the signal must appear `right` bars later.
    """
    highs = df['high'].values
    lows = df['low'].values
    
    # We'll stick these in a Series aligned with df.index
    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)
    
    # We iterate from (left) to (len - right)
    # This is slow in pure python loop, but safe. 
    # For optimization, we can use rolling().max() but handling the "future" 
    # confirmation specifically is tricky with pandas rolling.
    
    for i in range(left, len(df) - right):
        # Check Swing High
        current_high = highs[i]
        # Max of left neighbors
        if current_high > np.max(highs[i-left:i]):
            # Max of right neighbors
            if current_high > np.max(highs[i+1:i+right+1]):
                # Found a Swing High at [i]
                # It is confirmed at [i + right]
                confirm_idx = df.index[i + right]
                
                # We record the PRICE of the swing, at the time of confirmation
                # Note: You might want to store it at 'i' if your backtester looks back,
                # but for event-driven, we usually push an event. 
                # Here we will mark the EVENT at i+right, but store the PRICE.
                swing_highs.iloc[i + right] = current_high

        # Check Swing Low
        current_low = lows[i]
        if current_low < np.min(lows[i-left:i]):
            if current_low < np.min(lows[i+1:i+right+1]):
                # Found Swing Low
                swing_lows.iloc[i + right] = current_low
                
    return swing_highs, swing_lows

def detect_fair_value_gaps(df):
    """
    Detect plain 3-bar Fair Value Gaps.
    Bullish FVG: Low[n] > High[n-2]
    Bearish FVG: High[n] < Low[n-2]
    """
    # Vectorized approach
    highs = df['high']
    lows = df['low']
    
    # Shifted arrays for comparison
    # candle[i] is current. candle[i-2] is 2 bars ago.
    
    # Bullish: Low[i] > High[i-2]
    # We compare Current Low vs High shifted by 2
    # But strictly, FVG is created at close of candle [i].
    
    # Create boolean masks
    bull_fvg_mask = lows > highs.shift(2)
    bear_fvg_mask = highs < lows.shift(2)
    
    # Store the Gap Price Range
    # Bullish: Gap is between High[i-2] (Bottom) and Low[i] (Top) 
    # WAIT: Usually defined as:
    # Bullish FVG: Candle 1 High ... Gap ... Candle 3 Low. 
    # So Gap Bottom = High[i-2], Gap Top = Low[i]
    
    fvg_list = []
    
    # We iterate to create objects, or return a DataFrame
    # Let's return a DataFrame of FVGs
    
    # Create objects with explicit columns
    bull_gaps = pd.DataFrame(index=df.index[bull_fvg_mask], columns=['type', 'top', 'bottom'])
    if not bull_gaps.empty:
        bull_gaps['type'] = 'bull'
        bull_gaps['top'] = lows[bull_fvg_mask]
        bull_gaps['bottom'] = highs.shift(2)[bull_fvg_mask]
    
    bear_gaps = pd.DataFrame(index=df.index[bear_fvg_mask], columns=['type', 'top', 'bottom'])
    if not bear_gaps.empty:
        bear_gaps['type'] = 'bear'
        bear_gaps['top'] = lows.shift(2)[bear_fvg_mask]
        bear_gaps['bottom'] = highs[bear_fvg_mask]
        
    return pd.concat([bull_gaps, bear_gaps]).sort_index()

def detect_breaker_blocks(df, swing_highs, swing_lows):
    """
    Identifies potential Breaker Blocks.
    A Bullish Breaker is a Swing High that was broken (Close > High).
    A Bearish Breaker is a Swing Low that was broken (Close < Low).
    
    Returns: DataFrame of Breakers [time_of_break, type, level, top, bottom]
    """
    # This is a heavy calculation if done per bar. 
    # We will return a list of CONFIRMED breakers derived from Swings.
    # Logic: 
    # 1. Iterate confirmed Swing Highs. 
    # 2. Find when they are broken.
    # 3. Mark that area as a Breaker Zone.
    
    breakers = []
    
    # Analyze Highs (Potential Bullish Breakers)
    # We look for price closing above a Swing High.
    # Using vectorization is hard because "Broken" happens at dynamic times.
    
    # We will perform a simplified check:
    # return a DataFrame checked in strategy? No, too slow.
    
    # Let's map Swing Highs to their Break Times.
    # Optimization: Only look at major swings?
    
    # Given the constraint of python loops, we might do this inside the strategy loop selectively?
    # Or optimize:
    
    # Let's try iterating logic only.
    # For a Swing High at time T with price P:
    # Find first bar T2 > T where Close > P.
    # Record Breaker(Time=T2, Price=P, Type=Bullish).
    
    # Efficient search:
    # prices = df['close'].values
    # times = df.index
    
    # This is still O(N*M).
    # Since we have ~40k bars, maybe okay.
    pass 
    
    return pd.DataFrame() # Placeholder for now, implemented inside Unicorn logic for efficiency.

def calculate_atr(df, length=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return pd.Series(true_range).rolling(length).mean()
