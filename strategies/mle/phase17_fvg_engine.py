"""
PHASE 17: FVG / SILVER BULLET ENGINE
Evolution of Phase 16 to include Fair Value Gap (FVG) detection.

Silver Bullet Logic:
1. Time Window: 10:00 - 11:00 AM ET.
2. Signal: Formation of an FVG that aligns with Session Trend.
3. Entry: Limit order at the FVG proximal (start) line.
"""

import pandas as pd
import numpy as np

# Inherit / Import common logic from Phase 16 if possible, or duplicate for independence?
# Duplicating for stability and independence.

# ==============================================================================
# FEATURE ENGINEERING: FVG
# ==============================================================================
def engineer_fvg(df_bars):
    """
    Computes Bullish & Bearish FVGs.
    Bullish FVG: Low[n] > High[n-2]
    Bearish FVG: High[n] < Low[n-2]
    """
    df = df_bars.copy()
    
    # Shifted Prices for comparison
    # Bar N is current (completed)
    # Bar N-1 is middle
    # Bar N-2 is start
    
    # We want to know if Bar N *completed* an FVG
    
    # Bullish FVG Check
    # Gap exists between High of N-2 and Low of N
    prev_high = df['high'].shift(2)
    curr_low = df['low']
    
    df['fvg_bull'] = (curr_low > prev_high)
    df['fvg_bull_top'] = curr_low
    df['fvg_bull_btm'] = prev_high
    
    # Bearish FVG Check
    # Gap exists between Low of N-2 and High of N
    prev_low = df['low'].shift(2)
    curr_high = df['high']
    
    df['fvg_bear'] = (curr_high < prev_low)
    df['fvg_bear_top'] = prev_low
    df['fvg_bear_btm'] = curr_high
    
    return df

# ==============================================================================
# SIGNAL DETECTION: SILVER BULLET
# ==============================================================================
def detect_silver_bullet_signals(df_bars, tracker, current_date):
    """
    Detects valid FVG entries in the 10:00 - 11:00 window.
    """
    signals = []
    
    # Strict Window: 10:00 to 11:00
    mask_sb = (df_bars['hour'] == 10) & (df_bars['date'] == current_date)
    sb_bars = df_bars[mask_sb]
    
    # Also need context (daily bias)? For now, test BOTH directions.
    
    for i, row in sb_bars.iterrows():
        # Check if THIS bar created an FVG
        
        # Bullish Silver Bullet
        if row['fvg_bull']:
            # Entry is at the TOP of the gap (Limit Buy)
            # Stop is below the 3-candle cluster? Or fixed?
            # Standard ICT: Stop below the swing low (Bar 1 low)
            entry_price = row['fvg_bull_top']
            sl_price = row['low'] - 10.0 # Default buffer for search
            tp_price = entry_price + 40.0 # Target 40 pts
            
            signals.append({
                'time': row['time'],
                'type': 'FVG_BULL',
                'entry': entry_price,
                'sl': sl_price,
                'tp': tp_price
            })
            
        # Bearish Silver Bullet
        if row['fvg_bear']:
            entry_price = row['fvg_bear_btm']
            sl_price = row['high'] + 10.0
            tp_price = entry_price - 40.0
            
            signals.append({
                'time': row['time'],
                'type': 'FVG_BEAR',
                'entry': entry_price,
                'sl': sl_price,
                'tp': tp_price
            })
            
    return signals
