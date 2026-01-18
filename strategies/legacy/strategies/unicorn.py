from .base_strategy import Strategy
from strategies.legacy.ict_library import calculate_swings, detect_fair_value_gaps, calculate_atr
from strategies.legacy.config import is_in_killzone
import pandas as pd
import numpy as np

class UnicornStrategy(Strategy):
    """
    ICT Unicorn Model:
    Confluence of a Breaker Block and a Fair Value Gap.
    
    Long Setup:
    1. Formation of a Bearish Order Block / Swing High.
    2. Price breaks ABOVE this high (turning it into a Bullish Breaker).
    3. The displacement leg that broke the high MUST leave a Bullish FVG.
    4. The FVG must overlap or be adjacent to the Breaker level.
    5. Entry at the FVG/Breaker retest.
    """
    
    def __init__(self, backtester):
        super().__init__(backtester)
        self.swing_len = 5 # Longer Swing for Breakers usually (5m context on 1m chart)
        self.risk_pct = 0.01
        
        # State
        self.breakers = [] # List of confirmed breakers
        self.last_clean_time = None

    def on_start(self, df):
        print("Calculating Indicators for Unicorn Strategy...")
        self.swing_highs, self.swing_lows = calculate_swings(df, left=self.swing_len, right=self.swing_len)
        self.fvgs = detect_fair_value_gaps(df)
        self.atr = calculate_atr(df)
        print(f"DEBUG: Found {self.swing_highs.count()} Swing Highs")

    def on_bar(self, i, bar):
        current_time = bar.name
        if not is_in_killzone(current_time):
            return

        # Position Guard
        if self.backtester.position_size != 0:
            return

        # 1. Detect New Breakers (Dynamic)
        # Look at recent swing highs (e.g. last 100 bars)
        # Verify if ANY swing high is broken by THIS bar's CLOSE
        
        recent_highs = self.swing_highs.iloc[i-50:i].dropna()
        if not recent_highs.empty:
            # Check for Break: Close > High
            # BUT we want a High that was previously RESPECTED (not already broken).
            # Simplified: Just find the Highest recent swing that is below current Close.
            
            # Optimization: 
            # Iterate recent highs. If bar['close'] > high AND bar['open'] < high? 
            # (Displacement Candle)
            pass
            
        # --- SIMPLIFIED UNICORN ---
        # 1. Identify MSS (Close > Recent High)
        # 2. Check overlap with FVG.
        
        # A. Find nearest un-broken Swing High
        # We assume recent highs are valid points.
        
        targets = recent_highs[recent_highs < bar['close']] 
        # These are highs we are currently ABOVE.
        # Ideally we want a High that we JUST broke recently.
        
        # Let's verify standard pattern:
        # High formed at T1.
        # Low formed at T2.
        # Break of T1 at T3.
        # FVG at T3.
        
        # Check FVG at this bar
        fvg = None
        if current_time in self.fvgs.index:
            try: 
                item = self.fvgs.loc[current_time]
                if isinstance(item, pd.DataFrame): item = item.iloc[0]
                if item['type'] == 'bull': fvg = item
            except: pass
            
        if fvg is not None:
            # We have a Bullish FVG.
            # Does it overlap with a broken Swing High?
            # Check if any Swing High is within (FVG Bottom - buffer, FVG Top + buffer).
            
            fvg_top = fvg['top']
            fvg_bot = fvg['bottom']
            
            # Look for recent highs that are roughly at this price level
            # Or highs that are slightly below FVG Top (Breaker).
            
            # Authentic Unicorn: The FVG is often RIGHT ABOVE the Breaker.
            # Breaker Level (High) ~ FVG Bottom.
            
            # Scan recent highs
            nearby_highs = recent_highs[ (recent_highs >= fvg_bot * 0.999) & (recent_highs <= fvg_top * 1.001) ]
            
            if not nearby_highs.empty:
                # We have a High inside the FVG? Or matching?
                # This suggests the gap formed AS we broke the high.
                
                # ENTRY
                entry_price = fvg_top
                breaker_level = nearby_highs.iloc[-1]
                
                print(f"[{current_time}] UNICORN LONG: FVG {fvg_top}-{fvg_bot} overlaps Breaker {breaker_level}")
                
                stop_loss = fvg_bot - (fvg_top - fvg_bot) # Stop below FVG
                tp = entry_price + (entry_price - stop_loss) * 3
                
                self.backtester.cancel_all_orders()
                self.backtester.place_limit_order('BUY', entry_price, sl=stop_loss, tp=tp)
        
        # --- SHORT UNICORN ---
        # Symmetric
        fvg_bear = None
        if current_time in self.fvgs.index:
            try:
                item = self.fvgs.loc[current_time]
                if isinstance(item, pd.DataFrame): item = item.iloc[0]
                if item['type'] == 'bear': fvg_bear = item
            except: pass
            
        if fvg_bear is not None:
            recent_lows = self.swing_lows.iloc[i-50:i].dropna()
            fvg_top = fvg_bear['top']
            fvg_bot = fvg_bear['bottom']
            
            # Check overlap
            nearby_lows = recent_lows[ (recent_lows >= fvg_bot * 0.999) & (recent_lows <= fvg_top * 1.001) ]
            
            if not nearby_lows.empty:
                breaker_level = nearby_lows.iloc[-1]
                print(f"[{current_time}] UNICORN SHORT: FVG {fvg_top}-{fvg_bot} overlaps Breaker {breaker_level}")
                
                entry_price = fvg_bot
                stop_loss = fvg_top + (fvg_top - fvg_bot)
                tp = entry_price - (stop_loss - entry_price) * 3
                
                self.backtester.cancel_all_orders()
                self.backtester.place_limit_order('SELL', entry_price, sl=stop_loss, tp=tp)
