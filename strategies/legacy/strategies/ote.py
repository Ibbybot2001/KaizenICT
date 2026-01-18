from strategies.legacy.strategies.base_strategy import Strategy
from strategies.legacy.ict_library import calculate_swings, calculate_atr
from strategies.legacy.backtester import OrderType
import pandas as pd
import math

class OTEStrategy(Strategy):
    def on_start(self, df):
        self.data = df
        self.swings_high, self.swings_low = calculate_swings(self.data, left=5, right=5)
        self.atr = calculate_atr(self.data)
        
        # State variables
        self.last_swing_high = None # (price, idx)
        self.last_swing_low = None  # (price, idx)
        
        self.market_structure = "NEUTRAL" # BULLISH, BEARISH
        self.anchor_point = None # The Low (for Bullish) or High (for Bearish) that started the move
        self.leg_peak = None # The highest/lowest point of the displacement leg
        
        self.pending_order_id = None

    def on_bar(self, idx, row):
        timestamp = row.name
        current_close = row['close']
        current_high = row['high']
        current_low = row['low']
        
        # 1. Update Swings
        if not pd.isna(self.swings_high.iloc[idx]):
            self.last_swing_high = (self.swings_high.iloc[idx], idx)
        if not pd.isna(self.swings_low.iloc[idx]):
            self.last_swing_low = (self.swings_low.iloc[idx], idx)
            
        # 2. Check for MSS (Market Structure Shift)
        # Only if we have recent swings
        if self.last_swing_high and self.last_swing_low:
            sh_price, sh_idx = self.last_swing_high
            sl_price, sl_idx = self.last_swing_low
            
            # BULLISH MSS: Breaking a Swing High
            # We need the Swing Low to be AFTER the Swing High?
            # Sequence: High -> Low -> Break High.
            # So sl_idx > sh_idx.
            if sl_idx > sh_idx:
                if current_close > sh_price:
                    # New Bullish Structure
                    if self.market_structure != "BULLISH":
                        self.market_structure = "BULLISH"
                        self.anchor_point = sl_price
                        self.leg_peak = current_high
                        self.cancel_pending()
                        print(f"[OTE] Bullish MSS at {timestamp}. Anchor: {self.anchor_point}")

            # BEARISH MSS: Breaking a Swing Low
            # Sequence: Low -> High -> Break Low
            # sh_idx > sl_idx
            if sh_idx > sl_idx:
                if current_close < sl_price:
                    if self.market_structure != "BEARISH":
                        self.market_structure = "BEARISH"
                        self.anchor_point = sh_price
                        self.leg_peak = current_low
                        self.cancel_pending()
                        print(f"[OTE] Bearish MSS at {timestamp}. Anchor: {self.anchor_point}")

        # 3. Manage Active Structure
        if self.backtester.position_size == 0:
            if self.market_structure == "BULLISH":
                # Update Peak
                if current_high > self.leg_peak:
                    self.leg_peak = current_high
                    self.cancel_pending() # Re-adjust limit
                
                # Check Invalidation (Price broke anchor)
                if current_low < self.anchor_point:
                    self.market_structure = "NEUTRAL"
                    self.cancel_pending()
                    return

                # Place/Update Limit Order at 0.705 (OTE)
                ote_level = self.anchor_point + (self.leg_peak - self.anchor_point) * 0.705
                # Ideally place order if not exists
                if not self.pending_order_id:
                    sl = self.anchor_point
                    tp = self.leg_peak + (self.leg_peak - self.anchor_point) * 0.5 # -0.5 SD Target
                    
                    self.backtester.place_order(
                        price=ote_level,
                        size=1,
                        side='BUY',
                        order_type=OrderType.LIMIT,
                        sl=sl,
                        tp=tp
                    )
                    self.pending_order_id = "ACTIVE" # Ideally track ID
            
            elif self.market_structure == "BEARISH":
                # Update Trough
                if current_low < self.leg_peak:
                    self.leg_peak = current_low
                    self.cancel_pending()

                if current_high > self.anchor_point:
                    self.market_structure = "NEUTRAL"
                    self.cancel_pending()
                    return

                ote_level = self.anchor_point - (self.anchor_point - self.leg_peak) * 0.705
                if not self.pending_order_id:
                    sl = self.anchor_point
                    tp = self.leg_peak - (self.anchor_point - self.leg_peak) * 0.5
                    
                    self.backtester.place_order(
                        price=ote_level,
                        size=1,
                        side='SELL',
                        order_type=OrderType.LIMIT,
                        sl=sl,
                        tp=tp
                    )
                    self.pending_order_id = "ACTIVE"

    def cancel_pending(self):
        self.backtester.orders = [] # Naive clear all orders
        self.pending_order_id = None
