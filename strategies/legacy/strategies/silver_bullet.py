from strategies.legacy.strategies.base_strategy import Strategy
from strategies.legacy.ict_library import detect_fair_value_gaps, calculate_swings, calculate_atr
from strategies.legacy.config import is_in_silver_bullet, TICK_SIZE
import pandas as pd
from strategies.legacy.backtester import OrderType

class SilverBulletStrategy(Strategy):
    def on_start(self, df):
        self.data = df
        self.fvgs = detect_fair_value_gaps(self.data)
        self.atr = calculate_atr(self.data)
        self.swings_high, self.swings_low = calculate_swings(self.data, left=5, right=5)
        self.trades_in_session = set()
        
    def on_bar(self, idx, row):
        timestamp = row.name
        # 1. Check Session Window
        if not is_in_silver_bullet(timestamp):
            return

        # 2. Check if we already traded this session
        session_id = f"{timestamp.date()}_{'AM' if timestamp.hour < 12 else 'PM'}"
        if session_id in self.trades_in_session:
            return
            
        # 3. Check for Active Positions
        if self.backtester.position_size != 0:
            return

        # 4. Check for New FVG Formation
        # If the current bar *completed* an FVG, it will be in self.fvgs index
        if timestamp in self.fvgs.index:
            fvg_data = self.fvgs.loc[timestamp]
            
            # Handle duplicates? usually one per bar.
            if isinstance(fvg_data, pd.DataFrame): 
                # Very rare to have both bull and bear gap close on same bar? Impossible for 3-bar logic.
                # But assume Series.
                fvg_data = fvg_data.iloc[0]

            fvg_type = fvg_data['type']
            fvg_top = fvg_data['top']
            fvg_bottom = fvg_data['bottom']
            
            # 5. Determine Entry, SL, TP
            # Entry: Limit at FVG "open" (Top for Bull, Bottom for Bear) or Market?
            # Silver Bullet often enters 'at market' upon close of FVG candle or limit at FVG start.
            # We'll place a LIMIT order at the FVG overlap to get better price?
            # Or Market aggressively? 
            # Given we are bar-by-bar, we can assume we enter on the CLOSE of this bar (Market).
            # But the 'fvg_top'/'bottom' defines the gap.
            # Bullish FVG: Gap is between High[i-2] and Low[i]. Price is currently at Close[i].
            # Close[i] is typically ABOVE the gap if it's a valid gap? 
            # No, Bullish FVG: Low[i] > High[i-2]. The gap is "below" the current price. 
            # So looking for a retrace into the gap.
            
            # Bullish: Place LIMIT buy at fvg_top (Top of the gap).
            # Bearish: Place LIMIT sell at fvg_bottom (Bottom of the gap).
            
            atr = self.atr.iloc[idx]
            if pd.isna(atr): atr = 5.0

            if fvg_type == 'bull':
                entry_price = fvg_top
                sl_price = fvg_bottom - (TICK_SIZE * 4) # Below the gap + buffer
                
                # Target: Recent Swing High or Fixed 2R
                # Find recent swing high
                recent_highs = self.swings_high.iloc[:idx].dropna() # Very slow?
                # Optimization: Look back last 120 bars
                slice_highs = self.swings_high.iloc[max(0, idx-120):idx].dropna()
                if not slice_highs.empty:
                    target = slice_highs.max() # Target highest liquidity
                else:
                    target = entry_price + (entry_price - sl_price) * 2 # 2R default
                
                # Minimum 2R
                if (target - entry_price) < (entry_price - sl_price) * 1.5:
                     target = entry_price + (entry_price - sl_price) * 2

                # Market Entry Logic for Silver Bullet
                # We want to catch the expansion. 
                # Waiting for a retrace to the "Top" of the FVG might be missing the move if momentum is strong.
                # However, ICT teaches 50% retrace or just "Fair Value".
                # Let's try MARKET entry to prove validity, then refine.
                
                self.backtester.place_order(
                    price=row['close'], # Current close
                    size=1, 
                    side='BUY',
                    order_type=OrderType.MARKET, # Aggressive
                    sl=sl_price,
                    tp=target
                )
                self.trades_in_session.add(session_id)
                print(f"[SilverBullet] {timestamp} MARKET BUY @ {row['close']}, SL: {sl_price:.2f}, TP: {target:.2f}")

            elif fvg_type == 'bear':
                entry_price = fvg_bottom
                sl_price = fvg_top + (TICK_SIZE * 4)
                
                slice_lows = self.swings_low.iloc[max(0, idx-120):idx].dropna()
                if not slice_lows.empty:
                    target = slice_lows.min()
                else:
                    target = entry_price - (sl_price - entry_price) * 2
                
                if (entry_price - target) < (sl_price - entry_price) * 1.5:
                    target = entry_price - (sl_price - entry_price) * 2

                self.backtester.place_order(
                    price=row['close'],
                    size=1, 
                    side='SELL',
                    order_type=OrderType.MARKET,
                    sl=sl_price,
                    tp=target
                )
                self.trades_in_session.add(session_id)
                print(f"[SilverBullet] {timestamp} MARKET SELL @ {row['close']}, SL: {sl_price:.2f}, TP: {target:.2f}")
