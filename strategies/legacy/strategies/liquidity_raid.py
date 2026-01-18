from .base_strategy import Strategy
from strategies.legacy.ict_library import calculate_swings, calculate_atr
from strategies.legacy.config import is_in_killzone
from strategies.legacy.backtester import OrderType
import pandas as pd
import numpy as np

class LiquidityRaidStrategy(Strategy):
    """
    ICT Liquidity Raid (Turtle Soup):
    Aggressive reversal strategy targeting fakeouts of key levels.
    
    FIXED VERSION:
    - Signal at bar i, ENTRY at bar i+1 open
    - SL/TP calculated using bar i+1 prices (execution bar)
    - Stale signals are cleared if not executed immediately
    """
    
    def __init__(self, backtester):
        super().__init__(backtester)
        self.swing_len = 20
        self.risk_pct = 0.01
        
        # Pending signal: just direction, SL/TP calculated on execution
        self.pending_side = None  # 'BUY' or 'SELL'
        self.signal_bar_idx = None  # To detect stale signals

    def on_start(self, df):
        print("Calculating Indicators for Liquidity Raid...")
        self.swing_highs, self.swing_lows = calculate_swings(df, left=self.swing_len, right=self.swing_len)
        self.atr = calculate_atr(df)
        print(f"DEBUG: Found {self.swing_highs.count()} Major Swing Highs")

    def on_bar(self, i, bar):
        current_time = bar.name
        
        # 1. Execute pending signal from PREVIOUS bar (must be exactly 1 bar ago)
        if self.pending_side and self.signal_bar_idx == i - 1:
            if self.backtester.position_size == 0:
                # Calculate SL/TP based on THIS bar (execution bar)
                atr_val = self.atr.iloc[i] if not pd.isna(self.atr.iloc[i]) else 5.0
                
                if self.pending_side == 'SELL':
                    # Entry at this bar's open (via Market order)
                    entry_price = bar['open']
                    stop_loss = bar['high'] + atr_val * 0.5  # Above this bar's high
                    tp = entry_price - (stop_loss - entry_price) * 3
                else:  # BUY
                    entry_price = bar['open']
                    stop_loss = bar['low'] - atr_val * 0.5  # Below this bar's low
                    tp = entry_price + (entry_price - stop_loss) * 3
                
                self.backtester.place_order(
                    price=entry_price,
                    size=1,
                    side=self.pending_side,
                    order_type=OrderType.MARKET,
                    sl=stop_loss,
                    tp=tp
                )
            
            # Clear signal regardless of execution
            self.pending_side = None
            self.signal_bar_idx = None
            return
        
        # Clear any stale signal (older than 1 bar)
        if self.pending_side and self.signal_bar_idx != i - 1:
            self.pending_side = None
            self.signal_bar_idx = None
        
        # 2. Killzone check
        if not is_in_killzone(current_time):
            return

        # 3. Position Guard
        if self.backtester.position_size != 0:
            return

        # 4. Signal Detection
        lookback_start = max(0, i - 500)
        
        # --- LIQUIDITY RAID SHORT ---
        recent_highs = self.swing_highs.iloc[lookback_start:i].dropna()
        if not recent_highs.empty:
            raided_levels = recent_highs[(recent_highs < bar['high']) & (recent_highs > bar['close'])]
            if not raided_levels.empty:
                self.pending_side = 'SELL'
                self.signal_bar_idx = i
                return
                
        # --- LIQUIDITY RAID LONG ---
        recent_lows = self.swing_lows.iloc[lookback_start:i].dropna()
        if not recent_lows.empty:
            raided_levels = recent_lows[(recent_lows > bar['low']) & (recent_lows < bar['close'])]
            if not raided_levels.empty:
                self.pending_side = 'BUY'
                self.signal_bar_idx = i
