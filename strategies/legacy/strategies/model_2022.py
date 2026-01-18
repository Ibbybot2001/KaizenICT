from .base_strategy import Strategy
from strategies.legacy.ict_library import calculate_swings, detect_fair_value_gaps, calculate_atr
from strategies.legacy.config import is_in_killzone
import pandas as pd
import numpy as np

class Model2022(Strategy):
    def __init__(self, backtester):
        super().__init__(backtester)
        # Parameters
        self.swing_len = 3
        self.risk_pct = 0.01
        self.mss_window = 20 # Bars to wait for MSS after sweep
        self.entry_window = 60 # Bars to wait for FVG entry after MSS
        
        # State
        self.reset_state()

    def reset_state(self):
        self.sweep_time = None
        self.sweep_type = None
        self.sweep_price = None
        self.mss_time = None
        self.mss_price = None
        self.active_setup = False 

    def on_start(self, df):
        print("Calculating Indicators for Model 2022 (Authentic)...")
        self.swing_highs, self.swing_lows = calculate_swings(df, left=self.swing_len, right=self.swing_len)
        print(f"DEBUG: Found {self.swing_highs.count()} Swing Highs and {self.swing_lows.count()} Swing Lows")
        self.fvgs = detect_fair_value_gaps(df)
        print(f"DEBUG: FVG Columns: {self.fvgs.columns}")
        # print(self.fvgs.head())
        self.atr = calculate_atr(df)
        
    def on_bar(self, i, bar):
        current_time = bar.name
        
        # 1. Killzone Filter (Strict)
        # We only LOOK for setups in Killzone.
        # But if a setup started inside, we can execute slightly outside?
        # Strict: Setup (Sweep) must happen in Killzone.
        if not is_in_killzone(current_time) and not self.active_setup:
            return

        # 1.5 Position Guard 
        # CRITICAL: Do not hunt for new setups if we are already in a trade.
        if self.backtester.position_size != 0:
            self.reset_state() # Clear any pending setup state if we are managing a trade
            return

        # 2. State Machine
        
        # PHASE A: Detect Sweep (if not already active)
        if not self.active_setup:
            # --- LONG SETUP SEARCH ---
            recent_lows = self.swing_lows.iloc[i-100:i].dropna()
            if not recent_lows.empty:
                target_liq_low = recent_lows.iloc[-1]
                if bar['low'] < target_liq_low:
                     self.active_setup = True
                     self.sweep_time = current_time
                     self.sweep_type = 'long'
                     self.sweep_price = bar['low']
                     return

            # --- SHORT SETUP SEARCH ---
            recent_highs = self.swing_highs.iloc[i-100:i].dropna()
            if not recent_highs.empty:
                target_liq_high = recent_highs.iloc[-1]
                if bar['high'] > target_liq_high:
                     self.active_setup = True
                     self.sweep_time = current_time
                     self.sweep_type = 'short'
                     self.sweep_price = bar['high']
                     return

        # PHASE B: Wait for MSS (Market Structure Shift)
        if self.active_setup and not self.mss_time:
            # Timeout
            if (current_time - self.sweep_time).seconds/60 > self.mss_window:
                self.reset_state()
                return

            if self.sweep_type == 'long':
                # MSS Long: Close > Recent Swing High
                recent_highs = self.swing_highs.iloc[i-50:i].dropna() 
                if not recent_highs.empty:
                    last_swing_high = recent_highs.iloc[-1]
                    if bar['close'] > last_swing_high:
                        self.mss_time = current_time
                        self.mss_price = last_swing_high
                        # print(f"[{current_time}] MSS LONG CONFIRMED")
            
            elif self.sweep_type == 'short':
                # MSS Short: Close < Recent Swing Low
                recent_lows = self.swing_lows.iloc[i-50:i].dropna()
                if not recent_lows.empty:
                    last_swing_low = recent_lows.iloc[-1]
                    if bar['close'] < last_swing_low:
                        self.mss_time = current_time
                        self.mss_price = last_swing_low
                        # print(f"[{current_time}] MSS SHORT CONFIRMED")
            return

        # PHASE C: Entry on FVG
        if self.mss_time:
            # Timeout
            if (current_time - self.mss_time).seconds/60 > self.entry_window:
                self.reset_state()
                return

            # Check FVGs exists at current time
            if current_time in self.fvgs.index:
                fvg = self.fvgs.loc[current_time]
                
                fvg_type = None
                if isinstance(fvg, pd.DataFrame): fvg = fvg.iloc[0]
                try: fvg_type = fvg['type']
                except: return

                # --- LONG ENTRY ---
                if self.sweep_type == 'long' and fvg_type == 'bull':
                    entry_price = fvg['top']
                    stop_loss = self.sweep_price 
                    
                    # Target: Recent High
                    recent_highs = self.swing_highs.iloc[i-100:i].dropna()
                    if not recent_highs.empty: tp = recent_highs.max()
                    else: tp = entry_price + (entry_price - stop_loss) * 2.5 
                    
                    self.backtester.cancel_all_orders()
                    self.backtester.place_limit_order('BUY', entry_price, sl=stop_loss, tp=tp)
                    self.reset_state()
                    
                # --- SHORT ENTRY ---
                elif self.sweep_type == 'short' and fvg_type == 'bear':
                    entry_price = fvg['bottom'] # Entry at bottom of Bearish FVG
                    stop_loss = self.sweep_price # Stop above sweep high
                    
                    # Target: Recent Low
                    recent_lows = self.swing_lows.iloc[i-100:i].dropna()
                    if not recent_lows.empty: tp = recent_lows.min()
                    else: tp = entry_price - (stop_loss - entry_price) * 2.5
                    
                    self.backtester.cancel_all_orders()
                    self.backtester.place_limit_order('SELL', entry_price, sl=stop_loss, tp=tp)
                    self.reset_state()
