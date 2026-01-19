import pandas as pd
import numpy as np
from ib_insync import *
import time
import asyncio
from datetime import datetime
import pytz
import os
from dashboard_logger import DashboardLogger
from live.execution_bridge import TradersPostBroker
from live.pool_fsm import TradeDirection # Needed for bridge compatibility

# Patch asyncio to allow nested event loops (fixes Jupyter/Windows issues)
util.patchAsyncio()

# -------------------------------------------------------------------------
# STRATEGY CONFIGURATION (The "Golden Half-Hour")
# -------------------------------------------------------------------------
SYMBOL = 'MNQ'
EXCHANGE = 'CME'
CURRENCY = 'USD'
CONTRACT_MONTH = '202603' # Correct Front Month for 2026 

# SAFETY LOCK
DATA_ONLY_MODE = False 

# Risk
SL_POINTS = 15
TP_POINTS = 40 
CONTRACTS = 1

# Triggers
LONG_BODY_MIN_TICKS = 5.0
LONG_DIST_BIPS = 7.2
SHORT_BODY_MIN_TICKS = 6.8
SHORT_DIST_BIPS = 10.0

# Sniper Filters (Hyper-Dive)
MAX_WICK_RATIO = 0.25
MIN_REL_VOL = 1.5

# Session (NY Time)
START_HOUR = 9
START_MINUTE = 30
END_HOUR = 15
END_MINUTE = 55

# Safety Limit (Matched to Research)
MAX_DAILY_TRADES = 2

class GoldenBot:
    def __init__(self):
        self.ib = IB()
        self.df = pd.DataFrame()
        self.contract = None
        self.in_position = False
        self.csv_file = 'live_dashboard.csv'
        self.gs_logger = DashboardLogger() # Google Sheets Logger
        self.latest_metrics = {
            "vol": 0, "wick": 0, "body": 0,
            "dh": 0, "dl": 0, "dist_h": 0, "dist_l": 0
        }
        self.broker = TradersPostBroker() # Execution Bridge
        
        # Consistent Heartbeat State
        self.last_known_price = None
        self.last_hb_time = 0
        
        # Fidelity Tracking
        self.ibkr_tick_count = 0
        
        # Trade Limiter
        self.daily_trade_count = 0
        self.last_trade_day = None
        
        # Stateful Sweep Flags (Persist until reset)
        self.swept_ib_low = False
        self.swept_ib_high = False
        self.swept_asia_low = False
        self.swept_asia_high = False
        
        # Init CSV if not exists
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w') as f:
                f.write("timestamp,price,action,signal_type,rel_vol,wick_ratio,body,dh,dl,dist_h,dist_l\n")
        
    async def run(self):
        # Connect: Try Paper (7497) then Live (7496)
        print("Connecting to IBKR...")
        connected = False
        for port in [7497, 7496]:
            try:
                print(f"Trying Port {port}...")
                await self.ib.connectAsync('127.0.0.1', port, clientId=115)
                connected = True
                print(f"✅ Success on Port {port}")
                break
            except Exception as e:
                print(f"❌ Failed Port {port}: {e}")
                
        if not connected:
            print("CRITICAL: IBKR Connection Failed. TWS Open?")
            return

        # Define Contract
        self.contract = Future(SYMBOL, CONTRACT_MONTH, exchange=EXCHANGE, currency=CURRENCY)
        await self.ib.qualifyContractsAsync(self.contract)
        print(f"Tracking: {self.contract}")

        # Subscribe to 1 min bars (Real-time updates)
        self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr='2 D',
            barSizeSetting='1 min', whatToShow='TRADES', useRTH=False,
            keepUpToDate=True
        )
        
        self.ib.barUpdateEvent += self.on_bar
        
        # Subscribe to Streaming Ticker (1Hz+ frequency)
        self.ib.reqMktData(self.contract, '', False, False)
        self.ib.pendingTickersEvent += self.on_ticker
        
        # Start Heartbeat Task
        asyncio.create_task(self.heartbeat_loop())
        # Start Stats Export Task
        asyncio.create_task(self.stats_loop())
        
        print(f"✅ Bridge Running. Feeding: strategies/mle/data/live_ticks.csv")
        
        # Keep alive
        while self.ib.isConnected():
            await asyncio.sleep(1)

    def on_bar(self, bars, has_new_bar):
        """Called by IBKR on every bar update (real-time every 5s)."""
        # 1. Update Internal DataFrame
        self.df = util.df(bars)
        if len(self.df) < 2: return 
        
        last_bar = self.df.iloc[-1]
        
        # 2. ALWAYS Log Raw Tick (Feeds Engine/Dashboard every 5s)
        self.log_tick_only(last_bar.date, last_bar.close)

        # 3. Finalized Bar Processing
        # [DE-DUPLICATION]: Log only raw ticks here. 
        # The Live Engine is the sole authority for 1-minute bars and logic.
        if has_new_bar:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] IBKR Bar Closed: {last_bar.close}")
            
            # --- CRITICAL: EXECUTE STRATEGY ---
            try:
                action, details, strat_id = self.check_signals(self.df)
                if action:
                    self.execute_trade(action, last_bar.close, strat_id)
            except Exception as e:
                print(f"Signal Check Error: {e}")
            # ----------------------------------

    async def heartbeat_loop(self):
        """Ensures a gapless 1Hz data stream even when market is quiet."""
        while self.ib.isConnected():
            await asyncio.sleep(1)
            if self.last_known_price:
                # Log heartbeat if we haven't seen an update in >0.9s
                now = time.time()
                if now - self.last_hb_time >= 0.95:
                    self.ibkr_tick_count += 1
                    self.log_tick_only(datetime.now(), self.last_known_price)
                    # Note: log_tick_only handles GS as well if desired, 
                    # but we usually only want GS to see true updates
                    # Actually, user wants perfect sheet consistency, so we log to GS too.
                    if self.gs_logger.enabled:
                        self.gs_logger.log_tick(datetime.now(), self.last_known_price, 1)

    def on_ticker(self, tickers):
        """Called by IBKR on every price update (fast)."""
        for t in tickers:
            if t.contract == self.contract and t.last:
                self.ibkr_tick_count += 1
                self.last_known_price = t.last
                self.last_hb_time = time.time()
                # MAXIMUM DATA: Log every single update instantly
                self.log_tick_only(t.time or datetime.now(), t.last)
                if self.gs_logger.enabled:
                    self.gs_logger.log_tick(t.time or datetime.now(), t.last, 1)

    def log_tick_only(self, ts, price):
        """Minimal logging for raw tick stream."""
        try:
            tick_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_path = "strategies/mle/data/live_ticks.csv"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(f"{tick_str},{price},1\n")
        except: pass

    async def stats_loop(self):
        """Exports bridge performance metrics for the dashboard."""
        while self.ib.isConnected():
            await asyncio.sleep(1)
            try:
                import json
                stats = {
                    "ibkr_ticks": self.ibkr_tick_count,
                    "gs_ticks": self.gs_logger.gs_tick_count,
                    "gs_rpm": self.gs_logger.get_requests_per_min()
                }
                with open("bridge_stats.json", "w") as f:
                    json.dump(stats, f)
            except: pass

    def log_event(self, ts, price, action, details, vol, wick, body, dh, dl, disth, distl):
        # 1. Dashboard Log (Legacy Rich Data)
        try:
            with open(self.csv_file, 'a') as f:
                f.write(f"{ts},{price},{action},{details},{vol:.2f},{wick:.2f},{body:.1f},{dh:.2f},{dl:.2f},{disth:.1f},{distl:.1f}\n")
        except Exception as e:
            print(f"Bridge Write Error: {e}")

        # 3. Google Sheet Log
        if self.gs_logger.enabled:
            # Timezone Handling
            if isinstance(ts, datetime):
                # Ensure it has timezone info if not present
                if ts.tzinfo is None:
                    ny_tz = pytz.timezone('America/New_York')
                    ts = ny_tz.localize(ts)
                else:
                    ts = ts.astimezone(pytz.timezone('America/New_York'))
                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            else:
                ts_str = str(ts)
            
            # NaN Safety for Sheets (JSON doesn't support NaN/Inf)
            def sf(v):
                try:
                    val = float(v)
                    return 0.0 if np.isnan(val) or np.isinf(val) else val
                except: return 0.0

            self.gs_logger.log_min_data(
                ts_str, sf(price), action, details, 
                sf(vol), sf(wick), sf(body), sf(dh), sf(dl), sf(disth), sf(distl)
            )

            self.gs_logger.log_min_data(
                ts_str, sf(price), action, details, 
                sf(vol), sf(wick), sf(body), sf(dh), sf(dl), sf(disth), sf(distl)
            )

    def engineer_live_features(self, df):
        """Calculates IB and ASIA levels + FVG for the active session."""
        try:
            # Ensure NY Time
            df['ny_time'] = df['date'].dt.tz_convert('America/New_York')
            df['hour'] = df['ny_time'].dt.hour
            df['minute'] = df['ny_time'].dt.minute
            df['date_only'] = df['ny_time'].dt.date
            
            # 1. IB Levels (09:30 - 10:00 Today)
            today = df['date_only'].iloc[-1]
            ib_mask = (df['date_only'] == today) & (
                ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] == 10) & (df['minute'] == 0))
            )
            ib_bars = df[ib_mask]
            
            df['IB_H'] = ib_bars['high'].max() if not ib_bars.empty else 0
            df['IB_L'] = ib_bars['low'].min() if not ib_bars.empty else 0
            
            # 2. ASIA Levels (18:00 Prev - 00:00 Today)
            # Simplistic: Just look at 18-00 in the last 24h window
            # Harder in live stream. We can just take lookback?
            # Let's use strict time:
            # Asia is 18:00 (D-1) -> 00:00 (D)
            # Find the most recent completed Asia session
            
            # If current time > 00:00 today, Asia was yesterday 18:00 to today 00:00
            # Since we have 2 days data, we can filter for that window
            
            # Construct Asia Start/End logic?
            # Shortcut: Use fixed hour checks on the full dataframe
            asia_mask = (df['hour'] >= 18) | (df['hour'] < 0) # 0 is midnight?
            # This is tricky with rolling.
            # Alternative: Use simple rolling min/max of ~6 hours if in morning?
            # Standard: 18:00 - 00:00.
            
            # Let's isolate the 'Last Asia Session' based on current time
            # If now is 10:00 AM, Asia ended 10 hours ago via midnight.
            
            # Filter for last 24h
            last_24h = df.iloc[-1440:] 
            asia_session = last_24h[ (last_24h['hour'] >= 18) | (last_24h['hour'] < 0) ] # <0? No, 0-23.
            # Asia: 18, 19, 20, 21, 22, 23. (Start 18:00, End 00:00 aka 23:59)
            
            asia_bars = last_24h[ (last_24h['hour'] >= 18) ]
            # Note: This ignores the 00:00 bar if it exists? Usually Asia end is New Day Open.
            
            if not asia_bars.empty:
                df['ASIA_H'] = asia_bars['high'].max()
                df['ASIA_L'] = asia_bars['low'].min()
            else:
                df['ASIA_H'] = 0
                df['ASIA_L'] = 0
            
            # 3. FVG Logic
            # Bullish FVG: Low > High[n-2]
            prev_high = df['high'].shift(2)
            curr_low = df['low'] # Shift 0
            # Wait, live engine usually evaluates on CLOSED bars?
            # current_bar is incomplete?
            # check_signals uses df.iloc[-1]. If 'on_bar' is called, bar is closed.
            # So df.iloc[-1] is the JUST CLOSED bar.
            
            df['fvg_bull'] = (curr_low > prev_high)
            df['fvg_bull_top'] = curr_low
            df['fvg_bull_btm'] = prev_high
            
            prev_low = df['low'].shift(2)
            curr_high = df['high']
            df['fvg_bear'] = (curr_high < prev_low)
            df['fvg_bear_top'] = prev_low 
            df['fvg_bear_btm'] = curr_high
            
            # 4. TREND FILTER (SMA 200)
            df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            return df
        except Exception as e:
            print(f"Feature Eng Error: {e}")
            return df

    def check_signals(self, df):
        # 1. Engineer Features
        df = self.engineer_live_features(df)
        
        # 2. Get Current State (Just Closed Bar)
        row = df.iloc[-1]
        
        # 3. Time Filter (US Session: 10:00 - 16:00)
        # We allow 10:00 to 15:30?
        # User wants "Intraday". Let's stick to 09:30 - 16:00 generally, 
        # but strategy specific?
        # IB Strategy starts AFTER IB (10:00).
        # Asia Strategy can trade anytime NY session?
        # Let's use 10:00 - 15:55.
        
        if current_hour < 10 or current_hour >= 16:
            return None, "Outside Session (10-16)", None
            
        # Trade Limiter & State Reset
        today_date = row['date_only']
        if self.last_trade_day != today_date:
            self.daily_trade_count = 0 # Reset for new day
            self.last_trade_day = today_date
            # Reset Sweep Flags for new day
            self.swept_ib_low = False
            self.swept_ib_high = False
            self.swept_asia_low = False
            self.swept_asia_high = False
            
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return None, f"Max Trades Reached ({self.daily_trade_count}/{MAX_DAILY_TRADES})", None

            
        # 4. Strategy Evaluation
        signal = None
        strategy_id = None
        
        # --- STRATEGY A : IB HYBRID ---
        # Trigger: Sweep IB + FVG (Stateful)
        ib_signal = False
        ib_dir = 0
        
        if row['IB_H'] > 0 and row['IB_L'] > 0:
            # Update State
            if row['low'] < row['IB_L']: self.swept_ib_low = True
            if row['high'] > row['IB_H']: self.swept_ib_high = True
            
            # Check Trigger (Memory + FVG + Trend)
            # Trend Rules: Long > SMA, Short < SMA
            trend_ok_long = row['close'] > row['SMA_200'] if pd.notna(row['SMA_200']) else True
            trend_ok_short = row['close'] < row['SMA_200'] if pd.notna(row['SMA_200']) else True
            
            if self.swept_ib_low and row['fvg_bull'] and trend_ok_long:
                ib_signal = True
                ib_dir = 1 # BUY
            elif self.swept_ib_high and row['fvg_bear'] and trend_ok_short:
                ib_signal = True
                ib_dir = -1 # SELL
                
        # --- STRATEGY B : ASIA HYBRID ---
        asia_signal = False
        asia_dir = 0
        
        if row['ASIA_H'] > 0 and row['ASIA_L'] > 0:
            # Update State
            if row['low'] < row['ASIA_L']: self.swept_asia_low = True
            if row['high'] > row['ASIA_H']: self.swept_asia_high = True
            
            # Trend Check (Recalculate or reuse)
            trend_ok_long = row['close'] > row['SMA_200'] if pd.notna(row['SMA_200']) else True
            trend_ok_short = row['close'] < row['SMA_200'] if pd.notna(row['SMA_200']) else True

            # Check Trigger
            if self.swept_asia_low and row['fvg_bull'] and trend_ok_long:
                asia_signal = True
                asia_dir = 1
            elif self.swept_asia_high and row['fvg_bear'] and trend_ok_short:
                asia_signal = True
                asia_dir = -1
                
        # 5. Conflict Resolution (ASIA Priority)
        final_action = None
        final_strat = None
        
        if asia_signal and ib_signal:
            # OVERLAP -> Pick ASIA
            final_strat = "ASIA_Hybrid"
            final_action = "BUY" if asia_dir == 1 else "SELL"
        elif asia_signal:
            final_strat = "ASIA_Hybrid"
            final_action = "BUY" if asia_dir == 1 else "SELL"
        elif ib_signal:
            final_strat = "IB_Hybrid"
            final_action = "BUY" if ib_dir == 1 else "SELL"
            
        details = f"IB={row['IB_H']:.1f}/{row['IB_L']:.1f}, ASIA={row['ASIA_H']:.1f}/{row['ASIA_L']:.1f}"
        
        # Return Tuple
        return final_action, details, final_strat

    def execute_trade(self, action, price, strategy_id="GOLDEN_DEFAULT"):
        if DATA_ONLY_MODE:
            print(f"🛑 DATA ONLY MODE: Trade {action} @ {price} BLOCKED.")
            return

        print(f"🚀 LIVE EXECUTION: {action} @ {price} | Strat: {strategy_id}")
        
        # CONFIG MAP (Portfolio)
        # IB: TP 55, SL 5
        # ASIA: TP 85, SL 5
        
        if strategy_id == "ASIA_Hybrid":
            param_tp = 85
            param_sl = 5
        elif strategy_id == "IB_Hybrid":
            param_tp = 55
            param_sl = 5
        else:
            # Fallback
            param_tp = 40
            param_sl = 15
        
        # Calculate SL/TP
        if action == "BUY":
            sl = price - param_sl
            tp = price + param_tp
            direction = TradeDirection.LONG
        else:
            sl = price + param_sl
            tp = price - param_tp
            direction = TradeDirection.SHORT
            
        # Send to TradersPost
        res = self.broker.execute_order(strategy_id, direction, CONTRACTS, sl, tp)
        
        # Increment Daily Count on Success
        if res == "TP-OK":
            self.daily_trade_count += 1
            print(f"✅ Trade Count: {self.daily_trade_count}/{MAX_DAILY_TRADES}")
        
        # Log to GS
        if res == "TP-OK":
            m = self.latest_metrics
            self.gs_logger.log_trade(
                pool_id=strategy_id, 
                direction=action, 
                entry=price, 
                sl=sl, 
                tp=tp, 
                status="OPEN",
                wick=m['wick'],
                vol=m['vol'],
                body=m['body']
            )

if __name__ == '__main__':
    bot = GoldenBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bridge stopping...")
        if bot.gs_logger.enabled:
            bot.gs_logger.flush_ticks()
