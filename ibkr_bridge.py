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
END_HOUR = 10
END_MINUTE = 00

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
            self.contract, endDateTime='', durationStr='1 D',
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
            # The engine handles rich logging to Google Sheets to avoid 1Hz conflict
            pass

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

    def check_signals(self, df):
        # 1. Check Time
        ny_tz = pytz.timezone('America/New_York')
        now_ny = datetime.now(ny_tz)
        
        # USER REQUEST: Full Day Tracking
        # Valid Time is basically always True for Data-Only/Sim mode
        # But we still want to label it properly?
        # User said: "I want full day tracking"
        valid_time = True 
        
        # ... logic ...
        
        # LOGIC RE-CALC (for signal strictness)
        current_bar = df.iloc[-1]
        close = current_bar['close']
        open_p = current_bar['open']
        high = current_bar['high']
        low = current_bar['low']
        
        is_bull = close > open_p
        is_bear = close < open_p
        
        lookback = 20
        rolling_high = df['high'].rolling(lookback).max().iloc[-1]
        rolling_low = df['low'].rolling(lookback).min().iloc[-1]
        dist_to_high_bips = (abs(rolling_high - close) / close) * 10000.0
        dist_to_low_bips = (abs(rolling_low - close) / close) * 10000.0
        
        tick_size = 0.25 
        body_ticks = abs(close - open_p) / tick_size
        
        # Filters
        wick_ratio = 999
        if is_bull and abs(close - open_p) > 0:
            wick_upper = high - close
            wick_ratio = wick_upper / abs(close - open_p)
        elif is_bear and abs(close - open_p) > 0:
            wick_lower = close - low
            wick_ratio = wick_lower / abs(close - open_p)
            
        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        rel_vol = current_bar['volume'] / avg_vol if avg_vol > 0 else 0
        
        # Logic
        long_base = is_bull and (body_ticks >= LONG_BODY_MIN_TICKS) and (dist_to_high_bips <= LONG_DIST_BIPS)
        short_base = is_bear and (body_ticks >= SHORT_BODY_MIN_TICKS) and (dist_to_low_bips <= SHORT_DIST_BIPS)
        
        sniper_filter = (wick_ratio <= MAX_WICK_RATIO) and (rel_vol >= MIN_REL_VOL)
        
        # Store for execute_trade
        self.latest_metrics = {
            "vol": rel_vol, "wick": wick_ratio, "body": body_ticks,
            "dh": rolling_high, "dl": rolling_low,
            "dist_h": dist_to_high_bips, "dist_l": dist_to_low_bips
        }
        
        details = f"Body={body_ticks:.1f},DH={dist_to_high_bips:.1f},DL={dist_to_low_bips:.1f}"
        
        signal = None
        if valid_time:
            if long_base and sniper_filter:
                signal = "BUY"
            elif short_base and sniper_filter:
                signal = "SELL"
            
        return signal, details

    def execute_trade(self, action, price):
        if DATA_ONLY_MODE:
            print(f"🛑 DATA ONLY MODE: Trade {action} @ {price} BLOCKED.")
            return

        print(f"🚀 LIVE EXECUTION: {action} @ {price}")
        
        # Calculate SL/TP
        if action == "BUY":
            sl = price - SL_POINTS
            tp = price + TP_POINTS
            direction = TradeDirection.LONG
        else:
            sl = price + SL_POINTS
            tp = price - TP_POINTS
            direction = TradeDirection.SHORT
            
        # Send to TradersPost
        res = self.broker.execute_order("GOLDEN_HALF_HOUR", direction, CONTRACTS, sl, tp)
        
        # Log to GS
        if res == "TP-OK":
            m = self.latest_metrics
            self.gs_logger.log_trade(
                pool_id="GOLDEN_HALF_HOUR", 
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
