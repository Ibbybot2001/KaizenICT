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
from strategies.mle.strategy_loader import get_strategy_config # Dynamic Loader

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
TRUST_SIGNALS_MODE = True # User uses TradersPost; Local IBKR API may not see the trade. Trust Logic. 

# Risk
SL_POINTS = 5       # Sniper Precision (BH Closed)
TP_POINTS = 40      # Target (1:8 R:R)
CONTRACTS = 2

# Triggers
LONG_BODY_MIN_TICKS = 5.0
LONG_DIST_BIPS = 7.2
SHORT_BODY_MIN_TICKS = 6.8
SHORT_DIST_BIPS = 10.0

# Sniper Filters (Hyper-Dive)
# TODO: These should come from strategy_loader for discovered params parity
# Currently hardcoded - live trading may not match backtest filters exactly
MAX_WICK_RATIO = 0.25
MIN_REL_VOL = 1.5

# Session (NY Time: 24/7 Pivot)
START_HOUR = 0
START_MINUTE = 0
END_HOUR = 23
END_MINUTE = 59

# Safety Limit (Matched to Research)
MAX_DAILY_TRADES = 99 # Uncapped for Holy Grail Strategy

class GoldenBot:
    def __init__(self):
        self.ib = IB()
        self.df = pd.DataFrame()
        self.contract = None
        self.in_position = False
        self.csv_file = 'live_dashboard.csv'
        self.audit_file = 'trades_audit.csv'
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
        self.traded_signals = set() # Signature guard for entries
        self.logged_minutes = set() # Signature guard for telemetry logs
        
        # Stateful Sweep Flags (Persist until reset)
        self.swept_ib_low = False
        self.swept_ib_high = False
        self.swept_asia_low = False
        self.swept_asia_high = False
        
        # Performance Tracking
        self.current_trade = None # Stores {entry, direction, strat, sl, tp, gs_row}
        self.last_trade_time = 0 # Safety timer for race conditions
        self.empty_pos_count = 0 # Persistence counter
        
        # Init CSV if not exists
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w') as f:
                f.write("timestamp,price,action,signal_type,rel_vol,wick_ratio,body,dh,dl,dist_h,dist_l\n")
        if not os.path.exists(self.audit_file):
            with open(self.audit_file, 'w') as f:
                f.write("timestamp,strat,direction,entry,sl,tp,exit,pnl_pts,status\n")
        
    async def run(self):
        # Connect: Try Paper (7497) then Live (7496)
        print("Connecting to IBKR...")
        connected = False
        import random
        client_id = random.randint(100, 999)
        for port in [7497, 7496]:
            try:
                print(f"Trying Port {port} (Client ID: {client_id})...")
                await self.ib.connectAsync('127.0.0.1', port, clientId=client_id)
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

        # Subscribe to 1 min bars (Deep Memory Pivot: 1 Week)
        self.ib.reqHistoricalData(
            self.contract, endDateTime='', durationStr='1 W',
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
        # Start Position Monitor Task
        asyncio.create_task(self.monitor_positions_loop())
        
        print(f"✅ Bridge Running. Feeding: strategies/mle/data/live_ticks.csv")

        # INITIAL AUDIT: Check if we are starting with a position
        try:
            print(f"🔒 ACCOUNTS VISIBLE: {self.ib.managedAccounts()}")
            print("🔍 STARTUP SCAN: Forcing Position Audit...")
            self.ib.reqPositions() # Force refresh
            await asyncio.sleep(2) # Allow sync
            
            # Retry loop to catch slow API sync
            for i in range(3):
                positions = self.ib.positions()
                print(f"[DEBUG] Startup Scan {i+1}/3: Raw Positions: {positions}")
                if positions: break
                await asyncio.sleep(1)

            for p in positions:
                if p.contract.symbol == SYMBOL or p.contract.localSymbol.startswith(SYMBOL):
                    if p.position != 0:
                        self.in_position = True
                        self.current_trade = {
                            "strat": "RECOVERED",
                            "direction": "BUY" if p.position > 0 else "SELL",
                            "entry": p.avgCost,
                            "sl": 0, "tp": 0, "gs_row": 0
                        }
                        print(f"🛡️ GUARDIAN: Internalized ACTIVE POSITION {p.position} contract(s) on startup.")
        except Exception as e:
            print(f"Audit Error: {e}")
        
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
        if has_new_bar:
            # --- ENGINE THOUGHT STREAM (The Narrative) ---
            try:
                # Engineering features for the narrative
                self.df = self.engineer_live_features(self.df)
                row = self.df.iloc[-1]
                
                # Check Filters First
                time_ok = True # 24/7 ENGAGEMENT ENABLED
                trades_left = MAX_DAILY_TRADES - self.daily_trade_count
                
                # Build the "Thought"
                thoughts = []
                if not time_ok: # Legacy catch
                    thoughts.append(f"⏱️ SLEEPING.")
                elif trades_left <= 0:
                    thoughts.append(f"🔒 LOCKED: Daily trade limit reached ({MAX_DAILY_TRADES}/{MAX_DAILY_TRADES}).")
                elif self.in_position or self.current_trade:
                    thoughts.append(f"🛡️ GUARDIAN: Position Active. Watching for Target/Stop.")
                else:
                    # Capture State
                    high_dist = row['close'] - row['IB_H']
                    low_dist = row['close'] - row['IB_L']
                    trend_score = "BULLISH" if row['close'] > row['SMA_200'] else "BEARISH"
                    
                    if not self.swept_ib_high and not self.swept_ib_low:
                        thoughts.append(f"🎯 HUNTING: Price is INSIDE IB Range ({row['IB_H']:.1f} - {row['IB_L']:.1f}). Waiting for a Liquidity Sweep.")
                    
                    if self.swept_ib_high:
                        thoughts.append(f"⚠️ DETECTED: IB High ({row['IB_H']:.1f}) has been SWEPT.")
                        if trend_score == "BULLISH":
                            thoughts.append(f"🛡️ FILTER: Trend is {trend_score} (SMA 200). Refusing Shorts for safety.")
                        else:
                            thoughts.append(f"🏹 SIGNAL: Scanning for Bearish FVG to confirm Short entry.")
                            
                    if self.swept_ib_low:
                        thoughts.append(f"⚠️ DETECTED: IB Low ({row['IB_L']:.1f}) has been SWEPT.")
                        if trend_score == "BEARISH":
                            thoughts.append(f"🛡️ FILTER: Trend is {trend_score} (SMA 200). Refusing Longs for safety.")
                        else:
                            thoughts.append(f"🏹 SIGNAL: Scanning for Bullish FVG to confirm Long entry.")

                # Final Print Block
                ny_now = self.gs_logger.get_ny_time_str()
                print(f"\n🧠 [ENGINE THOUGHTS] {ny_now}")
                print(f"   ► {' | '.join(thoughts)}")
                print(f"   ► P:{row['close']:.1f} | SMA:{row['SMA_200']:.1f} | D:{trades_left} Left")
                
                # Weekly Pos Report
                w_h = row.get('WEEK_H', 0)
                w_l = row.get('WEEK_L', 0)
                w_range = w_h - w_l
                rel_pos = (row['close'] - w_l) / w_range if w_range > 0 else 0.5
                
                exhaustion = ""
                if rel_pos > 0.90: exhaustion = "⚠️ [EXHAUSTED HIGH - LONGS BLOCKED]"
                elif rel_pos < 0.10: exhaustion = "⚠️ [EXHAUSTED LOW - SHORTS BLOCKED]"
                
                print(f"   ► WEEKLY: Pos:{rel_pos*100:.1f}% | H:{w_h:.1f} | L:{w_l:.1f} {exhaustion}\n")

            except Exception as e:
                print(f"Thought Stream Error: {e}")

            # --- CRITICAL: EXECUTE STRATEGY ---
            try:
                action, details, strat_id, sig_ts = self.check_signals(self.df)
                
                # 4. Log Event (1-Min Summary for Sheets)
                # Signature Guard: Only log each NY timestamp ONCE
                ny_ts_str = row['ny_time'].strftime('%Y-%m-%d %H:%M')
                if ny_ts_str not in self.logged_minutes:
                    self.log_event(
                        ts=row['ny_time'], 
                        price=row['close'], 
                        action=action or "HUNTING", 
                        details=details or "Scanning...",
                        vol=row.get('rel_vol', 0),
                        wick=row.get('wick_ratio', 0),
                        body=row.get('body_ticks', 0),
                        dh=row.get('IB_H', 0),
                        dl=row.get('IB_L', 0),
                        disth=0, distl=0
                    )
                    self.logged_minutes.add(ny_ts_str)

                if action:
                    self.execute_trade(action, last_bar.close, strat_id, sig_ts)
            except Exception as e:
                print(f"Signal Check Error: {e}")
            # ----------------------------------

    async def monitor_positions_loop(self):
        """Syncs self.in_position and logs exits to GS."""
        while self.ib.isConnected():
            await asyncio.sleep(2) # Poll every 2s
            try:
                positions = self.ib.positions()
                found = False
                for p in positions:
                    # Robust Symbol Check: Handles MNQ, MNQH6, etc.
                    # IBKR Contract object's symbol is usually the base 'MNQ'
                    if p.contract.symbol == SYMBOL or p.contract.localSymbol.startswith(SYMBOL):
                        if p.position != 0:
                            found = True
                            break
                
                # REINFORCED GUARD: Persistence Logic
                if found:
                    self.empty_pos_count = 0
                    self.in_position = True
                if TRUST_SIGNALS_MODE:
                     # In Signal Trust Mode, we DO NOT use API feedback to close trades.
                     # We only close on Price Check (SL/TP) or internal logic.
                     pass 
                else:
                    self.empty_pos_count += 1
                    if self.in_position:
                         # print(f"[DEBUG] Drift {self.empty_pos_count}/30...")
                         pass
                    
                # Only process EXIT if we were in a position AND we have 30 consecutive empty polls
                # (60 seconds of persistent FLAT status)
                # AND we are past the 30s settlement window
                if self.in_position and not found and self.empty_pos_count >= 30:
                    if time.time() - self.last_trade_time < 30:
                        # Still in the "Fog of War" after entry - FORCE LOCK
                        pass 
                    else:
                        print(f"📉 EXIT DETECTED: Position confirmed closed after {self.empty_pos_count} polls.")
                        # Use the centralized exit logic
                        exit_price = self.last_known_price or 0
                        self.finalize_internal_exit(exit_price, "BROKER_CONFIRMED")
            except Exception as e:
                print(f"Position Monitor Error: {e}")

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
                    # DISABLED: GS tick logging to reduce API usage
                    # 1-min data and TradeLog still logged to GS
                    # if self.gs_logger.enabled:
                    #     self.gs_logger.log_tick(datetime.now(), self.last_known_price, 1)

    def on_ticker(self, tickers):
        """Called by IBKR on every price update (fast)."""
        for t in tickers:
            if t.contract == self.contract and t.last:
                self.ibkr_tick_count += 1
                self.last_known_price = t.last
                self.last_hb_time = time.time()
                # Local CSV tick logging (lightweight)
                self.log_tick_only(t.time or datetime.now(), t.last)
                # DISABLED: GS tick logging to reduce API usage
                # if self.gs_logger.enabled:
                #     self.gs_logger.log_tick(t.time or datetime.now(), t.last, 1)
                
                # REAL-TIME SL/TP PROBE
                if self.current_trade:
                    self.check_active_trade_sl_tp(t.last)

    def check_active_trade_sl_tp(self, current_price):
        """Proactively monitors ticks for SL/TP breaches."""
        trade = self.current_trade
        if not trade: return
        
        sl = trade['sl']
        tp = trade['tp']
        side = trade['direction']
        
        breached = False
        reason = ""
        
        if side == "BUY":
            if current_price <= sl:
                breached = True
                reason = "STOP_LOSS"
            elif current_price >= tp:
                breached = True
                reason = "TAKE_PROFIT"
        else: # SELL
            if current_price >= sl:
                breached = True
                reason = "STOP_LOSS"
            elif current_price <= tp:
                breached = True
                reason = "TAKE_PROFIT"
                
        if breached:
            # We don't close here (broker handles it usually), but we LOG the audit event
            if not trade.get('audit_logged'):
                print(f"⚠️ [AUDIT] Price touched {reason} level ({current_price:.1f}).")
                trade['audit_logged'] = True
                self.log_audit_event(trade, current_price, f"TOUCHED_{reason}")

            # NEW: TRUST_SIGNALS_MODE decouple
            if TRUST_SIGNALS_MODE:
                print(f"🔓 TRUST_SIGNALS: Releasing internal lock due to {reason}.")
                self.finalize_internal_exit(current_price, reason)
            else:
                 print("   ... Waiting for broker exit confirmation")

    def finalize_internal_exit(self, exit_price, reason="CLOSED"):
        """Forcefully closes the internal trade state and logs to GS."""
        trade_to_close = self.current_trade
        if not trade_to_close: return

        try:
            if trade_to_close.get('gs_row'):
                row_idx = trade_to_close['gs_row']
                strat_id = trade_to_close['strat']
                entry_px = trade_to_close['entry']
                side = trade_to_close['direction']
                
                pnl_pts = (exit_price - entry_px) if side == "BUY" else (entry_px - exit_price)
                
                if self.gs_logger.enabled:
                    self.gs_logger.update_trade_close(
                        row_index=row_idx,
                        exit_price=exit_price,
                        pnl_pts=pnl_pts
                    )
                
                # Log audit if not already logged as CLOSED
                self.log_audit_event(trade_to_close, exit_price, f"{reason}_FINAL")
                print(f"✅ GS Updated: {strat_id} CLOSED (Row {row_idx}). PnL Pts: {pnl_pts:.2f}")

        except Exception as e:
            print(f"Finalize Exit Error: {e}")

        # CRITICAL: Reset internal state
        self.current_trade = None
        self.in_position = False
        self.empty_pos_count = 0
        self.last_trade_time = time.time() # Reset cooldown timer

    def log_audit_event(self, trade, exit_px, status):
        """Logs trade lifecycle events to local CSV."""
        try:
            ts = self.gs_logger.get_ny_time_str()
            pnl = (exit_px - trade['entry']) if trade['direction'] == "BUY" else (trade['entry'] - exit_px)
            with open(self.audit_file, 'a') as f:
                f.write(f"{ts},{trade['strat']},{trade['direction']},{trade['entry']},{trade['sl']},{trade['tp']},{exit_px},{pnl:.2f},{status}\n")
        except: pass

    def log_tick_only(self, ts, price):
        """Minimal logging for raw tick stream."""
        try:
            # Use GS logger's NY time helper for consistency
            tick_str = self.gs_logger.get_ny_time_str(ts)
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
        # 1. Local CSV Dashboard Log
        try:
            ts_str = self.gs_logger.get_ny_time_str(ts)
            with open(self.csv_file, 'a') as f:
                # Local CSV usually likes simple strings
                f.write(f"{ts_str},{price},{action},{details},{vol:.2f},{wick:.2f},{body:.1f},{dh:.2f},{dl:.2f},{disth:.1f},{distl:.1f}\n")
        except Exception as e:
            print(f"Bridge Write Error: {e}")

        # 2. Google Sheet Log
        if self.gs_logger.enabled:
            # log_min_data now handles its own NY timezone and NaN safety
            self.gs_logger.log_min_data(
                ts, price, action, details, vol, wick, 
                body, dh, dl, disth, distl
            )

    def engineer_live_features(self, df):
        """Calculates IB and ASIA levels + FVG for the active session."""
        try:
            # OPTIMIZATION: Slice to last ~5.5 days (8000 mins) to capture WEEKLY LEVELS while getting speed
            # 1 Week = ~7000 trading minutes (24/5). 8000 is a safe buffer.
            if len(df) > 8000:
                df = df.iloc[-8000:].copy()
            
            # Ensure NY Time
            df['ny_time'] = df['date'].dt.tz_convert('America/New_York')
            df['hour'] = df['ny_time'].dt.hour
            df['minute'] = df['ny_time'].dt.minute
            df['date_only'] = df['ny_time'].dt.date
            
            # 1. IB Levels (Most recent 09:30 - 10:00 window)
            ib_mask = (
                ((df['hour'] == 9) & (df['minute'] >= 30)) | ((df['hour'] == 10) & (df['minute'] == 0))
            )
            ib_bars = df[ib_mask]
            
            # Use the most recent day's IB if we have multiple days
            if not ib_bars.empty:
                last_ib_date = ib_bars['date_only'].iloc[-1]
                latest_ib_bars = ib_bars[ib_bars['date_only'] == last_ib_date]
                df['IB_H'] = latest_ib_bars['high'].max()
                df['IB_L'] = latest_ib_bars['low'].min()
            else:
                df['IB_H'] = 0
                df['IB_L'] = 0
            
            # 2. ASIA Levels (Most recent 18:00 - 00:00 window)
            # Find bars from the most recent session starting at 18:00
            asia_mask = (df['hour'] >= 18)
            asia_bars = df[asia_mask]
            
            if not asia_bars.empty:
                last_asia_date = asia_bars['date_only'].iloc[-1]
                latest_asia_bars = asia_bars[asia_bars['date_only'] == last_asia_date]
                df['ASIA_H'] = latest_asia_bars['high'].max()
                df['ASIA_L'] = latest_asia_bars['low'].min()
            else:
                # Fallback: If no 18:00 bars today, look for any bars before 09:30 
                # (which would be the tail end of the overnight Asia session)
                pre_market = df[df['hour'] < 9]
                if not pre_market.empty:
                    df['ASIA_H'] = pre_market['high'].max()
                    df['ASIA_L'] = pre_market['low'].min()
                else:
                    df['ASIA_H'] = 0
                    df['ASIA_L'] = 0
            
            # 3. WEEKLY Levels (Across the full 1-Week dataframe)
            df['WEEK_H'] = df['high'].max()
            df['WEEK_L'] = df['low'].min()
            
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
            
            # 5. MICRO-METRICS (For the Ledger)
            # Body Ticks (Points for now)
            df['body_ticks'] = (df['close'] - df['open']).abs()
            
            # Wick Ratio (MUST MATCH BACKTEST: max_wick / range)
            df['range'] = (df['high'] - df['low']).replace(0, 1e-6)
            df['body'] = (df['close'] - df['open']).abs()
            df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
            df['max_wick'] = df[['upper_wick', 'lower_wick']].max(axis=1)
            df['wick_ratio'] = df['max_wick'] / df['range']
            
            # Relative Volume (10-bar avg)
            df['avg_vol'] = df['volume'].rolling(window=10).mean().replace(0, 1e-6)
            df['rel_vol'] = df['volume'] / df['avg_vol']
            
            # Displacement Z-Score (Placeholder or simple diff)
            df['disp_z'] = df['body'] / df['body'].rolling(window=20).std()
            
            return df
        except Exception as e:
            print(f"Feature Eng Error: {e}")
            return df

    def check_signals(self, df):
        # 0. Double-Wall Position Guard
        # Blocks if IBKR sees a position OR if bot is still processing an internal trade state
        # Blocks if IBKR sees a position OR if bot is still processing an internal trade state
        if self.in_position or self.current_trade:
            return None, "Position/Internal State Active", None, None

        # 1. Engineer Features
        df = self.engineer_live_features(df)
        if df.empty or len(df) < 5: return None, "Insufficient Data", None, None
        row = df.iloc[-1]
        
        # 3. Time Filter (REMOVED for 24/7 Pivot)
        pass 
        
        # 4. Weekly Exhaustion Filter (Phase 26 Optimized)
        # Blocks Longs in top 10% of week; blocks Shorts in bottom 10% of week.
        w_h = row.get('WEEK_H', 0)
        w_l = row.get('WEEK_L', 0)
        w_range = w_h - w_l
        rel_pos = (row['close'] - w_l) / w_range if w_range > 0 else 0.5
        
        # Trade Limiter & State Reset
        today_date = row['date_only']
        if self.last_trade_day != today_date:
            self.daily_trade_count = 0 # Reset for new day
            self.last_trade_day = today_date
            self.traded_signals = set() # Clear signal memory
            self.logged_minutes = set() # Clear logging memory
            
            # Reset Sweep Flags for new day
            self.swept_ib_low = False
            self.swept_ib_high = False
            self.swept_asia_low = False
            self.swept_asia_high = False
            
        if self.daily_trade_count >= MAX_DAILY_TRADES:
            return None, f"Max Trades Reached ({self.daily_trade_count}/{MAX_DAILY_TRADES})", None, None

            
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
            
            # Weekly Exhaustion Filter (Phase 26)
            long_allowed = rel_pos <= 0.90 # Block near Weekly High
            short_allowed = rel_pos >= 0.10 # Block near Weekly Low
            
            if self.swept_ib_low and row['fvg_bull'] and trend_ok_long and long_allowed:
                ib_signal = True
                ib_dir = 1 # BUY
            elif self.swept_ib_high and row['fvg_bear'] and trend_ok_short and short_allowed:
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
            # Weekly Exhaustion Filter (Phase 26)
            long_allowed = rel_pos <= 0.90
            short_allowed = rel_pos >= 0.10

            # Check Trigger
            if self.swept_asia_low and row['fvg_bull'] and trend_ok_long and long_allowed:
                asia_signal = True
                asia_dir = 1
            elif self.swept_asia_high and row['fvg_bear'] and trend_ok_short and short_allowed:
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
            
        # 6. Signature Guard (Mechanical suppression of redundant signals)
        if final_action:
            sig_ts = str(row['ny_time']) if 'ny_time' in row else str(row.name)
            signature = (sig_ts, final_strat, final_action)
            
            if signature in self.traded_signals:
                return None, f"Signal Already Executed ({sig_ts})", None, None
            
            # 7. Settlement Cooldown (Quick 30s buffer for order flight)
            if (time.time() - self.last_trade_time) < 30:
                return None, "Settlement Cooldown (Order in Flight)", None, None

        details = f"IB={row['IB_H']:.1f}/{row['IB_L']:.1f}, ASIA={row['ASIA_H']:.1f}/{row['ASIA_L']:.1f}"
        
        # Capture Metrics for the ledger
        self.latest_metrics = {
            'wick': row['wick_ratio'] if 'wick_ratio' in row else 0,
            'vol': row['rel_vol'] if 'rel_vol' in row else 0,
            'body': row['body_ticks'] if 'body_ticks' in row else 0,
            'zscore': row['disp_z'] if 'disp_z' in row else 0
        }

        # Return Tuple (Action, Details, Strat_ID, Signal_TS)
        return final_action, details, final_strat, str(row['ny_time']) if final_action and 'ny_time' in row else None

    def execute_trade(self, action, price, strategy_id="GOLDEN_DEFAULT", sig_ts=None):
        if DATA_ONLY_MODE:
            print(f"🛑 DATA ONLY MODE: Trade {action} @ {price} BLOCKED.")
            return

        print(f"🚀 LIVE EXECUTION: {action} @ {price} | Strat: {strategy_id}")
        
        # CONFIG MAP (Portfolio)
        # IB: TP 80, SL 5
        # ASIA: TP 85, SL 5
        
        # CONFIG MAP (Robust Pivot) -- DYNAMIC LOAD
        config = get_strategy_config(strategy_id)
        param_tp = config.get('tp', TP_POINTS)
        param_sl = config.get('sl', SL_POINTS)
        
        print(f"   ► Dynamic Config: SL={param_sl}, TP={param_tp}")
        
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
            self.in_position = True # Immediate Lock
            self.empty_pos_count = 0 # Reset Persistence Guard
            self.last_trade_time = time.time() # Start settlement timer
            
            # Log to GS immediately and capture row index
            m = self.latest_metrics
            gs_row = self.gs_logger.log_trade(
                pool_id=strategy_id, 
                direction=action, 
                entry=price, 
                sl=sl, 
                tp=tp, 
                status="OPEN",
                wick=m['wick'],
                vol=m['vol'],
                body=m['body'],
                zscore=m['zscore']
            )
            
            self.current_trade = {
                'entry': price,
                'direction': action,
                'strat': strategy_id,
                'sl': sl,
                'tp': tp,
                'gs_row': gs_row # Store row index for closing update
            }
            # Initial Audit Log
            self.log_audit_event(self.current_trade, price, "OPEN")
            # Record the signature immediately!
            if sig_ts:
                self.traded_signals.add((sig_ts, strategy_id, action))
                
            self.daily_trade_count += 1
            print(f"✅ Trade Count: {self.daily_trade_count}/{MAX_DAILY_TRADES}")

if __name__ == '__main__':
    MAX_RECONNECT_ATTEMPTS = 999  # Essentially infinite reconnects
    RECONNECT_DELAY_SECONDS = 10
    
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        bot = GoldenBot()
        try:
            print(f"\n🔄 IBKR Bridge Start (Attempt {attempt}/{MAX_RECONNECT_ATTEMPTS})")
            asyncio.run(bot.run())
            # If run() exits cleanly (isConnected() became False), we reconnect
            print(f"⚠️ Connection Lost. Reconnecting in {RECONNECT_DELAY_SECONDS}s...")
            time.sleep(RECONNECT_DELAY_SECONDS)
        except KeyboardInterrupt:
            print("Bridge stopping (User Interrupt)...")
            if bot.gs_logger.enabled:
                bot.gs_logger.flush_ticks()
            break  # Exit cleanly on Ctrl+C
        except Exception as e:
            print(f"❌ Bridge Error: {e}. Reconnecting in {RECONNECT_DELAY_SECONDS}s...")
            time.sleep(RECONNECT_DELAY_SECONDS)
    
    print("🛑 IBKR Bridge Terminated.")
