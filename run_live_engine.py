"""
PJ/ICT LIVE EXECUTION ENGINE
Final Integration Script

Connects:
1. Data Source: Live Ticks (Tail `live_ticks.csv` or IBKR API)
2. Bar Builder: Immutability & Time Authority
3. Pool FSM: Logic & State
4. Risk Guard: Circuit Breaker & Safety

Run this script to START the engine.
"""

import time
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import logging
import threading
import sys
import os
import requests

# Import components
from live.bar_builder import BarBuilder, Tick, TimeAuthority
from live.pool_fsm import SessionPoolManager, BarEventProcessor, TradeDirection, PoolState
from live.risk_guard import RiskGuard, RiskConfig
from live.execution_bridge import TradersPostBroker
from dashboard_logger import DashboardLogger

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LIVE_LOG_DIR = Path("live_logs")
# Point to the artifact directory where user said the data is
LIVE_DATA_FILE = Path("strategies/mle/data/live_ticks.csv")
POLL_INTERVAL = 0.5  # Seconds

# Use IBKR Server time (simulated via NTP for now)
TIME_AUTHORITY = TimeAuthority(mode='ntp')

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
LIVE_LOG_DIR.mkdir(exist_ok=True)
log_file = LIVE_LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_engine.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiveEngine")

# ==============================================================================
# LIVE ENGINE CLASS
# ==============================================================================
class LiveEngine:
    def __init__(self):
        logger.info("Initializing PJ/ICT Live Engine...")
        
        # 1. Components
        self.risk_guard = RiskGuard()
        self.session_manager = SessionPoolManager(datetime.now(timezone.utc))
        self.broker = TradersPostBroker()
        
        # 2. Google Sheets Logger
        self.gs_logger = DashboardLogger()
        if self.gs_logger.enabled:
            logger.info("✅ Google Sheets Logging: ENABLED (RawTicks + OneMinuteData)")
        else:
            logger.warning("❌ Google Sheets Logging: DISABLED")

        # 3. Bar Builder (Wall Clock Driven)
        self.bar_builder = BarBuilder(
            interval_seconds=60,
            time_authority=TIME_AUTHORITY,
            on_bar_close=self.on_bar_close
        )
        
        # 4. Logic Processor
        self.processor = BarEventProcessor(self.session_manager)
        
        # 5. State
        self.last_tick_time = None
        self.is_running = False
        
        # Continuity Tracking
        self.stream_start_time = None
        self.last_tick_time = None
        
        # Initialize Pools (Example - would load from config daily)
        self.init_pools()
        
    def init_pools(self):
        """Define today's liquidity pools."""
        # Example: Previous Day High/Low
        self.session_manager.add_pool("PDH", 21500.0, TradeDirection.SHORT, 21400.0)
        self.session_manager.add_pool("PDL", 21400.0, TradeDirection.LONG, 21500.0)
        logger.info("Pools Initialized: PDH, PDL")

    def on_bar_close(self, bar):
        """Called when a bar is finalized (IMMUTABLE SNAPSHOT)."""
        logger.info(f"📊 [BAR] Closed: {bar.end_time.strftime('%H:%M:%S')} | C: {bar.close:.2f} | V: {bar.volume}")
        
        # 1. Log Bar to Google Sheets
        if self.gs_logger.enabled:
             # Using "Bar Closed" as action and Volume for vol field
             # Basic metrics for now
            self.gs_logger.log_min_data(
                timestamp=datetime.now(), # Or bar.end_time
                price=bar.close,
                action="BAR_CLOSE",
                details=f"O:{bar.open} H:{bar.high} L:{bar.low} C:{bar.close}",
                vol=bar.volume,
                wick=0, # Metrics not calculated in bar struct yet
                body_ticks=bar.tick_count
            )

        # 2. Update FSMs
        self.processor.process_bar(bar)
        
        # 3. Reconcile Risk (Check for closed trades)
        for pool_id, fsm in self.session_manager.pools.items():
            if fsm.pool.state == PoolState.CLOSED and not hasattr(fsm, '_risk_synced'):
                # Sync point-pnl to dollar-pnl ($2 per pt for MNQ)
                dollar_pnl = fsm.pool.pnl * 2.0
                self.risk_guard.on_trade_closed(dollar_pnl)
                fsm._risk_synced = True # Mark as synced to prevent double counting
                logger.info(f"💰 [RISK-SYNC] Pool {pool_id} CLOSED. PnL: ${dollar_pnl:.2f}. Daily: ${self.risk_guard.daily_pnl:.2f}")
                
                # 3. Log Exit to GS
                if self.gs_logger.enabled:
                    self.gs_logger.log_trade(
                        pool_id=pool_id,
                        direction=fsm.pool.direction.name,
                        entry=fsm.pool.entry_price,
                        sl=fsm.pool.stop_loss,
                        tp=fsm.pool.take_profit,
                        status=f"CLOSED ({fsm.pool.exit_reason})",
                        pnl_pts=fsm.pool.pnl,
                        wick=fsm.pool.setup_wick_ratio,
                        vol=fsm.pool.setup_rel_vol,
                        body=fsm.pool.setup_body_ticks,
                        zscore=fsm.pool.setup_disp_zscore
                    )

        # 4. Check for Signals
        signals = self.session_manager.get_all_ready_signals()
        for sig in signals:
            self.execute_signal(sig)
            
    def execute_signal(self, sig):
        """Execute a valid signal."""
        pool_id = sig['pool_id']
        fsm = self.session_manager.get_pool(pool_id)
        
        # 1. Risk Check
        allowed, reason = self.risk_guard.check_trade_allowed(1)
        if not allowed:
            logger.critical(f"🛑 [RISK] Trade Blocked: {reason}")
            return
        
        # 2. Update FSM State (Entry Pending)
        # Note: In real system, wait for fill confirmation
        fsm.on_order_sent(datetime.now(timezone.utc))
        
        # 3. Send to Broker
        self.broker.execute_order(
            pool_id, 
            sig['direction'], 
            1, 
            sig['stop_loss'], 
            sig['take_profit']
        )
        
        # 4. Assume Fill (Demo Only)
        fsm.on_fill(sig['level_price'])  # Assume filled at level for now
        logger.info(f"✅ [TRADE] Entered {pool_id} {sig['direction'].name}")
        
        # 5. Log Entry to GS
        if self.gs_logger.enabled:
            self.gs_logger.log_trade(
                pool_id=pool_id,
                direction=sig['direction'].name,
                entry=sig['level_price'],
                sl=sig['stop_loss'],
                tp=sig['take_profit'],
                status="OPEN",
                wick=sig.get('wick_ratio', 0),
                vol=sig.get('rel_vol', 0),
                body=sig.get('body_ticks', 0),
                zscore=sig.get('disp_zscore', 0)
            )

    def run(self):
        """Main Loop: Tail CSV and feed BarBuilder."""
        print("\n[ENG] Entering run()...")
        self.is_running = True
        logger.info(f"Watching data file: {LIVE_DATA_FILE}")
        
        # Ensure file exists
        if not LIVE_DATA_FILE.exists():
            print(f"[ENG] Creating data file: {LIVE_DATA_FILE}")
            # Create dummy file for testing if not exists
            LIVE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LIVE_DATA_FILE, 'w') as f:
                f.write("time,price,size\n")
            logger.warning("Created empty data file.")
        
        # Sync Time
        print("[ENG] Syncing with NTP...")
        TIME_AUTHORITY.sync_with_ntp()
        print("[ENG] Sync complete.")
        
        # Tail Loop
        print(f"[ENG] Opening {LIVE_DATA_FILE} for tailing...")
        with open(LIVE_DATA_FILE, 'r') as f:
            print("[ENG] File opened. Seeking EOF...")
            # Go to end
            f.seek(0, 2)
            print("[ENG] Main loop STARTING.")
            
            while self.is_running:
                # --- DATA CONTINUITY CHECK ---
                warmup_mins = 0
                if self.stream_start_time:
                    warmup_mins = (datetime.now() - self.stream_start_time).total_seconds() / 60.0
                
                if warmup_mins >= 30: # Use RiskConfig.MIN_CONTINUITY_MINUTES if possible
                    self.risk_guard.data_stream_ready = True
                    self.risk_guard.warmup_status = "READY"
                else:
                    self.risk_guard.data_stream_ready = False
                    self.risk_guard.warmup_status = f"WARMING_UP ({warmup_mins:.1f}/30m)"

                # --- DASHBOARD EXPORT (Every 1s) ---
                # Run this regardless of whether a tick just arrived
                curr_ts = time.time()
                if not hasattr(self, 'last_dash_dump') or (curr_ts - self.last_dash_dump > 1.0):
                    try:
                        import json 
                        state = {
                            "time": datetime.now().strftime('%H:%M:%S'),
                            "pnl": self.risk_guard.daily_pnl,
                            "trades": self.risk_guard.daily_loss_count, 
                            "status": self.risk_guard.warmup_status if not self.risk_guard.is_halted else "HALTED",
                            "pools": [p for p in self.session_manager.pools.keys()],
                            "last_price": self.bar_builder.get_last_finalized_bar().close if self.bar_builder.get_last_finalized_bar() else 0,
                            "gs_rpm": self.gs_logger.get_requests_per_min() if self.gs_logger.enabled else 0
                        }
                        # Add current bar if open
                        curr_bar = self.bar_builder.get_current_bar()
                        if curr_bar:
                            state['last_price'] = curr_bar.close

                        with open("live_dashboard.json", "w") as f_out:
                            json.dump(state, f_out)
                        self.last_dash_dump = curr_ts
                        # Debug Heartbeat
                        sys.stdout.write(f"\r💓 Dashboard Heartbeat: {state['time']}   ")
                        sys.stdout.flush()
                    except Exception as e:
                        pass 

                line = f.readline()
                if not line:
                    # No new data: Check for time-based bar close
                    self.bar_builder.force_close()  # Checks wall-clock
                    time.sleep(POLL_INTERVAL)
                    continue
                
                try:
                    # Parse Tick
                    # Expected format: YYYY-MM-DD HH:MM:SS.mmm, price, size
                    parts = line.strip().split(',')
                    if len(parts) < 2: continue
                    
                    # Standardize tick_time to naive local for comparison
                    tick_time = pd.to_datetime(parts[0]).replace(tzinfo=None)
                    price = float(parts[1])
                    size = int(parts[2]) if len(parts) > 2 else 1
                    
                    # Update Continuity
                    if self.stream_start_time is None:
                        self.stream_start_time = datetime.now()
                    self.last_tick_time = datetime.now()
                    
                    tick = Tick(tick_time, price, size=size)
                    
                    # Log Raw Tick to Google Sheets
                    if self.gs_logger.enabled:
                        self.gs_logger.log_tick(tick_time, price, size)
                    
                    # Process tick
                    self.bar_builder.process_tick(tick)
                    
                except Exception as e:
                    logger.error(f"Error parsing line: {line.strip()} - {e}")

if __name__ == "__main__":
    engine = LiveEngine()
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Engine stopping...")
        if engine.gs_logger.enabled:
            engine.gs_logger.flush_ticks()
