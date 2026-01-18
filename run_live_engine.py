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

# Import components
from live.bar_builder import BarBuilder, Tick, TimeAuthority
from live.pool_fsm import SessionPoolManager, BarEventProcessor, TradeDirection, PoolState
from live.risk_guard import RiskGuard, RiskConfig

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
# ORDER EXECUTION STUB (Replace with IBKR / TraderPost)
# ==============================================================================
class MockBroker:
    def execute_order(self, pool_id, direction, size, sl, tp):
        logger.info(f"⚡ [EXEC] SENDING ORDER: {direction.name} {size} @ MKT | SL: {sl:.2f} | TP: {tp:.2f}")
        # In real impl, return order_id
        return "ORD-123"

# ==============================================================================
# LIVE ENGINE CLASS
# ==============================================================================
class LiveEngine:
    def __init__(self):
        logger.info("Initializing PJ/ICT Live Engine...")
        
        # 1. Components
        self.risk_guard = RiskGuard()
        self.session_manager = SessionPoolManager(datetime.now(timezone.utc))
        self.broker = MockBroker()
        
        # 2. Bar Builder (Wall Clock Driven)
        self.bar_builder = BarBuilder(
            interval_seconds=60,
            time_authority=TIME_AUTHORITY,
            on_bar_close=self.on_bar_close
        )
        
        # 3. Logic Processor
        self.processor = BarEventProcessor(self.session_manager)
        
        # 4. State
        self.last_tick_time = None
        self.is_running = False
        
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
        
        # 1. Update FSMs
        self.processor.process_bar(bar)
        
        # 2. Check for Signals
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

    def run(self):
        """Main Loop: Tail CSV and feed BarBuilder."""
        self.is_running = True
        logger.info(f"Watching data file: {LIVE_DATA_FILE}")
        
        # Ensure file exists
        if not LIVE_DATA_FILE.exists():
            # Create dummy file for testing if not exists
            LIVE_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(LIVE_DATA_FILE, 'w') as f:
                f.write("time,price,size\n")
            logger.warning("Created empty data file.")
        
        # Sync Time
        TIME_AUTHORITY.sync_with_ntp()
        
        # Tail Loop
        with open(LIVE_DATA_FILE, 'r') as f:
            # Go to end
            f.seek(0, 2)
            
            while self.is_running:
                line = f.readline()
                if not line:
                    # No new data: Check for time-based bar close
                    self.bar_builder.force_close()  # Checks wall-clock
                    
                    # --- DASHBOARD EXPORT (Every 1s) ---
                    curr_ts = time.time()
                    if not hasattr(self, 'last_dash_dump') or (curr_ts - self.last_dash_dump > 1.0):
                        try:
                            import json # Safe localized import
                            state = {
                                "time": datetime.now().strftime('%H:%M:%S'),
                                "pnl": self.risk_guard.daily_pnl,
                                "trades": self.risk_guard.daily_trades,
                                "status": "ACTIVE" if self.risk_guard.can_trade()[0] else "HALTED",
                                "pools": [p for p in self.session_manager.pools.keys()],
                                "last_price": self.bar_builder.current_bar.close if self.bar_builder.current_bar else 0
                            }
                            with open("live_dashboard.json", "w") as f:
                                json.dump(state, f)
                            self.last_dash_dump = curr_ts
                        except Exception:
                            pass # Don't crash engine for dashboard
                    
                    time.sleep(POLL_INTERVAL)
                    continue
                
                try:
                    # Parse Tick
                    # Expected format: YYYY-MM-DD HH:MM:SS.mmm, price, size
                    parts = line.strip().split(',')
                    if len(parts) < 2: continue
                    
                    tick_time = pd.to_datetime(parts[0])
                    price = float(parts[1])
                    size = int(parts[2]) if len(parts) > 2 else 1
                    
                    tick = Tick(tick_time, price, size=size)
                    
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
