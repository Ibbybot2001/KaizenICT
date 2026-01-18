
import datetime
import time
import sys
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

# Import Config
# Assuming running from root, python strategies/production/chronos_bot.py
# Adjust path if needed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from strategies.production import config
except ImportError:
    # Fallback for local testing if path issues
    import config

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chronos_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ChronosBot")

@dataclass
class Signal:
    concept: str
    direction: int # 1 = Long, -1 = Short
    price: Optional[float] = None
    timestamp: Optional[datetime.datetime] = None

class ChronosEngine:
    def __init__(self, time_source=None):
        self.symbol = config.SYMBOL
        self.stop_loss = config.STOP_LOSS_PTS
        self.take_profit = config.TAKE_PROFIT_PTS
        self.time_source = time_source # Function returning datetime
        
        # State
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.is_active = True
        
        logger.info(f"ChronosBot Initialized for {self.symbol}")
        logger.info(f"Risk: SL {self.stop_loss} / TP {self.take_profit}")
        logger.info(f"Schedule: {config.SCHEDULE}")

    def get_current_time(self) -> datetime.datetime:
        """Get current time in target timezone (simplified for MVP to local/UTC)."""
        if self.time_source:
            return self.time_source()
        return datetime.datetime.now()

    def check_schedule(self, now: datetime.datetime) -> List[str]:
        """Check if current HH:MM matches any trigger(s). Returns ALL matches."""
        current_time_str = now.strftime("%H:%M")
        triggers = []
        
        # We also want to fire ONLY at the start of the minute (Seconds == 00-05)
        if now.second > 5:
            return []
            
        for concept_name, trigger_time in config.SCHEDULE.items():
            if current_time_str == trigger_time:
                triggers.append(concept_name)
        return triggers

    def execute_logic(self, concept_name: str, price_data: dict) -> Optional[Signal]:
        """
        Execute the specific logic for the accepted concept.
        Requires `price_data` populated with: {'current': float, 'open_0930': float, 'high_0930_0944': float, ...}
        """
        logger.info(f"Triggering Logic for {concept_name}...")
        
        # For Phase 5 Build, we assume price_data is passed by the Datafeed (to be implemented).
        # Here we define the EXACT logic required.
        
        if concept_name == "C1_NY_ORB":
            # Logic: Breakout of 09:30-09:44 Range.
            # Needs: Current Close, High(09:30-09:44), Low(09:30-09:44)
            current_price = price_data.get('current', 0.0)
            range_high = price_data.get('high_0930_0944', float('inf')) 
            range_low = price_data.get('low_0930_0944', float('-inf'))
            
            if current_price > range_high:
                return Signal(concept_name, 1, current_price, datetime.datetime.now())
            elif current_price < range_low:
                return Signal(concept_name, -1, current_price, datetime.datetime.now())
            else:
                logger.info(f"{concept_name}: Price {current_price} inside range {range_low}-{range_high}. No Trade.")
                return None
            
        elif concept_name == "C3_3PM_MACRO":
            # 15:00 Macro. Momentum of 14:50-15:00.
            # Needs: Current Close, Close(14:50).
            current_price = price_data.get('current', 0.0)
            close_1450 = price_data.get('close_1450', 0.0)
            
            if current_price > close_1450:
                return Signal(concept_name, 1, current_price, datetime.datetime.now())
            elif current_price < close_1450:
                return Signal(concept_name, -1, current_price, datetime.datetime.now())
            
        elif concept_name == "C8_LAST_HOUR_MOMENTUM":
            # 15:00 Last Hour. Trend of the Day.
            # Needs: Current Close, Open(09:30).
            current_price = price_data.get('current', 0.0)
            open_0930 = price_data.get('open_0930', 0.0)
            
            if current_price > open_0930:
                return Signal(concept_name, 1, current_price, datetime.datetime.now())
            elif current_price < open_0930:
                return Signal(concept_name, -1, current_price, datetime.datetime.now())
        
        elif concept_name == "C14_SILVER_BULLET":
            # 10:15 Silver Bullet. 
            # Logic: Breakout of 10:00 Price (Momentum Continuation).
            # Needs: Current Close, Close(10:00).
            current_price = price_data.get('current', 0.0)
            close_1000 = price_data.get('close_1000', 0.0)
            
            if current_price > close_1000:
                return Signal(concept_name, 1, current_price, datetime.datetime.now())
            elif current_price < close_1000:
                return Signal(concept_name, -1, current_price, datetime.datetime.now())
            
        return None

    def execute_trade(self, signal: Signal):
        """Send Order (Mock)."""
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            logger.warning("Max Daily Trades Reached. Skipping.")
            return

        logger.info(f"*** EXECUTING TRADE ***")
        logger.info(f"Concept: {signal.concept}")
        logger.info(f"Direction: {'LONG' if signal.direction > 0 else 'SHORT'}")
        logger.info(f"Params: SL {self.stop_loss}pt / TP {self.take_profit}pt")
        
        self.daily_trades += 1
        # In production: Send IBKR API order here.

    def run(self, single_step: bool = False):
        """Main Loop."""
        logger.info("Starting Main Loop...")
        last_minute_checked = -1
        
        try:
            while self.is_active:
                now = self.get_current_time()
                
                # Check Schedule
                triggers = self.check_schedule(now)
                
                # Only run logic once per minute
                # Note: `check_schedule` returns empty if seconds > 5, 
                # but we also dedupe by `now.minute` to be safe/efficient.
                if len(triggers) > 0 and now.minute != last_minute_checked:
                    last_minute_checked = now.minute
                    
                    # Get Data (Mock - In Prod this comes from Datafeed)
                    # We inject dummy data that GUARANTEES a trigger for testing
                    price_data = {
                        'current': 100.0,
                        'high_0930_0944': 90.0,
                        'low_0930_0944': 80.0,
                        'close_1450': 95.0,
                        'open_0930': 95.0
                    }
                    
                    for concept in triggers:
                        # Decide
                        signal = self.execute_logic(concept, price_data)
                        
                        # Execute
                        if signal:
                            self.execute_trade(signal)
                        
                if single_step:
                    break
                
                time.sleep(0.1) # low latency sleep
                
        except KeyboardInterrupt:
            logger.info("Stopping ChronosBot...")

if __name__ == "__main__":
    bot = ChronosEngine()
    bot.run()
