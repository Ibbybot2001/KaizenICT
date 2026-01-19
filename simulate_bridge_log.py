import sys
sys.path.append("C:/Users/CEO/ICT reinforcement")
from dashboard_logger import DashboardLogger
from datetime import datetime

print("Testing Logger Strategy Formatting...")

logger = DashboardLogger()
if logger.enabled:
    print("Log Enabled. Writing Test Trades...")
    
    # 1. Simulate IB Hybrid Trade
    logger.log_trade(
        pool_id="IB_Hybrid_TEST",
        direction="BUY",
        entry=21000.0,
        sl=21000-5,
        tp=21000+55,
        status="OPEN",
        wick=0.1, vol=2.5, body=10.0
    )
    print("Logged IB Hybrid Trade.")
    
    # 2. Simulate ASIA Hybrid Trade
    logger.log_trade(
        pool_id="ASIA_Hybrid_TEST",
        direction="SELL", 
        entry=21050.0,
        sl=21050+5,
        tp=21050-85,
        status="OPEN",
        wick=0.2, vol=3.0, body=12.0
    )
    print("Logged ASIA Hybrid Trade.")
    
    print("Check Google Sheet > TradeLog for entries.")
else:
    print("Logger Disabled (Check Creds).")
