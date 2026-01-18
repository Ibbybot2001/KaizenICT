from strategies.legacy.backtester import Backtester
from strategies.legacy.data_loader import load_data
import pandas as pd

# Import Strategies
from strategies.legacy.strategies.liquidity_raid import LiquidityRaidStrategy
from strategies.legacy.strategies.silver_bullet import SilverBulletStrategy
from strategies.legacy.strategies.ote import OTEStrategy
from strategies.legacy.strategies.unicorn import UnicornStrategy

def run_debug(strategy_cls, name, days=5):
    print(f"--- DEBUGGING {name} ---")
    
    # Load small slice
    path = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
    print("Loading data...")
    # Read last N lines efficiently or just load and slice
    # We'll load full and slice last 5 days for speed/relevance
    df = load_data(path) 
    df = df.iloc[-5000:] # Approx 3-4 days of 1m data
    
    print(f"Data Slice: {len(df)} bars. {df.index[0]} to {df.index[-1]}")
    
    backtester = Backtester(df)
    strategy = strategy_cls(backtester)
    backtester.strategy = strategy
    
    # Run
    backtester.run()
    
    # Analyze
    print(f"{name} Trades: {len(backtester.trades)}")
    for t in backtester.trades[:5]: # Print first 5 trades details
        print(f"Trade: {t}")

if __name__ == "__main__":
    # Uncomment the one to debug
    
    # 1. Debug Liquidity Raid (The anomaly)
    # run_debug(LiquidityRaidStrategy, "Liquidity Raid")
    
    # 2. Debug Silver Bullet (The ghost)
    # run_debug(SilverBulletStrategy, "Silver Bullet")
    
    # 3. Debug OTE
    run_debug(OTEStrategy, "OTE")
