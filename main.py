
import sys
import os
import pandas as pd

# Add root to path
sys.path.append(os.getcwd())

from strategies.legacy.data_loader import load_data
from strategies.legacy.backtester import Backtester
from strategies.legacy.model_2022 import Model2022
from strategies.legacy.unicorn import UnicornStrategy


def main():
    print("=== ICT Master Suite Python Backtester ===")
    
    # 1. Load Data
    path = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
    # To use GOLDEN DATA (Superior), uncomment below:
    # path = r"C:\Users\CEO\ICT reinforcement\data\GOLDEN_DATA\USTEC_2025_CLEAN_TICKS\USTEC_2025-01.csv"
    df = load_data(path)
    
    # Optional: Slice data for speed during dev
    # df = df.iloc[-40000:] # Last 1 month approx
    # df = df.last('1M') 
    print(f"Data Loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}") 
    
    # 2. Init Strategy
    # We pass None to init, then attach strategy later? 
    # Or pass class to Backtester? 
    # 2. Init Strategy
    # from src.strategies.model_2022 import Model2022
    
    # strategy = Model2022(None) # Strategy attached later? 
    # Backtester takes strategy class or instance?
    # Inspecting init: Backtester(data, strategy=None)
    # Strategy(backtester)
    
    # Circular dependency if we pass backtester instance to strategy init?
    # Pattern used in main.py:
    # strategy = Model2022(None)
    # backtester = Backtester(df, strategy)
    # strategy.backtester = backtester
    
    # Let's fix cleaner:
    backtester = Backtester(df)
    
    # Select Strategy to Run (Uncomment one):
    
    # Select Strategy to Run (Uncomment one):
    
    # 1. Model 2022 (Sweep -> MSS -> FVG)
    # from src.strategies.model_2022 import Model2022
    # strategy = Model2022(backtester)

    # 2. Unicorn Model (Breaker + FVG)
    # from src.strategies.unicorn import UnicornStrategy
    # strategy = UnicornStrategy(backtester)

    # 3. Liquidity Raid (Turtle Soup / Sweep)
    from strategies.legacy.liquidity_raid import LiquidityRaidStrategy
    strategy = LiquidityRaidStrategy(backtester)

    # 4. Silver Bullet (Time-based FVG)
    # from strategies.legacy.silver_bullet import SilverBulletStrategy
    # strategy = SilverBulletStrategy(backtester)

    # 5. Optimal Trade Entry (OTE)
    # from strategies.legacy.ote import OTEStrategy
    # strategy = OTEStrategy(backtester)

    backtester.strategy = strategy
    
    # 3. Run
    backtester.run()
    
    # 4. Results
    print(f"\nTotal Trades: {len(backtester.trades)}")
    
    if len(backtester.trades) > 0:
        trades_df = pd.DataFrame([vars(t) for t in backtester.trades])
        
        # Win Rate
        wins = trades_df[trades_df['pnl'] > 0]
        wr = len(wins) / len(trades_df) * 100
        total_pnl = trades_df['pnl'].sum()
        
        print(f"Net PnL: ${total_pnl:,.2f}")
        print(f"Win Rate: {wr:.2f}%")
        print(f"Avg Trade: ${trades_df['pnl'].mean():.2f}")
        
        # Save to CSV
        output_path = "ict_backtest/output/trades_model2022.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        trades_df.to_csv(output_path, index=False)
        print(f"Trade log saved to {output_path}")

if __name__ == "__main__":
    main()
