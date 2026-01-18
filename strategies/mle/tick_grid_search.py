from strategies.mle.tick_realistic_backtester import TickRealisticBacktester
from pathlib import Path
import pandas as pd
import time

def main():
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    tick_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet"
    bar_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    out_path = base_dir / "output/tick_grid_results.csv"
    
    print("Initializing Tick Backtester...")
    # Initialize
    tester = TickRealisticBacktester(str(tick_path), str(bar_path))
    
    # Grid Parameters for "Testing It All"
    hours = [15]
    minutes = [0, 1]
    directions = [1] # Long only for C3
    stops = [5.0, 10.0, 15.0, 20.0]
    targets = [10.0, 20.0, 40.0, 60.0, 100.0]
    be_levels = [None, 5.0, 10.0, 15.0, 20.0]
    
    start_time = time.time()
    
    # Run Grid Search
    df = tester.run_grid_search(
        hours=hours,
        minutes=minutes,
        directions=directions,
        stops=stops,
        targets=targets,
        be_levels=be_levels
    )
    
    # Save
    df.to_csv(out_path, index=False)
    elapsed = time.time() - start_time
    
    print(f"\nGrid Search Complete in {elapsed:.2f}s")
    print(f"Combinations Tested: {len(df)}")
    print(f"Saved to {out_path}")
    
    # Print Top 5 by PnL
    print("\nTOP 5 CONFIGURATIONS:")
    print(df.sort_values('pnl', ascending=False).head(5).to_string())

if __name__ == "__main__":
    main()
