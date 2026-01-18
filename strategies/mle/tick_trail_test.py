from strategies.mle.tick_realistic_backtester import TickRealisticBacktester
from pathlib import Path
import pandas as pd
import time

def main():
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    tick_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet"
    bar_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    out_path = base_dir / "output/tick_trail_results.csv"
    
    # Initialize
    tester = TickRealisticBacktester(str(tick_path), str(bar_path))
    
    # Grid Parameters - Focused on confirming Home Run vs Trailing
    hours = [15]
    minutes = [0, 1]
    directions = [1]
    stops = [15.0] # Best stop from previous test
    targets = [100.0, 200.0] # Aiming big
    be_levels = [None] # BE was bad, keeping it off
    
    # Trailing Configs: (Trigger, Distance)
    trail_configs = [
        (None, None),          # Baseline (No Trail)
        (20.0, 10.0),          # Early protection
        (40.0, 20.0),          # Mid-run protection
        (60.0, 30.0),          # Late-run protection
        (100.0, 50.0)          # Ultra-late protection (runner management)
    ]
    
    start_time = time.time()
    
    # Run Grid Search
    df = tester.run_grid_search(
        hours=hours,
        minutes=minutes,
        directions=directions,
        stops=stops,
        targets=targets,
        be_levels=be_levels,
        trail_configs=trail_configs
    )
    
    
    # Save Grid Search Results
    df.to_csv(out_path, index=False)
    elapsed = time.time() - start_time
    
    summary_path = base_dir / "output/tick_trail_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Grid Search Complete in {elapsed:.2f}s\n")
        f.write(f"Combinations Tested: {len(df)}\n")
        
        # Write Top 10 by PnL
        f.write("\nTOP 10 CONFIGURATIONS (Trailing vs Fixed):\n")
        f.write(df.sort_values('pnl', ascending=False).head(10)[
            ['hour','minute','stop_pts','target_pts','trail_trig','trail_dist','pnl','pf']
        ].to_string())
    
    print(f"\nSaved summary to {summary_path}")
    
    # Read and print explicitly line by line
    with open(summary_path, "r") as f:
        print(f.read())

if __name__ == "__main__":
    main()
