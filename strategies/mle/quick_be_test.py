from strategies.mle.tick_realistic_backtester import TickRealisticBacktester
from pathlib import Path
import sys

def main():
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    tick_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet"
    bar_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    # Initialize
    tester = TickRealisticBacktester(str(tick_path), str(bar_path))
    
    params = [
        ("Standard (No BE)", None),
        ("BE @ 5 pts", 5.0),
        ("BE @ 10 pts", 10.0)
    ]
    
    print("--- START RESULTS ---")
    for name, be_val in params:
        res = tester.backtest_strategy(15, 0, 1, 10.0, 40.0, move_to_be_pts=be_val)
        print(f"{name}|PnL:{res['pnl']:.2f}|PF:{res['pf']:.2f}|WR:{res['wr']:.1f}|Trades:{res['trades']}")
        sys.stdout.flush()
    print("--- END RESULTS ---")

if __name__ == "__main__":
    main()
