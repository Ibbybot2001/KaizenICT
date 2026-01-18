"""
Pure GPU Backtester (Titan III)
Zero Python Loops - Full CUDA Utilization
Target: 500,000+ Strategies/Second on RTX 4080 Super
"""

import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class StrategyGenome:
    """Tensor-based strategy representation."""
    hours: torch.Tensor      # [S] - Trigger Hour
    minutes: torch.Tensor    # [S] - Trigger Minute
    hold_bars: torch.Tensor  # [S] - Hold Duration (bars)
    directions: torch.Tensor # [S] - 1=Long, -1=Short


class PureGPUBacktester:
    """
    Fully vectorized backtester using broadcast operations.
    No Python loops in the hot path.
    """
    
    def __init__(self, data_path: str, chunk_size: int = 2000):
        self.chunk_size = chunk_size  # Strategies per batch
        self.load_data(data_path)
        
    def load_data(self, data_path: str):
        print(f"[GPU Backtester] Loading data on {DEVICE}...")
        t0 = time.time()
        
        df = pd.read_parquet(data_path)
        times = pd.to_datetime(df.index)
        
        # Core price data [T]
        self.closes = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        self.T = len(self.closes)
        
        # Time features [T]
        self.hours = torch.tensor(times.hour.values, dtype=torch.int16, device=DEVICE)
        self.minutes = torch.tensor(times.minute.values, dtype=torch.int16, device=DEVICE)
        
        # Pre-calculate returns for various hold periods
        # We'll support holds of 5, 10, 15, 30, 60 bars
        self.hold_options = [5, 10, 15, 30, 60]
        self.returns_by_hold = {}
        
        for h in self.hold_options:
            future_close = torch.roll(self.closes, -h)
            ret = future_close - self.closes
            ret[-h:] = 0  # Zero out wrapped values
            self.returns_by_hold[h] = ret
            
        print(f"[GPU Backtester] Loaded {self.T:,} bars in {time.time()-t0:.2f}s")
        
    def generate_random_population(self, size: int) -> StrategyGenome:
        """Generate random strategy genomes on GPU."""
        return StrategyGenome(
            hours=torch.randint(9, 17, (size,), dtype=torch.int16, device=DEVICE),
            minutes=torch.randint(0, 60, (size,), dtype=torch.int16, device=DEVICE),
            hold_bars=torch.tensor(
                np.random.choice(self.hold_options, size), 
                dtype=torch.int16, device=DEVICE
            ),
            directions=torch.randint(0, 2, (size,), dtype=torch.int16, device=DEVICE) * 2 - 1
        )
    
    def evaluate_batch(self, genomes: StrategyGenome) -> torch.Tensor:
        """
        Evaluate a batch of strategies using pure broadcast operations.
        
        Args:
            genomes: StrategyGenome with S strategies
            
        Returns:
            Tensor [S] of PnL scores
        """
        S = len(genomes.hours)
        
        # Expand time features to [1, T] for broadcasting
        hours_exp = self.hours.unsqueeze(0)      # [1, T]
        minutes_exp = self.minutes.unsqueeze(0)  # [1, T]
        
        # Expand strategy params to [S, 1] for broadcasting
        strat_hours = genomes.hours.unsqueeze(1).to(torch.int16)    # [S, 1]
        strat_mins = genomes.minutes.unsqueeze(1).to(torch.int16)   # [S, 1]
        strat_dirs = genomes.directions.unsqueeze(1).float()        # [S, 1]
        
        # Broadcast comparison: [S, T]
        # This creates S*T boolean comparisons in ONE kernel
        mask = (hours_exp == strat_hours) & (minutes_exp == strat_mins)
        
        # Now we need returns based on each strategy's hold period
        # This is tricky because hold_bars varies per strategy
        # Solution: Process by unique hold values
        
        scores = torch.zeros(S, device=DEVICE)
        
        for hold_val in self.hold_options:
            # Find strategies with this hold value
            hold_mask = genomes.hold_bars == hold_val
            if not hold_mask.any():
                continue
                
            # Get indices
            indices = torch.where(hold_mask)[0]
            
            # Get the returns for this hold
            returns = self.returns_by_hold[hold_val]  # [T]
            returns_exp = returns.unsqueeze(0)        # [1, T]
            
            # Get subset of masks and directions
            subset_mask = mask[indices]               # [S_sub, T]
            subset_dirs = strat_dirs[indices]         # [S_sub, 1]
            
            # Trade returns: where mask is True, multiply by returns and direction
            trade_returns = subset_mask.float() * returns_exp * subset_dirs  # [S_sub, T]
            
            # Apply cost per trade
            cost = 0.5  # 2 ticks
            num_trades = subset_mask.sum(dim=1)  # [S_sub]
            total_cost = num_trades.float() * cost
            
            # Sum returns and subtract cost
            raw_pnl = trade_returns.sum(dim=1)  # [S_sub]
            net_pnl = raw_pnl - total_cost
            
            # Store back
            scores[indices] = net_pnl
            
        return scores
    
    def run_full_search(self, total_strategies: int) -> Tuple[torch.Tensor, StrategyGenome]:
        """
        Run massive scale search with chunked processing.
        
        Returns:
            Tuple of (all_scores, all_genomes)
        """
        print(f"[GPU Backtester] Starting search of {total_strategies:,} strategies...")
        print(f"[GPU Backtester] Chunk size: {self.chunk_size}")
        
        all_scores = []
        all_genomes_hours = []
        all_genomes_mins = []
        all_genomes_holds = []
        all_genomes_dirs = []
        
        t0 = time.time()
        processed = 0
        
        while processed < total_strategies:
            batch_size = min(self.chunk_size, total_strategies - processed)
            
            # Generate random genomes
            genomes = self.generate_random_population(batch_size)
            
            # Evaluate
            scores = self.evaluate_batch(genomes)
            
            # Store
            all_scores.append(scores)
            all_genomes_hours.append(genomes.hours)
            all_genomes_mins.append(genomes.minutes)
            all_genomes_holds.append(genomes.hold_bars)
            all_genomes_dirs.append(genomes.directions)
            
            processed += batch_size
            
            if processed % 10000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                print(f"[GPU] Processed {processed:,} | Rate: {rate:,.0f}/sec")
                
        # Concatenate all
        final_scores = torch.cat(all_scores)
        final_genomes = StrategyGenome(
            hours=torch.cat(all_genomes_hours),
            minutes=torch.cat(all_genomes_mins),
            hold_bars=torch.cat(all_genomes_holds),
            directions=torch.cat(all_genomes_dirs)
        )
        
        total_time = time.time() - t0
        final_rate = total_strategies / total_time
        
        print(f"\n[GPU Backtester] COMPLETE")
        print(f"[GPU Backtester] Total: {total_strategies:,} strategies in {total_time:.2f}s")
        print(f"[GPU Backtester] Rate: {final_rate:,.0f} strategies/second")
        
        return final_scores, final_genomes


def benchmark():
    """Run benchmark to measure GPU performance."""
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    output_dir = base_dir / "output/gpu_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    backtester = PureGPUBacktester(str(data_path), chunk_size=10000)
    
    # Warm-up
    print("\n[Benchmark] Warm-up run...")
    _ = backtester.run_full_search(10000)
    
    # FULL SCALE: 100 Million Strategies
    print("\n" + "=" * 60)
    print("[Benchmark] LAUNCHING 100 MILLION STRATEGY SEARCH")
    print("=" * 60)
    
    TOTAL_STRATEGIES = 100_000_000
    
    scores, genomes = backtester.run_full_search(TOTAL_STRATEGIES)
    
    # Find Top 100 Best
    k = min(100, len(scores))
    top_vals, top_indices = torch.topk(scores, k)
    
    results = []
    for i in range(k):
        idx = top_indices[i]
        results.append({
            'rank': i + 1,
            'pnl': top_vals[i].item(),
            'hour': genomes.hours[idx].item(),
            'minute': genomes.minutes[idx].item(),
            'hold_bars': genomes.hold_bars[idx].item(),
            'direction': 'LONG' if genomes.directions[idx].item() == 1 else 'SHORT'
        })
    
    # Save to CSV
    import pandas as pd
    df_results = pd.DataFrame(results)
    out_path = output_dir / "top_100_strategies_100M.csv"
    df_results.to_csv(out_path, index=False)
    
    print(f"\n[Benchmark] TOP 100 SAVED TO: {out_path}")
    print(f"\n[Benchmark] TOP 5 STRATEGIES:")
    print(df_results.head().to_string())
    
    # Also print the absolute best
    best = results[0]
    print(f"\n{'='*60}")
    print(f"GLOBAL OPTIMUM (100 Million Search):")
    print(f"  Time: {best['hour']:02d}:{best['minute']:02d}")
    print(f"  Hold: {best['hold_bars']} bars")
    print(f"  Direction: {best['direction']}")
    print(f"  PnL: {best['pnl']:.2f} pts")
    print(f"{'='*60}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires GPU.")
        exit(1)
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    benchmark()
