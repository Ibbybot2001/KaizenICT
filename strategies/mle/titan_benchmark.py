
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from itertools import product
import random

# Force Device (Check availability, default to CPU for benchmark if fails)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Titan Benchmark running on: {DEVICE}")

class TitanBenchmark:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        t0 = time.time()
        df = pd.read_parquet(self.data_path)
        self.times = pd.to_datetime(df.index)
        
        # Move to Tensor
        self.closes = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        self.opens = torch.tensor(df['open'].values, dtype=torch.float32, device=DEVICE)
        self.highs = torch.tensor(df['high'].values, dtype=torch.float32, device=DEVICE)
        self.lows = torch.tensor(df['low'].values, dtype=torch.float32, device=DEVICE)
        
        # Time Features
        self.hours = torch.tensor(self.times.hour.values, dtype=torch.int32, device=DEVICE)
        self.minutes = torch.tensor(self.times.minute.values, dtype=torch.int32, device=DEVICE)
        
        # Pre-calc returns for speed
        self.next_close = torch.roll(self.closes, -1)
        self.price_change = self.next_close - self.closes
        
        print(f"Data Loaded in {time.time() - t0:.2f}s. Rows: {len(df)}")

    def generate_random_configs(self, n=1000):
        """Generate N random strategy configurations."""
        configs = []
        for _ in range(n):
            cfg = {
                'concept': random.choice(['C1', 'C3', 'C8', 'C14']),
                'trigger_hour': random.choice([9, 10, 14, 15]),
                'trigger_min': random.choice([0, 15, 30, 45]),
                'stop_loss': random.choice([5, 10, 20, 50]),
                'take_profit': random.choice([10, 20, 40, 100]),
                'direction': random.choice([1, -1]) # 1=Long, -1=Short (Twist: Inverse)
            }
            configs.append(cfg)
        return configs

    def backtest_single(self, cfg):
        """Run a single strategy variant (Vectorized Entry, Iterative Exit or Simplified)."""
        # simplified vectorized pnl for speed benchmark
        # Logic: Enter at Trigger Time. Hold until TP or SL hit?
        # Vectorized Exit is hard. Let's do "Fixed Hold" or "Bar Completion" for benchmark speed.
        # Actually, for the 10M run, we want 'Realistic' exits.
        # Let's approximate: 
        # Entry = Trigger Time.
        # Exit = Close[t+exit_bars] or PnL check.
        
        # Trigger Mask
        time_mask = (self.hours == cfg['trigger_hour']) & (self.minutes == cfg['trigger_min'])
        
        # Signals
        signals = torch.zeros_like(self.closes)
        signals[time_mask] = cfg['direction']
        
        # PnL (Simplified: Trade return of the *Next Bar* only for benchmark)
        # In real Titan, we'd simulate the full trade path.
        trade_returns = signals * self.price_change
        
        # Metrics
        total_pnl = trade_returns.sum().item()
        return total_pnl

    def run_benchmark(self, n_tests=1000):
        print(f"\n--- Starting Titan Benchmark ({n_tests} Tests) ---")
        configs = self.generate_random_configs(n_tests)
        
        t_start = time.time()
        
        results = []
        for i, cfg in enumerate(configs):
            pnl = self.backtest_single(cfg)
            results.append(pnl)
            
            if i % 100 == 0:
                print(f"Processed {i}/{n_tests}...", end="\r")
                
        t_total = time.time() - t_start
        
        avg_time = t_total / n_tests
        est_10m = (avg_time * 10_000_000) / 3600 # hours
        
        output_str = f"""
Benchmark Complete.
Total Time: {t_total:.4f}s
Time Per Test: {avg_time:.6f}s
Tests Per Second: {1/avg_time:.1f}
--------------------------------------------------
ESTIMATED TIME FOR 10,000,000 TESTS (Single Core):
   {est_10m:.2f} HOURS
   ({est_10m/24:.2f} DAYS)
--------------------------------------------------
Device: {DEVICE}
"""
        print(output_str)
        with open("strategies/mle/benchmark_output.txt", "w") as f:
            f.write(output_str)
        
        if DEVICE.type == 'cpu':
            print("WARNING: Running on CPU. Massive slowdown expected.")

if __name__ == "__main__":
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    bencher = TitanBenchmark(str(data_path))
    bencher.run_benchmark(n_tests=1000)
