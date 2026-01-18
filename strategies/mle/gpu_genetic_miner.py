
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
import random

# Device Configuration (Will use CPU if CUDA not found, but warns user)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPUGeneticMiner:
    def __init__(self, data_path, population_size=10000, generations=100):
        self.data_path = data_path
        self.pop_size = population_size
        self.generations = generations
        self.load_data()
        
        # Genome Definition:
        # 0: Hour (0-23)
        # 1: Minute (0-59)
        # 2: HoldTime (Bars)
        # 3: Direction (1, -1)
        # 4: SMA_Period (0=None, 1=20, 2=50, 3=200) - Index
        # 5: Volatility_Filter (0=None, 1=High, 2=Low) - Index
        self.genome_size = 6
        
    def load_data(self):
        print(f"Loading data from {self.data_path} on {DEVICE}...")
        t0 = time.time()
        df = pd.read_parquet(self.data_path)
        
        # Base Tensors
        self.closes = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        self.hours = torch.tensor(pd.to_datetime(df.index).hour.values, dtype=torch.int32, device=DEVICE)
        self.minutes = torch.tensor(pd.to_datetime(df.index).minute.values, dtype=torch.int32, device=DEVICE)
        
        # Pre-Calc Features (To avoid re-computing in loop)
        # 1. EMAs/SMAs
        # Simple conv implementation for speed
        def calc_sma(data, window):
            cs = data.cumsum(0)
            cs_shift = torch.roll(cs, window)
            sma = (cs - cs_shift) / window
            sma[:window] = data[:window] # Fix edge
            return sma
            
        self.ma_20 = calc_sma(self.closes, 20)
        self.ma_50 = calc_sma(self.closes, 50)
        self.ma_200 = calc_sma(self.closes, 200)
        
        # 2. Volatility (ATR-ish)
        # High - Low approx
        highs = torch.tensor(df['high'].values, dtype=torch.float32, device=DEVICE)
        lows = torch.tensor(df['low'].values, dtype=torch.float32, device=DEVICE)
        tr = highs - lows
        self.atr_14 = calc_sma(tr, 14)
        
        # Pre-calc Returns for max Hold Time (e.g. up to 60)
        # We can't pre-calc ALL holds efficiently if continuous.
        # But we can pre-calc a lookup table for common holds: 5, 10, 15, 30, 60
        self.valid_holds = [5, 10, 15, 30, 60]
        self.hold_returns = {}
        for h in self.valid_holds:
            fut = torch.roll(self.closes, -h)
            ret = fut - self.closes
            ret[-h:] = 0
            self.hold_returns[h] = ret
            
        print(f"Data Loaded in {time.time()-t0:.2f}s")
        
    def init_population(self):
        """Create random genomes."""
        # Random tensors
        hours = torch.randint(0, 24, (self.pop_size,), device=DEVICE)
        mins = torch.randint(0, 60, (self.pop_size,), device=DEVICE)
        
        # Random index into valid_holds
        hold_indices = torch.randint(0, len(self.valid_holds), (self.pop_size,), device=DEVICE)
        
        dirs = torch.randint(0, 2, (self.pop_size,), device=DEVICE) * 2 - 1 # -1 or 1
        
        ma_idx = torch.randint(0, 4, (self.pop_size,), device=DEVICE) # 0-3
        vol_idx = torch.randint(0, 3, (self.pop_size,), device=DEVICE) # 0-2
        
        # Stack
        population = torch.stack([hours, mins, hold_indices, dirs, ma_idx, vol_idx], dim=1)
        return population
        
    def evaluate(self, population):
        """
        Evaluate full population in parallel.
        This is the complex part: We need to score 10,000 strategies against 500,000 bars.
        
        Naive Approach: Loop population.
        Vectorized Approach: Broadcast? [Pop, Time]. Too big memory (10k * 500k * 4B = 20GB).
        
        Hybrid: Process small batches of Population against full time.
        """
        scores = torch.zeros(self.pop_size, device=DEVICE)
        
        # We can't broadcast full pop. Loop through chunks of Pop.
        CHUNK_SIZE = 100 # Evaluate 100 strats at a time vs full history
        
        # Pre-Calc Boolean Filters to speed up
        # MA Filters: 0None, 1(P>MA20), 2(P>MA50), 3(P>MA200)
        ma_filters = torch.stack([
            torch.ones_like(self.closes, dtype=torch.bool), # Dummy True
            self.closes > self.ma_20,
            self.closes > self.ma_50,
            self.closes > self.ma_200
        ])
        
        # Vol Filters: 0None, 1(ATR>Mean), 2(ATR<Mean)
        atr_mean = self.atr_14.mean()
        vol_filters = torch.stack([
            torch.ones_like(self.closes, dtype=torch.bool),
            self.atr_14 > atr_mean,
            self.atr_14 < atr_mean
        ])
        
        for i in range(0, self.pop_size, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, self.pop_size)
            batch = population[i:end] # [CHUNK, 6]
            
            # For each strat in batch:
            # Reconstruct mask
            # This inner loop is still Python but 10,000/100 = 100 iters. 100 iters * 500k ops is fast on GPU.
            
            for j in range(len(batch)):
                strat = batch[j]
                h, m, h_idx, d, ma_i, vol_i = strat[0], strat[1], strat[2], strat[3], strat[4], strat[5]
                
                # 1. Time Mask
                mask = (self.hours == h) & (self.minutes == m)
                
                # 2. MA Filter
                if ma_i > 0:
                    mask = mask & ma_filters[ma_i]
                    
                # 3. Vol Filter
                if vol_i > 0:
                    mask = mask & vol_filters[vol_i]
                    
                # 4. Returns
                hold_val = self.valid_holds[h_idx]
                raw_rets = self.hold_returns[hold_val]
                
                hits = torch.masked_select(raw_rets, mask)
                
                if hits.numel() == 0:
                    scores[i+j] = -9999
                    continue
                    
                # PnL
                net = hits * d - 0.5 # Cost
                total = net.sum()
                
                scores[i+j] = total
                
        return scores

    def mutate(self, elites):
        """Create next gen from elites."""
        num_elites = len(elites)
        needed = self.pop_size - num_elites
        
        # Clone elites to fill
        indices = torch.randint(0, num_elites, (needed,), device=DEVICE)
        next_gen = elites[indices].clone()
        
        # Mutate: Randomly change columns with 10% prob
        prob = 0.1
        mask = torch.rand_like(next_gen, dtype=torch.float32) < prob
        
        # Generate random noise for all params
        rand_h = torch.randint(0, 24, (needed,), device=DEVICE)
        rand_m = torch.randint(0, 60, (needed,), device=DEVICE)
        rand_ho = torch.randint(0, len(self.valid_holds), (needed,), device=DEVICE)
        rand_d = torch.randint(0, 2, (needed,), device=DEVICE) * 2 - 1
        rand_ma = torch.randint(0, 4, (needed,), device=DEVICE)
        rand_vol = torch.randint(0, 3, (needed,), device=DEVICE)
        
        noise = torch.stack([rand_h, rand_m, rand_ho, rand_d, rand_ma, rand_vol], dim=1)
        
        # Apply
        next_gen = torch.where(mask, noise, next_gen)
        
        return torch.cat([elites, next_gen], dim=0)

    def run(self):
        print(f"Starting Genetic Miner (Pop: {self.pop_size}, Gens: {self.generations})")
        pop = self.init_population()
        
        for g in range(self.generations):
            t0 = time.time()
            scores = self.evaluate(pop)
            
            # Select Top 10%
            k = int(self.pop_size * 0.1)
            top_vals, top_indices = torch.topk(scores, k)
            
            best_score = top_vals[0].item()
            best_strat = pop[top_indices[0]] # Tensor
            
            avg_score = top_vals.mean().item()
            
            # Print Stats
            print(f"Gen {g+1}/{self.generations} | Time: {time.time()-t0:.2f}s | Best: {best_score:.2f} | Avg Elite: {avg_score:.2f}")
            print(f"   Best Genome: Time={best_strat[0]}:{best_strat[1]}, HoldIdx={best_strat[2]}, Dir={best_strat[3]}")
            
            # Evolve
            elites = pop[top_indices]
            pop = self.mutate(elites)
            
        print("Done.")

if __name__ == "__main__":
    # Check CUDA again
    if not torch.cuda.is_available():
        print("WARNING: CUDA Not Available. Running on CPU (Will be slow for Genetic Mining).")
        input("Press Enter to continue anyway (or Ctrl+C to abort)...")
        
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    miner = GPUGeneticMiner(str(data_path), population_size=10000, generations=100) # Full Scale for GPU
    miner.run()
