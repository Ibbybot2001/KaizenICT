
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
import itertools
import heapq
from dataclasses import dataclass, asdict

# Force Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Titan Search running on: {DEVICE}")

@dataclass
class StratResult:
    config: dict
    total_return: float
    win_rate: float
    trades: int
    score: float  # Custom ranking metric

class TitanSearch:
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        
        # Track Top 20 per Hour to find diversity
        self.hours = range(9, 17)
        self.top_results_by_hour = {h: [] for h in self.hours}
        self.counter = 0
        
    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        df = pd.read_parquet(self.data_path)
        self.times = pd.to_datetime(df.index)
        
        # Tensors
        self.closes = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        self.opens = torch.tensor(df['open'].values, dtype=torch.float32, device=DEVICE)
        self.highs = torch.tensor(df['high'].values, dtype=torch.float32, device=DEVICE)
        self.lows = torch.tensor(df['low'].values, dtype=torch.float32, device=DEVICE)
        
        # Features
        self.hours = torch.tensor(self.times.hour.values, dtype=torch.int32, device=DEVICE)
        self.minutes = torch.tensor(self.times.minute.values, dtype=torch.int32, device=DEVICE)
        
        # Pre-calc Returns for various Hold Times
        self.hold_returns = {}
        self.hold_times = [5, 10, 15, 30, 60]
        
        for k in self.hold_times:
            # Return k bars later.
            # Shift closes backward by k.
            future_close = torch.roll(self.closes, -k)
            ret = future_close - self.closes
            # Mask the last k bars (wraparound) to 0
            ret[-k:] = 0
            self.hold_returns[k] = ret
        
    def get_configs_generator(self):
        """Yields batches of configurations."""
        # --- Variables (Optimized for Hold Time) ---
        hours = range(9, 17) # 09:00 to 16:00 (8)
        minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55] # (12)
        
        hold_times = self.hold_times # (5) values
        
        directions = [1, -1] # (2)
        offsets = [0, 1, 2, 3, 4, 5] # (6)
        dows = [0, 1, 2, 3, 4, 7] # (6)
        trends = [-1, 0, 1] # (3)
        
        # Total: 8 * 12 * 5 * 2 * 6 * 6 * 3 = ~103,680 Permutations.
        # This is extremely fast (<10s).
        # We can increase density or add more filters if needed.
        # Let's keep it lean to get the "3-5 trades" fast.
        
        iterator = itertools.product(hours, minutes, hold_times, directions, offsets, dows, trends)
        
        batch = []
        BATCH_SIZE = 10000
        
        for h, m, hold, d, off, dow, tr in iterator:
            cfg = {
                'hour': h, 'min': m, 'hold': hold, 'dir': d, 
                'offset': off, 'dow': dow, 'trend': tr
            }
            batch.append(cfg)
            if len(batch) >= BATCH_SIZE:
                yield batch
                batch = []
        if batch:
            yield batch

    def run_search(self):
        print("Starting Titan Search (Multi-Bar Hold Mode)...")
        t0 = time.time()
        
        # Pre-calc Trend Filter (SMA50)
        cs = self.closes.cumsum(0)
        cs_shift = torch.roll(cs, 50)
        sma50 = (cs - cs_shift) / 50
        trend_up = self.closes > sma50
        
        total_processed = 0
        
        for i, batch in enumerate(self.get_configs_generator()):
            batch_results_by_hour = {h: [] for h in self.hours}
            
            for cfg in batch:
                # 1. Time Mask
                mask = (self.hours == cfg['hour']) & (self.minutes == cfg['min'])
                
                # 2. Day of Week Filter - Simplified Skip for speed in this demo
                
                # 3. Offset
                if cfg['offset'] > 0:
                    mask = torch.roll(mask, cfg['offset'])
                    
                # 4. Trend Filter
                if cfg['trend'] != -1:
                    if cfg['trend'] == 1:
                        mask = mask & trend_up
                    else:
                        mask = mask & (~trend_up)
                
                # 5. Extract Returns
                # Use the pre-calc hold return
                raw_returns = self.hold_returns[cfg['hold']]
                hits = torch.masked_select(raw_returns, mask)
                
                if hits.numel() == 0:
                    continue
                    
                # 6. Apply Direction
                returns = hits * cfg['dir']
                
                # 7. Apply Approximate SL/TP expectancy
                # For Fixed Hold, cost is impact.
                cost = 0.5 # 2 ticks
                net_returns = returns - cost
                
                total_ret = net_returns.sum().item()
                
                # Score = Total Return
                if total_ret > 0:
                    batch_results_by_hour[cfg['hour']].append((total_ret, cfg))
            
            # Update Top 20 Per Hour
            for h in self.hours:
                for score, cfg in batch_results_by_hour[h]:
                    self.counter += 1
                    target_heap = self.top_results_by_hour[h]
                    if len(target_heap) < 20: # Keep top 20 diverse options per hour
                        heapq.heappush(target_heap, (score, self.counter, cfg))
                    else:
                        heapq.heappushpop(target_heap, (score, self.counter, cfg))
            
            total_processed += len(batch)
            if i % 10 == 0:
                elapsed = time.time() - t0
                rate = total_processed / elapsed
                
                # Show Best of a sample hour
                best_15 = self.top_results_by_hour[15][-1][0] if self.top_results_by_hour[15] else 0.0
                best_09 = self.top_results_by_hour[9][-1][0] if self.top_results_by_hour[9] else 0.0
                print(f"Processed {total_processed}. Speed: {rate:.0f}/s. Best 09h: {best_09:.1f}, Best 15h: {best_15:.1f}")

        # Save Results
        print("\nSaving Top Hourly Diversity...")
        results_data = []
        
        for h in self.hours:
            final_list = sorted(self.top_results_by_hour[h], key=lambda x: x[0], reverse=True)
            for score, _, cfg in final_list:
                row = cfg.copy()
                row['score_pnl'] = score
                results_data.append(row)
            
        df_res = pd.DataFrame(results_data)
        df_res.to_csv(self.output_dir / "titan_top_100_diverse.csv", index=False)
        print("Done.")

if __name__ == "__main__":
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    output_dir = base_dir / "output/titan"
    
    titan = TitanSearch(str(data_path), str(output_dir))
    titan.run_search()
