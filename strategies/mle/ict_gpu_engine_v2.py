"""
ICT Pattern GPU Engine V2 (Optimized)
Pre-computed Pattern Masks + Tensor Indexing
Target: 100,000+ Strategies/Second
"""

import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pattern and Filter counts
NUM_PATTERNS = 16
NUM_TIME_WINDOWS = 10
NUM_HOLDS = 6

@dataclass 
class ICTGenomeV2:
    pattern_idx: torch.Tensor      # [S] - 0-15
    time_window_idx: torch.Tensor  # [S] - 0-9
    hold_idx: torch.Tensor         # [S] - 0-5 (maps to [5,10,15,30,60,120])
    direction: torch.Tensor        # [S] - 1 or -1


class ICTEngineV2:
    """
    Optimized ICT Engine with pre-computed pattern masks.
    All indexing via tensors - minimal Python loops.
    """
    
    def __init__(self, data_path: str, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        self.hold_options = [5, 10, 15, 30, 60, 120]
        self.load_data(data_path)
        self.precompute_all_masks()
        
    def load_data(self, data_path: str):
        print(f"[ICT V2] Loading data...")
        df = pd.read_parquet(data_path)
        times = pd.to_datetime(df.index)
        
        self.opens = torch.tensor(df['open'].values, dtype=torch.float16, device=DEVICE)
        self.highs = torch.tensor(df['high'].values, dtype=torch.float16, device=DEVICE)
        self.lows = torch.tensor(df['low'].values, dtype=torch.float16, device=DEVICE)
        self.closes = torch.tensor(df['close'].values, dtype=torch.float16, device=DEVICE)
        self.T = len(self.closes)
        
        self.hours = torch.tensor(times.hour.values, dtype=torch.int16, device=DEVICE)
        self.minutes = torch.tensor(times.minute.values, dtype=torch.int16, device=DEVICE)
        
        print(f"[ICT V2] Loaded {self.T:,} bars")
        
    def precompute_all_masks(self):
        """Pre-compute ALL pattern and time window masks."""
        print("[ICT V2] Pre-computing all masks...")
        t0 = time.time()
        
        # ============================================================
        # PATTERN MASKS: [NUM_PATTERNS, T]
        # ============================================================
        pattern_masks = []
        
        # Pattern 0: FVG Bullish (Large)
        high_2back = torch.roll(self.highs, 2)
        fvg_bull = (self.lows - high_2back) >= 3
        pattern_masks.append(fvg_bull)
        
        # Pattern 1: FVG Bearish (Large)
        low_2back = torch.roll(self.lows, 2)
        fvg_bear = (low_2back - self.highs) >= 3
        pattern_masks.append(fvg_bear)
        
        # Pattern 2: Displacement Up
        body = self.closes - self.opens
        disp_up = body > 10
        pattern_masks.append(disp_up)
        
        # Pattern 3: Displacement Down
        disp_down = body < -10
        pattern_masks.append(disp_down)
        
        # Pattern 4: 09:45 Time Entry
        pattern_masks.append((self.hours == 9) & (self.minutes == 45))
        
        # Pattern 5: 10:00 Time Entry
        pattern_masks.append((self.hours == 10) & (self.minutes == 0))
        
        # Pattern 6: 10:15 Time Entry (Silver Bullet)
        pattern_masks.append((self.hours == 10) & (self.minutes == 15))
        
        # Pattern 7: 14:00 Time Entry
        pattern_masks.append((self.hours == 14) & (self.minutes == 0))
        
        # Pattern 8: 15:00 Time Entry (Power Hour)
        pattern_masks.append((self.hours == 15) & (self.minutes == 0))
        
        # Pattern 9: 15:01 Time Entry (Global Optimum!)
        pattern_masks.append((self.hours == 15) & (self.minutes == 1))
        
        # Pattern 10: Gap Up Open
        prev_close = torch.roll(self.closes, 1)
        gap_up = (self.opens - prev_close) > 5
        pattern_masks.append(gap_up)
        
        # Pattern 11: Gap Down Open
        gap_down = (prev_close - self.opens) > 5
        pattern_masks.append(gap_down)
        
        # Pattern 12: Strong Bullish Candle
        pattern_masks.append(body > 5)
        
        # Pattern 13: Strong Bearish Candle
        pattern_masks.append(body < -5)
        
        # Pattern 14: High Volume (proxy: large range)
        candle_range = self.highs - self.lows
        avg_range = candle_range.mean()
        pattern_masks.append(candle_range > avg_range * 1.5)
        
        # Pattern 15: Any Time (Always True for testing)
        pattern_masks.append(torch.ones(self.T, dtype=torch.bool, device=DEVICE))
        
        # Stack: [16, T]
        self.pattern_masks = torch.stack(pattern_masks)
        
        # ============================================================
        # TIME WINDOW MASKS: [NUM_TIME_WINDOWS, T]
        # ============================================================
        time_masks = []
        
        # 0: London Open 03:00-05:00
        time_masks.append((self.hours >= 3) & (self.hours < 5))
        
        # 1: London Session 03:00-08:00
        time_masks.append((self.hours >= 3) & (self.hours < 8))
        
        # 2: NY Pre-Market 08:00-09:30
        time_masks.append((self.hours == 8) | ((self.hours == 9) & (self.minutes < 30)))
        
        # 3: NY Open 09:30-10:00
        time_masks.append((self.hours == 9) & (self.minutes >= 30))
        
        # 4: Silver Bullet AM 10:00-11:00
        time_masks.append(self.hours == 10)
        
        # 5: Lunch 11:30-13:30
        time_masks.append(((self.hours == 11) & (self.minutes >= 30)) | 
                          (self.hours == 12) | 
                          ((self.hours == 13) & (self.minutes < 30)))
        
        # 6: Silver Bullet PM 14:00-15:00
        time_masks.append(self.hours == 14)
        
        # 7: Power Hour 15:00-16:00
        time_masks.append(self.hours == 15)
        
        # 8: Full NY 09:30-16:00
        time_masks.append(((self.hours == 9) & (self.minutes >= 30)) | 
                          ((self.hours >= 10) & (self.hours < 16)))
        
        # 9: All Day
        time_masks.append(torch.ones(self.T, dtype=torch.bool, device=DEVICE))
        
        # Stack: [10, T]
        self.time_masks = torch.stack(time_masks)
        
        # ============================================================
        # RETURNS: [NUM_HOLDS, T]
        # ============================================================
        returns_list = []
        for h in self.hold_options:
            future = torch.roll(self.closes, -h)
            ret = future - self.closes
            ret[-h:] = 0
            returns_list.append(ret)
        
        self.all_returns = torch.stack(returns_list)  # [6, T]
        
        print(f"[ICT V2] Masks computed in {time.time()-t0:.2f}s")
        print(f"[ICT V2] Pattern masks: {self.pattern_masks.shape}")
        print(f"[ICT V2] Time masks: {self.time_masks.shape}")
        print(f"[ICT V2] Returns: {self.all_returns.shape}")
        
    def generate_random_population(self, size: int) -> ICTGenomeV2:
        return ICTGenomeV2(
            pattern_idx=torch.randint(0, NUM_PATTERNS, (size,), device=DEVICE),
            time_window_idx=torch.randint(0, NUM_TIME_WINDOWS, (size,), device=DEVICE),
            hold_idx=torch.randint(0, NUM_HOLDS, (size,), device=DEVICE),
            direction=torch.randint(0, 2, (size,), device=DEVICE).to(torch.float16) * 2 - 1
        )
    
    def evaluate_batch(self, genomes: ICTGenomeV2) -> torch.Tensor:
        """
        Fully vectorized evaluation using pre-computed masks.
        """
        S = len(genomes.pattern_idx)
        
        # Index into pre-computed masks
        # pattern_masks[pattern_idx] -> [S, T]
        selected_patterns = self.pattern_masks[genomes.pattern_idx]  # [S, T]
        selected_times = self.time_masks[genomes.time_window_idx]    # [S, T]
        selected_returns = self.all_returns[genomes.hold_idx]        # [S, T]
        
        # Combine masks
        entry_mask = selected_patterns & selected_times  # [S, T]
        
        # Apply direction and compute returns
        dirs = genomes.direction.unsqueeze(1)  # [S, 1]
        trade_returns = entry_mask.float() * selected_returns * dirs  # [S, T]
        
        # Sum returns and subtract cost
        raw_pnl = trade_returns.sum(dim=1)  # [S]
        num_trades = entry_mask.sum(dim=1).float()
        cost = num_trades * 0.5
        
        scores = raw_pnl - cost
        
        return scores.float()
    
    def run_search(self, total_strategies: int) -> Tuple[torch.Tensor, ICTGenomeV2]:
        """Memory-efficient streaming search."""
        print(f"[ICT V2] Starting {total_strategies:,} strategies...")
        
        top_scores = torch.full((100,), -float('inf'), device=DEVICE)
        top_patterns = torch.zeros(100, dtype=torch.int64, device=DEVICE)
        top_times = torch.zeros(100, dtype=torch.int64, device=DEVICE)
        top_holds = torch.zeros(100, dtype=torch.int64, device=DEVICE)
        top_dirs = torch.zeros(100, dtype=torch.float16, device=DEVICE)
        
        t0 = time.time()
        processed = 0
        
        torch.cuda.synchronize()
        
        while processed < total_strategies:
            batch_size = min(self.chunk_size, total_strategies - processed)
            genomes = self.generate_random_population(batch_size)
            scores = self.evaluate_batch(genomes)
            
            # Merge with Top 100
            all_scores = torch.cat([top_scores, scores])
            all_patterns = torch.cat([top_patterns, genomes.pattern_idx])
            all_times = torch.cat([top_times, genomes.time_window_idx])
            all_holds = torch.cat([top_holds, genomes.hold_idx])
            all_dirs = torch.cat([top_dirs, genomes.direction])
            
            top_vals, top_idx = torch.topk(all_scores, 100)
            
            top_scores = top_vals
            top_patterns = all_patterns[top_idx]
            top_times = all_times[top_idx]
            top_holds = all_holds[top_idx]
            top_dirs = all_dirs[top_idx]
            
            processed += batch_size
            
            if processed % 1000000 == 0:
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                rate = processed / elapsed
                print(f"[ICT V2] {processed:,} | Rate: {rate:,.0f}/sec | Best: {top_scores[0].item():.1f}")
        
        torch.cuda.synchronize()
        total_time = time.time() - t0
        
        print(f"\n[ICT V2] COMPLETE: {total_strategies:,} in {total_time:.2f}s")
        print(f"[ICT V2] RATE: {total_strategies/total_time:,.0f} strategies/second")
        
        final_genomes = ICTGenomeV2(
            pattern_idx=top_patterns,
            time_window_idx=top_times,
            hold_idx=top_holds,
            direction=top_dirs
        )
        
        return top_scores, final_genomes


def benchmark():
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    engine = ICTEngineV2(str(data_path), chunk_size=50000)
    
    print("\n[Benchmark] Warm-up...")
    _ = engine.run_search(50000)
    
    print("\n[Benchmark] MAIN RUN: 10 Million ICT strategies")
    scores, genomes = engine.run_search(10_000_000)
    
    print(f"\nTOP 5 ICT STRATEGIES:")
    patterns = ['FVG_Bull', 'FVG_Bear', 'Disp_Up', 'Disp_Down', '09:45', '10:00', 
                '10:15_SB', '14:00', '15:00', '15:01', 'Gap_Up', 'Gap_Down',
                'Bull_Candle', 'Bear_Candle', 'High_Vol', 'Any']
    times = ['London_Open', 'London', 'NY_Pre', 'NY_Open', 'SB_AM', 'Lunch', 
             'SB_PM', 'Power_Hour', 'Full_NY', 'All_Day']
    holds = [5, 10, 15, 30, 60, 120]
    
    for i in range(5):
        p = patterns[genomes.pattern_idx[i].item()]
        t = times[genomes.time_window_idx[i].item()]
        h = holds[genomes.hold_idx[i].item()]
        d = 'LONG' if genomes.direction[i].item() > 0 else 'SHORT'
        s = scores[i].item()
        print(f"  #{i+1}: {p} | {t} | {h}bar | {d} | PnL: {s:.1f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        exit(1)
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    benchmark()
