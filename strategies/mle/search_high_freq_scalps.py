"""
High Frequency Scalping Search
Focus: Short Holds (5-15 mins), Broad Sessions, FVG/Disp Patterns
Goal: High Trade Count + Positive Expectancy
"""

import torch
import time
from strategies.mle.ict_gpu_engine_v2 import ICTEngineV2, ICTGenomeV2, DEVICE, NUM_HOLDS

class ScalpingICTEngine(ICTEngineV2):
    def generate_random_population(self, size: int) -> ICTGenomeV2:
        # Scalping Patterns: FVG, Displacement, Candles, High Vol (No specific times like 9:45)
        # 0: FVG Bull, 1: FVG Bear, 2: Disp Up, 3: Disp Down
        # 12: Bull Candle, 13: Bear Candle, 14: High Vol
        allowed_patterns = torch.tensor([0, 1, 2, 3, 12, 13, 14], device=DEVICE)
        p_idx = allowed_patterns[torch.randint(0, len(allowed_patterns), (size,), device=DEVICE)]
        
        # Broad Sessions Only (No single minutes)
        # 1: London Sess, 3: NY Open, 5: Lunch, 6: SB PM, 8: Full NY, 9: All Day
        allowed_times = torch.tensor([1, 3, 5, 6, 8, 9], device=DEVICE)
        t_idx = allowed_times[torch.randint(0, len(allowed_times), (size,), device=DEVICE)]
        
        # Short Holds Only (5, 10, 15 bars)
        # 0: 5 bars, 1: 10 bars, 2: 15 bars
        allowed_holds = torch.tensor([0, 1, 2], device=DEVICE)
        h_idx = allowed_holds[torch.randint(0, len(allowed_holds), (size,), device=DEVICE)]
        
        return ICTGenomeV2(
            pattern_idx=p_idx,
            time_window_idx=t_idx,
            hold_idx=h_idx,
            direction=torch.randint(0, 2, (size,), device=DEVICE).to(torch.float16) * 2 - 1
        )

def run_scalping_search():
    base_dir = "C:/Users/CEO/ICT reinforcement"
    data_path = f"{base_dir}/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    print("Initializing Scalping Engine...")
    engine = ScalpingICTEngine(data_path, chunk_size=50000)
    
    print("\n[Search] Launching 1 Million Scalping Strategies (Quick Probe)...")
    scores, genomes = engine.run_search(1_000_000)
    
    out_path = f"{base_dir}/output/scalping_results.txt"
    
    patterns = ['FVG_Bull', 'FVG_Bear', 'Disp_Up', 'Disp_Down', '09:45', '10:00', 
                '10:15_SB', '14:00', '15:00', '15:01', 'Gap_Up', 'Gap_Down',
                'Bull_Candle', 'Bear_Candle', 'High_Vol', 'Any']
    times = ['London_Open', 'London', 'NY_Pre', 'NY_Open', 'SB_AM', 'Lunch', 
             'SB_PM', 'Power_Hour', 'Full_NY', 'All_Day']
    holds = [5, 10, 15, 30, 60, 120]
    
    with open(out_path, "w") as f:
        f.write("TOP 10 HIGH FREQUENCY SCALPING STRATEGIES (With Trade Counts):\n")
        f.write("-" * 60 + "\n")
        
        print(f"\nTOP 10 HIGH FREQUENCY SCALPING STRATEGIES:")
        for i in range(10):
            p_idx = genomes.pattern_idx[i].item()
            t_idx = genomes.time_window_idx[i].item()
            h_idx = genomes.hold_idx[i].item()
            dir_val = genomes.direction[i].item()
            
            # Re-evaluate to get trade count
            # Create a mini-batch of 1
            mini_genome = ICTGenomeV2(
                pattern_idx=torch.tensor([p_idx], device=DEVICE),
                time_window_idx=torch.tensor([t_idx], device=DEVICE),
                hold_idx=torch.tensor([h_idx], device=DEVICE),
                direction=torch.tensor([dir_val], device=DEVICE)
            )
            
            # Manual Eval to get count
            pat_mask = engine.pattern_masks[p_idx]
            time_mask = engine.time_masks[t_idx]
            entry_mask = pat_mask & time_mask
            num_trades = entry_mask.sum().item()
            
            p = patterns[p_idx]
            t = times[t_idx]
            h = holds[h_idx]
            d = 'LONG' if dir_val > 0 else 'SHORT'
            s = scores[i].item()
            
            # Approx trades per day (Jan 2025 has ~20 trading days)
            tpd = num_trades / 20.0
            
            line = f"#{i+1}: {p} | {t} | {h}bar | {d} | PnL: {s:.1f} | Trades: {num_trades} ({tpd:.1f}/day)"
            print(line)
            f.write(line + "\n")
            
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_scalping_search()
