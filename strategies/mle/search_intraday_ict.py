"""
Intraday ICT Search (Excluding 3pm)
Focus: London, NY AM, Silver Bullet
"""

import torch
import time
from strategies.mle.ict_gpu_engine_v2 import ICTEngineV2, ICTGenomeV2, DEVICE, NUM_HOLDS

class IntradayICTEngine(ICTEngineV2):
    def generate_random_population(self, size: int) -> ICTGenomeV2:
        # Allowed Pattern Indices (No 15:00 specific patterns)
        # 0-7, 10-14 (Excluding 8:15:00, 9:15:01, 15:Any)
        allowed_patterns = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14], device=DEVICE)
        p_idx = allowed_patterns[torch.randint(0, len(allowed_patterns), (size,), device=DEVICE)]
        
        # Allowed Time Indices (No Power Hour, No Full Day)
        # 0: Lon Open, 1: Lon Sess, 2: NY Pre, 3: NY Open, 4: SB AM, 5: Lunch, 6: SB PM
        allowed_times = torch.tensor([0, 1, 2, 3, 4, 5, 6], device=DEVICE)
        t_idx = allowed_times[torch.randint(0, len(allowed_times), (size,), device=DEVICE)]
        
        return ICTGenomeV2(
            pattern_idx=p_idx,
            time_window_idx=t_idx,
            hold_idx=torch.randint(0, NUM_HOLDS, (size,), device=DEVICE),
            direction=torch.randint(0, 2, (size,), device=DEVICE).to(torch.float16) * 2 - 1
        )

def run_intraday_search():
    base_dir = "C:/Users/CEO/ICT reinforcement"
    data_path = f"{base_dir}/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    print("Initializing Intraday Engine...")
    engine = IntradayICTEngine(data_path, chunk_size=50000)
    
    print("\n[Search] Launching 5 Million 'Normal' Intraday Strategies...")
    scores, genomes = engine.run_search(5_000_000)
    
    print(f"\nTOP 10 NORMAL INTRADAY STRATEGIES (No 3pm):")
    
    patterns = ['FVG_Bull', 'FVG_Bear', 'Disp_Up', 'Disp_Down', '09:45', '10:00', 
                '10:15_SB', '14:00', '15:00', '15:01', 'Gap_Up', 'Gap_Down',
                'Bull_Candle', 'Bear_Candle', 'High_Vol', 'Any']
    times = ['London_Open', 'London', 'NY_Pre', 'NY_Open', 'SB_AM', 'Lunch', 
             'SB_PM', 'Power_Hour', 'Full_NY', 'All_Day']
    holds = [5, 10, 15, 30, 60, 120]
    
    out_path = f"{base_dir}/output/intraday_results_vol.txt"
    with open(out_path, "w") as f:
        f.write("TOP 10 NORMAL INTRADAY STRATEGIES (With Trade Counts):\n")
        f.write("-" * 60 + "\n")
        
        for i in range(10):
            p_idx = genomes.pattern_idx[i].item()
            t_idx = genomes.time_window_idx[i].item()
            h_idx = genomes.hold_idx[i].item()
            dir_val = genomes.direction[i].item()
            
            # Re-evaluate to get trade count
            mini_genome = ICTGenomeV2(
                pattern_idx=torch.tensor([p_idx], device=DEVICE),
                time_window_idx=torch.tensor([t_idx], device=DEVICE),
                hold_idx=torch.tensor([h_idx], device=DEVICE),
                direction=torch.tensor([dir_val], device=DEVICE)
            )
            
            pat_mask = engine.pattern_masks[p_idx]
            time_mask = engine.time_masks[t_idx]
            entry_mask = pat_mask & time_mask
            num_trades = entry_mask.sum().item()
            
            p = patterns[p_idx]
            t = times[t_idx]
            h = holds[h_idx]
            d = 'LONG' if dir_val > 0 else 'SHORT'
            s = scores[i].item()
            
            tpd = num_trades / 20.0
            
            line = f"#{i+1}: {p} | {t} | {h}bar | {d} | PnL: {s:.1f} | Trades: {num_trades} ({tpd:.1f}/day)"
            print(line)
            f.write(line + "\n")
            
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_intraday_search()
    else:
        print("CUDA required")
