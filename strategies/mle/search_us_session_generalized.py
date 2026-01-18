"""
Generalized US Session Search (09:30 - 16:00)
Focus: US Session Only, No Magic Minutes
Constraints: SL > 10pts, Duration < 2H
Goal: Robustness across session (PJ Style)
"""

import torch
import time
from strategies.mle.ict_gpu_engine_v2 import ICTEngineV2, ICTGenomeV2, DEVICE, NUM_HOLDS

class GeneralizedUSSessionEngine(ICTEngineV2):
    def generate_random_population(self, size: int) -> ICTGenomeV2:
        # Generalized Patterns (No Time Triggers)
        # 0: FVG Bull, 1: FVG Bear, 2: Disp Up, 3: Disp Down
        # 10: Gap Up, 11: Gap Down, 12: Bull Candle, 13: Bear Candle, 14: High Vol
        allowed_patterns = torch.tensor([0, 1, 2, 3, 10, 11, 12, 13, 14], device=DEVICE)
        p_idx = allowed_patterns[torch.randint(0, len(allowed_patterns), (size,), device=DEVICE)]
        
        # US Session Intervals Only (Broad Windows)
        # 3: NY Open (09:30-10:00), 4: SB AM (10:00-11:00), 5: Lunch (11:30-13:30)
        # 6: SB PM (14:00-15:00), 7: Power Hour (15:00-16:00), 8: Full NY
        # Note: We want generalized, so 'Full NY' is the best target if logic holds all day.
        # But we search all US windows to see if "All NY" is viable or if we must split.
        allowed_times = torch.tensor([3, 4, 5, 6, 7, 8], device=DEVICE)
        t_idx = allowed_times[torch.randint(0, len(allowed_times), (size,), device=DEVICE)]
        
        # Medium to Long Holds (15, 30, 60, 120 mins) - Matches User's 5-30m to 2H request
        # 2: 15, 3: 30, 4: 60, 5: 120
        allowed_holds = torch.tensor([2, 3, 4, 5], device=DEVICE)
        h_idx = allowed_holds[torch.randint(0, len(allowed_holds), (size,), device=DEVICE)]
        
        return ICTGenomeV2(
            pattern_idx=p_idx,
            time_window_idx=t_idx,
            hold_idx=h_idx,
            direction=torch.randint(0, 2, (size,), device=DEVICE).to(torch.float16) * 2 - 1
        )

def run_generalized_search():
    base_dir = "C:/Users/CEO/ICT reinforcement"
    data_path = f"{base_dir}/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    print("Initializing Generalized US Engine...")
    engine = GeneralizedUSSessionEngine(data_path, chunk_size=50000)
    
    print("\n[Search] Launching 2 Million Generalized US Strategies (Quick Probe)...")
    scores, genomes = engine.run_search(2_000_000)
    
    out_path = f"{base_dir}/output/generalized_us_results.txt"
    
    patterns = ['FVG_Bull', 'FVG_Bear', 'Disp_Up', 'Disp_Down', '09:45', '10:00', 
                '10:15_SB', '14:00', '15:00', '15:01', 'Gap_Up', 'Gap_Down',
                'Bull_Candle', 'Bear_Candle', 'High_Vol', 'Any']
    times = ['London_Open', 'London', 'NY_Pre', 'NY_Open', 'SB_AM', 'Lunch', 
             'SB_PM', 'Power_Hour', 'Full_NY', 'All_Day']
    holds = [5, 10, 15, 30, 60, 120]
    
    with open(out_path, "w") as f:
        f.write("TOP 10 GENERALIZED US SESSION STRATEGIES (09:30-16:00):\n")
        f.write("-" * 60 + "\n")
        
        print(f"\nTOP 10 GENERALIZED US SESSION STRATEGIES:")
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
        run_generalized_search()
