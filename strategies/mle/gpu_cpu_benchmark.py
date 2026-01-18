
import torch
import pandas as pd
import time
from pathlib import Path

def benchmark_device(device_name):
    """Run identical workload on specified device."""
    device = torch.device(device_name)
    
    # Load Data
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    df = pd.read_parquet(data_path)
    closes = torch.tensor(df['close'].values, dtype=torch.float32, device=device)
    hours = torch.tensor(pd.to_datetime(df.index).hour.values, dtype=torch.int32, device=device)
    minutes = torch.tensor(pd.to_datetime(df.index).minute.values, dtype=torch.int32, device=device)
    
    # Warm-up
    _ = closes * 2
    if device_name == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark: 10,000 Strategy Evaluations (Vectorized)
    NUM_STRATEGIES = 10000
    
    # Random Strategy Configs (on device)
    strat_hours = torch.randint(9, 17, (NUM_STRATEGIES,), device=device)
    strat_mins = torch.randint(0, 60, (NUM_STRATEGIES,), device=device)
    strat_dirs = torch.randint(0, 2, (NUM_STRATEGIES,), device=device) * 2 - 1
    
    # Pre-calc Returns
    next_close = torch.roll(closes, -30)
    returns = next_close - closes
    
    # Start Timer
    t0 = time.time()
    
    scores = torch.zeros(NUM_STRATEGIES, device=device)
    
    for i in range(NUM_STRATEGIES):
        mask = (hours == strat_hours[i]) & (minutes == strat_mins[i])
        hits = torch.masked_select(returns, mask)
        if hits.numel() > 0:
            scores[i] = (hits * strat_dirs[i]).sum()
    
    if device_name == 'cuda':
        torch.cuda.synchronize()
        
    elapsed = time.time() - t0
    rate = NUM_STRATEGIES / elapsed
    
    return elapsed, rate

if __name__ == "__main__":
    results = []
    results.append("=" * 60)
    results.append("GPU vs CPU Benchmark (10,000 Strategy Evaluations)")
    results.append("=" * 60)
    
    # CPU
    print("\nRunning on CPU...")
    cpu_time, cpu_rate = benchmark_device('cpu')
    results.append(f"CPU: {cpu_time:.2f}s | {cpu_rate:.0f} strategies/sec")
    
    # GPU
    if torch.cuda.is_available():
        print("\nRunning on GPU (RTX 4080 Super)...")
        gpu_time, gpu_rate = benchmark_device('cuda')
        results.append(f"GPU: {gpu_time:.2f}s | {gpu_rate:.0f} strategies/sec")
        
        speedup = cpu_time / gpu_time
        results.append(f"\n>>> GPU SPEEDUP: {speedup:.1f}x FASTER <<<")
    else:
        results.append("GPU Not Available")
    
    results.append("=" * 60)
    
    output = "\n".join(results)
    print(output)
    
    with open("strategies/mle/benchmark_results.txt", "w") as f:
        f.write(output)
