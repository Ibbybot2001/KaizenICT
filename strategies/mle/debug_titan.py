
import torch
import pandas as pd
import numpy as np
from pathlib import Path

# Force CPU for consistency
DEVICE = torch.device('cpu')

class DebugTitan:
    def __init__(self, data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        self.df = df
        self.times = pd.to_datetime(df.index)
        
        self.closes = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        self.hours = torch.tensor(self.times.hour.values, dtype=torch.int32, device=DEVICE)
        self.minutes = torch.tensor(self.times.minute.values, dtype=torch.int32, device=DEVICE)
        
        self.next_close = torch.roll(self.closes, -1)
        self.price_change = self.next_close - self.closes
        
        print(f"Data Loaded. Rows: {len(df)}")
        
    def test_config(self, hour, minute, direction):
        print(f"\nTesting Config: Hour={hour}, Min={minute}, Dir={direction}")
        
        mask = (self.hours == hour) & (self.minutes == minute)
        hits = torch.masked_select(self.price_change, mask)
        
        count = hits.numel()
        print(f"Found {count} triggers.")
        
        if count == 0:
            print("ZERO TRIGGERS FOUND!")
            return
            
        returns = hits * direction
        cost = 0.5
        net_returns = returns - cost
        
        total_pnl = net_returns.sum().item()
        avg_pnl = total_pnl / count
        
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Avg PnL: {avg_pnl:.2f}")
        print(f"Raw Returns (First 5): {returns[:5]}")

if __name__ == "__main__":
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    debugger = DebugTitan(str(data_path))
    
    # Test C1 (NY ORB) equivalent: 09:45 Breakout Long
    debugger.test_config(9, 45, 1)
    
    # Test 15:00 Momentum Long (C3/C8)
    debugger.test_config(15, 0, 1)
