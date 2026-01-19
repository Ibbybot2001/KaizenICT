import pandas as pd
import sys
from pathlib import Path

# Setup
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase16_pj_engine import engineer_pools, detect_pj_signals

BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")

# Load 1 Month
df = pd.read_parquet(BASE_DIR / "USTEC_2025-01_clean_1m.parquet")
print(f"Loaded {len(df)} bars")

# Engineer
df = engineer_pools(df)
print("\nColumns:", df.columns)

# Check IB Pools
print("\n--- CHECKING IB POOLS ---")
sample_day = df['date'].unique()[0]
day_data = df[df['date'] == sample_day]

# Check 09:30-10:00 Data
ib_data = day_data[(day_data['hour'] == 9) & (day_data['minute'] >= 30)]
print(f"IB Data Rows: {len(ib_data)}")
if not ib_data.empty:
    print(f"Calculated IB High: {ib_data['high'].max()}")
    print(f"Calculated IB Low: {ib_data['low'].min()}")

# Check Merged Columns
print(f"Merged IB_H: {day_data['IB_H'].iloc[-1]}")
print(f"Merged IB_L: {day_data['IB_L'].iloc[-1]}")

# Check Signals
class MockTracker:
    def can_trade(self, p): return True
    def mark_traded(self, p): pass

tracker = MockTracker()
signals = detect_pj_signals(df, tracker, sample_day)
print(f"\nSignals Found: {len(signals)}")
for s in signals:
    print(s)
