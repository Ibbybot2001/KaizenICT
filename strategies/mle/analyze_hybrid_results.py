import pandas as pd
import sys
from pathlib import Path

csv_path = Path("C:/Users/CEO/ICT reinforcement/output/hybrid_search/hybrid_results_10k.csv")

if not csv_path.exists():
    print("No results yet.")
    sys.exit()

df = pd.read_csv(csv_path)
print(f"Total Rows: {len(df)}")

if len(df) == 0: sys.exit()

# Sort by PF
df_sorted = df.sort_values('test_pf', ascending=False)

print("\n--- TOP 10 BY PROFIT FACTOR ---")
print(df_sorted[['params', 'test_pf', 'test_trades']].head(10))

print("\n--- TOP 5 HIGH VOLUME (Trades > 100) ---")
high_vol = df[df['test_trades'] > 100].sort_values('test_pf', ascending=False)
print(high_vol[['params', 'test_pf', 'test_trades']].head(5))
