import pandas as pd
import sys
from pathlib import Path
import json
import ast

# Paths
PATHS = [
    Path("C:/Users/CEO/ICT reinforcement/output/hybrid_search/hybrid_results_10k.csv"),
    Path("C:/Users/CEO/ICT reinforcement/output/overnight_search/validated_strategies.csv"),
    Path("C:/Users/CEO/ICT reinforcement/output/highvol_search/results_highvol.csv"),
    # Path("C:/Users/CEO/ICT reinforcement/output/intraday_search/intraday_results_raw.csv"), # Low PF
]

all_strats = []

for p in PATHS:
    if not p.exists(): continue
    try:
        df = pd.read_csv(p)
        
        # Normalize Columns
        # We need: 'params', 'pf', 'trades', 'source'
        
        if 'test_pf' in df.columns:
            df['pf'] = df['test_pf']
            df['trades'] = df['test_trades']
        elif 'pf_test' in df.columns:
            df['pf'] = df['pf_test']
            df['trades'] = df['trades_test']
        
        # Source Name
        source_name = p.parent.name
        df['source'] = source_name
        
        # Select
        if 'params' in df.columns:
            subset = df[['params', 'pf', 'trades', 'source']].copy()
        elif 'config' in df.columns:
            subset = df[['config', 'pf', 'trades', 'source']].rename(columns={'config': 'params'}).copy()
        
        all_strats.append(subset)
    except Exception as e:
        print(f"Error reading {p}: {e}")

if not all_strats:
    print("No strategies found.")
    sys.exit()

combined = pd.concat(all_strats)

# FILTER: >= 200 Trades
valid = combined[combined['trades'] >= 200].copy()

# Sort
valid = valid.sort_values('pf', ascending=False)

print(f"## Top 10 Strategies (Trades >= 200) [Total Candidates: {len(valid)}]")
print("| Rank | Source | Profit Factor | Trades | Settings |")
print("|---|---|---|---|---|")

for i, row in valid.head(10).iterrows():
    # Format Params
    try:
        p_str = row['params']
        # Try parse to json/dict to format nicely
        # p_dict = ast.literal_eval(p_str)
        # short_p = str(p_dict)
        short_p = p_str.replace('\n', ' ')
    except:
        short_p = str(row['params'])
        
    print(f"| {i+1} | {row['source']} | **{row['pf']:.2f}** | {row['trades']} | `{short_p[:50]}...` |")

# Also show ones close to 200 (150-200) just in case
near_miss = combined[(combined['trades'] >= 150) & (combined['trades'] < 200)].sort_values('pf', ascending=False)
if not near_miss.empty:
    print("\n## Honorable Mentions (150-199 Trades)")
    print("| Source | Profit Factor | Trades | Settings |")
    print("|---|---|---|---|")
    for i, row in near_miss.head(5).iterrows():
        print(f"| {row['source']} | **{row['pf']:.2f}** | {row['trades']} | `{str(row['params'])[:50]}...` |")
