import pandas as pd
df = pd.read_csv('output/overnight_search/validated_strategies.csv')

# Calculate trades per day
df['total_trades'] = df['train_trades'] + df['test_trades']
df['train_per_day'] = df['train_trades'] / 168  # ~168 trading days in 8 months
df['test_per_day'] = df['test_trades'] / 84  # ~84 trading days in 4 months

# Filter for 3-5 trades/day (or close to it)
high_vol = df[df['train_per_day'] >= 2.5]

print(f"Total winners: {len(df)}")
print(f"Strategies with 3+ trades/day: {len(high_vol)}")
print()

if len(high_vol) > 0:
    print("=== TOP 10 HIGH-VOLUME STRATEGIES ===")
    for i, row in high_vol.nlargest(10, 'test_pf').iterrows():
        print(f"Trades/Day: {row['train_per_day']:.1f} (train) / {row['test_per_day']:.1f} (test)")
        print(f"PF: {row['test_pf']:.2f} | Exp: ${row['test_exp']:.2f}")
        print(f"{row['params'][:120]}")
        print()
else:
    print("No 3+ trades/day strategies found yet.")
    print()
    print("Current highest volume strategies:")
    top = df.nlargest(5, 'train_per_day')
    for i, row in top.iterrows():
        print(f"Trades/Day: {row['train_per_day']:.1f} | PF: {row['test_pf']:.2f} | Exp: ${row['test_exp']:.2f}")
