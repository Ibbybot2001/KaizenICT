import pandas as pd

df = pd.read_csv('output/overnight_search/validated_strategies.csv')

print(f"TOTAL WINNERS SO FAR: {len(df)}")
print()

print("=== TOP 5 BY PROFIT FACTOR ===")
top_pf = df.nlargest(5, 'test_pf')
for i, row in top_pf.iterrows():
    print(f"PF: {row['test_pf']:.2f} | Exp: ${row['test_exp']:.2f} | Trades: {row['train_trades']+row['test_trades']}")
    print(f"   {row['params'][:120]}")
    print()

print("=== TOP 5 BY TRADE VOLUME ===")
df['total_trades'] = df['train_trades'] + df['test_trades']
top_vol = df.nlargest(5, 'total_trades')
for i, row in top_vol.iterrows():
    print(f"Trades: {row['total_trades']} | PF: {row['test_pf']:.2f} | Exp: ${row['test_exp']:.2f}")
    print(f"   {row['params'][:120]}")
    print()
