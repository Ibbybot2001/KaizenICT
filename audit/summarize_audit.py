
import pandas as pd
df = pd.read_csv("ml_lab/audit/continuum_audit_log.csv")
print(f"Total Trades: {len(df)}")
print(f"Total Net R: {df['r_net'].sum():.2f}")
print("-" * 30)
print(df.groupby('regime')['r_net'].agg(['count', 'sum', 'mean']))
print("-" * 30)
print("Win Rate:", (df['outcome'] == 'TP').mean())
