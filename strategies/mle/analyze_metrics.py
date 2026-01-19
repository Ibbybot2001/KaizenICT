import pandas as pd
import numpy as np

# Load trade results
res = pd.read_csv('output/Phase16A_All_Trades.csv')

# Core Metrics
total_trades = len(res)
winners = res[res['pnl'] > 0]
losers = res[res['pnl'] < 0]

win_rate = len(winners) / total_trades * 100
avg_win = winners['pnl'].mean()
avg_loss = abs(losers['pnl'].mean())

# Expectancy
expectancy_pts = res['pnl'].mean()
expectancy_usd = expectancy_pts * 2  # MNQ = $2/pt

# Profit Factor
gross_profit = winners['pnl'].sum()
gross_loss = abs(losers['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

# Sharpe Ratio (annualized, assuming ~252 trading days)
daily_pnl = res.groupby('date')['pnl'].sum()
sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

# Max Drawdown
res['cum_pnl'] = res['pnl'].cumsum()
res['peak'] = res['cum_pnl'].cummax()
res['dd'] = res['peak'] - res['cum_pnl']
max_dd = res['dd'].max()

# Win/Loss Ratio
wl_ratio = avg_win / avg_loss if avg_loss > 0 else 999

print('='*60)
print('COMPREHENSIVE STRATEGY METRICS')
print('='*60)
print(f'Total Trades:      {total_trades}')
print(f'Win Rate:          {win_rate:.1f}%')
print(f'Avg Win (pts):     {avg_win:.2f}')
print(f'Avg Loss (pts):    {avg_loss:.2f}')
print(f'Win/Loss Ratio:    {wl_ratio:.2f}')
print()
print(f'Expectancy (pts):  {expectancy_pts:.2f}')
print(f'Expectancy (USD):  ${expectancy_usd:.2f} per trade')
print()
print(f'Profit Factor:     {profit_factor:.2f}')
print(f'Sharpe Ratio:      {sharpe:.2f}')
print(f'Max Drawdown:      {max_dd:.2f} pts (${max_dd * 2:.2f})')
print()
print(f'Gross Profit:      {gross_profit:.2f} pts')
print(f'Gross Loss:        {gross_loss:.2f} pts')
print(f'Net Profit:        {res["pnl"].sum():.2f} pts (${res["pnl"].sum() * 2:,.2f} USD)')
