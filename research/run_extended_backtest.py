"""Extended backtest runner - generates more trade examples"""
import sys
sys.path.insert(0, '.')
from ict_fvg_strategy import load_data, run_backtest, plot_trade
import pandas as pd
import os

OUTPUT_DIR = r'ict_fvg_charts_extended'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = load_data()
trades = run_backtest(df, max_trades=300)

print('='*60)
print('EXTENDED BACKTEST RESULTS')
print('='*60)
print(f'Total Trades: {len(trades)}')

if trades:
    tdf = pd.DataFrame(trades)
    
    for d in ['LONG', 'SHORT']:
        s = tdf[tdf['direction'] == d]
        if len(s) == 0:
            continue
        wins = (s['outcome'] == 'WIN').sum()
        tr = s['total_r'].sum()
        wr = wins / len(s) * 100
        tp1r = s['tp1_hit'].sum() / len(s) * 100
        avg_stop = s['stop_distance'].mean()
        print(f'\n{d}:')
        print(f'  Trades: {len(s)}')
        print(f'  Win Rate: {wr:.1f}%')
        print(f'  TP1 Rate: {tp1r:.1f}%')
        print(f'  Avg Stop: {avg_stop:.0f} pts')
        print(f'  Total R: {tr:+.2f}')
        for det in s['detail'].unique():
            cnt = (s['detail'] == det).sum()
            print(f'    - {det}: {cnt}')
    
    print()
    print('='*60)
    wr = (tdf['outcome']=='WIN').sum()/len(tdf)*100
    print(f'OVERALL: {len(trades)} trades | Win Rate: {wr:.1f}%')
    print(f'TOTAL P&L: {tdf["total_r"].sum():+.2f}R')
    print(f'Avg R per trade: {tdf["total_r"].mean():+.3f}R')
    print('='*60)
    
    # Generate more charts - 5 wins, 5 losses per direction
    print('\nGenerating charts...')
    for d in ['LONG', 'SHORT']:
        s = tdf[tdf['direction'] == d]
        for idx, (_, t) in enumerate(s[s['outcome']=='WIN'].head(5).iterrows(), 1):
            p = plot_trade(df, t, f'{d}_win_{idx}', OUTPUT_DIR)
            if p: print(f'  Saved: {os.path.basename(p)}')
        for idx, (_, t) in enumerate(s[s['outcome']=='LOSS'].head(5).iterrows(), 1):
            p = plot_trade(df, t, f'{d}_loss_{idx}', OUTPUT_DIR)
            if p: print(f'  Saved: {os.path.basename(p)}')
    
    print(f'\nCharts saved to: {OUTPUT_DIR}')
