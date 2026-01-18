"""
Run Portfolio Metrics
Validates Strategies A, B, C, D and detailed metrics.
"""

from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import timedelta

def compute_detailed_stats(trade_list, initial_capital=100000):
    if not trade_list:
        return {}
        
    df = pd.DataFrame(trade_list)
    df['cum_pnl'] = df['pnl'].cumsum()
    df['equity'] = initial_capital + df['cum_pnl']
    df['drawdown'] = df['equity'] - df['equity'].cummax()
    df['drawdown_pct'] = df['drawdown'] / df['equity'].cummax()
    
    # 1. Profit Factor
    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    
    # 2. Max Drawdown
    max_dd_pts = abs(df['drawdown'].min())
    
    # 3. Max DD Duration
    # This requires full equity curve (daily/hourly). 
    # Approx: Count consecutive trades in DD? Or Time difference?
    # Let's count max time between equity peaks.
    equity_peaks = df[df['equity'] == df['equity'].cummax()]
    # If only 1 peak at end, duration is full period.
    # Logic: For each trade, check time since last peak.
    current_peak_time = df.iloc[0]['entry_time'] # Init
    max_duration = timedelta(0)
    current_peak_val = -9999999
    
    for _, row in df.iterrows():
        if row['equity'] > current_peak_val:
            current_peak_val = row['equity']
            current_peak_time = row['exit_time']
        else:
            duration = row['exit_time'] - current_peak_time
            if duration > max_duration:
                max_duration = duration
                
    # 4. Longest Trade (Duration)
    df['duration'] = df['exit_time'] - df['entry_time']
    longest_trade = df['duration'].max()
    
    # 5. Streaks
    wins = (df['pnl'] > 0).astype(int)
    # Group consecutive 1s or 0s
    streaks = df['pnl'].gt(0).ne(df['pnl'].gt(0).shift()).cumsum()
    counts = df.groupby(streaks)['pnl'].apply(lambda x: (x > 0).sum() if (x > 0).any() else -((x <= 0).sum()))
    max_win_streak = counts.max()
    max_lose_streak = abs(counts.min())
    
    # 6. Sharpe Ratio (Annualized)
    # Approx: (Mean Return / Std Dev) * sqrt(252 * trades/day) ??
    # Better: Daily PnL
    df['date'] = df['exit_time'].dt.date
    daily_pnl = df.groupby('date')['pnl'].sum()
    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std()
    
    # Annualized Sharpe (assuming 252 trading days)
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0.0
    
    return {
        'total_trades': len(df),
        'total_pnl': df['pnl'].sum(),
        'pf': pf,
        'max_dd_pts': max_dd_pts,
        'max_dd_duration': max_duration,
        'longest_trade': longest_trade,
        'max_win_streak': max_win_streak,
        'max_lose_streak': max_lose_streak,
        'sharpe': sharpe,
        'avg_win': df[df['pnl'] > 0]['pnl'].mean(),
        'avg_loss': df[df['pnl'] < 0]['pnl'].mean()
    }

def main():
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    tick_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet"
    bar_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    print("Loading Data...")
    df_bars = pd.read_parquet(bar_path)
    df_bars['time'] = pd.to_datetime(df_bars.index)
    df_bars['hour'] = df_bars['time'].dt.hour
    df_bars['minute'] = df_bars['time'].dt.minute
    
    # Pre-compute Indicators
    print(f"Columns: {df_bars.columns.tolist()}")
    vol_col = 'tick_volume' if 'tick_volume' in df_bars.columns else 'volume'
    print(f"Using Volume Column: {vol_col}")
    print(f"Rows: {len(df_bars)}")
    
    df_bars['range'] = df_bars['high'] - df_bars['low']
    df_bars['avg_range'] = df_bars['range'].rolling(20).mean()
    df_bars['close_change'] = df_bars['close'].diff()
    df_bars['vol_spike'] = df_bars[vol_col] > (df_bars[vol_col].rolling(20).mean() * 1.5)
    print(f"Vol Spikes: {df_bars['vol_spike'].sum()}")
    
    # Strategy Logic Definitions
    
    # Strat A: 15:00 Breakout (Fixed)
    mask_a = (df_bars['hour'] == 15) & (df_bars['minute'] == 1)
    # Entry direction dictated by 15:00 candle (already encoded in valid logic? No, let's assume Long for simplicity or check 15:00 candle)
    # For A, we know it's "Follow 15:00 Direction".
    # Implementation: Get direction from 15:00 candle.
    # To keep it simple for this script, I'll hardcode Long if 15:00 closed Green.
    signals_a = []
    
    # Strat B: 14:00 Trend (PM Silver Bullet)
    # High Vol + Long
    mask_b = (df_bars['hour'] == 14) & df_bars['vol_spike'] & (df_bars['close'] > df_bars['open'])
    signals_b = df_bars[mask_b]['time'].tolist()
    
    # Strat C: PM Scalp (PM Fade)
    # Disp Down + Long (Fade)
    mask_c = (df_bars['hour'].isin([14, 15])) & (df_bars['close_change'] < -10) # Disp Down > 10pts
    signals_c = df_bars[mask_c]['time'].tolist()
    
    # Strat D: Generalized US (DoL)
    # Trigger: Range Expansion (Range > 1.5x Avg Range) -- Matches Validation Script
    # 09:30-16:00
    mask_d_time = ((df_bars['hour'] == 9) & (df_bars['minute'] >= 30)) | ((df_bars['hour'] >= 10) & (df_bars['hour'] < 16))
    
    # Recalculate Range Spike used in Validation
    mask_range_spike = df_bars['range'] > (df_bars['avg_range'] * 1.5)
    
    mask_d = mask_d_time & mask_range_spike & (df_bars['close'] > df_bars['open'])
    idx_d = np.where(mask_d)[0]
    
    # DoL Logic for D
    highs = df_bars['high'].values
    is_swing_high = (highs[2:] < highs[1:-1]) & (highs[:-2] < highs[1:-1])
    swing_indices = np.where(is_swing_high)[0] + 1
    swing_highs = highs[swing_indices]
    
    signals_d = []
    for idx in idx_d:
        row = df_bars.iloc[idx]
        entry = row['close']
        targets = swing_highs[swing_highs > entry]
        valid = targets[targets > entry + 10]
        tp = valid.min() if len(valid) > 0 else (entry + 50)
        signals_d.append({'time': row['time'], 'target_price': tp})

    # Backtester
    tester = TickGeneralizedBacktester(str(tick_path), str(bar_path))
    
    # Generate Singals A correctly
    # 15:01 entry if 15:00 was Green
    signals_a = []
    mask_1500 = (df_bars['hour'] == 15) & (df_bars['minute'] == 0)
    idxs_1500 = np.where(mask_1500)[0]
    for idx in idxs_1500:
        if df_bars.iloc[idx]['close'] > df_bars.iloc[idx]['open']:
            # Signal at 15:01:00
            sig_time = df_bars.iloc[idx]['time'] + timedelta(minutes=1)
            signals_a.append(sig_time)

    strategies = [
        ("A: 15:00 Sniper", signals_a, 1, 15, 200, None, 0.0, False),
        ("B: 14:00 Trend", signals_b, 1, 15, 100, None, 0.0, False),
        ("C: PM Scalp", signals_c, 1, 10, 20, None, 0.0, False),
        ("D: General DoL", signals_d, 1, 20, 50, None, 0.0, False)
    ]
    
    print(f"DEBUG: Signals A: {len(signals_a)}")
    print(f"DEBUG: Signals B: {len(signals_b)}")
    print(f"DEBUG: Signals C: {len(signals_c)}")
    print(f"DEBUG: Signals D: {len(signals_d)}")
    
    print(f"{'Strategy':<20} | {'PF':<5} | {'Sharpe':<6} | {'MaxDD':<8} | {'DD Dur':<10} | {'Longest Trd':<12} | {'Win Strk':<8} | {'Lose Strk':<8}")
    print("-" * 120)
    
    for name, sigs, dire, sl, tp, tp1, pct, be in strategies:
        res = tester.backtest_signals(sigs, direction=dire, stop_pts=sl, target_pts=tp, tp1_pts=tp1, tp1_pct=pct, move_to_be=be)
        stats = compute_detailed_stats(res['trade_list'])
        
        if not stats:
            print(f"{name:<20} | NO TRADES SPECIFIED OR FOUND")
            continue
            
        print(f"{name:<20} | {stats['pf']:<5.2f} | {stats['sharpe']:<6.2f} | {stats['max_dd_pts']:<8.1f} | {str(stats['max_dd_duration']):<10} | {str(stats['longest_trade']):<12} | {stats['max_win_streak']:<8} | {stats['max_lose_streak']:<8}")

if __name__ == "__main__":
    main()
