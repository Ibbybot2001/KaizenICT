"""
ICT Order Flow Strategy v4
==========================
Full ICT entry model with:
1. FVG Detection (Fair Value Gaps)
2. Order Block identification
3. FVG Retest Entry (mitigation)
4. SL at FVG edge (min 20 pts)
5. TP1 at prior major swing, TP2 at 1.5R
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from engine.event_engine import EventEngine, SimulationConfig
from engine.trade import Side, OrderType, ExitReason
from primitives.zones import ZoneDetector
from primitives.displacement import DisplacementDetector

# Configuration
DATA_PATH = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
OUTPUT_DIR = r"C:\Users\CEO\ICT reinforcement\output\charts\ict_fvg_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy Parameters
SWING_LOOKBACK = 20
MIN_SWEEP_DEPTH = 5.0
MIN_STOP = 20.0              # MINIMUM 20 pts to breathe
MAX_STOP = 45.0
TP2_MULT = 1.5
MIN_FVG_SIZE = 2.0           # Minimum FVG gap size
MIN_TP1_DISTANCE = 15.0      # TP1 must be at least 15-20 pts away


def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    return df


def find_fvg(df, start_idx, end_idx, direction):
    """
    Find Fair Value Gap (FVG) in the given range
    
    Bearish FVG: Gap between candle[i-1] low and candle[i+1] high (down move)
    Bullish FVG: Gap between candle[i-1] high and candle[i+1] low (up move)
    
    Returns the most recent FVG
    """
    fvgs = []
    
    for i in range(start_idx + 1, min(end_idx, len(df) - 1)):
        if i < 1:
            continue
            
        prev_bar = df.iloc[i - 1]
        curr_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        
        if direction == 'SHORT':
            # Bearish FVG: gap between prev low and next high (price dropped through)
            if prev_bar['low'] > next_bar['high']:
                gap_size = prev_bar['low'] - next_bar['high']
                if gap_size >= MIN_FVG_SIZE:
                    fvgs.append({
                        'type': 'bearish',
                        'top': prev_bar['low'],
                        'bottom': next_bar['high'],
                        'size': gap_size,
                        'idx': i,
                        'time': df.index[i]
                    })
        else:
            # Bullish FVG: gap between prev high and next low (price rallied through)
            if prev_bar['high'] < next_bar['low']:
                gap_size = next_bar['low'] - prev_bar['high']
                if gap_size >= MIN_FVG_SIZE:
                    fvgs.append({
                        'type': 'bullish',
                        'top': next_bar['low'],
                        'bottom': prev_bar['high'],
                        'size': gap_size,
                        'idx': i,
                        'time': df.index[i]
                    })
    
    # Return most recent FVG
    return fvgs[-1] if fvgs else None


def find_order_block(df, sweep_idx, direction, lookback=20):
    """Find order block before the sweep"""
    start = max(0, sweep_idx - lookback)
    window = df.iloc[start:sweep_idx]
    
    if direction == 'LONG':
        # Bullish OB: last bearish candle before up move
        for i in range(len(window) - 1, -1, -1):
            bar = window.iloc[i]
            if bar['close'] < bar['open']:
                body = bar['open'] - bar['close']
                if body > 1.5:
                    return {
                        'high': bar['high'],
                        'low': bar['low'],
                        'idx': start + i,
                        'time': window.index[i]
                    }
    else:
        # Bearish OB: last bullish candle before down move
        for i in range(len(window) - 1, -1, -1):
            bar = window.iloc[i]
            if bar['close'] > bar['open']:
                body = bar['close'] - bar['open']
                if body > 1.5:
                    return {
                        'high': bar['high'],
                        'low': bar['low'],
                        'idx': start + i,
                        'time': window.index[i]
                    }
    
    return None


def check_fvg_retest_entry(df, setup_idx, direction, fvg):
    """
    PROPER FVG RETEST ENTRY:
    1. Price must FIRST retrace INTO the FVG zone (retest)
    2. THEN close above (longs) or below (shorts) as confirmation
    
    This ensures we're not entering AT the FVG, but AFTER the retest.
    """
    if fvg is None:
        return False, None, None
    
    fvg_touched = False  # Track if FVG has been tested
    
    for i in range(setup_idx, min(setup_idx + 20, len(df))):
        bar = df.iloc[i]
        
        if direction == 'SHORT':
            # Step 1: Wait for price to retrace UP INTO the bearish FVG
            if bar['high'] >= fvg['bottom']:
                fvg_touched = True
            
            # Step 2: After FVG touched, wait for bearish close BELOW FVG bottom
            if fvg_touched and bar['close'] < bar['open']:  # Bearish candle
                if bar['close'] < fvg['bottom']:  # Closes below FVG
                    return True, i, bar['close']
        else:
            # Step 1: Wait for price to retrace DOWN INTO the bullish FVG
            if bar['low'] <= fvg['top']:
                fvg_touched = True
            
            # Step 2: After FVG touched, wait for bullish close ABOVE FVG top
            if fvg_touched and bar['close'] > bar['open']:  # Bullish candle
                if bar['close'] > fvg['top']:  # Closes above FVG
                    return True, i, bar['close']
    
    return False, None, None


def find_major_swing(df, entry_idx, direction, lookback=60):
    """Find prior major swing for TP1"""
    start = max(0, entry_idx - lookback)
    window = df.iloc[start:entry_idx]
    
    if len(window) < 10:
        return None
    
    swings = []
    
    for i in range(3, len(window) - 3):
        if direction == 'LONG':
            # Find swing highs above entry
            if (window.iloc[i]['high'] > window.iloc[i-1]['high'] and
                window.iloc[i]['high'] > window.iloc[i-2]['high'] and
                window.iloc[i]['high'] > window.iloc[i+1]['high']):
                swings.append(window.iloc[i]['high'])
        else:
            # Find swing lows below entry
            if (window.iloc[i]['low'] < window.iloc[i-1]['low'] and
                window.iloc[i]['low'] < window.iloc[i-2]['low'] and
                window.iloc[i]['low'] < window.iloc[i+1]['low']):
                swings.append(window.iloc[i]['low'])
    
    if not swings:
        return None
    
    # Return second swing if available (more significant), else first
    swings.sort(reverse=(direction == 'LONG'))
    return swings[1] if len(swings) > 1 else swings[0]


def calculate_outcome(df, entry_idx, direction, entry_price, stop_price, tp1_price, tp2_price):
    """
    50/50 scaling - FIXED TP ORDER:
    - TP1 is always closer to entry (50% out)
    - TP2 is always further (remaining 50%)
    """
    max_bars = 80
    tp1_hit = False
    total_r = 0
    stop_distance = abs(entry_price - stop_price)
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        bars = i - entry_idx
        
        if direction == 'LONG':
            # Stop check
            if not tp1_hit and bar['low'] <= stop_price:
                return {'outcome': 'LOSS', 'total_r': -1.0, 'tp1_hit': False,
                        'exit_bar': bars, 'detail': 'FULL_STOP'}
            if tp1_hit and bar['low'] <= entry_price:
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'TP1_THEN_BE'}
            
            # TP1 first (closer target)
            if not tp1_hit and bar['high'] >= tp1_price:
                tp1_hit = True
                r1 = (tp1_price - entry_price) / stop_distance
                total_r = 0.5 * r1
            
            # TP2 second (further target)
            if tp1_hit and bar['high'] >= tp2_price:
                r2 = (tp2_price - entry_price) / stop_distance
                total_r += 0.5 * r2
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'FULL_TARGET'}
        else:  # SHORT
            if not tp1_hit and bar['high'] >= stop_price:
                return {'outcome': 'LOSS', 'total_r': -1.0, 'tp1_hit': False,
                        'exit_bar': bars, 'detail': 'FULL_STOP'}
            if tp1_hit and bar['high'] >= entry_price:
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'TP1_THEN_BE'}
            
            # TP1 (closer - higher price for shorts since it's first target down)
            if not tp1_hit and bar['low'] <= tp1_price:
                tp1_hit = True
                r1 = (entry_price - tp1_price) / stop_distance
                total_r = 0.5 * r1
            
            # TP2 (further down)
            if tp1_hit and bar['low'] <= tp2_price:
                r2 = (entry_price - tp2_price) / stop_distance
                total_r += 0.5 * r2
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'FULL_TARGET'}
    
    return {'outcome': 'LOSS' if total_r <= 0 else 'WIN', 'total_r': total_r,
            'tp1_hit': tp1_hit, 'exit_bar': max_bars, 'detail': 'TIME_EXIT'}


def detect_setup(df, swing_low_idx, swing_low_price):
    """Full ICT setup detection with FVG"""
    
    for i in range(swing_low_idx + 1, min(swing_low_idx + 50, len(df))):
        bar = df.iloc[i]
        
        # Check for sweep
        if bar['low'] < swing_low_price - MIN_SWEEP_DEPTH:
            sweep_low = bar['low']
            sweep_depth = swing_low_price - sweep_low
            sweep_idx = i
            
            # Wait for close back above swing low
            for j in range(i + 1, min(i + 30, len(df))):
                if df.iloc[j]['close'] > swing_low_price:
                    # Measure recovery
                    pre_sweep = df.iloc[max(0, i-20):i+1]
                    recent_high = pre_sweep['high'].max()
                    drop_size = recent_high - sweep_low
                    
                    if drop_size < 10:
                        break
                    
                    recovery_end = min(j + 15, len(df))
                    recovery_window = df.iloc[j:recovery_end]
                    
                    if len(recovery_window) < 3:
                        break
                    
                    recovery_high = recovery_window['high'].max()
                    recovery_ratio = (recovery_high - sweep_low) / drop_size
                    
                    # Determine direction
                    if recovery_ratio >= 0.50:
                        direction = 'LONG'
                    else:
                        direction = 'SHORT'
                    
                    # Find FVG in the move
                    fvg = find_fvg(df, max(0, sweep_idx - 30), sweep_idx + 5, direction)
                    
                    # Find order block
                    order_block = find_order_block(df, sweep_idx, direction)
                    
                    # Check for FVG retest entry
                    setup_start = j if direction == 'LONG' else j + len(recovery_window) - 1
                    has_entry, entry_idx, entry_price = check_fvg_retest_entry(
                        df, setup_start, direction, fvg
                    )
                    
                    # Fallback to OB entry if no FVG entry
                    if not has_entry and order_block:
                        for k in range(setup_start, min(setup_start + 10, len(df))):
                            bar_k = df.iloc[k]
                            if direction == 'SHORT':
                                if bar_k['close'] < bar_k['open'] and bar_k['close'] < order_block['low']:
                                    has_entry, entry_idx, entry_price = True, k, bar_k['close']
                                    break
                            else:
                                if bar_k['close'] > bar_k['open'] and bar_k['close'] > order_block['high']:
                                    has_entry, entry_idx, entry_price = True, k, bar_k['close']
                                    break
                    
                    if not has_entry:
                        break
                    
                    entry_time = df.index[entry_idx]
                    
                    # TACTICAL SL PLACEMENT (not arbitrary)
                    # Place SL beyond key liquidity levels to avoid stop hunts
                    if direction == 'LONG':
                        # For longs: SL below FVG bottom OR sweep low (whichever is lower/safer)
                        if fvg and fvg['type'] == 'bullish':
                            fvg_stop = fvg['bottom'] - 3  # Just below FVG
                        else:
                            fvg_stop = entry_price - 25
                        
                        sweep_stop = sweep_low - 3  # Just below sweep low
                        
                        # Use the LOWER of the two (more protection from stop hunts)
                        raw_stop = min(fvg_stop, sweep_stop)
                        
                        stop_distance = entry_price - raw_stop
                        # Only enforce MIN, don't cap with MAX (tactical > arbitrary)
                        stop_distance = max(MIN_STOP, stop_distance)
                        stop_price = entry_price - stop_distance
                        
                        tp1_price = find_major_swing(df, entry_idx, 'LONG')
                        if tp1_price is None or tp1_price <= entry_price:
                            tp1_price = entry_price + stop_distance * 0.8
                        
                        # Ensure TP1 is at least MIN_TP1_DISTANCE away
                        if tp1_price - entry_price < MIN_TP1_DISTANCE:
                            tp1_price = entry_price + MIN_TP1_DISTANCE
                        
                        tp2_price = entry_price + stop_distance * TP2_MULT
                        
                        # Ensure TP1 < TP2 for longs
                        if tp1_price >= tp2_price:
                            tp1_price = entry_price + MIN_TP1_DISTANCE
                    else:
                        # For shorts: SL above FVG top OR recovery high (whichever is higher/safer)
                        if fvg and fvg['type'] == 'bearish':
                            fvg_stop = fvg['top'] + 3  # Just above FVG
                        else:
                            fvg_stop = entry_price + 25
                        
                        recovery_stop = recovery_high + 3  # Just above recovery high
                        
                        # Use the HIGHER of the two (more protection from stop hunts)
                        raw_stop = max(fvg_stop, recovery_stop)
                        
                        stop_distance = raw_stop - entry_price
                        # Only enforce MIN, don't cap with MAX (tactical > arbitrary)
                        stop_distance = max(MIN_STOP, stop_distance)
                        stop_price = entry_price + stop_distance
                        
                        tp1_price = find_major_swing(df, entry_idx, 'SHORT')
                        if tp1_price is None or tp1_price >= entry_price:
                            tp1_price = entry_price - stop_distance * 0.8
                        
                        # Ensure TP1 is at least MIN_TP1_DISTANCE away
                        if entry_price - tp1_price < MIN_TP1_DISTANCE:
                            tp1_price = entry_price - MIN_TP1_DISTANCE
                        
                        tp2_price = entry_price - stop_distance * TP2_MULT
                        
                        # Ensure TP1 > TP2 for shorts (TP1 closer to entry)
                        if tp1_price <= tp2_price:
                            tp1_price = entry_price - MIN_TP1_DISTANCE
                    
                    result = calculate_outcome(df, entry_idx, direction,
                                               entry_price, stop_price, tp1_price, tp2_price)
                    
                    return {
                        'swing_low_price': swing_low_price,
                        'swing_low_idx': swing_low_idx,
                        'sweep_low': sweep_low,
                        'sweep_depth': sweep_depth,
                        'recovery_ratio': recovery_ratio,
                        'direction': direction,
                        'fvg': fvg,
                        'order_block': order_block,
                        'entry_idx': entry_idx,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'stop_distance': stop_distance,
                        'tp1_price': tp1_price,
                        'tp2_price': tp2_price,
                        **result
                    }
            break
    
    return None


def run_backtest(df, max_trades=200):
    print("\nFinding swing lows...")
    swing_lows = []
    for i in range(SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
        window = df.iloc[i - SWING_LOOKBACK:i + SWING_LOOKBACK + 1]
        if df.iloc[i]['low'] == window['low'].min():
            swing_lows.append({'idx': i, 'price': df.iloc[i]['low']})
    
    print(f"Found {len(swing_lows)} swing lows")
    
    trades = []
    last_idx = 0
    
    print("\nScanning for setups...")
    for sl in swing_lows:
        if sl['idx'] < last_idx + 60:
            continue
        
        result = detect_setup(df, sl['idx'], sl['price'])
        
        if result:
            trades.append(result)
            last_idx = result['entry_idx']
            
            if len(trades) % 25 == 0:
                print(f"  {len(trades)} trades...")
            
            if len(trades) >= max_trades:
                break
    
    return trades


def plot_trade(df, trade, idx, output_dir):
    """Chart with FVG visualization"""
    start = max(0, trade['swing_low_idx'] - 60)
    end = min(len(df), trade['entry_idx'] + 80)
    window = df.iloc[start:end].copy()
    
    if len(window) < 40:
        return None
    
    fig, ax = plt.subplots(figsize=(18, 11))
    bg = '#1a1a2e'
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    
    # Candles
    for i, (t, row) in enumerate(window.iterrows()):
        c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([i, i], [row['low'], row['high']], color=c, lw=1)
        b = min(row['open'], row['close'])
        h = max(abs(row['close'] - row['open']), 0.25)
        ax.add_patch(plt.Rectangle((i-0.35, b), 0.7, h, fc=c, ec=c))
    
    time_to_x = {t: i for i, t in enumerate(window.index)}
    
    # FVG visualization
    if trade['fvg']:
        fvg = trade['fvg']
        fvg_x = time_to_x.get(fvg['time'], 50)
        fc = '#FF6B6B' if fvg['type'] == 'bearish' else '#26a69a'
        ax.add_patch(plt.Rectangle((fvg_x, fvg['bottom']), 30, fvg['top']-fvg['bottom'],
                                    fc=fc, alpha=0.3, ec=fc, lw=2))
        ax.text(fvg_x+1, fvg['top']+2, f"FVG ({fvg['type']})", color=fc, fontsize=9, fontweight='bold')
    
    # Order block
    if trade['order_block']:
        ob = trade['order_block']
        ob_x = time_to_x.get(ob['time'], 40)
        ax.add_patch(plt.Rectangle((ob_x, ob['low']), 10, ob['high']-ob['low'],
                                    fc='#9C27B0', alpha=0.3, ec='#9C27B0'))
    
    # Levels
    ax.axhline(trade['swing_low_price'], color='#FFD700', ls='--', lw=2, alpha=0.7)
    
    entry_c = '#00FF7F' if trade['direction']=='LONG' else '#FF6B6B'
    ax.axhline(trade['entry_price'], color=entry_c, lw=1.5, alpha=0.6)
    ax.axhline(trade['stop_price'], color='#FF4444', ls='--', lw=2)
    ax.axhline(trade['tp1_price'], color='#00BFFF', ls='-.', lw=2)
    ax.axhline(trade['tp2_price'], color='#00FF7F', ls='--', lw=2)
    
    # Entry marker
    entry_x = time_to_x.get(trade['entry_time'], 60)
    ax.annotate(f"ENTRY ({trade['direction']})", xy=(entry_x, trade['entry_price']),
                xytext=(entry_x+3, trade['entry_price']+8), fontsize=11, fontweight='bold',
                color=entry_c, arrowprops=dict(arrowstyle='->', color=entry_c, lw=2))
    
    # Labels
    ax.text(len(window)-8, trade['stop_price'], f"SL: {trade['stop_distance']:.0f}pts",
            color='#FF4444', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round', fc=bg, ec='#FF4444'))
    ax.text(len(window)-8, trade['tp1_price'], "TP1 (50%)", color='#00BFFF', fontsize=9)
    ax.text(len(window)-8, trade['tp2_price'], "TP2 (50%)", color='#00FF7F', fontsize=9)
    
    # Title
    result_c = '#00FF7F' if trade['outcome'] == 'WIN' else '#FF4444'
    fvg_info = f"FVG: {trade['fvg']['type']}" if trade['fvg'] else "No FVG"
    title = (f"{trade['direction']} | {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}\n"
             f"{fvg_info} | Stop: {trade['stop_distance']:.0f}pts | "
             f"Result: {trade['detail']} ({trade['total_r']:+.2f}R)")
    ax.set_title(title, color=result_c, fontsize=13, fontweight='bold', pad=20)
    
    ax.set_xlim(-1, len(window))
    ax.set_ylim(window['low'].min()-10, window['high'].max()+10)
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, color='#2a2a4a', alpha=0.3)
    
    plt.tight_layout()
    
    tag = 'win' if trade['outcome'] == 'WIN' else 'loss'
    path = os.path.join(output_dir, f"{tag}_{trade['direction']}_{idx}_{trade['entry_time'].strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(path, dpi=150, facecolor=bg, bbox_inches='tight')
    plt.close()
    
    return path


def main():
    df = load_data()
    print(f"Loaded {len(df)} bars")
    
    trades = run_backtest(df, max_trades=200)
    
    print(f"\n{'='*60}")
    print("ICT ORDER FLOW + FVG STRATEGY RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades: {len(trades)}")
    
    if not trades:
        print("No trades!")
        return
    
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
        
        print(f"\n{d}:")
        print(f"  Trades: {len(s)}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  TP1 Rate: {tp1r:.1f}%")
        print(f"  Avg Stop: {avg_stop:.0f} pts")
        print(f"  Total R: {tr:+.2f}")
        
        for det in s['detail'].unique():
            print(f"    - {det}: {(s['detail']==det).sum()}")
    
    print(f"\n{'='*60}")
    wr = (tdf['outcome']=='WIN').sum()/len(tdf)*100
    print(f"OVERALL: {len(trades)} trades | Win Rate: {wr:.1f}%")
    print(f"TOTAL P&L: {tdf['total_r'].sum():+.2f}R")
    print(f"{'='*60}")
    
    # Charts
    print("\nGenerating charts...")
    for d in ['LONG', 'SHORT']:
        s = tdf[tdf['direction'] == d]
        for idx, (_, t) in enumerate(s[s['outcome']=='WIN'].head(3).iterrows(), 1):
            p = plot_trade(df, t, f"{d}_win_{idx}", OUTPUT_DIR)
            if p: print(f"  {p}")
        for idx, (_, t) in enumerate(s[s['outcome']=='LOSS'].head(3).iterrows(), 1):
            p = plot_trade(df, t, f"{d}_loss_{idx}", OUTPUT_DIR)
            if p: print(f"  {p}")
    
    print(f"\nCharts: {OUTPUT_DIR}")
    return tdf


if __name__ == "__main__":
    main()
