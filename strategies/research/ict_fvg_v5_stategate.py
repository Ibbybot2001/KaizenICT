"""
ICT Order Flow Strategy v5 - WITH MARKET STATE GATE
====================================================
Adds regime detection layer to filter context:
1. TREND STATE - Expansion vs Digestion
2. HTF ALIGNMENT - Is trade direction aligned with structure?
3. DISPLACEMENT QUALITY - True displacement before FVG?

Only trades that pass the state gate are allowed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration
DATA_PATH = r"C:\Users\CEO\ICT reinforcement\data\kaizen_1m_data_ibkr_2yr.csv"
OUTPUT_DIR = r"C:\Users\CEO\ICT reinforcement\output\charts\ict_v5_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Strategy Parameters
SWING_LOOKBACK = 20
MIN_SWEEP_DEPTH = 5.0
MIN_STOP = 20.0
MIN_TP1_DISTANCE = 15.0
TP2_MULT = 1.5
MIN_FVG_SIZE = 2.0

# Market State Parameters
HTF_LOOKBACK = 100          # Bars to assess higher timeframe structure (increased for better trend detection)
TREND_THRESHOLD = 0.6       # % of HTF that must align
MIN_DISPLACEMENT = 15.0     # Minimum points for true displacement


def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df['time'] = df['time'].dt.tz_localize(None)
    df = df.set_index('time')
    return df


def detect_market_state(df, entry_idx, entry_price, direction):
    """
    REFINED MARKET STATE GATE v2
    
    Key improvements:
    1. Lower highs OVERRIDE higher lows (descending resistance = bearish)
    2. Resistance proximity check for LONGs
    3. Must have clear structure break for reversal trades
    
    Returns: (is_valid, state_label, reason)
    """
    lookback_start = max(0, entry_idx - HTF_LOOKBACK)
    htf_window = df.iloc[lookback_start:entry_idx]
    
    if len(htf_window) < 30:
        return False, "INSUFFICIENT_DATA", "Not enough HTF context"
    
    # 1. FIND SWING STRUCTURE
    highs = []
    lows = []
    
    for i in range(5, len(htf_window) - 5):
        bar = htf_window.iloc[i]
        window_slice = htf_window.iloc[i-5:i+6]
        
        if bar['high'] == window_slice['high'].max():
            highs.append({'price': bar['high'], 'idx': i, 'time': htf_window.index[i]})
        if bar['low'] == window_slice['low'].min():
            lows.append({'price': bar['low'], 'idx': i, 'time': htf_window.index[i]})
    
    if len(highs) < 2 or len(lows) < 2:
        return False, "UNCLEAR", "Not enough swing points"
    
    # 2. SWING HIGH STRUCTURE (PRIORITY FOR TREND DIRECTION)
    # Lower highs = descending resistance = BEARISH (even if lows are higher)
    recent_highs = sorted(highs, key=lambda x: x['idx'])[-4:]
    recent_lows = sorted(lows, key=lambda x: x['idx'])[-4:]
    
    # Count swing high pattern
    lower_highs = 0
    higher_highs = 0
    for i in range(1, len(recent_highs)):
        if recent_highs[i]['price'] < recent_highs[i-1]['price']:
            lower_highs += 1
        elif recent_highs[i]['price'] > recent_highs[i-1]['price']:
            higher_highs += 1
    
    # Count swing low pattern
    higher_lows = 0
    lower_lows = 0
    for i in range(1, len(recent_lows)):
        if recent_lows[i]['price'] > recent_lows[i-1]['price']:
            higher_lows += 1
        elif recent_lows[i]['price'] < recent_lows[i-1]['price']:
            lower_lows += 1
    
    # 3. DETERMINE TREND STATE (SWING HIGHS TAKE PRIORITY)
    # Lower highs = bearish, EVEN IF there are higher lows (just a pullback)
    # Also check: if price is well below window high AND we have any lower highs → BEARISH
    window_high = htf_window['high'].max()
    window_low = htf_window['low'].min()
    entry_bar = df.iloc[entry_idx]
    price_position = (entry_price - window_low) / (window_high - window_low) if window_high > window_low else 0.5
    
    # If price is in lower half of range AND we have lower highs → clearly bearish
    if price_position < 0.5 and lower_highs >= 1:
        trend_state = "BEARISH"
    elif lower_highs >= 2:
        trend_state = "BEARISH"
    elif higher_highs >= 2 and higher_lows >= 1:
        trend_state = "BULLISH"
    elif lower_lows >= 2:
        trend_state = "BEARISH"
    elif higher_lows >= 2 and lower_highs == 0:
        trend_state = "BULLISH"
    else:
        trend_state = "RANGING"
    
    # ADDITIONAL CHECK: If entry is far below window high, we're in a downtrend
    # This catches the 1431 pattern where we're trying to LONG well below resistance
    distance_from_high = window_high - entry_price
    if direction == 'LONG' and distance_from_high > 35:
        # Entry is 40+ pts below the recent high - we're clearly in distribution, not accumulation
        return False, "FAR_FROM_HIGH", f"LONG entry {distance_from_high:.0f}pts below window high - likely downtrend"
    
    # 4. RESISTANCE/SUPPORT PROXIMITY CHECK
    # For LONGS: Reject if price is near/below descending resistance
    if direction == 'LONG' and len(recent_highs) >= 2:
        # Calculate descending resistance line
        last_high = recent_highs[-1]['price']
        prev_high = recent_highs[-2]['price']
        
        if last_high < prev_high:  # Descending resistance confirmed
            # Entry should not be near this resistance
            resistance_level = last_high
            if entry_price > resistance_level * 0.998:  # Within 0.2% of resistance
                return False, "AT_RESISTANCE", f"LONG at descending resistance {resistance_level:.0f} - rejected"
    
    # For SHORTS: Reject if price is near/above ascending support
    if direction == 'SHORT' and len(recent_lows) >= 2:
        last_low = recent_lows[-1]['price']
        prev_low = recent_lows[-2]['price']
        
        if last_low > prev_low:  # Ascending support confirmed
            support_level = last_low
            if entry_price < support_level * 1.002:  # Within 0.2% of support
                return False, "AT_SUPPORT", f"SHORT at ascending support {support_level:.0f} - rejected"
    
    # 5. HTF ALIGNMENT CHECK
    if direction == 'LONG':
        if trend_state == "BEARISH":
            return False, "COUNTER_TREND", f"LONG in {trend_state} structure (LH count: {lower_highs}) - rejected"
    else:  # SHORT
        if trend_state == "BULLISH":
            return False, "COUNTER_TREND", f"SHORT in {trend_state} structure (HH count: {higher_highs}) - rejected"
    
    # 6. DISPLACEMENT CHECK
    recent_range = htf_window.iloc[-20:]
    
    if direction == 'LONG':
        high_before = recent_range['high'].max()
        low_recent = recent_range['low'].min()
        displacement = high_before - low_recent
    else:
        low_before = recent_range['low'].min()
        high_recent = recent_range['high'].max()
        displacement = high_recent - low_before
    
    if displacement < MIN_DISPLACEMENT:
        return False, "WEAK_DISPLACEMENT", f"Displacement {displacement:.1f} < {MIN_DISPLACEMENT} required"
    
    # All checks passed
    return True, trend_state, f"HTF aligned ({trend_state}), LH:{lower_highs} HH:{higher_highs}"


def find_fvg(df, start_idx, end_idx, direction):
    """Find Fair Value Gap"""
    fvgs = []
    
    for i in range(start_idx + 1, min(end_idx, len(df) - 1)):
        if i < 1:
            continue
            
        prev_bar = df.iloc[i - 1]
        next_bar = df.iloc[i + 1]
        
        if direction == 'SHORT':
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
    
    return fvgs[-1] if fvgs else None


def check_fvg_retest_entry(df, setup_idx, direction, fvg):
    """Wait for FVG retest then confirmation close"""
    if fvg is None:
        return False, None, None
    
    fvg_touched = False
    
    for i in range(setup_idx, min(setup_idx + 20, len(df))):
        bar = df.iloc[i]
        
        if direction == 'SHORT':
            if bar['high'] >= fvg['bottom']:
                fvg_touched = True
            if fvg_touched and bar['close'] < bar['open']:
                if bar['close'] < fvg['bottom']:
                    return True, i, bar['close']
        else:
            if bar['low'] <= fvg['top']:
                fvg_touched = True
            if fvg_touched and bar['close'] > bar['open']:
                if bar['close'] > fvg['top']:
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
            if (window.iloc[i]['high'] > window.iloc[i-1]['high'] and
                window.iloc[i]['high'] > window.iloc[i+1]['high']):
                swings.append(window.iloc[i]['high'])
        else:
            if (window.iloc[i]['low'] < window.iloc[i-1]['low'] and
                window.iloc[i]['low'] < window.iloc[i+1]['low']):
                swings.append(window.iloc[i]['low'])
    
    if not swings:
        return None
    
    swings.sort(reverse=(direction == 'LONG'))
    return swings[1] if len(swings) > 1 else swings[0]


def calculate_outcome(df, entry_idx, direction, entry_price, stop_price, tp1_price, tp2_price):
    """50/50 scaling with SL to BE after TP1"""
    max_bars = 120
    tp1_hit = False
    total_r = 0
    stop_distance = abs(entry_price - stop_price)
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        bars = i - entry_idx
        
        if direction == 'LONG':
            if not tp1_hit and bar['low'] <= stop_price:
                return {'outcome': 'LOSS', 'total_r': -1.0, 'tp1_hit': False,
                        'exit_bar': bars, 'detail': 'FULL_STOP'}
            if tp1_hit and bar['low'] <= entry_price:
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'TP1_THEN_BE'}
            if not tp1_hit and bar['high'] >= tp1_price:
                tp1_hit = True
                r1 = (tp1_price - entry_price) / stop_distance
                total_r = 0.5 * r1
            if tp1_hit and bar['high'] >= tp2_price:
                r2 = (tp2_price - entry_price) / stop_distance
                total_r += 0.5 * r2
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'FULL_TARGET'}
        else:
            if not tp1_hit and bar['high'] >= stop_price:
                return {'outcome': 'LOSS', 'total_r': -1.0, 'tp1_hit': False,
                        'exit_bar': bars, 'detail': 'FULL_STOP'}
            if tp1_hit and bar['high'] >= entry_price:
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'TP1_THEN_BE'}
            if not tp1_hit and bar['low'] <= tp1_price:
                tp1_hit = True
                r1 = (entry_price - tp1_price) / stop_distance
                total_r = 0.5 * r1
            if tp1_hit and bar['low'] <= tp2_price:
                r2 = (entry_price - tp2_price) / stop_distance
                total_r += 0.5 * r2
                return {'outcome': 'WIN', 'total_r': total_r, 'tp1_hit': True,
                        'exit_bar': bars, 'detail': 'FULL_TARGET'}
    
    return {'outcome': 'LOSS' if total_r <= 0 else 'WIN', 'total_r': total_r,
            'tp1_hit': tp1_hit, 'exit_bar': max_bars, 'detail': 'TIME_EXIT'}


def detect_setup(df, swing_low_idx, swing_low_price):
    """Full setup detection with Market State Gate"""
    
    for i in range(swing_low_idx + 1, min(swing_low_idx + 50, len(df))):
        bar = df.iloc[i]
        
        if bar['low'] < swing_low_price - MIN_SWEEP_DEPTH:
            sweep_low = bar['low']
            sweep_idx = i
            
            for j in range(i + 1, min(i + 30, len(df))):
                if df.iloc[j]['close'] > swing_low_price:
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
                    
                    direction = 'LONG' if recovery_ratio >= 0.50 else 'SHORT'
                    
                    # Find FVG
                    fvg = find_fvg(df, max(0, sweep_idx - 30), sweep_idx + 5, direction)
                    
                    # Check FVG retest entry
                    setup_start = j if direction == 'LONG' else j + len(recovery_window) - 1
                    has_entry, entry_idx, entry_price = check_fvg_retest_entry(
                        df, setup_start, direction, fvg
                    )
                    
                    if not has_entry:
                        break
                    
                    trade_data = {
                        'direction': direction,
                        'swing_low_idx': swing_low_idx,
                        'swing_low_price': swing_low_price,
                        'sweep_low': sweep_low,
                        'recovery_ratio': recovery_ratio,
                        'fvg': fvg,
                        'entry_idx': entry_idx,
                        'entry_time': df.index[entry_idx],
                        'entry_price': entry_price,
                    }

                    # Tactical SL Calculation (Needed for plotting even if rejected)
                    if direction == 'LONG':
                        if fvg and fvg['type'] == 'bullish':
                            fvg_stop = fvg['bottom'] - 3
                        else:
                            fvg_stop = entry_price - 25
                        sweep_stop = sweep_low - 3
                        raw_stop = min(fvg_stop, sweep_stop)
                        stop_distance = max(MIN_STOP, entry_price - raw_stop)
                        stop_price = entry_price - stop_distance
                        
                        tp1_price = find_major_swing(df, entry_idx, 'LONG')
                        if tp1_price is None or tp1_price <= entry_price:
                            tp1_price = entry_price + stop_distance * 0.8
                        if tp1_price - entry_price < MIN_TP1_DISTANCE:
                            tp1_price = entry_price + MIN_TP1_DISTANCE
                        tp2_price = entry_price + stop_distance * TP2_MULT
                        if tp1_price >= tp2_price:
                            tp1_price = entry_price + MIN_TP1_DISTANCE
                    else:
                        if fvg and fvg['type'] == 'bearish':
                            fvg_stop = fvg['top'] + 3
                        else:
                            fvg_stop = entry_price + 25
                        recovery_stop = recovery_high + 3
                        raw_stop = max(fvg_stop, recovery_stop)
                        stop_distance = max(MIN_STOP, raw_stop - entry_price)
                        stop_price = entry_price + stop_distance
                        
                        tp1_price = find_major_swing(df, entry_idx, 'SHORT')
                        if tp1_price is None or tp1_price >= entry_price:
                            tp1_price = entry_price - stop_distance * 0.8
                        if entry_price - tp1_price < MIN_TP1_DISTANCE:
                            tp1_price = entry_price - MIN_TP1_DISTANCE
                        tp2_price = entry_price - stop_distance * TP2_MULT
                        if tp1_price <= tp2_price:
                            tp1_price = entry_price - MIN_TP1_DISTANCE

                    # Update trade data with calculated levels
                    trade_data.update({
                        'stop_price': stop_price,
                        'stop_distance': stop_distance,
                        'tp1_price': tp1_price,
                        'tp2_price': tp2_price
                    })

                    # ===== MARKET STATE GATE =====
                    state_valid, state_label, state_reason = detect_market_state(
                        df, entry_idx, entry_price, direction
                    )
                    
                    trade_data['market_state'] = state_label
                    trade_data['state_reason'] = state_reason

                    if not state_valid:
                        # Trade rejected - Return with REJECTED outcome
                        trade_data.update({
                            'outcome': 'REJECTED',
                            'total_r': 0.0,
                            'tp1_hit': False, 
                            'exit_bar': 0, 
                            'detail': state_reason
                        })
                        return trade_data
                    # =============================
                    
                    # If passed, calculate actual outcome
                    result = calculate_outcome(df, entry_idx, direction,
                                               entry_price, stop_price, tp1_price, tp2_price)
                    
                    trade_data.update(result)
                    return trade_data

            break
    
    return None


def run_backtest(df, max_trades=1000):
    print("\nFinding swing lows...")
    swing_lows = []
    for i in range(SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
        window = df.iloc[i - SWING_LOOKBACK:i + SWING_LOOKBACK + 1]
        if df.iloc[i]['low'] == window['low'].min():
            swing_lows.append({'idx': i, 'price': df.iloc[i]['low']})
    
    print(f"Found {len(swing_lows)} swing lows")
    
    trades = []
    last_idx = 0
    
    print("\nScanning for setups (with Market State Gate)...")
    for sl in swing_lows:
        if sl['idx'] < last_idx + 60:
            continue
        
        result = detect_setup(df, sl['idx'], sl['price'])
        
        if result:
            trades.append(result)
            # Only advance index if trade was TAKEN (not rejected)
            # Or should we advance anyway? Better to advance to avoid overlap.
            last_idx = result['entry_idx']
            
            if len(trades) % 50 == 0:
                print(f"  {len(trades)} setups found...")
            
            if len(trades) >= max_trades:
                break
    
    return trades


def plot_trade(df, trade, idx, output_dir):
    """Chart with market state info"""
    start = max(0, trade['swing_low_idx'] - 60)
    end = min(len(df), trade['entry_idx'] + 80)
    window = df.iloc[start:end].copy()
    
    if len(window) < 40:
        return None
    
    fig, ax = plt.subplots(figsize=(18, 11))
    bg = '#1a1a2e'
    fig.set_facecolor(bg)
    ax.set_facecolor(bg)
    
    for i, (t, row) in enumerate(window.iterrows()):
        c = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
        ax.plot([i, i], [row['low'], row['high']], color=c, lw=1)
        b = min(row['open'], row['close'])
        h = max(abs(row['close'] - row['open']), 0.25)
        ax.add_patch(plt.Rectangle((i-0.35, b), 0.7, h, fc=c, ec=c))
    
    time_to_x = {t: i for i, t in enumerate(window.index)}
    
    if trade['fvg']:
        fvg = trade['fvg']
        fvg_x = time_to_x.get(fvg['time'], 50)
        fc = '#FF6B6B' if fvg['type'] == 'bearish' else '#26a69a'
        ax.add_patch(plt.Rectangle((fvg_x, fvg['bottom']), 30, fvg['top']-fvg['bottom'],
                                    fc=fc, alpha=0.3, ec=fc, lw=2))
    
    ax.axhline(trade['swing_low_price'], color='#FFD700', ls='--', lw=2, alpha=0.7)
    
    entry_c = '#00FF7F' if trade['direction']=='LONG' else '#FF6B6B'
    if trade['outcome'] == 'REJECTED':
        entry_c = '#AAAAAA'
        
    ax.axhline(trade['entry_price'], color=entry_c, lw=1.5, alpha=0.6)
    ax.axhline(trade['stop_price'], color='#FF4444', ls='--', lw=2)
    ax.axhline(trade['tp1_price'], color='#00BFFF', ls='-.', lw=2)
    ax.axhline(trade['tp2_price'], color='#00FF7F', ls='--', lw=2)
    
    entry_x = time_to_x.get(trade['entry_time'], 60)
    ax.annotate(f"ENTRY ({trade['direction']})", xy=(entry_x, trade['entry_price']),
                xytext=(entry_x+3, trade['entry_price']+8), fontsize=11, fontweight='bold',
                color=entry_c, arrowprops=dict(arrowstyle='->', color=entry_c, lw=2))
    
    ax.text(len(window)-8, trade['stop_price'], f"SL: {trade['stop_distance']:.0f}pts",
            color='#FF4444', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', fc=bg, ec='#FF4444'))
    
    # Title Color
    if trade['outcome'] == 'WIN':
        result_c = '#00FF7F'
        tag = 'win'
    elif trade['outcome'] == 'REJECTED':
        result_c = '#FFA500' # Orange for rejected
        tag = 'rejected'
    else:
        result_c = '#FF4444'
        tag = 'loss'
        
    title = (f"{trade['direction']} | {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}\n"
             f"STATE: {trade['market_state']} ({trade.get('state_reason', '')}) | Stop: {trade['stop_distance']:.0f}pts | "
             f"Result: {trade['detail']} ({trade['total_r']:+.2f}R)")
    ax.set_title(title, color=result_c, fontsize=13, fontweight='bold', pad=20)
    
    ax.set_xlim(-1, len(window))
    ax.set_ylim(window['low'].min()-10, window['high'].max()+10)
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, color='#2a2a4a', alpha=0.3)
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, f"{tag}_{trade['direction']}_{idx}_{trade['entry_time'].strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(path, dpi=150, facecolor=bg, bbox_inches='tight')
    plt.close()
    
    return path


def main():
    df = load_data()
    print(f"Loaded {len(df)} bars")
    
    trades = run_backtest(df, max_trades=1000)
    
    print(f"\n{'='*60}")
    print("ICT FVG v5 - WITH REJECTION TRACKING")
    print(f"{'='*60}")
    
    if not trades:
        print("No setups found!")
        return
    
    tdf_all = pd.DataFrame(trades)
    
    # Split Active vs Rejected
    active = tdf_all[tdf_all['outcome'] != 'REJECTED'].copy()
    rejected = tdf_all[tdf_all['outcome'] == 'REJECTED'].copy()
    
    print(f"Total Setups Found: {len(tdf_all)}")
    print(f"  - Executed: {len(active)}")
    print(f"  - Rejected: {len(rejected)} ({(len(rejected)/len(tdf_all)*100):.1f}%)")
    
    # --- Rejected Stats ---
    if not rejected.empty:
        print("\nRejection Reasons:")
        print(rejected['detail'].value_counts().to_string())
    
    # --- Active Stats ---
    if not active.empty:
        print(f"\n{'='*30} PERFORMANCE {'='*30}")
        for d in ['LONG', 'SHORT']:
            s = active[active['direction'] == d]
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
            
        print(f"\n{'='*60}")
        wr = (active['outcome']=='WIN').sum()/len(active)*100
        print(f"OVERALL: {len(active)} trades | Win Rate: {wr:.1f}%")
        print(f"TOTAL P&L: {active['total_r'].sum():+.2f}R")
        print(f"Avg R/trade: {active['total_r'].mean():+.3f}R")
        print(f"{'='*60}")
    
    print("\nGenerating charts...")
    # Plot sampled Wins (20 random across full period)
    wins = active[active['outcome']=='WIN']
    if not wins.empty:
        for i, (_, t) in enumerate(wins.sample(n=min(20, len(wins)), random_state=42).sort_values('entry_time').iterrows(), 1):
            p = plot_trade(df, t, f"win_{i}", OUTPUT_DIR)
            if p: print(f"  [WIN] {os.path.basename(p)}")
        
    # Plot sampled Losses (20 random across full period)
    losses = active[active['outcome']=='LOSS']
    if not losses.empty:
        for i, (_, t) in enumerate(losses.sample(n=min(20, len(losses)), random_state=42).sort_values('entry_time').iterrows(), 1):
            p = plot_trade(df, t, f"loss_{i}", OUTPUT_DIR)
            if p: print(f"  [LOSS] {os.path.basename(p)}")
        
    # Plot sampled Rejected Examples (20 random)
    if not rejected.empty:
        for i, (_, t) in enumerate(rejected.sample(n=min(20, len(rejected)), random_state=42).sort_values('entry_time').iterrows(), 1):
            p = plot_trade(df, t, f"rejected_{i}", OUTPUT_DIR)
            if p: print(f"  [REJ] {os.path.basename(p)}")

    print(f"\nCharts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
