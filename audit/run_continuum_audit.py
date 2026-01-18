"""
Continuum Execution Audit
"The Lie Detector"

Simulates the 'Continuum' Playbook mechanically to establish a baseline for 
perfect compliance.

Rules:
1. Regimes: Wake (Displacement) / Coil (Compression -> Zone)
2. Anchor Rule: Wake Body must be largest in last 100 bars.
3. Window: 20 bars after detection.
4. Execution: Color match (Green for Long, Red for Short).
5. Management: Fixed SL (10pt), Fixed TP (15pt), Time Stop (20 bars).
"""

import pandas as pd
import numpy as np
import os
from ml.feature_builder import FeatureBuilder
from ml.zone_relative import build_zone_relative_features
from engine.constants import MIN_SL_POINTS

DATA_PATH = "data/kaizen_1m_data_ibkr_2yr.csv"

# --- Constants ---
SL_PTS = 10.0
TP_PTS = 15.0
TIME_STOP_BARS = 20
WINDOW_BARS = 20
ANCHOR_WINDOW = 100
FRICTION_R = 0.05

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, parse_dates=['time'])
        df = df.set_index('time')
        df = df.sort_index()
        df.columns = [c.lower() for c in df.columns]
        # Use last 200k bars (Avoid early data corruption/SystemError)
        return df.iloc[-200000:]
    else: 
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

def run_audit(data, prim, zone):
    trades = []
    
    # State
    permission_state = None # 'WAKE_LONG', 'WAKE_SHORT', 'COIL_LONG', 'COIL_SHORT', None
    permission_timer = 0
    
    in_trade = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    trade_start_idx = 0
    trade_type = None # 'LONG', 'SHORT'
    regime_trigger = None # 'WAKE', 'COIL'
    
    # Pre-calc candle bodies for Anchor Rule
    data['body'] = abs(data['close'] - data['open'])
    # Rolling max body (shift 1 to not include current, though for anchor we compare current to past? 
    # "is the displacement candle largest". So compare current to rolling previous.)
    data['max_body_prev'] = data['body'].rolling(ANCHOR_WINDOW).max().shift(1)
    
    print("Running Mechanical Simulation...")
    
    for i in range(ANCHOR_WINDOW, len(data)):
        # Market Data
        row = data.iloc[i]
        bar_open = row['open']
        bar_high = row['high']
        bar_low = row['low']
        bar_close = row['close']
        
        # Features
        p_row = prim.iloc[i]
        z_row = zone.iloc[i]
        
        # --- 1. Trade Management (If In Trade) ---
        if in_trade:
            outcome = None
            exit_price = 0.0
            
            # Check SL/TP
            if trade_type == 'LONG':
                if bar_low <= sl_price:
                    outcome = 'SL'
                    exit_price = min(bar_open, sl_price) # Realistic slip
                elif bar_high >= tp_price:
                    outcome = 'TP'
                    exit_price = max(bar_open, tp_price)
            elif trade_type == 'SHORT':
                if bar_high >= sl_price:
                    outcome = 'SL'
                    exit_price = max(bar_open, sl_price)
                elif bar_low <= tp_price:
                    outcome = 'TP'
                    exit_price = min(bar_open, tp_price)
            
            # Check Time Stop
            bars_in_trade = i - trade_start_idx
            if outcome is None and bars_in_trade >= TIME_STOP_BARS:
                outcome = 'TIME'
                exit_price = bar_open # Exit at open of 21st bar (or close of 20th? "If 20 bars pass... exit at market") -> exit at current open
                
            if outcome:
                # Log Trade
                pnl_pts = (exit_price - entry_price) if trade_type == 'LONG' else (entry_price - exit_price)
                risk_pts = abs(entry_price - sl_price)
                r_multiple = (pnl_pts / risk_pts) - FRICTION_R # Subtract Friction check
                
                trades.append({
                    'entry_time': data.index[trade_start_idx],
                    'exit_time': data.index[i],
                    'type': trade_type,
                    'regime': regime_trigger,
                    'outcome': outcome,
                    'r_net': r_multiple,
                    'duration': bars_in_trade
                })
                
                in_trade = False
                entry_price = 0.0
                permission_state = None # Reset permission on trade completion? Or keep it if window open? 
                # Playbook doesn't say. Usually assume "one shot" per permission or hold means busy. 
                # "Walk Away" implies one trade per setup.
            
            continue # Skip detection if in trade
            
        # --- 2. Permission Decay ---
        if permission_state:
            permission_timer += 1
            if permission_timer > WINDOW_BARS:
                permission_state = None
                permission_timer = 0
        
        # --- 3. Regime Detection (If No Permission) ---
        # Only look for new regime if we don't have one? Or can new override old? 
        # Assume new overrides or refreshes.
        
        # WAKE Detection
        # Disp > 2.0 AND Anchor Rule (Body > Max Prev 100)
        is_wake_long = p_row['disp_is_displacement'] and p_row['disp_range_zscore'] > 2.0 and row['close'] > row['open']
        is_wake_short = p_row['disp_is_displacement'] and p_row['disp_range_zscore'] > 2.0 and row['close'] < row['open']
        
        if is_wake_long:
            if row['body'] > row['max_body_prev']: # Anchor Rule
                permission_state = 'WAKE_LONG'
                permission_timer = 0
        elif is_wake_short:
             if row['body'] > row['max_body_prev']: # Anchor Rule
                permission_state = 'WAKE_SHORT'
                permission_timer = 0
                
        # COIL Detection
        # Comp > 0.8 (High) and Approaching Zone
        is_coil = p_row['comp_score'] > 0.8 and z_row['approaching']
        
        if is_coil and not permission_state: # Wake takes precedence? Or Coil? 
            # Check direction based on zone?
            # "Approaching a known level".
            # If dist to zone is positive (above zone?), we are approaching support -> Long?
            # `dist_to_nearest_zone` is signed.
            # Usually: Positive dist means price > zone (Support). Negative dist means price < zone (Resist).
            
            dist = z_row['dist_to_nearest_zone']
            if dist > 0: # Above support, approaching -> Anticipate Bounce (Long) or Break (Short)?
                # Coil implies reversal/breakout. Playbook says "Reversal/Breakout". 
                # Let's assume Reversal for "Compression to Structure" (Coil usually implies storing energy to reverse or pop).
                # Actually, Coil can be continuation.
                # Let's simplify: Permission to trade BOTH ways if coiled at zone? 
                # Playbook: "Look for reversal/breakout".
                # Let's set generic COIL permission and let Color Match decide direction.
                permission_state = 'COIL_ANY' 
                permission_timer = 0
            elif dist < 0: # Below resistance
                 permission_state = 'COIL_ANY'
                 permission_timer = 0

        # --- 4. Rejection Rule (Coil Fail) ---
        # If Coil permission active, check if we blasted through zone
        if permission_state == 'COIL_ANY':
            # If Disp > 3.0 (Example) and we crossed the zone line?
            # Or simply if we are no longer approaching and valid?
            # Start simple: If strict displacement AWAY from zone happens, that's the trade trigger (Breakout).
            pass 

        # --- 5. Entry Trigger (Execution) ---
        if permission_state:
            is_green = row['close'] > row['open']
            is_red = row['close'] < row['open']
            
            if 'LONG' in permission_state or permission_state == 'COIL_ANY':
                if is_green: # Confirmation
                    # Enter Long Next Bar
                    in_trade = True
                    trade_type = 'LONG'
                    trade_start_idx = i
                    # Exec at current close (Simpler for backtest) or next open. 
                    # Script rules say "Market or Limit at candle close". 
                    # We'll use Close for Entry price.
                    entry_price = row['close']
                    sl_price = entry_price - SL_PTS
                    tp_price = entry_price + TP_PTS
                    regime_trigger = 'WAKE' if 'WAKE' in permission_state else 'COIL'
                    
                    # Consumed permission
                    permission_state = None
                    continue

            if 'SHORT' in permission_state or permission_state == 'COIL_ANY':
                if is_red: # Confirmation
                    # Enter Short
                    in_trade = True
                    trade_type = 'SHORT'
                    trade_start_idx = i
                    entry_price = row['close']
                    sl_price = entry_price + SL_PTS
                    tp_price = entry_price - TP_PTS
                    regime_trigger = 'WAKE' if 'WAKE' in permission_state else 'COIL'
                    
                    permission_state = None
                    continue

    return pd.DataFrame(trades)

def main():
    data = load_data()
    # data = data.iloc[-100000:] # Last ~100k bars (approx 6-8 months)
    
    print("Building Primitives...")
    fb = FeatureBuilder()
    prim = fb.build_feature_matrix(data)
    
    print("Building Zone Features...")
    zone = build_zone_relative_features(data)
    
    trades_df = run_audit(data, prim, zone)
    
    if len(trades_df) == 0:
        print("No trades found matching strict criteria.")
        return

    print("="*60)
    print("CONTINUUM EXECUTION AUDIT")
    print("="*60)
    print(f"Total Trades: {len(trades_df)}")
    print(f"Net Reward: {trades_df['r_net'].sum():.2f} R")
    print(f"Avg Reward: {trades_df['r_net'].mean():.3f} R/Trade")
    print(f"Win Rate: {(trades_df['outcome'] == 'TP').mean()*100:.1f}%")
    print(f"Time Stops: {(trades_df['outcome'] == 'TIME').mean()*100:.1f}%")
    print("-" * 30)
    print("By Regime:")
    print(trades_df.groupby('regime')['r_net'].agg(['count', 'sum', 'mean']))
    print("="*60)
    
    # Save log
    os.makedirs("audit", exist_ok=True)
    trades_df.to_csv("audit/continuum_audit_log.csv")
    print("Audit log saved to audit/continuum_audit_log.csv")

if __name__ == "__main__":
    main()
