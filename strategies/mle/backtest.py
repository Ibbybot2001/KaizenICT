"""
MLE Backtest Orchestrator
Stitches implementation of SignalEngine and ExecutionEngine.
Simulates the 'Golden Engine' logic (66% WR) using M1 data with synthetic tick generation.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings

from strategies.mle.signals import SignalEngine, LiquidityManager
from strategies.mle.execution import TickSimulator, TradeIntent, TradeResult

import os
import gc

from dataclasses import dataclass

@dataclass
class BacktestConfig:
    # Core
    tick_size: float = 0.25
    expiry_minutes: int = 20
    stop_loss_padding: float = 2.0
    
    # Realism Parameters
    fvg_tolerance_ticks: int = 2
    latency_ms: int = 500
    execution_mode: str = 'LIMIT' # 'LIMIT', 'MARKET', 'MARKET_DELAYED', 'BREAKER'
    
    # Advanced Filters (Concept Library)
    filter_killzone: bool = False
    filter_trend_ema: bool = False
    filter_dealing_range: bool = False
    
    # Entry Logic Override
    entry_strategy: str = 'FVG' # 'FVG', 'BREAKER', 'WICK_50', 'CLOSE'

# Defaults
DEFAULT_CONFIG = BacktestConfig()

# Global Constants (Restored for load_data)
BASE_DIR = r"C:\Users\CEO\ICT reinforcement\data\GOLDEN_DATA\USTEC_2025_GOLDEN_PARQUET"
MONTH = "2025-01"
M1_FILE = f"USTEC_{MONTH}_clean_1m.parquet"
TICK_FILE = f"USTEC_{MONTH}_clean_ticks.parquet"

M1_PATH = os.path.join(BASE_DIR, M1_FILE)
TICK_PATH = os.path.join(BASE_DIR, TICK_FILE)
TICK_SIZE = 0.25 # Also needed

def load_data():
    print(f"Loading M1 Data: {M1_PATH}")
    df_m1 = pd.read_parquet(M1_PATH)
    
    if not isinstance(df_m1.index, pd.DatetimeIndex):
        if 'time' in df_m1.columns:
            df_m1 = df_m1.set_index('time').sort_index()
            
    if df_m1.index.tz is not None:
        df_m1.index = df_m1.index.tz_localize(None)
    
    df_m1.columns = [c.lower() for c in df_m1.columns]
    
    print(f"Loading Tick Data: {TICK_PATH}")
    df_ticks = pd.read_parquet(TICK_PATH)
    
    if not isinstance(df_ticks.index, pd.DatetimeIndex):
         if 'time' in df_ticks.columns:
            df_ticks = df_ticks.set_index('time').sort_index()

    if df_ticks.index.tz is not None:
        df_ticks.index = df_ticks.index.tz_localize(None)
        
    df_ticks.columns = [c.lower() for c in df_ticks.columns]
    
    if 'bid' not in df_ticks.columns:
        if 'price' in df_ticks.columns:
            df_ticks['bid'] = df_ticks['price']
            df_ticks['ask'] = df_ticks['price'] + TICK_SIZE
        elif 'close' in df_ticks.columns: 
             df_ticks['bid'] = df_ticks['close']
             df_ticks['ask'] = df_ticks['close'] + TICK_SIZE
    
    return df_m1, df_ticks

def run_backtest(config: BacktestConfig = DEFAULT_CONFIG, df_m1=None, df_ticks=None):
    print(f"Initializing Metrics | Mode: {config.execution_mode} | Entry: {config.entry_strategy} | Latency: {config.latency_ms}ms")
    
    if df_m1 is None or df_ticks is None:
        try:
            df_m1, df_ticks = load_data()
        except Exception as e:
            print(f"CRITICAL ERROR LOADING DATA: {e}")
            return None

    print(f"Data Ready. M1: {len(df_m1)}, Ticks: {len(df_ticks)}")
    
    print("Pre-computing Signals (Signal Engine)...")
    se = SignalEngine(df_m1)
    # Inject filters into SE if needed, or handle in loop
    sig_df = se.precompute_market_structure()
    
    # Pre-calc Indicators
    if config.filter_trend_ema:
        print("Calculating EMA(50)...")
        # Ensure we don't overwrite if re-using DF?
        # Actually df_m1 is passed in. Modifying it is efficient for Batch.
        if 'ema50' not in df_m1.columns:
            df_m1['ema50'] = df_m1['close'].ewm(span=50, adjust=False).mean()
            
    # Pre-calc Time Features if needed
    
    lm = LiquidityManager()
    ts = TickSimulator(tick_size=config.tick_size, slippage_ticks=0)
    
    trades = []
    
    # State
    active_intent = None
    total_bars = len(sig_df)
    
    for i in range(50, total_bars - 1): # Start after 50 for EMA
        if i % 5000 == 0:
            print(f"Processing Bar {i}/{total_bars}...")
            
        row = sig_df.iloc[i]
        curr_time = row.name
        
        # --- GLOBAL FILTERS ---
        # 1. Killzone (Strict NY AM: 09:30 - 11:00)
        # Assuming timestamps are consistent. If data is 24h, this filters significant noise.
        if config.filter_killzone:
             # Naive check: Hour/Min
             t = curr_time
             # 9:30 to 11:00
             in_kz = False
             if t.hour == 9 and t.minute >= 30: in_kz = True
             elif t.hour == 10: in_kz = True
             elif t.hour == 11 and t.minute == 0: in_kz = True # Exactly 11:00
             
             if not in_kz:
                 continue # Skip processing
                 
        # 1. Update Liquidity 
        t_minus_2 = sig_df.iloc[i-2]
        if t_minus_2['is_swing_high']:
            lm.add_swing(t_minus_2['high'], 'High', i-2)
        if t_minus_2['is_swing_low']:
            lm.add_swing(t_minus_2['low'], 'Low', i-2)
            
        # 2. Check Triggers
        swept_pools = lm.check_sweeps(row['high'], row['low'])
        
        trigger_long = False
        trigger_short = False
        
        if swept_pools:
            if row['morphology'] > 0.30:
                swept_lows = [p for p in swept_pools if p.type == 'Low']
                swept_highs = [p for p in swept_pools if p.type == 'High']
                
                # --- TREND FILTER ---
                if config.filter_trend_ema:
                    ema_val = df_m1['ema50'].iloc[i]
                    # Only Long if Price > EMA ? Or Reversal: Price WAS below, now closing above?
                    # Standard Trend Following: Trade WITH Trend.
                    # If EMA is rising? Or Price > EMA.
                    # Let's use Price > EMA for Longs.
                    if row['close'] < ema_val: swept_lows = [] # Disable Longs
                    if row['close'] > ema_val: swept_highs = [] # Disable Shorts
                
                # --- DEALING RANGE FILTER ---
                if config.filter_dealing_range:
                    # Lookback 60 mins
                    window = df_m1.iloc[i-60:i]
                    if not window.empty:
                        r_high = window['high'].max()
                        r_low = window['low'].min()
                        mid = (r_high + r_low) / 2
                        
                        # Longs must be in Discount (Price < Mid)
                        if row['close'] > mid: swept_lows = []
                        # Shorts must be in Premium (Price > Mid)
                        if row['close'] < mid: swept_highs = []

                if swept_lows and row['close'] > row['open']: 
                    trigger_long = True
                if swept_highs and row['close'] < row['open']:
                    trigger_short = True

        # 3. Generate Intent
        if trigger_long or trigger_short:
            limit_price = 0.0
            sl_price = 0.0
            valid_setup = False
            
            # --- CONCEPT LOGIC BRANCHING ---
            
            if trigger_long:
                # ENTRY STRATEGY SELECTOR
                if config.entry_strategy == 'MARKET_CLOSE': # Concept 1, 2
                    valid_setup = True
                    limit_price = row['close']
                    sl_price = row['low'] - config.stop_loss_padding
                    
                elif config.entry_strategy == 'BREAKER': # Concept 3
                    # Breaker = The Swing High that was broken? 
                    # No, for Long, Retest of Broken Swing High (if reversal)
                    # Actually, standard Breaker is: High-Low-HigherHigh-LowerLow.
                    # Simplified Breaker here: Limit at the Swing Low that was swept? No.
                    # Limit at the High of the candle that created the Low? 
                    # Let's use: Limit at Open of T (Retest of Open)
                    limit_price = row['open']
                    sl_price = row['low'] - config.stop_loss_padding
                    valid_setup = True

                elif config.entry_strategy == 'FVG': # Original
                    fvg_bottom = row['fvg_bull_bottom']
                    tolerance_val = config.fvg_tolerance_ticks * config.tick_size
                    if row['low'] >= (fvg_bottom - tolerance_val):
                         valid_setup = True
                         limit_price = row['low']
                         sl_price = fvg_bottom - config.stop_loss_padding
                         
                if valid_setup:
                    risk = limit_price - sl_price
                    if risk <= 0: continue
                    tp1 = limit_price + 2.0 * risk
                    tp2 = limit_price + 4.0 * risk
                    
                    intent = TradeIntent(
                        ticket_id=f"L_{curr_time}",
                        direction=1,
                        entry_price=limit_price,
                        sl_price=sl_price,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        start_time=curr_time + timedelta(minutes=1), 
                        expiry_time=curr_time + timedelta(minutes=config.expiry_minutes)
                    )
                    intent.order_type = config.execution_mode
                    active_intent = intent

            elif trigger_short:
                 # ENTRY STRATEGY SELECTOR
                if config.entry_strategy == 'MARKET_CLOSE':
                    valid_setup = True
                    limit_price = row['close']
                    sl_price = row['high'] + config.stop_loss_padding
                    
                elif config.entry_strategy == 'BREAKER':
                    limit_price = row['open']
                    sl_price = row['high'] + config.stop_loss_padding
                    valid_setup = True

                elif config.entry_strategy == 'FVG':
                    fvg_top = row['fvg_bear_top'] 
                    tolerance_val = config.fvg_tolerance_ticks * config.tick_size
                    if row['high'] <= (fvg_top + tolerance_val):
                         valid_setup = True
                         limit_price = row['high']
                         sl_price = fvg_top + config.stop_loss_padding

                if valid_setup:
                    risk = sl_price - limit_price
                    if risk <= 0: continue
                    tp1 = limit_price - 2.0 * risk
                    tp2 = limit_price - 4.0 * risk
                    
                    intent = TradeIntent(
                        ticket_id=f"S_{curr_time}",
                        direction=-1,
                        entry_price=limit_price,
                        sl_price=sl_price,
                        tp1_price=tp1,
                        tp2_price=tp2,
                        start_time=curr_time + timedelta(minutes=1),
                        expiry_time=curr_time + timedelta(minutes=config.expiry_minutes)
                    )
                    intent.order_type = config.execution_mode
                    active_intent = intent
        
        # 4. Simulate Active Intent
        if active_intent:
            sim_start = active_intent.start_time + timedelta(milliseconds=config.latency_ms)
            sim_end = active_intent.expiry_time
            
            try:
                tick_slice = df_ticks[sim_start:sim_end]
                if not tick_slice.empty:
                    result = ts.simulate_trade(active_intent, tick_slice)
                    if result.outcome not in ['CANCELLED', 'FAIL']:
                        trades.append(result)
            except Exception:
                pass
            active_intent = None
            
    # Reporting
    if not trades:
        res = {'trades': 0, 'wr': 0.0, 'pf': 0.0, 'pnl': 0.0}
        print(f"RESULT: {res}")
        return res

    res_data = []
    for t in trades:
        res_data.append({
            'outcome': t.outcome,
            'pnl_ticks': t.pnl_ticks
        })
        
    df_res = pd.DataFrame(res_data)
    wins = df_res[df_res['pnl_ticks'] > 0]
    losses = df_res[df_res['pnl_ticks'] < 0]
    
    wr = len(wins) / len(df_res) * 100
    gross_win = wins['pnl_ticks'].sum()
    gross_loss = abs(losses['pnl_ticks'].sum())
    pf = gross_win / gross_loss if gross_loss != 0 else 999.0
    
    final_res = {
        'trades': len(trades), 
        'wr': wr, 
        'pf': pf, 
        'pnl': df_res['pnl_ticks'].sum()
    }
    print(f"RESULT: {final_res}")
    return final_res

if __name__ == "__main__":
    run_backtest()
