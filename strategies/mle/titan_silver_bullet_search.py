"""
TITAN SEARCH: SILVER BULLET (Phase 17)
Targeting 10:00 AM - 11:00 AM FVG Entries.

PROTOCOL: SILENT RUNNING + INITIALIZER
"""

import pandas as pd
import itertools
import multiprocessing
from multiprocessing import Pool
import logging
import sys
from pathlib import Path

# Engine
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase17_fvg_engine import engineer_fvg, detect_silver_bullet_signals
# Borrow feature engineering from P16 if needed (HLs etc), but SB is mostly price action.

# ==============================================================================
# CONFIG
# ==============================================================================
NUM_CORES = 14
OUTPUT_DIR = Path("C:/Users/CEO/ICT reinforcement/output/silver_bullet_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=OUTPUT_DIR / "sb_search.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'
)
logger = logging.getLogger()

BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_MONTHS = [9, 10, 11, 12]

# PARAM SPACE
TP_TARGETS = [20, 30, 40, 50, 60]
SL_BUFFERS = [5.0, 10.0, 15.0]
ENTRY_OFFSETS = [0.0, 2.0] # 0.0 = Proximal Line, 2.0 = Deep in gap? No, 2.0 buffer entry?
# Actually SB is Limit Entry at Proximal.

# Filter: Trend Alignment?
# Simple moving average trend?
TREND_FILTERS = [False] # Start raw

MAX_TRADES_DAILY = [2] # Only 1-2 shots in the 10-11am hour

# ==============================================================================
# WORKER
# ==============================================================================
TRAIN_DF = None
TEST_DF = None

def init_worker(train_df, test_df):
    global TRAIN_DF, TEST_DF
    TRAIN_DF = train_df
    TEST_DF = test_df

def evaluate_params(args):
    idx, params = args
    
    try:
        # TRAIN
        train_res = run_backtest(TRAIN_DF, params)
        # Filters disabled for Pilot
        
        # TEST
        test_res = run_backtest(TEST_DF, params)
        
        return {
            'param_id': idx,
            'params': params,
            'train_trades': train_res['trades'],
            'train_pf': train_res['pf'],
            'train_exp': train_res['expectancy'],
            'test_trades': test_res['trades'],
            'test_pf': test_res['pf'],
            'test_exp': test_res['expectancy']
        }
        
    except Exception as e:
        return None

def run_backtest(bars_df, params):
    # Silver Bullet Logic
    # 1. 10am-11am window.
    # 2. Detect FVG.
    # 3. Enter Limit.
    
    current_date = None
    daily_trades = 0
    trades = []
    
    # Iterate Days? 
    # Or Iterate Bars (Slow)?
    # Faster: Detect All Signals First?
    # No, signals depend on params? 
    # Actually, FVG existence is constant. Entry/SL/TP varies.
    
    # Pre-filter Session Bars: 10:00 - 11:00
    # Actually logic allows entries slightly after? No, SB is window.
    
    mask_sb = (bars_df['hour'] == 10)
    sb_bars = bars_df[mask_sb]
    
    grouped = sb_bars.groupby('date')
    
    for date, day_bars in grouped:
        daily_trades = 0
        
        # We need the FULL day bars to sim exit (it might run past 11am)
        # This is the tricky part. 'sb_bars' assumes exit within hour? No.
        # We need to access full 'bars_df' for the day.
        
        # Optimization: Pass full bars_df but use 'date' to slice?
        # Slow slicing. 
        # Better: Pre-slice full days in memory?
        pass # To handle inside loop
    
    # RE-THINK: 
    # If we iterate ALL bars, it's slow.
    # But checking 14 cores means we can afford some slowness.
    
    # Let's iterate DAYS in full_df (grouped).
    grouped_full = bars_df.groupby('date')
    
    for date, day_rows in grouped_full:
        daily_trades = 0
        
        # Slice 10am-11am for Signals
        # Assuming sorted by time
        
        # Find 10am index
        # This is getting complex for a 1-file script.
        # Let's rely on 'detect_silver_bullet_signals' logic which iterates rows.
        
        # To make it fast:
        # 1. Signals detected on 10am rows.
        # 2. If Signal -> Sim outcome on remaining rows.
        
        # Optimization: Pre-compute FVG columns 'fvg_bull', 'fvg_bear' in INIT.
        # Yes, assume 'engineer_fvg' was called globally once.
        
        # Iterate over 10am rows only
        # day_rows is a DF.
        
        # Window: Hour 10
        am_rows = day_rows[day_rows['hour'] == 10]
        if am_rows.empty: continue
        
        for i, row in am_rows.iterrows():
            if daily_trades >= params['max_trades']: break
            
            signal = None
            entry, sl, tp = 0,0,0
            direction = 0
            
            # Check FVG (Pre-computed)
            if row['fvg_bull']:
                entry = row['fvg_bull_top']
                sl = row['low'] - params['sl_buffer']
                tp = entry + params['tp_target']
                direction = 1
                signal = True
            elif row['fvg_bear']:
                entry = row['fvg_bear_btm']
                sl = row['high'] + params['sl_buffer']
                tp = entry - params['tp_target']
                direction = -1
                signal = True
            
            # TREND FILTER
            if signal and params.get('trend_col'):
                trend_val = row.get(params['trend_col'])
                if pd.isna(trend_val): 
                    signal = False
                else:
                    # Long must be > Trend
                    if direction == 1 and row['close'] < trend_val: signal = False
                    # Short must be < Trend
                    if direction == -1 and row['close'] > trend_val: signal = False
                
            if signal:
                # SIMULATE
                # We need to see if Price HITS 'entry' (Limit Order) in FUTURE bars
                # Then if it hits SL or TP.
                
                # Get future rows in this day
                # Since 'i' is index in full DF?
                # Yes, i is global index.
                # slice: day_rows.loc[i+1:]
                
                future = day_rows.loc[i:].iloc[1:] # Bars AFTER signal bar
                
                filled = False
                outcome = 0
                
                for _, f_row in future.iterrows():
                    # 1. Check Fill
                    if not filled:
                        if direction == 1:
                            if f_row['low'] <= entry: filled = True
                        else:
                            if f_row['high'] >= entry: filled = True
                            
                    # 2. Check Outcome (if filled)
                    if filled:
                        if direction == 1:
                            if f_row['low'] <= sl:
                                outcome = sl - entry
                                break
                            if f_row['high'] >= tp:
                                outcome = tp - entry
                                break
                        else:
                            if f_row['high'] >= sl:
                                outcome = entry - sl
                                break
                            if f_row['low'] <= tp:
                                outcome = entry - tp
                                break
                                
                    # Timeout? End of day close.
                    
                if filled:
                    trades.append(outcome)
                    daily_trades += 1
                    
    # Metrics
    if not trades: return {'trades':0, 'pf':0, 'expectancy':0}
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 10.0
    exp = sum(trades) / len(trades)
    return {'trades': len(trades), 'pf': pf, 'expectancy': exp}

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    # 1. LOAD & ENGINEER
    logger.info("Loading Data...")
    all_bars = []
    for m in TRAIN_MONTHS + TEST_MONTHS:
        try:
            p = BASE_DIR / f"USTEC_2025-{m:02d}_clean_1m.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df['time'] = pd.to_datetime(df.index)
                df['date'] = df['time'].dt.date
                df['hour'] = df['time'].dt.hour
                all_bars.append(df)
        except: pass
    full_df = pd.concat(all_bars)
    
    # ENGINEER FVG ONCE
    full_df = engineer_fvg(full_df)
    
    # ENGINEER TREND (SMA)
    # SMA 200 (1m) ~= 3 Hours
    # SMA 800 (1m) ~= 13 Hours (Daily Bias)
    full_df['SMA_200'] = full_df['close'].rolling(200).mean()
    full_df['SMA_800'] = full_df['close'].rolling(800).mean()
    
    # Split
    train_mask = full_df['time'].dt.month.isin(TRAIN_MONTHS)
    train_df = full_df[train_mask]
    test_df = full_df[~train_mask]
    
    # 2. COMBOS
    param_grid = {
        'tp_target': TP_TARGETS,
        'sl_buffer': SL_BUFFERS,
        'max_trades': MAX_TRADES_DAILY,
        'trend_col': [None, 'SMA_200', 'SMA_800']
    }
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 3. RUN
    work_items = [(i, c) for i, c in enumerate(combos)]
    print(f"Silver Bullet Search: {len(combos)} items.")
    
    with Pool(NUM_CORES, initializer=init_worker, initargs=(train_df, test_df)) as pool:
        for res in pool.imap_unordered(evaluate_params, work_items):
            if res:
                logger.info(f"DATA: {res['params']} | PF: {res['test_pf']:.2f} | Trades: {res['test_trades']}")
                pd.DataFrame([res]).to_csv(OUTPUT_DIR / "sb_results_raw.csv", mode='a', header=not (OUTPUT_DIR / "sb_results_raw.csv").exists(), index=False)
                
    print("Search Complete.")
