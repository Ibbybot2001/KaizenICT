"""
TITAN INTRADAY SEARCH ENGINE
Targeting High-Frequency US Session Trades (IB Sweep / Silver Bullet)

PROTOCOL: SILENT RUNNING (No Console Output to prevent crashes)
"""

import pandas as pd
import itertools
import multiprocessing
from multiprocessing import Pool
import logging
import time
import sys
from pathlib import Path

# Engine
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase16_pj_engine import engineer_pools, detect_pj_signals, PJBacktester

# ==============================================================================
# CONFIG
# ==============================================================================
NUM_CORES = 14
OUTPUT_DIR = Path("C:/Users/CEO/ICT reinforcement/output/intraday_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING SETUP (FILE ONLY)
logging.basicConfig(
    filename=OUTPUT_DIR / "intraday_search.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'
)
logger = logging.getLogger()

# DATA
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_MONTHS = [9, 10, 11, 12]

# PARAM SPACE (Targeting 3-5 trades/day)
POOLS = [
    ['IB_L', 'IB_H'], # Opening Range Breakout
    ['IB_L', 'IB_H', 'PDL', 'PDH'], # ORB + Daily Levels
    ['IB_L', 'IB_H', 'ONL', 'ONH'], # ORB + Overnight
] 

DISP_THRESHOLDS = [0.2, 0.3, 0.4]
DIRECTIONS = ['BOTH'] # Intraday requires flexibility
SL_BUFFERS = [3.0, 5.0, 7.5, 10.0]
SESSIONS = [(10, 0, 15, 30)] # Start after IB forms (10:00)

MAX_TRADES_DAILY = [5] # Cap at 5 to encourage quality

# ==============================================================================
# WORKER
# ==============================================================================
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
    idx, params = args # Data now in globals
    
    try:
        # TRAIN
        train_res = run_backtest(TRAIN_DF, params)
        # if train_res['trades'] < 150: return None # FILTER DISABLED
        # if train_res['pf'] < 1.1: return None    # FILTER DISABLED
        
        # TEST
        test_res = run_backtest(TEST_DF, params)
        # if test_res['pf'] < 1.0: return None     # FILTER DISABLED
        
        # RESULT
        result = {
            'param_id': idx,
            'params': params,
            'train_trades': train_res['trades'],
            'train_per_day': train_res['trades'] / 160.0, # Approx days
            'train_pf': train_res['pf'],
            'train_exp': train_res['expectancy'],
            'test_trades': test_res['trades'],
            'test_pf': test_res['pf'],
            'test_exp': test_res['expectancy']
        }
        return result
        
    except Exception as e:
        return None

def run_backtest(bars_df, params):
    # Setup
    # tester = PJBacktester(bars_df) 
    
    # dates = bars_df['date'].unique()
    trades = []
    
    # Fast iteration
    pools_active = set(params['pools'])
    h_start, m_start, h_end, m_end = params['session']
    
    # Filter bars in session
    mask_sess = ((bars_df['hour'] == h_start) & (bars_df['minute'] >= m_start)) | \
                ((bars_df['hour'] > h_start) & (bars_df['hour'] < h_end)) | \
                ((bars_df['hour'] == h_end) & (bars_df['minute'] <= m_end))
    
    # Only cycle days
    grouped = bars_df[mask_sess].groupby('date')
    
    for date, day_bars in grouped:
        daily_trades = 0
        pool_status = {p: 'DEFINED' for p in pools_active}
        
        for i, row in day_bars.iterrows():
            if daily_trades >= params['max_trades']: break
            
            # Check pools
            for pool in pools_active:
                if pool_status[pool] != 'DEFINED': continue
                level = row.get(pool)
                if pd.isna(level): continue
                
                # Logic: Sweep + Reclaim
                is_low_pool = 'L' in pool # Crude check
                
                trigger = False
                sl = 0.0
                tp = 0.0
                direction = 0
                
                if is_low_pool:
                    if row['low'] < level and row['close'] > level:
                         # Displacement check
                         body = abs(row['close'] - row['open'])
                         if body >= params['disp_threshold']:
                             trigger = True
                             direction = 1
                             sl = row['low'] - params['sl_buffer']
                             tp_target_col = pool.replace('L', 'H')
                             tp = row.get(tp_target_col, row['close'] + 50) 
                else:
                    if row['high'] > level and row['close'] < level:
                        body = abs(row['close'] - row['open'])
                        if body >= params['disp_threshold']:
                            trigger = True
                            direction = -1
                            sl = row['high'] + params['sl_buffer']
                            tp_target_col = pool.replace('H', 'L')
                            tp = row.get(tp_target_col, row['close'] - 50)
                
                if trigger:
                    # Sim outcome (Simplified)
                    # Look ahead in day_bars
                    outcome = 0
                    if direction == 1:
                        # Future bars
                        future = day_bars.loc[i:].iloc[1:]
                        for _, f_row in future.iterrows():
                            if f_row['low'] <= sl:
                                outcome = sl - row['close']
                                break
                            if f_row['high'] >= tp:
                                outcome = tp - row['close']
                                break
                    else:
                        future = day_bars.loc[i:].iloc[1:]
                        for _, f_row in future.iterrows():
                            if f_row['high'] >= sl:
                                outcome = row['close'] - sl
                                break
                            if f_row['low'] <= tp:
                                outcome = row['close'] - tp
                                break
                                
                    trades.append(outcome)
                    pool_status[pool] = 'TRADED'
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
    # 1. LOAD DATA
    logger.info("Loading 2025 Golden Data...")
    all_bars = []
    for m in TRAIN_MONTHS + TEST_MONTHS:
        try:
            p = BASE_DIR / f"USTEC_2025-{m:02d}_clean_1m.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df['time'] = pd.to_datetime(df.index)
                df['date'] = df['time'].dt.date
                df['hour'] = df['time'].dt.hour
                df['minute'] = df['time'].dt.minute
                all_bars.append(df)
        except: pass
    
    full_df = pd.concat(all_bars)
    
    # ENGINEER POOLS (Includes IB)
    full_df = engineer_pools(full_df)
    
    # SPLIT
    train_mask = full_df['time'].dt.month.isin(TRAIN_MONTHS)
    train_df = full_df[train_mask]
    test_df = full_df[~train_mask]
    
    logger.info(f"Data Loaded. Train: {len(train_df)} bars, Test: {len(test_df)} bars.")
    
    # 2. GENERATE COMBOS
    param_grid = {
        'pools': POOLS,
        'disp_threshold': DISP_THRESHOLDS,
        'direction': DIRECTIONS,
        'session': SESSIONS,
        'sl_buffer': SL_BUFFERS,
        'max_trades': MAX_TRADES_DAILY,
        'require_disp': [True]
    }
    
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    logger.info(f"Generated {len(combos)} combinations.")
    
    # 3. RUN PARALLEL
    logger.info(f"Starting Search on {NUM_CORES} cores...")
    
    # Pass only (i, c)
    work_items = [(i, c) for i, c in enumerate(combos)]
    print(f"Starting Search: {len(combos)} items. Check 'output/intraday_search/intraday_search.log' for progress.")
    
    results = []
    
    # Init worker with global data
    with Pool(NUM_CORES, initializer=init_worker, initargs=(train_df, test_df)) as pool:
        for res in pool.imap_unordered(evaluate_params, work_items):
            if res:
                results.append(res)
                logger.info(f"DATA: {res['params']} | PF: {res['test_pf']:.2f} | Trades: {res['test_trades']}")
                # Auto-save every winner to CSV to avoid data loss
                pd.DataFrame([res]).to_csv(OUTPUT_DIR / "intraday_results_raw.csv", mode='a', header=not (OUTPUT_DIR / "intraday_results_raw.csv").exists(), index=False)
    
    logger.info("Search Complete.")
    print("Search Complete.")
