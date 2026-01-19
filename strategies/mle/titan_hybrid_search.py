"""
TITAN SEARCH: HYBRID (Sweep + FVG)
Evolution of Intraday Strategy.
logic:
1. Wait for Liquidity Sweep (IB, ONL, PDL).
2. Wait for FVG in reversal direction.
3. Enter Limit at FVG.

PROTOCOL: SILENT RUNNING + INITIALIZER
"""

import pandas as pd
import itertools
import multiprocessing
from multiprocessing import Pool
import logging
import sys
from pathlib import Path

# Engines
sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
from strategies.mle.phase16_pj_engine import engineer_pools
from strategies.mle.phase17_fvg_engine import engineer_fvg

# ==============================================================================
# CONFIG
# ==============================================================================
NUM_CORES = 14
OUTPUT_DIR = Path("C:/Users/CEO/ICT reinforcement/output/hybrid_search")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=OUTPUT_DIR / "hybrid_search.log",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'
)
logger = logging.getLogger()

BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_MONTHS = [9, 10, 11, 12]

# PARAM SPACE
POOLS = [
    ['IB_L', 'IB_H'],
    ['ONL', 'ONH'],
    ['IB_L', 'IB_H', 'ONL', 'ONH'],
    ['PDL', 'PDH']
]
TP_TARGETS = [20, 30, 40, 50, 60, 80, 100]
SL_BUFFERS = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0]
MAX_TRADES = [2]

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
        train_res = run_backtest(TRAIN_DF, params)
        test_res = run_backtest(TEST_DF, params)
        return {
            'param_id': idx, 'params': params,
            'train_pf': train_res['pf'], 'train_trades': train_res['trades'],
            'test_pf': test_res['pf'], 'test_trades': test_res['trades']
        }
    except: return None

def run_backtest(bars_df, params):
    # Hybrid Logic: Sweep -> FVG -> Entry
    trades = []
    daily_trades = 0
    
    # Active Pools
    active_pools = set(params['pools'])
    
    # Filter Trading Hours (e.g. 10:00 - 16:00)
    # We allow sweep of IB (9:30-10) but trade usually happens after 10.
    mask_trade = (bars_df['hour'] >= 10) & (bars_df['hour'] < 16)
    
    grouped = bars_df[mask_trade].groupby('date')
    
    for date, day_bars in grouped:
        daily_trades = 0
        
        # State
        swept_low = False
        swept_high = False
        
        # Iterating bar by bar
        for i, row in day_bars.iterrows():
            if daily_trades >= params['max_trades']: break
            
            # 1. CHECK SWEEPS (Prices vs Levels)
            # Levels are columns: IB_H, IB_L, ONH, etc.
            
            # Check Low Pools
            if not swept_low:
                for p in active_pools:
                    if 'L' in p:
                        lvl = row.get(p)
                        if pd.notna(lvl) and row['low'] < lvl:
                            swept_low = True
            
            # Check High Pools
            if not swept_high:
                for p in active_pools:
                    if 'H' in p:
                        lvl = row.get(p)
                        if pd.notna(lvl) and row['high'] > lvl:
                            swept_high = True
                            
            # 2. CHECK FVG (If Swept)
            signal = False
            direction = 0
            entry, sl, tp = 0,0,0
            
            # Long Setup: Swept Low AND Bullish FVG
            if swept_low and row['fvg_bull']:
                entry = row['fvg_bull_top']
                sl = row['low'] - params['sl_buffer']
                tp = entry + params['tp_target']
                direction = 1
                signal = True
                # Reset sweep? Or allow multiple?
                # Standard: Allow multiple but limit trades.
                
            # Short Setup: Swept High AND Bearish FVG
            elif swept_high and row['fvg_bear']:
                entry = row['fvg_bear_btm']
                sl = row['high'] + params['sl_buffer']
                tp = entry - params['tp_target']
                direction = -1
                signal = True
                
            if signal:
                # SIMULATE (Look Ahead)
                outcome = 0
                filled = False
                future = day_bars.loc[i:].iloc[1:]
                
                for _, f_row in future.iterrows():
                    # Check Fill
                    if not filled:
                        if direction == 1 and f_row['low'] <= entry: filled = True
                        if direction == -1 and f_row['high'] >= entry: filled = True
                    
                    # Check Result
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
                
                if filled:
                    trades.append(outcome)
                    daily_trades += 1
                    # Reset state after trade? Use max_trades to limit.
                    
    # Metrics
    if not trades: return {'trades':0, 'pf':0}
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_win / gross_loss if gross_loss > 0 else 10.0
    return {'trades': len(trades), 'pf': pf}

if __name__ == '__main__':
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
    
    # ENGINEER BOTH
    full_df = engineer_pools(full_df)
    full_df = engineer_fvg(full_df)
    
    # Split
    train_mask = full_df['time'].dt.month.isin(TRAIN_MONTHS)
    train_df = full_df[train_mask]
    test_df = full_df[~train_mask]
    
    # Params
    param_grid = {
        'pools': POOLS,
        'tp_target': TP_TARGETS,
        'sl_buffer': SL_BUFFERS,
        'max_trades': MAX_TRADES
    }
    keys, values = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Hybrid Search: {len(combos)} items.")
    
    work_items = [(i, c) for i, c in enumerate(combos)]
    with Pool(NUM_CORES, initializer=init_worker, initargs=(train_df, test_df)) as pool:
        for res in pool.imap_unordered(evaluate_params, work_items):
            if res:
                logger.info(f"DATA: {res['params']} | PF: {res['test_pf']:.2f} | Trades: {res['test_trades']}")
                pd.DataFrame([res]).to_csv(OUTPUT_DIR / "hybrid_results_raw.csv", mode='a', header=not (OUTPUT_DIR / "hybrid_results_raw.csv").exists(), index=False)
    
    print("Search Complete.")
