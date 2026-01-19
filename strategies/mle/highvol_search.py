"""
TARGETED HIGH-VOLUME STRATEGY SEARCH
=====================================
Focus: Multi-pool combinations for 3-5 trades/day
Speed: Fast - only ~1000 combinations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from itertools import product
import time
import json
from datetime import datetime
import logging
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_CORES = 14
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
OUTPUT_DIR = Path("C:/Users/CEO/ICT reinforcement/output/highvol_search")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]
TEST_MONTHS = [9, 10, 11, 12]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'highvol_search.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HighVolSearch")

# FOCUSED PARAMETER SPACE - Multi-pool combos only
PARAM_SPACE = {
    'pools': [
        # High-volume multi-pool combos
        ['PDL', 'ONL'],
        ['PDL', 'ONL', 'ASIA_L'],
        ['PDL', 'ONL', 'ASIA_L', 'LON_L'],  # All lows
        ['PDH', 'ONH', 'ASIA_H', 'LON_H'],  # All highs
        ['PDL', 'PDH'],
        ['ONL', 'ONH'],
        ['PDL', 'PDH', 'ONL', 'ONH'],
        ['PDL', 'PDH', 'ONL', 'ONH', 'ASIA_L', 'ASIA_H'],
        ['PDL', 'PDH', 'ONL', 'ONH', 'ASIA_L', 'ASIA_H', 'LON_L', 'LON_H'],  # ALL POOLS
        ['ONL', 'ASIA_L', 'LON_L'],  # Overnight lows only
        ['ONH', 'ASIA_H', 'LON_H'],  # Overnight highs only
    ],
    'disp_threshold': [0.4, 0.5, 0.6],
    'direction': ['LONG', 'BOTH'],  # Skip SHORT-only based on prior results
    'session': [
        (9, 30, 15, 30),  # Full day
        (9, 30, 12, 0),   # Morning
    ],
    'sl_buffer': [2.0, 3.0, 5.0],
    'max_trades': [5, 10],  # Higher limits
    'require_disp': [False],  # Don't require displacement for volume
}

MIN_TRADES_TRAIN = 100  # ~0.6 trades/day minimum (relaxed from 300)
MIN_TRADES_TEST = 40
MIN_EXPECTANCY = 1.0    # Relaxed slightly
MIN_PF = 1.1            # Relaxed slightly

# ==============================================================================
# BACKTESTER (Same as before)
# ==============================================================================
def quick_backtest(df_bars, params):
    trades = []
    tracker = defaultdict(lambda: 'DEFINED')
    
    pools_to_trade = params['pools']
    disp_thresh = params['disp_threshold']
    direction_filter = params['direction']
    session = params['session']
    sl_buffer = params['sl_buffer']
    max_trades = params['max_trades']
    require_disp = params['require_disp']
    
    h_start, m_start, h_end, m_end = session
    mask = ((df_bars['hour'] == h_start) & (df_bars['minute'] >= m_start)) | \
           ((df_bars['hour'] > h_start) & (df_bars['hour'] < h_end)) | \
           ((df_bars['hour'] == h_end) & (df_bars['minute'] <= m_end))
    
    dates = df_bars['date'].unique()
    
    for d in dates:
        daily_trades = 0
        tracker.clear()
        
        day_bars = df_bars[(df_bars['date'] == d) & mask]
        
        for idx, row in day_bars.iterrows():
            if daily_trades >= max_trades:
                break
                
            for pool in pools_to_trade:
                if tracker[pool] != 'DEFINED':
                    continue
                    
                is_low = pool.endswith('L') or pool == 'PDL' or pool == 'ONL'
                level = row.get(pool)
                if pd.isna(level):
                    continue
                
                if is_low:
                    sweep = row['low'] < level
                    reclaim = row['close'] > level
                    direction = 1
                else:
                    sweep = row['high'] > level
                    reclaim = row['close'] < level
                    direction = -1
                
                if direction_filter == 'LONG' and direction == -1:
                    continue
                if direction_filter == 'SHORT' and direction == 1:
                    continue
                
                if sweep and reclaim:
                    body = abs(row['close'] - row['open'])
                    bar_range = row['high'] - row['low']
                    has_disp = body > (disp_thresh * bar_range) if bar_range > 0 else False
                    
                    if require_disp and not has_disp:
                        continue
                    
                    if direction == 1:
                        sl = row['low'] - sl_buffer
                        tp_col = pool.replace('L', 'H') if 'L' in pool else 'PDH'
                        tp = row.get(tp_col, row['high'] + 50)
                    else:
                        sl = row['high'] + sl_buffer
                        tp_col = pool.replace('H', 'L') if 'H' in pool else 'PDL'
                        tp = row.get(tp_col, row['low'] - 50)
                    
                    if pd.isna(tp):
                        tp = row['close'] + (50 if direction == 1 else -50)
                    
                    entry = row['close']
                    future_bars = df_bars[(df_bars['date'] == d) & (df_bars.index > idx)]
                    
                    pnl = 0
                    for _, fb in future_bars.iterrows():
                        if direction == 1:
                            if fb['low'] <= sl:
                                pnl = sl - entry
                                break
                            if fb['high'] >= tp:
                                pnl = tp - entry
                                break
                        else:
                            if fb['high'] >= sl:
                                pnl = entry - sl
                                break
                            if fb['low'] <= tp:
                                pnl = entry - tp
                                break
                    
                    trades.append({'pnl': pnl, 'disp': has_disp})
                    tracker[pool] = 'TRADED'
                    daily_trades += 1
    
    return trades

def evaluate_params(args):
    param_id, params, train_bars, test_bars = args
    
    try:
        train_trades = quick_backtest(train_bars, params)
        if len(train_trades) < MIN_TRADES_TRAIN:
            return None
        
        train_df = pd.DataFrame(train_trades)
        train_pnl = train_df['pnl'].sum()
        train_exp = train_df['pnl'].mean()
        train_winners = train_df[train_df['pnl'] > 0]
        train_losers = train_df[train_df['pnl'] < 0]
        train_pf = train_winners['pnl'].sum() / abs(train_losers['pnl'].sum()) if len(train_losers) > 0 and train_losers['pnl'].sum() != 0 else 0
        
        if train_exp < MIN_EXPECTANCY or train_pf < MIN_PF:
            return None
        
        test_trades = quick_backtest(test_bars, params)
        if len(test_trades) < MIN_TRADES_TEST:
            return None
        
        test_df = pd.DataFrame(test_trades)
        test_pnl = test_df['pnl'].sum()
        test_exp = test_df['pnl'].mean()
        test_winners = test_df[test_df['pnl'] > 0]
        test_losers = test_df[test_df['pnl'] < 0]
        test_pf = test_winners['pnl'].sum() / abs(test_losers['pnl'].sum()) if len(test_losers) > 0 and test_losers['pnl'].sum() != 0 else 0
        
        if test_exp < 1.0 or test_pf < 1.0:
            return None
        
        return {
            'param_id': param_id,
            'params': json.dumps(params, default=str),
            'train_trades': len(train_trades),
            'train_pnl': round(train_pnl, 2),
            'train_exp': round(train_exp, 2),
            'train_pf': round(train_pf, 2),
            'test_trades': len(test_trades),
            'test_pnl': round(test_pnl, 2),
            'test_exp': round(test_exp, 2),
            'test_pf': round(test_pf, 2),
            'trades_per_day_train': round(len(train_trades) / 168, 1),
            'trades_per_day_test': round(len(test_trades) / 84, 1),
        }
    except Exception as e:
        return None

def run_highvol_search():
    logger.info("=" * 60)
    logger.info("TARGETED HIGH-VOLUME STRATEGY SEARCH")
    logger.info("=" * 60)
    
    # Load data
    def load_months(months):
        bars = []
        for m in months:
            m_str = f"{m:02d}"
            try:
                df = pd.read_parquet(BASE_DIR / f"USTEC_2025-{m_str}_clean_1m.parquet")
                df['time'] = pd.to_datetime(df.index)
                df['date'] = df['time'].dt.date
                df['hour'] = df['time'].dt.hour
                df['minute'] = df['time'].dt.minute
                bars.append(df)
            except Exception as e:
                logger.warning(f"Could not load month {m_str}: {e}")
        return pd.concat(bars, ignore_index=False) if bars else pd.DataFrame()
    
    train_raw = load_months(TRAIN_MONTHS)
    test_raw = load_months(TEST_MONTHS)
    
    logger.info(f"Train bars: {len(train_raw):,}")
    logger.info(f"Test bars: {len(test_raw):,}")
    
    sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
    from strategies.mle.phase16_pj_engine import engineer_pools
    
    train_bars = engineer_pools(train_raw)
    test_bars = engineer_pools(test_raw)
    
    # Generate combinations
    param_keys = list(PARAM_SPACE.keys())
    param_values = list(PARAM_SPACE.values())
    all_combos = list(product(*param_values))
    
    logger.info(f"Total combinations: {len(all_combos)}")
    
    work_items = [(i, dict(zip(param_keys, combo)), train_bars, test_bars) for i, combo in enumerate(all_combos)]
    
    # Run search with real-time feedback
    results = []
    start_time = time.time()
    processed = 0
    
    logger.info("Starting real-time search...")
    
    with Pool(NUM_CORES) as pool:
        for r in pool.imap_unordered(evaluate_params, work_items):
            processed += 1
            if processed % 10 == 0:
                print(f"Progress: {processed}/{len(work_items)}", end='\r')
                
            if r is not None:
                results.append(r)
                logger.info(f"🏆 WINNER: {r['trades_per_day_train']} trades/day | PF: {r['test_pf']:.2f} | Exp: ${r['test_exp']:.2f}")
                logger.info(f"   {r['params']}")
    
    elapsed = time.time() - start_time
    logger.info(f"\nSearch completed in {elapsed:.1f} seconds")
    logger.info(f"Total winners: {len(results)}")
    
    if results:
        df_results = pd.DataFrame(results).sort_values('trades_per_day_train', ascending=False)
        df_results.to_csv(OUTPUT_DIR / 'highvol_strategies.csv', index=False)
        
        logger.info("\n=== TOP 10 HIGH-VOLUME STRATEGIES ===")
        for i, row in df_results.head(10).iterrows():
            logger.info(f"Trades/Day: {row['trades_per_day_train']} | PF: {row['test_pf']:.2f} | Exp: ${row['test_exp']:.2f}")
            logger.info(f"   {row['params'][:100]}")
        
        logger.info(f"\nResults saved to: {OUTPUT_DIR / 'highvol_strategies.csv'}")
    else:
        logger.info("No high-volume strategies found that passed validation.")

if __name__ == "__main__":
    run_highvol_search()
