"""
TITAN OVERNIGHT SEARCH ENGINE V3
================================
Massively parallel strategy search with WALK-FORWARD VALIDATION.

VALIDATION METHODOLOGY:
- 70% Training: Months 1-8 (Jan-Aug 2025)
- 30% Testing: Months 9-12 (Sep-Dec 2025) - UNSEEN DATA

AUTO CHECK-INS:
- Saves progress every 500 strategies tested
- Documents all winners to CSV and log file
- Final report generated on completion

CONCEPTS EXPLORED:
1. Pool combinations (which pools to trade)
2. Displacement thresholds (30%-80% body ratio)
3. Session windows (AM, PM, Full day, Power Hour)
4. Direction filters (Long only, Short only, Both)
5. SL buffers (1-5 points)
6. Max trades per day (1-10)
7. Displacement requirement (Yes/No)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import json
from datetime import datetime
import logging
import sys
import psutil
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
NUM_CORES = 14
MAX_RAM_GB = 59  # Auto-throttle if RAM exceeds this
LOW_POWER_CORES = 6  # Reduce to this many cores in low-power mode
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET")
OUTPUT_DIR = Path("C:/Users/CEO/ICT reinforcement/output/overnight_search")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Walk-Forward Split
TRAIN_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8]  # 70% - Jan to Aug
TEST_MONTHS = [9, 10, 11, 12]  # 30% - Sep to Dec (UNSEEN)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'overnight_search.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TitanSearch")

# Parameter Space
PARAM_SPACE = {
    'pools': [
        ['PDL'], ['PDH'], ['ONL'], ['ONH'], ['ASIA_L'], ['ASIA_H'], ['LON_L'], ['LON_H'],
        ['PDL', 'PDH'], ['ONL', 'ONH'], ['ASIA_L', 'ASIA_H'], ['LON_L', 'LON_H'],
        ['PDL', 'ONL'], ['PDL', 'ASIA_L'], ['PDL', 'LON_L'],
        ['PDL', 'PDH', 'ONL', 'ONH'],
        ['ASIA_L', 'LON_L', 'ONL'],
        ['ASIA_H', 'LON_H', 'ONH'],
        ['PDL', 'PDH', 'ONL', 'ONH', 'ASIA_L', 'ASIA_H', 'LON_L', 'LON_H'],
    ],
    'disp_threshold': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'direction': ['LONG', 'SHORT', 'BOTH'],
    'session': [
        (9, 30, 11, 0),
        (9, 30, 12, 0),
        (12, 0, 15, 30),
        (9, 30, 15, 30),
        (14, 0, 15, 30),
    ],
    'sl_buffer': [1.0, 2.0, 3.0, 5.0],
    'max_trades': [1, 2, 3, 5, 10],
    'require_disp': [True, False],
}

# Quality Thresholds
MIN_TRADES = 30
MIN_EXPECTANCY = 2.0
MIN_PF = 1.3

# ==============================================================================
# BACKTESTER
# ==============================================================================
def quick_backtest(df_bars, params):
    """Ultra-fast backtest for parameter search."""
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
    """Worker function with walk-forward validation."""
    param_id, params, train_bars, test_bars = args
    
    try:
        # TRAIN on 70%
        train_trades = quick_backtest(train_bars, params)
        if len(train_trades) < MIN_TRADES:
            return None
        
        train_df = pd.DataFrame(train_trades)
        train_pnl = train_df['pnl'].sum()
        train_exp = train_df['pnl'].mean()
        train_winners = train_df[train_df['pnl'] > 0]
        train_losers = train_df[train_df['pnl'] < 0]
        train_pf = train_winners['pnl'].sum() / abs(train_losers['pnl'].sum()) if len(train_losers) > 0 and train_losers['pnl'].sum() != 0 else 0
        
        # Filter on training performance
        if train_exp < MIN_EXPECTANCY or train_pf < MIN_PF:
            return None
        
        # TEST on 30% (UNSEEN)
        test_trades = quick_backtest(test_bars, params)
        if len(test_trades) < 10:
            return None
        
        test_df = pd.DataFrame(test_trades)
        test_pnl = test_df['pnl'].sum()
        test_exp = test_df['pnl'].mean()
        test_winners = test_df[test_df['pnl'] > 0]
        test_losers = test_df[test_df['pnl'] < 0]
        test_pf = test_winners['pnl'].sum() / abs(test_losers['pnl'].sum()) if len(test_losers) > 0 and test_losers['pnl'].sum() != 0 else 0
        
        # Only return if strategy holds up on unseen data
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
        }
    except Exception as e:
        return None

def run_overnight_search():
    """Main search with walk-forward validation."""
    logger.info("=" * 60)
    logger.info("TITAN OVERNIGHT SEARCH ENGINE V3")
    logger.info("=" * 60)
    logger.info(f"CPU Cores: {NUM_CORES}")
    logger.info(f"Training Months: {TRAIN_MONTHS} (70%)")
    logger.info(f"Testing Months: {TEST_MONTHS} (30% UNSEEN)")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading Golden Data...")
    
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
    
    # Engineer pools
    sys.path.insert(0, "C:/Users/CEO/ICT reinforcement")
    from strategies.mle.phase16_pj_engine import engineer_pools
    
    train_bars = engineer_pools(train_raw)
    test_bars = engineer_pools(test_raw)
    logger.info("Pools engineered")
    
    # Generate combinations
    param_keys = list(PARAM_SPACE.keys())
    param_values = list(PARAM_SPACE.values())
    all_combos = list(product(*param_values))
    
    logger.info(f"Total combinations to test: {len(all_combos):,}")
    
    # Prepare work
    work_items = []
    for i, combo in enumerate(all_combos):
        params = dict(zip(param_keys, combo))
        work_items.append((i, params, train_bars, test_bars))
    
    # Run parallel search with RAM monitoring
    results = []
    tested_count = 0
    batch_size = 500
    start_time = time.time()
    current_cores = NUM_CORES
    low_power_mode = False
    
    logger.info("Starting parallel search...")
    logger.info(f"RAM Limit: {MAX_RAM_GB}GB (auto-throttle enabled)")
    
    pool = Pool(current_cores)
    
    for batch_start in range(0, len(work_items), batch_size):
        # RAM CHECK
        ram_used_gb = psutil.virtual_memory().used / (1024**3)
        if ram_used_gb > MAX_RAM_GB and not low_power_mode:
            logger.warning(f"⚠️ RAM WARNING: {ram_used_gb:.1f}GB > {MAX_RAM_GB}GB limit")
            logger.warning(f"Switching to LOW POWER MODE ({LOW_POWER_CORES} cores)")
            pool.terminate()
            pool.join()
            current_cores = LOW_POWER_CORES
            low_power_mode = True
            pool = Pool(current_cores)
            time.sleep(2)  # Let system stabilize
        
        batch = work_items[batch_start:batch_start + batch_size]
        batch_results = pool.map(evaluate_params, batch)
        
        for r in batch_results:
            if r is not None:
                results.append(r)
                logger.info(f"🏆 WINNER #{len(results)}: Train Exp={r['train_exp']}, Test Exp={r['test_exp']}, PF={r['test_pf']}")
        
        tested_count = batch_start + len(batch)
        elapsed = time.time() - start_time
        rate = tested_count / elapsed if elapsed > 0 else 0
        eta = (len(work_items) - tested_count) / rate / 60 if rate > 0 else 0
        
        # Check-in with RAM status
        ram_gb = psutil.virtual_memory().used / (1024**3)
        logger.info(f"PROGRESS: {tested_count:,}/{len(work_items):,} | Winners: {len(results)} | Rate: {rate:.0f}/sec | RAM: {ram_gb:.1f}GB | ETA: {eta:.1f}min")
        
        # Save intermediate results
        if results:
            df_results = pd.DataFrame(results).sort_values('test_exp', ascending=False)
            df_results.to_csv(OUTPUT_DIR / 'validated_strategies.csv', index=False)
    
    pool.close()
    pool.join()
    
    # Final report
    logger.info("=" * 60)
    logger.info("SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total tested: {tested_count:,}")
    logger.info(f"Valid strategies: {len(results)}")
    
    if results:
        df_results = pd.DataFrame(results).sort_values('test_exp', ascending=False)
        df_results.to_csv(OUTPUT_DIR / 'validated_strategies.csv', index=False)
        
        logger.info("\nTOP 10 VALIDATED STRATEGIES:")
        logger.info(df_results.head(10).to_string())
        
        # Save detailed report
        with open(OUTPUT_DIR / 'final_report.txt', 'w') as f:
            f.write("TITAN OVERNIGHT SEARCH - FINAL REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total Combinations Tested: {tested_count:,}\n")
            f.write(f"Validated Strategies: {len(results)}\n")
            f.write(f"Training Period: Months {TRAIN_MONTHS}\n")
            f.write(f"Testing Period: Months {TEST_MONTHS}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("TOP 10 STRATEGIES:\n")
            f.write(df_results.head(10).to_string())
    
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("Overnight search complete. Good morning! ☀️")

if __name__ == "__main__":
    run_overnight_search()
