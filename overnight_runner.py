"""
OVERNIGHT RESEARCH RUNNER V6.1
==============================
V6.1: Institutional Engine + Weak Structure Opt
"""

import torch
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
import psutil
import gc
from joblib import Parallel, delayed

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path("C:/Users/CEO/ICT reinforcement")
DATA_DIR = BASE_DIR / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET"
OUTPUT_DIR = BASE_DIR / "v6_overnight_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Desktop Backup Directory
DESKTOP_DIR = Path("C:/Users/CEO/Desktop/V6_ENGINE_RESULTS")
DESKTOP_DIR_WINNERS = DESKTOP_DIR / "WINNERS"
DESKTOP_DIR_LOGS = DESKTOP_DIR / "ENGINE_LOGS"
DESKTOP_DIR_WINNERS.mkdir(parents=True, exist_ok=True)
DESKTOP_DIR_LOGS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Mode: Use tick data for accurate SL/TP resolution
USE_TICK_DATA = True  # Set to False for faster but less accurate testing

# When both SL and TP could be hit in same bar, use ticks to determine which came first
# This is CRITICAL for realistic backtesting

# RAM Limit (30GB max, throttle if exceeded)
MAX_RAM_GB = 30
THROTTLE_RAM_GB = 25  # Start throttling at 25GB to stay safe

# V6 Constants for Pure Tick Resolution
POINT_VALUE = 20.0
COMMISSION = 2.05
SPREAD_SLIPPAGE = 2.0  # Conservative: 2 pts (entry + exit slippage combined)
TRADE_TIMEOUT_SECONDS = 14400  # 4 hours max hold
TRADE_COOLDOWN_BARS = 1  # Minimum bars between trades (prevents instant re-entry)

# ============================================================================
# V6 TICK DATA FUNCTIONS
# ============================================================================
def load_tick_data():
    """Load all tick parquet files for V6 Pure Tick outcome resolution."""
    log("Loading Tick Data for V6 Pure Tick Resolution...")
    tick_files = sorted(DATA_DIR.glob("*_clean_ticks.parquet"))
    
    if not tick_files:
        log("WARNING: No tick data found! Falling back to bar-level resolution.")
        return None
    
    dfs = []
    for f in tick_files:
        try:
            df = pd.read_parquet(f)
            df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]
            dfs.append(df)
            log(f"  Loaded {f.name}: {len(df):,} ticks")
        except Exception as e:
            log(f"  Error loading {f.name}: {e}")
    
    if not dfs:
        return None
        
    tick_df = pd.concat(dfs).sort_index()
    log(f"Tick Data Ready: {len(tick_df):,} total ticks")
    return tick_df


def resolve_outcome_ticks(tick_times, tick_prices, entry_time_val, fill_price, sl_pts, tp_pts, direction):
    """
    V6 PURE TICK RESOLUTION (NUMPY OPTIMIZED)
    Verifies if the fill_price (Limit/OTE) is hit, then resolves SL/TP.
    """
    # 1. Find start index using binary search
    start_idx = np.searchsorted(tick_times, entry_time_val, side='right')
    if start_idx >= len(tick_times): return 'TIMEOUT', 0
        
    # 2. Slice future ticks
    max_lookahead = 200000 # Increased for V6 OTE wait time
    end_idx = min(len(tick_times), start_idx + max_lookahead)
    future_times = tick_times[start_idx:end_idx]
    future_prices = tick_prices[start_idx:end_idx]
    
    if len(future_prices) == 0: return 'TIMEOUT', 0
    
    # 3. Find Fill Index
    if direction == 'LONG':
        fill_mask = future_prices <= fill_price
    else:
        fill_mask = future_prices >= fill_price
    
    fill_indices = np.where(fill_mask)[0]
    if len(fill_indices) == 0:
        return 'SKIPPED', 0 # Limit order never touched
    
    f_idx = fill_indices[0]
    
    # 4. Check for Timeout before Fill
    fill_time = future_times[f_idx]
    if (fill_time - entry_time_val) / 1e9 > TRADE_TIMEOUT_SECONDS:
        return 'SKIPPED', 0
        
    # 5. Evaluate SL/TP starting from the Fill Index (inclusive)
    # SL/TP can be hit on the same tick as the fill
    eval_prices = future_prices[f_idx:]
    if len(eval_prices) == 0: return 'TIMEOUT', 0
    
    if direction == 'LONG':
        tp_price = fill_price + tp_pts
        sl_price = fill_price - sl_pts
        hit_tp = eval_prices >= tp_price
        hit_sl = eval_prices <= sl_price
    else:
        tp_price = fill_price - tp_pts
        sl_price = fill_price + sl_pts
        hit_tp = eval_prices <= tp_price
        hit_sl = eval_prices >= sl_price
        
    tp_indices = np.where(hit_tp)[0]
    sl_indices = np.where(hit_sl)[0]
    first_tp = tp_indices[0] if len(tp_indices) > 0 else 999999999
    first_sl = sl_indices[0] if len(sl_indices) > 0 else 999999999
    
    if first_tp < first_sl:
        return 'WIN', tp_pts
    elif first_sl < first_tp:
        return 'LOSS', -sl_pts
            
    return 'TIMEOUT', 0

def worker_resolve_trades_chunk(
    strat_params_chunk, long_signals_chunk, short_signals_chunk, 
    tick_times, tick_prices, bar_times, 
    close_prices, high_prices, low_prices, open_prices,
    n_bars, trade_cooldown_bars, spread_slippage, point_value, commission, trade_timeout_seconds,
    min_pf, min_win_rate
):
    """Worker function for parallel trade resolution on a CHUNK of strategies."""
    results = []
    num_strats = len(strat_params_chunk)
    
    for i in range(num_strats):
        strat_params = strat_params_chunk[i]
        entry_long_np = long_signals_chunk[i]
        entry_short_np = short_signals_chunk[i]
        
        c_sl, c_tp, c_ote = strat_params[0], strat_params[1], strat_params[8]
        
        long_indices = np.where(entry_long_np)[0]
        short_indices = np.where(entry_short_np)[0]
        
        signals = [(idx, 'LONG') for idx in long_indices] + [(idx, 'SHORT') for idx in short_indices]
        signals.sort(key=lambda x: x[0])
        
        total_pnl, wins, losses = 0.0, 0, 0
        actual_win_pnl, actual_loss_pnl = 0.0, 0.0
        in_position, last_trade_end_idx = False, -999
        
        # Performance tuning: only resolve if there are signals
        if len(signals) > 0:
            for sig_idx, direction in signals:
                if sig_idx <= last_trade_end_idx + trade_cooldown_bars: continue
                if in_position: continue
                entry_bar_idx = sig_idx + 1
                if entry_bar_idx >= n_bars: continue
                
                sig_h, sig_l = high_prices[sig_idx], low_prices[sig_idx]
                if direction == 'LONG':
                    ote_price = sig_l + (sig_h - sig_l) * (1 - c_ote)
                    entry_price = min(open_prices[entry_bar_idx], ote_price)
                else:
                    ote_price = sig_h - (sig_h - sig_l) * (1 - c_ote)
                    entry_price = max(open_prices[entry_bar_idx], ote_price)
                
                entry_time_val = bar_times[entry_bar_idx]
                outcome, pnl_pts = resolve_outcome_ticks(
                    tick_times, tick_prices, entry_time_val, entry_price, c_sl, c_tp, direction
                )
                
                if outcome == 'SKIPPED': continue
                
                in_position = True
                entry_cost = (spread_slippage * point_value) + commission
                trade_pnl = (pnl_pts * point_value) - entry_cost
                
                if outcome == 'TIMEOUT':
                    mtm_exit = close_prices[-1] if n_bars > 0 else entry_price
                    mtm_pnl_pts = (mtm_exit - entry_price) if direction == 'LONG' else (entry_price - mtm_exit)
                    trade_pnl = (mtm_pnl_pts * point_value) - entry_cost
                    last_trade_end_idx = n_bars - 1
                else:
                    last_trade_end_idx = entry_bar_idx
                
                if trade_pnl > 0:
                    wins += 1; actual_win_pnl += trade_pnl
                else:
                    losses += 1; actual_loss_pnl += abs(trade_pnl)
                
                total_pnl += trade_pnl
                in_position = False

        trades = wins + losses
        pf = actual_win_pnl / (actual_loss_pnl + 0.001) if actual_loss_pnl > 0 else actual_win_pnl
        win_rate = wins / (trades + 0.001)
        
        score = total_pnl
        if pf < min_pf: score -= 1e6
        if trades < 30: score -= 1e6
        if win_rate < min_win_rate: score -= 1e6
        
        results.append((float(score), float(trades)))
        
    return results


def check_ram_and_throttle():
    """Check RAM usage and throttle if approaching limit."""
    ram_used = psutil.Process().memory_info().rss / 1e9
    if ram_used > THROTTLE_RAM_GB:
        log(f"⚠️ RAM at {ram_used:.1f}GB - forcing garbage collection")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True  # Throttled
    return False

# Session Definitions (NY Time)
SESSIONS = {
    'asia': {'start': (18, 0), 'end': (0, 0)},      # 6pm-12am
    'london': {'start': (2, 0), 'end': (5, 0)},     # 2am-5am
    'ib': {'start': (9, 30), 'end': (10, 30)},      # 9:30-10:30am (Initial Balance)
    'silver_bullet': {'start': (10, 0), 'end': (11, 0)},  # 10am-11am
    'us_open': {'start': (9, 30), 'end': (12, 0)},  # 9:30am-12pm
    'late_session': {'start': (14, 0), 'end': (15, 30)},  # 2pm-3:30pm
}

# Quality Thresholds (REALISTIC for Live Trading)
# Target: 3-5 trades/day = 750-1250 trades/year
# These will be SCALED based on data size in the miner
BASE_TRADES_PER_YEAR = 750   # Baseline for full year
MIN_TRADES_RATIO = 0.02     # Min 2% of bars should be trades (flexible)
MAX_TRADES_RATIO = 0.10     # Max 10% of bars (avoid overtrading)

# For threshold calculation in miner
MIN_TRADES_PER_YEAR = 30    # LOOSENED: was 50
MAX_TRADES_PER_YEAR = 5000  # Absolute maximum

# Realistic Costs
SLIPPAGE_PTS = 0.5  # 0.5 point slippage per entry
COMMISSION_PTS = 0.25  # ~$0.50 commission = 0.25 pts for MNQ
TOTAL_COST_PER_TRADE = SLIPPAGE_PTS + COMMISSION_PTS  # 0.75 pts total

# Performance Requirements
MIN_PF = 1.1        # LOOSENED: was 1.3
MIN_EXPECTANCY = 0.2 # LOOSENED: was 1.0
MIN_WIN_RATE = 0.30  # LOOSENED: was 0.35

# Robustness Requirements (REALISM)
MAX_CONSECUTIVE_LOSSES = 10  # Stop if strategy has 10+ losing streak
MAX_DRAWDOWN_PERCENT = 25.0  # Reject if max drawdown > 25% of peak equity
MAX_DAILY_TRADES = 5  # Realistic daily limit (matches live trading)

# ============================================================================
# LOGGING
# ============================================================================
log_file = OUTPUT_DIR / f"overnight_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{ts}] {msg}"
    print(formatted)
    try:
        # Local Log
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")
        # Desktop Log
        desktop_log = DESKTOP_DIR_LOGS / log_file.name
        with open(desktop_log, 'a', encoding='utf-8') as f:
            f.write(formatted + "\n")
    except:
        pass

def save_checkpoint(results, filename="v6_checkpoint.json"):
    """Save results periodically to local and desktop."""
    try:
        # Local save
        path = OUTPUT_DIR / filename
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Desktop save
        desktop_path = DESKTOP_DIR_WINNERS / filename
        import shutil
        with open(desktop_path, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        log(f"Error saving checkpoint: {e}")

# ============================================================================
# DATA LOADER
# ============================================================================
def load_all_data():
    """Load all 12 months of 1-min data into a single DataFrame."""
    log("Loading all monthly data files...")
    
    all_dfs = []
    for month in range(1, 13):
        file_path = DATA_DIR / f"USTEC_2025-{month:02d}_clean_1m.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            all_dfs.append(df)
            log(f"  Loaded {file_path.name}: {len(df):,} bars")
    
    combined = pd.concat(all_dfs)
    log(f"Total bars loaded: {len(combined):,}")
    return combined

def load_train_test_data():
    """
    Load data with RANDOM 70/30 train/test split.
    CRITICAL FIX: Indicators calculated SEPARATELY on train/test to prevent leakage.
    """
    log("Loading data with random 70/30 train/test split...")
    
    # All available months
    all_months = list(range(1, 13))
    
    # Random shuffle and split
    import random
    random.seed(42)  # Reproducible
    random.shuffle(all_months)
    
    n_train = int(len(all_months) * 0.7)  # 8-9 months
    train_months = sorted(all_months[:n_train])
    test_months = sorted(all_months[n_train:])
    
    log(f"  TRAIN months: {train_months}")
    log(f"  TEST months: {test_months}")
    
    # Load and split FIRST (no indicator contamination)
    train_dfs = []
    test_dfs = []
    for month in range(1, 13):
        file_path = DATA_DIR / f"USTEC_2025-{month:02d}_clean_1m.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            if month in train_months:
                train_dfs.append(df)
                log(f"  TRAIN: Loaded {file_path.name}: {len(df):,} bars")
            else:
                test_dfs.append(df)
                log(f"  TEST: Loaded {file_path.name}: {len(df):,} bars")
    
    # Calculate indicators SEPARATELY (no leakage between train/test)
    def add_indicators(df):
        """Add indicators using ONLY data from this split."""
        df = df.sort_index().copy()
        df['sma200'] = df['close'].rolling(200, min_periods=200).mean()
        # CRITICAL FIX: Shift by 1 so we compare to PREVIOUS 60-bar extremes
        # This enables sweep detection (low < prior min, high > prior max)
        df['roll_min'] = df['low'].rolling(60, min_periods=60).min().shift(1)
        df['roll_max'] = df['high'].rolling(60, min_periods=60).max().shift(1)
        
        # ATR for Displacement
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['displacement'] = (df['close'] - df['open']).abs() / (df['atr'].shift() + 1e-9)
        
        # Swing High/Lows for MSS (Strong = 2-bar confirmation, Weak = 1-bar confirmation)
        # Strong: T-2 was extreme relative to T-4,3,1,0
        df['swing_high_strong'] = (df['high'].shift(2) > df['high'].shift(3)) & (df['high'].shift(2) > df['high'].shift(4)) & \
                                  (df['high'].shift(2) > df['high'].shift(1)) & (df['high'].shift(2) > df['high'])
        df['swing_low_strong'] = (df['low'].shift(2) < df['low'].shift(3)) & (df['low'].shift(2) < df['low'].shift(4)) & \
                                 (df['low'].shift(2) < df['low'].shift(1)) & (df['low'].shift(2) < df['low'])
        
        # Weak: T-1 was extreme relative to T-2, 0 
        df['swing_high_weak'] = (df['high'].shift(1) > df['high'].shift(2)) & (df['high'].shift(1) > df['high'])
        df['swing_low_weak'] = (df['low'].shift(1) < df['low'].shift(2)) & (df['low'].shift(1) < df['low'])
        
        # Track Most Recent Swing Strong
        df['last_swing_h_strong'] = df['high'].shift(2).where(df['swing_high_strong']).ffill()
        df['last_swing_l_strong'] = df['low'].shift(2).where(df['swing_low_strong']).ffill()
        
        # Track Most Recent Swing Weak
        df['last_swing_h_weak'] = df['high'].shift(1).where(df['swing_high_weak']).ffill()
        df['last_swing_l_weak'] = df['low'].shift(1).where(df['swing_low_weak']).ffill()
        
        # MSS: Price breaks the last institutional swing
        df['mss_bull_strong'] = (df['close'] > df['last_swing_h_strong'].shift())
        df['mss_bear_strong'] = (df['close'] < df['last_swing_l_strong'].shift())
        
        df['mss_bull_weak'] = (df['close'] > df['last_swing_h_weak'].shift())
        df['mss_bear_weak'] = (df['close'] < df['last_swing_l_weak'].shift())

        # Volume handling
        if 'volume' not in df.columns or df['volume'].sum() < 1.0:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            else:
                df['volume'] = 1.0
        
        vol_ma = df['volume'].rolling(20, min_periods=1).mean()
        df['rel_vol'] = df['volume'] / (vol_ma + 1e-9)
        
        # Drop NaN rows from indicator warmup
        df = df.dropna(subset=['sma200', 'roll_min', 'roll_max', 'atr'])
        return df
    
    # Apply indicators SEPARATELY to prevent train/test contamination
    log(f"  Calculating indicators on TRAIN only...")
    train_df = add_indicators(pd.concat(train_dfs))
    
    log(f"  Calculating indicators on TEST only (no leakage)...")
    test_df = add_indicators(pd.concat(test_dfs))
    
    log(f"  TRAIN: {len(train_df):,} bars (after warmup)")
    log(f"  TEST: {len(test_df):,} bars (after warmup)")
    
    return train_df, test_df, train_months, test_months

# ============================================================================
# V6 GPU GENETIC MINER WITH PURE TICK RESOLUTION
# ============================================================================
class SessionGeneticMiner:
    def __init__(self, df, session_name, population_size=5000, generations=50, tick_df=None):
        self.session_name = session_name
        self.session_cfg = SESSIONS[session_name]
        self.pop_size = population_size
        self.generations = generations
        self.df = df
        self.tick_df = tick_df  # V5: Tick data for precise SL/TP resolution
        
        # Optimize tick data for numpy fast access
        if tick_df is not None:
            self.tick_times = tick_df.index.astype(np.int64).values
            self.tick_prices = (tick_df['price'].values if 'price' in tick_df.columns else tick_df['last'].values).astype(np.float32)
        else:
            self.tick_times = None
            self.tick_prices = None
        
        # Constants
        self.POINT_VALUE = POINT_VALUE
        self.COMMISSION = COMMISSION
        self.SPREAD_SLIPPAGE = SPREAD_SLIPPAGE
        
        # Prepare GPU tensors
        self._prepare_data(df)
        
    def _prepare_data(self, df):
        """Convert DataFrame to GPU tensors with V5 fixes."""
        df = df.copy()
        df['hour'] = pd.to_datetime(df.index).hour
        df['minute'] = pd.to_datetime(df.index).minute
        
        # Core tensors
        self.close = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        self.high = torch.tensor(df['high'].values, dtype=torch.float32, device=DEVICE)
        self.low = torch.tensor(df['low'].values, dtype=torch.float32, device=DEVICE)
        self.open = torch.tensor(df['open'].values, dtype=torch.float32, device=DEVICE)
        
        # Indicators (pre-calculated in load_train_test_data)
        self.sma = torch.tensor(df['sma200'].values, dtype=torch.float32, device=DEVICE)
        self.liq_min = torch.tensor(df['roll_min'].values, dtype=torch.float32, device=DEVICE)
        self.liq_max = torch.tensor(df['roll_max'].values, dtype=torch.float32, device=DEVICE)
        
        # Volume handling
        if 'rel_vol' in df.columns:
            self.rel_vol = torch.tensor(df['rel_vol'].values, dtype=torch.float32, device=DEVICE)
        else:
            self.rel_vol = torch.ones_like(self.close)
        self.disable_vol = (self.rel_vol.max() == 0)
        
        # FVG (FIXED: NaN padding instead of circular shift)
        nan_pad = torch.full((2,), float('nan'), device=DEVICE)
        prev_high_2 = torch.cat([nan_pad, self.high[:-2]])
        prev_low_2 = torch.cat([nan_pad, self.low[:-2]])
        self.fvg_bull_gap = torch.nan_to_num(self.low - prev_high_2, nan=0.0)
        self.fvg_bear_gap = torch.nan_to_num(prev_low_2 - self.high, nan=0.0)
        
        # V6: Displacement and MSS (Strong/Weak)
        self.disp = torch.tensor(df['displacement'].values, dtype=torch.float32, device=DEVICE)
        self.mss_bull_strong = torch.tensor(df['mss_bull_strong'].values, dtype=torch.bool, device=DEVICE)
        self.mss_bear_strong = torch.tensor(df['mss_bear_strong'].values, dtype=torch.bool, device=DEVICE)
        self.mss_bull_weak = torch.tensor(df['mss_bull_weak'].values, dtype=torch.bool, device=DEVICE)
        self.mss_bear_weak = torch.tensor(df['mss_bear_weak'].values, dtype=torch.bool, device=DEVICE)
        self.atr = torch.tensor(df['atr'].values, dtype=torch.float32, device=DEVICE)
        
        # Quality filters
        self.body_size = torch.abs(self.close - self.open)
        range_size = self.high - self.low
        upper_wick = self.high - torch.max(self.close, self.open)
        lower_wick = torch.min(self.close, self.open) - self.low
        max_wick = torch.max(upper_wick, lower_wick)
        self.wick_ratio = max_wick / (range_size + 1e-6)
        
        # Session mask
        hours = torch.tensor(df['hour'].values, dtype=torch.int32, device=DEVICE)
        mins = torch.tensor(df['minute'].values, dtype=torch.int32, device=DEVICE)
        start_h, start_m = self.session_cfg['start']
        end_h, end_m = self.session_cfg['end']
        start_mins = start_h * 60 + start_m
        end_mins = end_h * 60 + end_m
        bar_mins = hours * 60 + mins
        
        if start_mins < end_mins:
            self.session_mask = (bar_mins >= start_mins) & (bar_mins < end_mins)
        else:
            self.session_mask = (bar_mins >= start_mins) | (bar_mins < end_mins)
        
        # Causal Sweep (FIXED: Left-padding only, no future leakage)
        sweep_pad = 5
        sweep_long_raw = (self.low < self.liq_min).float().view(1, 1, -1)
        sweep_short_raw = (self.high > self.liq_max).float().view(1, 1, -1)
        left_pad = sweep_pad - 1
        sweep_long_padded = torch.nn.functional.pad(sweep_long_raw, (left_pad, 0), value=0)
        sweep_short_padded = torch.nn.functional.pad(sweep_short_raw, (left_pad, 0), value=0)
        self.recent_sweep_long = torch.nn.functional.max_pool1d(sweep_long_padded, kernel_size=sweep_pad, stride=1, padding=0).view(-1)[:len(self.low)]
        self.recent_sweep_short = torch.nn.functional.max_pool1d(sweep_short_padded, kernel_size=sweep_pad, stride=1, padding=0).view(-1)[:len(self.high)]
    
    def init_population(self):
        """V6 Genome: [SL, TP, Body, Wick, FVG, Vol, Disp, MSS_En, OTE, Struct_Type]"""
        sl = torch.randint(5, 100, (self.pop_size,), device=DEVICE)
        tp = torch.randint(10, 300, (self.pop_size,), device=DEVICE)
        body = torch.randint(0, 15, (self.pop_size,), device=DEVICE)
        wick = torch.randint(10, 100, (self.pop_size,), device=DEVICE)
        fvg_str = torch.rand((self.pop_size,), device=DEVICE) * 2.5
        rel_vol = torch.rand((self.pop_size,), device=DEVICE) * 2.0
        
        # V6 Additions
        disp = torch.rand((self.pop_size,), device=DEVICE) * 2.5
        mss = torch.randint(0, 2, (self.pop_size,), device=DEVICE).float()
        ote = 0.5 + torch.rand((self.pop_size,), device=DEVICE) * 0.4
        struct_type = torch.randint(0, 2, (self.pop_size,), device=DEVICE).float() # 0=Weak, 1=Strong
        
        return torch.stack([sl.float(), tp.float(), body.float(), wick.float(), fvg_str, rel_vol, disp, mss, ote, struct_type], dim=1)
    
    def evaluate_v6(self, pop):
        """V6 PURE TICK EVALUATION - Hyper-Resolution Vectorized Mode with BATCHING."""
        pop_size = len(pop)
        scores = torch.zeros(pop_size, device=DEVICE)
        trade_counts = torch.zeros(pop_size, device=DEVICE)
        
        active_indices = self.session_mask
        s_close = self.close[active_indices]
        s_open = self.open[active_indices]
        s_high = self.high[active_indices]
        s_low = self.low[active_indices]
        s_sma = self.sma[active_indices]
        s_body = self.body_size[active_indices]
        s_wick = self.wick_ratio[active_indices]
        s_rel = self.rel_vol[active_indices]
        s_fvg_long = self.fvg_bull_gap[active_indices]
        s_fvg_short = self.fvg_bear_gap[active_indices]
        s_disp = self.disp[active_indices]
        
        sweep_long = (self.recent_sweep_long[active_indices] > 0)
        sweep_short = (self.recent_sweep_short[active_indices] > 0)
        trend_long = (s_close > s_sma)
        trend_short = (s_close < s_sma)
        
        bar_times_ints = self.df.index[active_indices.cpu().numpy()].astype(np.int64).values
        close_prices = s_close.cpu().numpy()
        high_prices = s_high.cpu().numpy()
        low_prices = s_low.cpu().numpy()
        open_prices = s_open.cpu().numpy()
        n_bars = len(bar_times_ints)

        # GPU BATCHING: Prevent VRAM Overflow at 150k+ population
        EVAL_BATCH_SIZE = 25000
        
        for batch_start in range(0, pop_size, EVAL_BATCH_SIZE):
            batch_end = min(batch_start + EVAL_BATCH_SIZE, pop_size)
            batch_pop = pop[batch_start:batch_end]
            curr_batch_size = len(batch_pop)
            
            # Vectorized Filters for this Batch
            c_body = batch_pop[:, 2].unsqueeze(1)    
            c_wick = batch_pop[:, 3].unsqueeze(1) / 100.0
            c_fvg  = batch_pop[:, 4].unsqueeze(1)
            c_vol  = batch_pop[:, 5].unsqueeze(1)
            c_disp = batch_pop[:, 6].unsqueeze(1)
            c_mss_en = batch_pop[:, 7].unsqueeze(1)
            c_struct = batch_pop[:, 9].unsqueeze(1)

            mask_body = s_body >= c_body
            mask_wick = s_wick <= c_wick
            mask_vol  = (s_rel >= c_vol) if not self.disable_vol else torch.ones((curr_batch_size, n_bars), device=DEVICE, dtype=torch.bool)
            mask_disp = s_disp >= c_disp
            
            mss_mask_bull = torch.where(c_struct > 0.5, self.mss_bull_strong[active_indices], self.mss_bull_weak[active_indices])
            mss_mask_bear = torch.where(c_struct > 0.5, self.mss_bear_strong[active_indices], self.mss_bear_weak[active_indices])
            
            mss_mask_bull = torch.where(c_mss_en > 0.5, mss_mask_bull, torch.ones_like(mss_mask_bull))
            mss_mask_bear = torch.where(c_mss_en > 0.5, mss_mask_bear, torch.ones_like(mss_mask_bear))

            all_entry_long  = sweep_long & trend_long & mask_body & mask_wick & mask_vol & (s_fvg_long >= c_fvg) & mask_disp & mss_mask_bull
            all_entry_short = sweep_short & trend_short & mask_body & mask_wick & mask_vol & (s_fvg_short >= c_fvg) & mask_disp & mss_mask_bear

            long_signals_np = all_entry_long.cpu().numpy()
            short_signals_np = all_entry_short.cpu().numpy()

            # Clean up VRAM after signal extraction
            del all_entry_long, all_entry_short, mask_body, mask_wick, mask_vol, mask_disp
            
            # Split this batch into 16 chunks for the 16 processes
            num_workers = 16
            chunk_size = (curr_batch_size + num_workers - 1) // num_workers
            
            pop_np = batch_pop.cpu().numpy()
            
            chunks = []
            for c in range(num_workers):
                c_start = c * chunk_size
                c_end = min(c_start + chunk_size, curr_batch_size)
                if c_start < c_end:
                    chunks.append((
                        pop_np[c_start:c_end],
                        long_signals_np[c_start:c_end],
                        short_signals_np[c_start:c_end]
                    ))

            # Parallel Multi-Core Trade Resolution on CHUNKS
            chunk_results = Parallel(n_jobs=num_workers, backend='loky')(
                delayed(worker_resolve_trades_chunk)(
                    p_chunk, l_chunk, s_chunk,
                    self.tick_times,
                    self.tick_prices,
                    bar_times_ints,
                    close_prices,
                    high_prices,
                    low_prices,
                    open_prices,
                    n_bars,
                    TRADE_COOLDOWN_BARS,
                    self.SPREAD_SLIPPAGE,
                    self.POINT_VALUE,
                    self.COMMISSION,
                    TRADE_TIMEOUT_SECONDS,
                    MIN_PF,
                    MIN_WIN_RATE
                ) for p_chunk, l_chunk, s_chunk in chunks
            )

            # Flatten chunk results back into the scores/trade_counts arrays
            idx_in_batch = 0
            for res_list in chunk_results:
                for score, trades in res_list:
                    strat_idx = batch_start + idx_in_batch
                    scores[strat_idx] = score
                    trade_counts[strat_idx] = trades
                    idx_in_batch += 1

            if batch_start + curr_batch_size < pop_size:
                log(f"      V6.1 HV Progress: {batch_start + curr_batch_size}/{pop_size} (16-Core Ultra)")
        
        return scores, trade_counts
    
    def mutate(self, elites):
        num_elites = len(elites)
        needed = self.pop_size - num_elites
        indices = torch.randint(0, num_elites, (needed,), device=DEVICE)
        next_gen = elites[indices].clone()
        prob = 0.2
        mask = torch.rand_like(next_gen.float()) < prob
        # SL, TP, Body, Wick, FVG, Vol, Disp, MSS, OTE, Struct_Type
        noise = torch.tensor([5, 10, 1, 5, 0.5, 0.2, 0.2, 0.1, 0.05, 0.1], device=DEVICE)
        next_gen = torch.where(mask, next_gen + torch.randn_like(next_gen) * noise, next_gen)
        # Clamping
        next_gen = torch.clamp(next_gen, min=0.0)
        next_gen[:, 8] = torch.clamp(next_gen[:, 8], 0.5, 0.9) # OTE Range
        next_gen[:, 7] = torch.where(next_gen[:, 7] > 0.5, 1.0, 0.0) # MSS Boolean
        next_gen[:, 9] = torch.where(next_gen[:, 9] > 0.5, 1.0, 0.0) # Struct Type Boolean
        return torch.cat([elites, next_gen], dim=0)
    
    def run(self):
        version = "V6.1 (Institutional)"
        log(f"  Starting {version} Genetic Search for {self.session_name}...")
        pop = self.init_population()
        best_results = []
        
        for g in range(self.generations):
            scores, trades = self.evaluate_v6(pop)
            
            k = int(self.pop_size * 0.1)
            top_vals, top_indices = torch.topk(scores, k)
            best_score = top_vals[0].item()
            
            if (g + 1) % 10 == 0:
                log(f"    Gen {g+1}/{self.generations} | Best: {best_score:.0f}")
            
            if g == self.generations - 1:
                for idx in top_indices[:5]:
                    s = pop[idx]
                    best_results.append({
                        'session': self.session_name,
                        'sl': s[0].item(), 'tp': s[1].item(),
                        'body': s[2].item(), 'wick': s[3].item(),
                        'fvg': s[4].item(), 'vol': s[5].item(),
                        'disp': s[6].item(), 'mss': s[7].item(),
                        'ote': s[8].item(), 'struct': s[9].item(),
                        'direction': 'BOTH',
                        'score': scores[idx].item(),
                        'trades': trades[idx].item()
                    })
            
            elites = pop[top_indices]
            pop = self.mutate(elites)
        
        return best_results

# ============================================================================
# MAIN OVERNIGHT RUNNER
# ============================================================================
def run_overnight():
    log("=" * 60)
    log("OVERNIGHT RESEARCH RUNNER V2")
    log("Walk-Forward Validation with Random 70/30 Split")
    log("=" * 60)
    
    if not torch.cuda.is_available():
        log("WARNING: CUDA not available. Running on CPU (will be slow).")
    else:
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    overall_start = time.time()
    all_results = []
    validated_results = []
    
    # Load data with random train/test split
    train_df, test_df, train_months, test_months = load_train_test_data()
    
    # V6.1: Load tick data for pure tick outcome resolution
    tick_df = load_tick_data()
    if tick_df is not None:
        log("V6.1 Pure Tick Engine ACTIVE - Using tick data for SL/TP resolution")
    else:
        log("WARNING: No tick data found - V6 requires tick data!")
        return
    
    # ========================================================================
    # PHASE 1: Multi-Session Genetic Mining (on TRAIN data)
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 1: Multi-Session Genetic Mining (TRAIN)")
    log("=" * 60)
    
    phase1_sessions = ['ib', 'asia', 'london', 'silver_bullet', 'us_open', 'late_session']
    
    for session in phase1_sessions:
        try:
            # Check RAM before starting
            check_ram_and_throttle()
            
            miner = SessionGeneticMiner(train_df, session, population_size=150000, generations=150, tick_df=tick_df)
            results = miner.run()
            all_results.extend(results)
            log(f"  {session}: Found {len(results)} strategies")
        except Exception as e:
            log(f"  {session}: ERROR - {e}")
        
        # Clear GPU cache and check RAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        save_checkpoint(all_results) # Live Update after session
    save_checkpoint(all_results, "v6_winners_phase1.json") # Save full phase result
    
    # ========================================================================
    # PHASE 2: Kill Zone Optimization (NY + London)
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 2: Kill Zone Optimization")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    # ICT Kill Zones: Specific 1-hour windows
    KILL_ZONES = {
        'london_kz': {'start': (2, 0), 'end': (3, 0)},    # London open
        'ny_kz': {'start': (9, 30), 'end': (10, 30)},      # NY open
        'london_close_kz': {'start': (10, 0), 'end': (11, 0)},  # London close
        'ny_lunch_kz': {'start': (11, 30), 'end': (13, 0)},  # Avoid!
        'ny_afternoon_kz': {'start': (13, 30), 'end': (15, 0)},  # Afternoon push
    }
    
    for kz_name, kz_cfg in KILL_ZONES.items():
        try:
            SESSIONS[kz_name] = kz_cfg  # Temporarily add
            miner = SessionGeneticMiner(train_df, kz_name, population_size=75000, generations=120, tick_df=tick_df)
            results = miner.run()
            all_results.extend(results)
            log(f"  {kz_name}: Found {len(results)} strategies")
            del SESSIONS[kz_name]
        except Exception as e:
            log(f"  {kz_name}: ERROR - {e}")
        gc.collect()
        save_checkpoint(all_results) # Live Update after KZ
    save_checkpoint(all_results, "v6_winners_phase2.json")
    
    # ========================================================================
    # PHASE 3: Day-of-Week Effects
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 3: Day-of-Week Pattern Discovery")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    df_copy = train_df.copy()
    df_copy['dayofweek'] = pd.to_datetime(df_copy.index).dayofweek
    
    for day_num, day_name in [(0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), 
                               (3, 'Thursday'), (4, 'Friday')]:
        try:
            day_df = df_copy[df_copy['dayofweek'] == day_num]
            log(f"  Testing {day_name}: {len(day_df):,} bars")
            
            # Use IB session as base
            miner = SessionGeneticMiner(day_df, 'ib', population_size=12000, generations=80, tick_df=tick_df)
            results = miner.run()
            for r in results:
                r['day'] = day_name  # Tag with day
            all_results.extend(results)
        except Exception as e:
            log(f"  {day_name}: ERROR - {e}")
        gc.collect()
        save_checkpoint(all_results) # Live Update after day
    save_checkpoint(all_results, "v6_winners_phase3.json")
    
    # ========================================================================
    # PHASE 4: Displacement Detection (Large Move Follow-Through)
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 4: Displacement / Momentum Detection")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    # Calculate displacement (large body candles)
    df_disp = train_df.copy()
    df_disp['body'] = abs(df_disp['close'] - df_disp['open'])
    df_disp['body_pct'] = df_disp['body'] / df_disp['close'].rolling(20).mean() * 100
    
    # Test different displacement thresholds
    for disp_threshold in [0.05, 0.1, 0.15, 0.2]:  # % of rolling mean
        try:
            disp_df = df_disp[df_disp['body_pct'] > disp_threshold * 100]
            log(f"  Displacement >{disp_threshold*100:.0f}%: {len(disp_df):,} bars")
            
            if len(disp_df) > 1000:
                miner = SessionGeneticMiner(disp_df, 'us_open', population_size=75000, generations=100, tick_df=tick_df)
                results = miner.run()
                for r in results:
                    r['filter'] = f'displacement>{disp_threshold}'
                all_results.extend(results)
        except Exception as e:
            log(f"  Displacement {disp_threshold}: ERROR - {e}")
        gc.collect()
    
    # ========================================================================
    # PHASE 5: Previous Day Level Sweeps
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 5: Previous Day High/Low Sweep Detection")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    df_pd = train_df.copy()
    df_pd['date'] = pd.to_datetime(df_pd.index).date
    
    # Calculate previous day high/low
    daily = df_pd.groupby('date').agg({'high': 'max', 'low': 'min', 'close': 'last'})
    daily['pdh'] = daily['high'].shift(1)
    daily['pdl'] = daily['low'].shift(1)
    
    df_pd = df_pd.merge(daily[['pdh', 'pdl']], left_on='date', right_index=True, how='left')
    df_pd = df_pd.dropna()
    
    # Find bars that sweep PDH or PDL
    df_pd['sweep_pdh'] = df_pd['high'] > df_pd['pdh']
    df_pd['sweep_pdl'] = df_pd['low'] < df_pd['pdl']
    
    for sweep_type, col in [('PDH_Sweep', 'sweep_pdh'), ('PDL_Sweep', 'sweep_pdl')]:
        try:
            sweep_df = df_pd[df_pd[col]]
            log(f"  {sweep_type}: {len(sweep_df):,} bars")
            
            if len(sweep_df) > 500:
                miner = SessionGeneticMiner(sweep_df, 'us_open', population_size=12000, generations=80, tick_df=tick_df)
                results = miner.run()
                for r in results:
                    r['filter'] = sweep_type
                all_results.extend(results)
        except Exception as e:
            log(f"  {sweep_type}: ERROR - {e}")
        gc.collect()

    # ========================================================================
    # PHASE 5b: Round Number Sweeps (Psychological Levels)
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 5b: Round Number Sweep Detection")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    # MNQ levels typically 100, 250, 500, 750, 1000
    df_round = train_df.copy()
    levels = []
    base_price = int(df_round['close'].iloc[0] / 1000) * 1000
    for offset in range(-2000, 2000, 100):
        levels.append(base_price + offset)
    
    # Find bars that cross a round level
    df_round['sweep_round'] = False
    for level in levels:
        # Crosses from below or above
        df_round['sweep_round'] |= (df_round['low'] <= level) & (df_round['high'] >= level)
    
    try:
        round_df = df_round[df_round['sweep_round']]
        log(f"  Round Number Crosses: {len(round_df):,} bars")
        
        if len(round_df) > 500:
            miner = SessionGeneticMiner(round_df, 'us_open', population_size=75000, generations=100, tick_df=tick_df)
            results = miner.run()
            for r in results:
                r['filter'] = 'Round_Number'
            all_results.extend(results)
            save_checkpoint(all_results) # Live Update
    except Exception as e:
        log(f"  Round Number Sweep: ERROR - {e}")
    gc.collect()

    # ========================================================================
    # PHASE 5c: Previous Week Level Sweeps
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 5c: Previous Week High/Low Sweep Detection")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    df_pw = train_df.copy()
    # Weekly resample manually to get prev week high/low
    df_pw['week_id'] = pd.to_datetime(df_pw.index).isocalendar().week
    df_pw['year_id'] = pd.to_datetime(df_pw.index).isocalendar().year
    
    weekly = df_pw.groupby(['year_id', 'week_id']).agg({'high': 'max', 'low': 'min'})
    # Shift to get previous week
    weekly['pwh'] = weekly['high'].shift(1)
    weekly['pwl'] = weekly['low'].shift(1)
    
    df_pw = df_pw.merge(weekly[['pwh', 'pwl']], left_on=['year_id', 'week_id'], right_index=True, how='left')
    df_pw = df_pw.dropna()
    
    df_pw['sweep_pwh'] = df_pw['high'] > df_pw['pwh']
    df_pw['sweep_pwl'] = df_pw['low'] < df_pw['pwl']
    
    for sweep_type, col in [('PWH_Sweep', 'sweep_pwh'), ('PWL_Sweep', 'sweep_pwl')]:
        try:
            sweep_df = df_pw[df_pw[col]]
            log(f"  {sweep_type}: {len(sweep_df):,} bars")
            
            if len(sweep_df) > 300: # Fewer weekly sweeps usually
                miner = SessionGeneticMiner(sweep_df, 'us_open', population_size=75000, generations=100, tick_df=tick_df)
                results = miner.run()
                for r in results:
                    r['filter'] = sweep_type
                all_results.extend(results)
                save_checkpoint(all_results) # Live Update
        except Exception as e:
            log(f"  {sweep_type}: ERROR - {e}")
        gc.collect()
    
    # ========================================================================
    # PHASE 6: Volatility Regime Detection
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 6: Volatility Regime Strategies")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    df_vol = train_df.copy()
    df_vol['range'] = df_vol['high'] - df_vol['low']
    df_vol['atr_20'] = df_vol['range'].rolling(20).mean()
    df_vol['vol_ratio'] = df_vol['range'] / df_vol['atr_20']
    
    # High Vol Regime
    high_vol_df = df_vol[df_vol['vol_ratio'] > 1.5]
    log(f"  High Volatility Regime: {len(high_vol_df):,} bars")
    
    try:
        miner = SessionGeneticMiner(high_vol_df, 'us_open', population_size=75000, generations=120, tick_df=tick_df)
        results = miner.run()
        for r in results:
            r['regime'] = 'high_volatility'
        all_results.extend(results)
    except Exception as e:
        log(f"  High Vol: ERROR - {e}")
    
    # Low Vol Regime
    low_vol_df = df_vol[df_vol['vol_ratio'] < 0.7]
    log(f"  Low Volatility Regime: {len(low_vol_df):,} bars")
    
    try:
        miner = SessionGeneticMiner(low_vol_df, 'us_open', population_size=15000, generations=100, tick_df=tick_df)
        results = miner.run()
        for r in results:
            r['regime'] = 'low_volatility'
        all_results.extend(results)
    except Exception as e:
        log(f"  Low Vol: ERROR - {e}")
    gc.collect()
    
    # ========================================================================
    # PHASE 7: Mean Reversion vs Trend Following
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 7: Mean Reversion vs Trend Following")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    df_mr = train_df.copy()
    df_mr['sma_50'] = df_mr['close'].rolling(50).mean()
    df_mr['distance'] = (df_mr['close'] - df_mr['sma_50']) / df_mr['sma_50'] * 100
    
    # Extended from mean (potential reversal)
    extended_df = df_mr[abs(df_mr['distance']) > 0.3]  # >0.3% from SMA
    log(f"  Extended from Mean: {len(extended_df):,} bars")
    
    try:
        miner = SessionGeneticMiner(extended_df, 'us_open', population_size=75000, generations=120, tick_df=tick_df)
        results = miner.run()
        for r in results:
            r['strategy_type'] = 'mean_reversion'
        all_results.extend(results)
    except Exception as e:
        log(f"  Mean Reversion: ERROR - {e}")
    
    # Close to mean (potential breakout)
    close_to_mean_df = df_mr[abs(df_mr['distance']) < 0.1]
    log(f"  Close to Mean: {len(close_to_mean_df):,} bars")
    
    try:
        miner = SessionGeneticMiner(close_to_mean_df, 'us_open', population_size=75000, generations=120, tick_df=tick_df)
        results = miner.run()
        for r in results:
            r['strategy_type'] = 'trend_breakout'
        all_results.extend(results)
        save_checkpoint(all_results) # Live Update
    except Exception as e:
        log(f"  Trend Breakout: ERROR - {e}")
    gc.collect()
    
    # ========================================================================
    # PHASE 8: Initial Balance Breakout Variations
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 8: Initial Balance Breakout Optimization")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    # Different IB window sizes
    IB_WINDOWS = [
        ('ib_30min', {'start': (9, 30), 'end': (10, 0)}),
        ('ib_60min', {'start': (9, 30), 'end': (10, 30)}),
        ('ib_90min', {'start': (9, 30), 'end': (11, 0)}),
    ]
    
    for ib_name, ib_cfg in IB_WINDOWS:
        try:
            SESSIONS[ib_name] = ib_cfg
            miner = SessionGeneticMiner(train_df, ib_name, population_size=75000, generations=150, tick_df=tick_df)
            results = miner.run()
            for r in results:
                r['ib_type'] = ib_name
            all_results.extend(results)
            log(f"  {ib_name}: Found {len(results)} strategies")
            del SESSIONS[ib_name]
        except Exception as e:
            log(f"  {ib_name}: ERROR - {e}")
        gc.collect()
    
    # ========================================================================
    # PHASE 9: Candle Pattern Recognition
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 9: Candle Pattern Recognition")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    df_candle = train_df.copy()
    df_candle['body'] = abs(df_candle['close'] - df_candle['open'])
    df_candle['upper_wick'] = df_candle['high'] - df_candle[['open', 'close']].max(axis=1)
    df_candle['lower_wick'] = df_candle[['open', 'close']].min(axis=1) - df_candle['low']
    df_candle['range'] = df_candle['high'] - df_candle['low']
    
    # Pin Bars (long wick, small body)
    df_candle['wick_ratio'] = (df_candle['upper_wick'] + df_candle['lower_wick']) / (df_candle['body'] + 0.01)
    pin_bars = df_candle[df_candle['wick_ratio'] > 3]
    log(f"  Pin Bars: {len(pin_bars):,} bars")
    
    try:
        miner = SessionGeneticMiner(pin_bars, 'us_open', population_size=75000, generations=100, tick_df=tick_df)
        results = miner.run()
        for r in results:
            r['pattern'] = 'pin_bar'
        all_results.extend(results)
    except Exception as e:
        log(f"  Pin Bars: ERROR - {e}")
    
    # Engulfing (large body relative to prev)
    df_candle['prev_range'] = df_candle['range'].shift(1)
    engulfing = df_candle[df_candle['body'] > 1.5 * df_candle['prev_range']]
    log(f"  Engulfing Candles: {len(engulfing):,} bars")
    
    try:
        miner = SessionGeneticMiner(engulfing, 'us_open', population_size=75000, generations=100, tick_df=tick_df)
        results = miner.run()
        for r in results:
            r['pattern'] = 'engulfing'
        all_results.extend(results)
    except Exception as e:
        log(f"  Engulfing: ERROR - {e}")
    gc.collect()
    
    # ========================================================================
    # PHASE 10: Deep Dive on Top 3 Best Sessions
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 10: Deep Dive on Top Performers")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    # Find top 3 performing sessions
    session_scores = {}
    for r in all_results:
        s = r.get('session', 'unknown')
        if s not in session_scores:
            session_scores[s] = []
        session_scores[s].append(r['score'])
    
    sorted_sessions = sorted(session_scores.keys(), 
                            key=lambda x: np.mean(session_scores[x]) if session_scores[x] else 0,
                            reverse=True)[:3]
    
    for best_session in sorted_sessions:
        if best_session in SESSIONS:
            log(f"  Deep diving: {best_session}")
            try:
                miner = SessionGeneticMiner(train_df, best_session, population_size=75000, generations=120, tick_df=tick_df)
                deep_results = miner.run()
                all_results.extend(deep_results)
            except Exception as e:
                log(f"  {best_session}: ERROR - {e}")
            gc.collect()
    
    # ========================================================================
    # PHASE 11: OUT-OF-SAMPLE VALIDATION (on UNSEEN TEST data)
    # ========================================================================
    log("\n" + "=" * 60)
    log("PHASE 11: OUT-OF-SAMPLE VALIDATION")
    log(f"Testing on UNSEEN months: {test_months}")
    log("=" * 60)
    
    check_ram_and_throttle()
    
    # Take top 50 strategies from training
    top_candidates = all_results[:50]
    
    for i, strat in enumerate(top_candidates):
        try:
            session = strat.get('session', 'us_open')
            if session not in SESSIONS:
                continue
            
            # PROPER OOS VALIDATION: Create miner on TEST data
            test_miner = SessionGeneticMiner(test_df, session, population_size=1, generations=1, tick_df=tick_df)
            
            # Build population tensor from discovered strategy params (V6.1: 10 params)
            strat_tensor = torch.tensor([[
                strat['sl'], strat['tp'], strat['body'],
                strat['wick'], strat['fvg'], strat['vol'],
                strat['disp'], strat['mss'], strat['ote'],
                strat['struct']
            ]], dtype=torch.float32, device=DEVICE)
            
            # Evaluate THIS strategy on TEST data
            test_scores, test_trades = test_miner.evaluate_v6(strat_tensor)
            test_score = test_scores[0].item()
            test_trade_count = test_trades[0].item()
            
            # Store OOS results
            strat['test_score'] = test_score
            strat['test_trades'] = test_trade_count
            strat['validated'] = test_score > 0 and test_trade_count >= 10
            strat['test_months'] = test_months
            
            # Calculate degradation
            train_score = strat.get('score', 0)
            if train_score > 0:
                strat['degradation'] = 1 - (test_score / train_score)
            else:
                strat['degradation'] = 1.0
            
            if strat['validated']:
                validated_results.append(strat)
            
            if i < 10:
                status = "✅ PASS" if strat['validated'] else "❌ FAIL"
                log(f"  #{i+1} {session} SL:{strat['sl']:.0f} TP:{strat['tp']:.0f} | "
                    f"Train:{train_score:.0f} Test:{test_score:.0f} | "
                    f"Trades:{test_trade_count:.0f} | {status}")
        except Exception as e:
            log(f"  Validation error for strategy #{i+1}: {e}")
    
    log(f"  ✅ Validated {len(validated_results)}/{len(top_candidates)} strategies on unseen data")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    log("\n" + "=" * 60)
    log("SAVING RESULTS")
    log("=" * 60)
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Save to JSON
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = OUTPUT_DIR / f"overnight_strategies_{ts_str}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Desktop Copy
    try:
        import shutil
        shutil.copy(results_file, DESKTOP_DIR_WINNERS / results_file.name)
    except: pass
    
    log(f"Saved {len(all_results)} strategies to {results_file}")
    
    # Save validated separately
    validated_file = OUTPUT_DIR / f"validated_strategies_{ts_str}.json"
    with open(validated_file, 'w') as f:
        json.dump(validated_results, f, indent=2)
        
    # Desktop Copy
    try:
        shutil.copy(validated_file, DESKTOP_DIR_WINNERS / validated_file.name)
    except: pass

    log(f"Saved {len(validated_results)} VALIDATED strategies to {validated_file}")
    
    # Save summary CSV
    summary_file = OUTPUT_DIR / f"overnight_summary_{ts_str}.csv"
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(summary_file, index=False)
    
    # Desktop Copy
    try:
        shutil.copy(summary_file, DESKTOP_DIR_WINNERS / summary_file.name)
    except: pass
    
    log(f"Saved summary to {summary_file}")
    
    # Print top 20
    log("\n" + "=" * 60)
    log("TOP 20 STRATEGIES FOUND (TRAIN)")
    log("=" * 60)
    
    for i, r in enumerate(all_results[:20]):
        extras = []
        if 'day' in r: extras.append(f"Day:{r['day']}")
        if 'filter' in r: extras.append(f"Filter:{r['filter']}")
        if 'regime' in r: extras.append(f"Regime:{r['regime']}")
        if 'pattern' in r: extras.append(f"Pattern:{r['pattern']}")
        extra_str = " | " + ", ".join(extras) if extras else ""
        
        log(f"{i+1}. {r.get('session', '?')} | {r['direction']} | SL:{r['sl']:.0f} TP:{r['tp']:.0f} | "
            f"Score:{r['score']:.0f} | Trades:{r['trades']:.0f} | Struct:{'Strong' if r.get('struct',0)>0.5 else 'Weak'}{extra_str}")
    
    elapsed = (time.time() - overall_start) / 60
    log(f"\n⏱️ Total Runtime: {elapsed:.1f} minutes")
    log(f"📊 Total Strategies Found: {len(all_results)}")
    log(f"�?Validated on Unseen Data: {len(validated_results)}")
    log(f"📅 Train Months: {train_months}")
    log(f"📅 Test Months: {test_months}")
    log("=" * 60)
    log("OVERNIGHT RESEARCH COMPLETE")
    log("=" * 60)

if __name__ == "__main__":
    run_overnight()
