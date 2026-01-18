"""
Evolutionary Strategy Engine V2 ("The Predator")
Autonomously designs, tests, and evolves trading strategies using Blind Split Verification.
SCALED TO 24 CORES (CPU PARALLELISM).
INSTITUTIONAL HARDENING: Regime Filters, Daily Limits, Stability Fitness.
INTERROGATION MODE: Telemetry Active.
"""

import pandas as pd
import numpy as np
import random
import json
import copy
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time
import multiprocessing

# Import Backtester
sys.path.append("C:/Users/CEO/ICT reinforcement")
from strategies.mle.tick_generalized_backtester import TickGeneralizedBacktester

# ==============================================================================
# 1. GENE DEFINITIONS (PREDATOR V2)
# ==============================================================================
OPTIONS = {
    'trigger': ['Vol_1.5x', 'Range_1.5x', 'Disp_60%'],
    'context': ['AM_Session', 'PM_Session', 'Full_US'],
    'sl': [10, 15, 20, 30],
    'tp': [20, 50, 100, 200], 
    'manager': ['Fixed', 'Scale_Out'],
    'be': [True, False],
    'regime_atr': ['Any', 'Low_Vol', 'High_Vol'], 
    'daily_limit': [1, 2, 3, 100],
}

# [INTERROGATION Q1] GENOME SIZE
GENOME_SIZE = 1
for k, v in OPTIONS.items(): GENOME_SIZE *= len(v)
print(f"  [INTERROGATION] Theoretical Genome Size: {GENOME_SIZE:,} Combinations")

class StrategyGene:
    def __init__(self, gene_dict=None):
        if gene_dict:
            self.dna = gene_dict
        else:
            self.dna = { k: random.choice(v) for k, v in OPTIONS.items() }
        self.fitness = 0.0
        self.stats = {}

    def __repr__(self):
        return f"{self.dna['trigger']}|{self.dna['context']}|ATR:{self.dna['regime_atr']}|Limit:{self.dna['daily_limit']}|SL{self.dna['sl']}|TP{self.dna['tp']}"

# ==============================================================================
# 2. WORKER LOGIC
# ==============================================================================
WORKER_TICKS = {}
WORKER_BARS = {}

def init_worker(tick_paths, bar_base_dir):
    """
    Worker Initializer: Loads data into Global Memory of the Process.
    """
    global WORKER_TICKS, WORKER_BARS
    
    for m, tick_path in tick_paths.items():
        if m not in WORKER_TICKS:
           try:
               df_ticks = pd.read_parquet(tick_path)
               bar_path = bar_base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m:02d}_clean_1m.parquet"
               df_bars = pd.read_parquet(bar_path)
               
               # Precompute Bar Features
               df_bars['time'] = pd.to_datetime(df_bars.index)
               df_bars['hour'] = df_bars['time'].dt.hour
               df_bars['minute'] = df_bars['time'].dt.minute
               
               vol_col = 'tick_volume' if 'tick_volume' in df_bars.columns else 'volume'
               df_bars['vol_ma'] = df_bars[vol_col].rolling(20).mean()
               
               df_bars['range'] = df_bars['high'] - df_bars['low']
               df_bars['range_ma'] = df_bars['range'].rolling(20).mean()
               
               # ATR Approximation (Range MA is close enough for speed)
               df_bars['is_high_vol'] = df_bars['range'] > df_bars['range_ma']
               
               WORKER_TICKS[m] = df_ticks
               WORKER_BARS[m] = df_bars
           except Exception as e:
               print(f"Worker Load Error {m}: {e}")

def eval_strategy_worker(strat_dna):
    """
    Worker Function: Evaluates ONE strategy against loaded data.
    Returns: (PnL, Trades, Max_DD, Eff_Score)
    """
    global WORKER_TICKS, WORKER_BARS
    
    total_pnl = 0.0
    all_trades = []
    
    for m in WORKER_TICKS:
        df_ticks = WORKER_TICKS[m]
        df_bars = WORKER_BARS[m]
        
        # 1. Calc Trigger Mask
        vol_col = 'tick_volume' if 'tick_volume' in df_bars.columns else 'volume'
        trig = strat_dna['trigger']
        if trig == 'Vol_1.5x': mask_t = df_bars[vol_col] > (df_bars['vol_ma'] * 1.5)
        elif trig == 'Range_1.5x': mask_t = df_bars['range'] > (df_bars['range_ma'] * 1.5)
        elif trig == 'Disp_60%': mask_t = (df_bars['close']-df_bars['open']).abs() > (df_bars['range']*0.6)
        else: mask_t = df_bars[vol_col] > -1
        
        # 2. Context Mask
        ctx = strat_dna['context']
        if ctx == 'Full_US': mask_c = ((df_bars['hour']==9)&(df_bars['minute']>=30)) | ((df_bars['hour']>=10)&(df_bars['hour']<16))
        elif ctx == 'PM_Session': mask_c = (df_bars['hour']>=13)&(df_bars['hour']<16)
        else: mask_c = (df_bars['hour']>=9)&(df_bars['hour']<12)
        
        # 3. Regime Mask
        reg = strat_dna['regime_atr']
        if reg == 'High_Vol': mask_r = df_bars['is_high_vol']
        elif reg == 'Low_Vol': mask_r = ~df_bars['is_high_vol']
        else: mask_r = True
        
        mask = mask_t & mask_c & mask_r & (df_bars['close'] > df_bars['open']) 
        raw_signals = df_bars[mask]['time'].tolist()
        
        if not raw_signals: continue
        
        # 4. Daily Shot Clock
        limit = strat_dna['daily_limit']
        if limit < 100:
            filtered_signals = []
            day_counts = {}
            for t in raw_signals:
                d = t.date()
                c = day_counts.get(d, 0)
                if c < limit:
                    filtered_signals.append(t)
                    day_counts[d] = c + 1
            signals = filtered_signals
        else:
            signals = raw_signals
            
        if not signals: continue

        # 5. Backtest
        tester = TickGeneralizedBacktester(df_ticks, df_bars)
        tp1 = 30 if strat_dna['manager'] == 'Scale_Out' else None
        pct = 0.5 if tp1 else 0.0
        move_be = strat_dna.get('be', True)
        
        res = tester.backtest_signals(signals, direction=1, 
                                    stop_pts=strat_dna['sl'], 
                                    target_pts=strat_dna['tp'], 
                                    tp1_pts=tp1, tp1_pct=pct, move_to_be=move_be)
        
        if res['trades'] > 0:
            total_pnl += res['pnl']
            if 'trades_df' in res and not res['trades_df'].empty:
                 all_trades.append(res['trades_df'])
    
    if not all_trades: return 0.0, 0, 0.0, 0.0
        
    full_df = pd.concat(all_trades).sort_values('fill_time')
    full_df['cum_pnl'] = full_df['pnl'].cumsum()
    
    peak = full_df['cum_pnl'].cummax()
    dd = (peak - full_df['cum_pnl'])
    max_dd = dd.max() if not dd.empty else 0.0
    
    return total_pnl, len(full_df), max_dd, 0.0

# ==============================================================================
# 3. EVOLUTION ENGINE CLASS
# ==============================================================================
class EvolutionEngine:
    def __init__(self, pop_size=200, train_months=[1,3,4,6,8,9,11,12], test_months=[2,5,7,10]):
        self.pop_size = pop_size
        self.train_months = train_months
        self.test_months = test_months
        self.population = []
        self.generation = 0
        self.base_dir = Path("C:/Users/CEO/ICT reinforcement/data/GOLDEN_DATA")
        self.df_train_proxy = self._load_data(train_months)
        self.test_tick_paths = self._get_tick_paths(test_months)

    def _load_data(self, months):
        print("Loading Training Data (1M Proxy)...")
        dfs = []
        files = list(self.base_dir.glob("USTEC_2025_GOLDEN_PARQUET/USTEC_2025-*_clean_1m.parquet"))
        for f in files:
            try:
                m = int(f.name.split('-')[1].split('_')[0])
                if m in months:
                    dfs.append(pd.read_parquet(f))
            except: pass
        if not dfs: raise ValueError("No training data found!")
        full = pd.concat(dfs).sort_index()
        return full 

    def _get_tick_paths(self, months):
        paths = {}
        for m in months:
            m_str = f"{m:02d}"
            p = self.base_dir / f"USTEC_2025_GOLDEN_PARQUET/USTEC_2025-{m_str}_clean_ticks.parquet"
            if p.exists(): paths[m] = p
        return paths

    def initialize_population(self):
        self.population = [StrategyGene() for _ in range(self.pop_size)]
        print(f"Gen 0 Initialized with {self.pop_size} strategies.")

    def evaluate_fitness(self):
        valid_candidates = [s.dna for s in self.population]
        
        # [INTERROGATION Q2] Failure Taxonomy
        death_stats = {
            'Low_Trades': 0,
            'Negative_PnL': 0,
            'Unstable_DD': 0,
            'Inefficient': 0
        }

        # Parallel Verify
        with multiprocessing.Pool(processes=24, initializer=init_worker, initargs=(self.test_tick_paths, self.base_dir)) as pool:
            results = pool.map(eval_strategy_worker, valid_candidates)
            
        # Fitness Assignment (FITNESS 2.0)
        for idx, (pnl, trades, dd, exp) in zip(range(len(results)), results):
            strat = self.population[idx]
            
            if trades < 10:
                strat.fitness = 0.0
                strat.stats = {'reason': 'Low Trades'}
                death_stats['Low_Trades'] += 1
                continue
                
            if pnl <= 0:
                strat.fitness = 0.0
                strat.stats = {'reason': 'Negative PnL'}
                death_stats['Negative_PnL'] += 1
                continue

            # Stability Checks
            dd_penalty = 1.0
            if dd > (pnl * 0.2): dd_penalty = 0.5 
            if dd > (pnl * 0.5): dd_penalty = 0.1 
            
            # Efficiency Score
            eff_score = pnl / trades
            
            strat.fitness = pnl * dd_penalty
            if eff_score < 10: strat.fitness *= 0.5
            
            # KILL CONDITIONS
            if dd_penalty < 0.2:
                strat.fitness = 0.0 
                death_stats['Unstable_DD'] += 1
                continue
            if eff_score < 5:
                 strat.fitness = 0.0
                 death_stats['Inefficient'] += 1
                 continue

            strat.stats = {'pnl': pnl, 'trades': trades, 'dd': dd, 'eff': eff_score}
            
        return death_stats

    def breed_next_gen(self):
        death_report = self.evaluate_fitness()
        
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        survivors = [s for s in self.population if s.fitness > 0]
        survival_rate = len(survivors) / self.pop_size
        
        # [INTERROGATION Q2 & Q3 Report]
        print(f"  [MORTUARY] {death_report} | Survivors: {len(survivors)} ({survival_rate:.1%})")

        if survivors:
            s = survivors[0]
            print(f"  >> Predator Alpha: {s} (Fit: {s.fitness:.0f} | PnL: {s.stats['pnl']:.0f} | DD: {s.stats['dd']:.0f})")
            elite = survivors[:int(self.pop_size*0.4)]
        else:
            print("  >> Extinction. Reseeding.")
            self.initialize_population()
            return

        children = []
        while len(children) < (self.pop_size - len(elite)):
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            child_dna = copy.deepcopy(p1.dna)
            for k in child_dna:
                if random.random() < 0.5: child_dna[k] = p2.dna[k]
            if random.random() < 0.2:
                key = random.choice(list(OPTIONS.keys()))
                child_dna[key] = random.choice(OPTIONS[key])
            children.append(StrategyGene(child_dna))
            
        self.population = elite + children
        self.generation += 1

    def run(self, generations=50):
        self.initialize_population()
        for g in range(generations):
            print(f"\n=== GENERATION {g} (PREDATOR MODE) ===")
            t0 = time.time()
            self.breed_next_gen()
            dt = time.time() - t0
            print(f"  Gen Time: {dt:.2f}s")
            
            if self.population and self.population[0].fitness > 0:
                try:
                    with open("output/evolution_best.json", "w") as f:
                        json.dump(self.population[0].dna, f)
                except: pass
        print(f"\nEOE. Restarting...")

if __name__ == "__main__":
    eng = EvolutionEngine(pop_size=200, train_months=[1,3,4,6,8,9,11,12], test_months=[2,5,7,10])
    eng.run(generations=50)
