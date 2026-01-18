"""
Tick-Level Realistic Backtester
- 500ms Latency Simulation
- Market Orders Only
- Actual Tick Fill Prices
"""

import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Optional

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENCY_MS = 500  # 500 millisecond latency


class TickRealisticBacktester:
    """
    Realistic backtester using tick data for execution.
    - Setups detected on 1-min bars
    - Entries executed at tick level with 500ms latency
    - Market orders fill at next available tick
    """
    
    def __init__(self, tick_path: str, bar_path: str):
        self.tick_path = tick_path
        self.bar_path = bar_path
        self.load_data()
        
    def load_data(self):
        print("[Tick Backtester] Loading data...")
        t0 = time.time()
        
        # Load 1-min bars for setup detection
        print("  Loading 1-min bars...")
        df_bars = pd.read_parquet(self.bar_path)
        self.bars = df_bars
        self.bar_times = pd.to_datetime(df_bars.index)
        self.bar_closes = df_bars['close'].values
        self.bar_opens = df_bars['open'].values
        self.bar_highs = df_bars['high'].values
        self.bar_lows = df_bars['low'].values
        
        # Pre-compute bar time features
        self.bar_hours = np.array([t.hour for t in self.bar_times])
        self.bar_minutes = np.array([t.minute for t in self.bar_times])
        
        # Load tick data for execution
        print("  Loading tick data...")
        df_ticks = pd.read_parquet(self.tick_path)
        self.ticks = df_ticks
        self.tick_times = pd.to_datetime(df_ticks.index)
        
        # Detect column names (bid/ask or price)
        if 'bid' in df_ticks.columns:
            self.tick_bids = df_ticks['bid'].values
            self.tick_asks = df_ticks['ask'].values if 'ask' in df_ticks.columns else df_ticks['bid'].values + 0.25
        else:
            # Use mid price if no bid/ask
            self.tick_bids = df_ticks['close'].values if 'close' in df_ticks.columns else df_ticks.iloc[:, 0].values
            self.tick_asks = self.tick_bids + 0.25
            
        print(f"[Tick Backtester] Loaded {len(self.bars):,} bars, {len(self.ticks):,} ticks in {time.time()-t0:.2f}s")
        
    def find_fill_price(self, signal_time: pd.Timestamp, direction: int) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
        """
        Find the fill price for a market order with 500ms latency.
        
        Args:
            signal_time: When the signal fired (bar close)
            direction: 1 for long (fill at ask), -1 for short (fill at bid)
            
        Returns:
            (fill_price, fill_time) or (None, None) if no tick found
        """
        # Add latency
        if isinstance(signal_time, np.datetime64):
             exec_time = signal_time + np.timedelta64(LATENCY_MS, 'ms')
        else:
             exec_time = signal_time + timedelta(milliseconds=LATENCY_MS)
        
        # Find first tick at or after exec_time
        idx = self.tick_times.searchsorted(exec_time)
        
        if idx >= len(self.tick_times):
            return None, None
            
        fill_time = self.tick_times[idx]
        
        # Market order: Long fills at ASK, Short fills at BID
        if direction == 1:
            fill_price = self.tick_asks[idx]
        else:
            fill_price = self.tick_bids[idx]
            
        return float(fill_price), fill_time
    
    def find_exit_price(self, entry_time: pd.Timestamp, direction: int, 
                        stop_pts: float, target_pts: float, 
                        move_to_be_pts: float = None,
                        trail_trigger_pts: float = None,
                        trail_dist_pts: float = None,
                        max_bars: int = 60) -> Tuple[float, str]:
        """
        Find exit price based on SL/TP, Break-Even, and Trailing Stop.
        """
        # Get entry bar index
        entry_bar_idx = self.bar_times.searchsorted(entry_time)
        
        if entry_bar_idx >= len(self.bar_times):
            return 0.0, "NO_DATA"
            
        entry_price = self.bar_closes[entry_bar_idx]
        
        # Initial SL
        current_sl_price = entry_price - stop_pts if direction == 1 else entry_price + stop_pts
        
        # State flags
        is_be_active = False
        is_trailing_active = False
        
        # Track max/min price reached since entry to calculate trail
        max_price_reached = -float('inf') if direction == 1 else float('inf')
        
        # Scan forward
        for i in range(entry_bar_idx + 1, min(entry_bar_idx + max_bars, len(self.bar_times))):
            bar_high = self.bar_highs[i]
            bar_low = self.bar_lows[i]
            
            if direction == 1:  # Long
                # Update Max Price
                max_price_reached = max(max_price_reached, bar_high)
                
                # 1. Check Trailing Stop Logic (Prioritize logic before checking current bar low)
                if trail_trigger_pts is not None and trail_dist_pts is not None:
                    if not is_trailing_active:
                        if max_price_reached >= entry_price + trail_trigger_pts:
                            is_trailing_active = True
                    
                    if is_trailing_active:
                        # Trail based on the *highest high seen so far*
                        potential_sl = max_price_reached - trail_dist_pts
                        current_sl_price = max(current_sl_price, potential_sl)
                
                # 2. Check Break Even (Only if trailing hasn't already moved SL above entry)
                if move_to_be_pts is not None and not is_be_active:
                    if max_price_reached >= entry_price + move_to_be_pts:
                        if current_sl_price < entry_price: # Only move if not already above
                            current_sl_price = entry_price
                        is_be_active = True
                
                # 3. Check if SL Hit (Current Bar Low)
                if bar_low <= current_sl_price:
                    pnl = current_sl_price - entry_price
                    reason = "TRAILING_STOP" if is_trailing_active else ("BREAK_EVEN" if is_be_active else "STOP_LOSS")
                    return pnl, reason
                
                # 4. Check Target
                if bar_high >= entry_price + target_pts:
                    return target_pts, "TARGET"
                
                        
            else:  # Short
                # Update Min Price
                max_price_reached = min(max_price_reached, bar_low) # "max_price_reached" acts as extreme price
                
                # 1. Check Trailing
                if trail_trigger_pts is not None and trail_dist_pts is not None:
                    if not is_trailing_active:
                        if max_price_reached <= entry_price - trail_trigger_pts:
                            is_trailing_active = True
                    
                    if is_trailing_active:
                        potential_sl = max_price_reached + trail_dist_pts
                        current_sl_price = min(current_sl_price, potential_sl)
                        
                # 2. Check Break Even
                if move_to_be_pts is not None and not is_be_active:
                    if max_price_reached <= entry_price - move_to_be_pts:
                        if current_sl_price > entry_price:
                            current_sl_price = entry_price
                        is_be_active = True
                        
                # 3. Check SL Hit
                if bar_high >= current_sl_price:
                    pnl = entry_price - current_sl_price
                    reason = "TRAILING_STOP" if is_trailing_active else ("BREAK_EVEN" if is_be_active else "STOP_LOSS")
                    return pnl, reason
                
                # 4. Check Target
                if bar_low <= entry_price - target_pts:
                    return target_pts, "TARGET"
        
        # Max bars reached
        final_price = self.bar_closes[min(entry_bar_idx + max_bars, len(self.bar_closes) - 1)]
        pnl = (final_price - entry_price) * direction
        return pnl, "MAX_BARS"
    
    def backtest_strategy(self, trigger_hour: int, trigger_minute: int, 
                          direction: int, stop_pts: float = 10.0, target_pts: float = 40.0,
                          move_to_be_pts: float = None,
                          trail_trigger_pts: float = None,
                          trail_dist_pts: float = None) -> dict:
        """
        Backtest a single strategy with tick-level execution.
        """
        trades = []
        
        # Find all bars matching the trigger time
        mask = (self.bar_hours == trigger_hour) & (self.bar_minutes == trigger_minute)
        signal_indices = np.where(mask)[0]
        
        for idx in signal_indices:
            signal_time = self.bar_times[idx]
            
            # Get fill price with 500ms latency
            fill_price, fill_time = self.find_fill_price(signal_time, direction)
            
            if fill_price is None:
                continue
                
            # Get exit
            exit_pnl, exit_reason = self.find_exit_price(
                fill_time, direction, stop_pts, target_pts, move_to_be_pts,
                trail_trigger_pts, trail_dist_pts
            )
            
            trades.append({
                'signal_time': signal_time,
                'fill_time': fill_time,
                'fill_price': fill_price,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'pnl': exit_pnl,
                'exit_reason': exit_reason
            })
        
        # Calculate stats
        if not trades:
            return {'trades': 0, 'pnl': 0, 'pf': 0, 'wr': 0}
            
        df_trades = pd.DataFrame(trades)
        total_pnl = df_trades['pnl'].sum()
        winners = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
        losers = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
        
        pf = winners / losers if losers > 0 else float('inf')
        wr = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100
        
        return {
            'trades': len(trades),
            'pnl': total_pnl,
            'pf': pf,
            'wr': wr,
            'avg_pnl': total_pnl / len(trades),
            'trades_df': df_trades
        }
    
    def run_grid_search(self, hours: list, minutes: list, directions: list,
                        stops: list, targets: list, be_levels: list = [None],
                        trail_configs: list = [(None, None)]) -> pd.DataFrame:
        """
        Grid search over strategy parameters.
        """
        results = []
        total = len(hours) * len(minutes) * len(directions) * len(stops) * len(targets) * len(be_levels) * len(trail_configs)
        i = 0
        
        print(f"[Tick Backtester] Running {total} strategy combinations...")
        
        for h in hours:
            for m in minutes:
                for d in directions:
                    for sl in stops:
                        for tp in targets:
                            for be in be_levels:
                                for (trail_trig, trail_dist) in trail_configs:
                                    stats = self.backtest_strategy(h, m, d, sl, tp, be, trail_trig, trail_dist)
                                    
                                    results.append({
                                        'hour': h,
                                        'minute': m,
                                        'direction': 'LONG' if d == 1 else 'SHORT',
                                        'stop_pts': sl,
                                        'target_pts': tp,
                                        'be_pts': be if be is not None else 0,
                                        'trail_trig': trail_trig if trail_trig is not None else 0,
                                        'trail_dist': trail_dist if trail_dist is not None else 0,
                                        'trades': stats['trades'],
                                        'pnl': stats['pnl'],
                                        'pf': stats['pf'],
                                        'wr': stats['wr']
                                    })
                                    
                                    i += 1
                                    if i % 100 == 0:
                                        print(f"  Progress: {i}/{total}")
        
        return pd.DataFrame(results)


def test_c3_with_latency():
    """Test the 15:00 strategy with realistic 500ms latency."""
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    tick_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_ticks.parquet"
    bar_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    out_path = base_dir / "output/tick_summary.txt"
    
    backtester = TickRealisticBacktester(str(tick_path), str(bar_path))
    
    results_str = []
    results_str.append("Testing C3 (15:00 Long) with 500ms Latency")
    results_str.append("=" * 60)
    
    # Test 1: Standard
    res1 = backtester.backtest_strategy(15, 0, 1, 10.0, 40.0, move_to_be_pts=None)
    results_str.append(f"\n1. Standard (No BE):")
    results_str.append(f"   Trades: {res1['trades']}")
    results_str.append(f"   PnL: {res1['pnl']:.2f}, PF: {res1['pf']:.2f}, WR: {res1['wr']:.1f}%")
                         
    # Test 2: BE at 5 pts
    res2 = backtester.backtest_strategy(15, 0, 1, 10.0, 40.0, move_to_be_pts=5.0)
    results_str.append(f"\n2. BE @ 5 pts:")
    results_str.append(f"   Trades: {res2['trades']}")
    results_str.append(f"   PnL: {res2['pnl']:.2f}, PF: {res2['pf']:.2f}, WR: {res2['wr']:.1f}%")
    
    # Test 3: BE at 10 pts
    res3 = backtester.backtest_strategy(15, 0, 1, 10.0, 40.0, move_to_be_pts=10.0)
    results_str.append(f"\n3. BE @ 10 pts:")
    results_str.append(f"   Trades: {res3['trades']}")
    results_str.append(f"   PnL: {res3['pnl']:.2f}, PF: {res3['pf']:.2f}, WR: {res3['wr']:.1f}%")
    
    final_output = "\n".join(results_str)
    print(final_output)
    
    with open(out_path, "w") as f:
        f.write(final_output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    test_c3_with_latency()
