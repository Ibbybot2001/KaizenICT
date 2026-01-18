"""
Tick Generalized Backtester
Extends TickRealisticBacktester to support:
1. Arbitrary Signal Lists (not just time-based)
2. Partial Take Profits (TP1)
3. Move to Break Even after TP1
"""

from strategies.mle.tick_realistic_backtester import TickRealisticBacktester, LATENCY_MS
from datetime import timedelta
import pandas as pd
import numpy as np
import time

class TickGeneralizedBacktester(TickRealisticBacktester):
    def __init__(self, tick_data_source, bar_data_source):
        """
        :param tick_data_source: Path to parquet file OR pd.DataFrame
        :param bar_data_source: Path to parquet file OR pd.DataFrame
        """
        self.latency_ms = 500
        
        # Load Ticks
        if isinstance(tick_data_source, pd.DataFrame):
            self.df_ticks = tick_data_source.copy()
            # Ensure index is datetime
            if not isinstance(self.df_ticks.index, pd.DatetimeIndex):
                self.df_ticks.index = pd.to_datetime(self.df_ticks.index)
            
            self.tick_times = self.df_ticks.index.values
            
            # Assuming columns 'bid' and 'ask' exist
            if 'bid' in self.df_ticks.columns and 'ask' in self.df_ticks.columns:
                 self.tick_bids = self.df_ticks['bid'].values
                 self.tick_asks = self.df_ticks['ask'].values
            else:
                 price = self.df_ticks['price'].values
                 self.tick_bids = price
                 self.tick_asks = price
                 
        elif isinstance(tick_data_source, str):
            print(f"[Tick Backtester] Loading tick data from {tick_data_source}...")
            self.df_ticks = pd.read_parquet(tick_data_source)
            self.tick_times = pd.to_datetime(self.df_ticks.index).values
            if 'bid' in self.df_ticks.columns:
                self.tick_bids = self.df_ticks['bid'].values
                self.tick_asks = self.df_ticks['ask'].values
            else:
                p = self.df_ticks['price'].values
                self.tick_bids = p
                self.tick_asks = p
        else:
            raise ValueError("Invalid tick_data_source type")
            
        # Load Bars
        if isinstance(bar_data_source, pd.DataFrame):
            self.df_bars = bar_data_source.copy()
        elif isinstance(bar_data_source, str):
            print(f"[Tick Backtester] Loading 1-min bars from {bar_data_source}...")
            self.df_bars = pd.read_parquet(bar_data_source)
        else:
            raise ValueError("Invalid bar_data_source type")
    def find_exit_price_with_partials(self, entry_time, direction, stop_pts, target_pts, 
                                      tp1_pts=None, tp1_pct=0.5, move_to_be_after_tp1=False,
                                      slippage_pts=0.0, spread_pts=0.0):
        """
        Calculates PnL with partial exit logic.
        Now includes Spread and Slippage simulation.
        Realized Entry = (Price +/- Slippage +/- HalfSpread)
        Realized Exit = (Price +/- Slippage +/- HalfSpread)
        But simplified: 
        We assume 'entry_price' passed in is already the raw tick price.
        The caller must adjust Entry Price.
        This function just adjusts Exit Prices and Checks.
        """
        idx = self.tick_times.searchsorted(entry_time)
        if idx >= len(self.tick_times):
            return 0.0, "NO_DATA", entry_time
            
        # RAW TICK PRICE (Mid/Last)
        raw_entry_tick = self.tick_asks[idx] if direction == 1 else self.tick_bids[idx]
        
        # COST APPLICATION (Entry)
        # Long: Buy at Ask (Mid+Spread/2). Plus Slippage.
        # Short: Sell at Bid (Mid-Spread/2). Minus Slippage.
        # But tick_asks IS the Ask.
        # So we just add slippage.
        # Wait, if tick data has bid/ask columns, tick_asks includes spread.
        # But we default to 'price' if bid/ask missing.
        # Let's assume passed tick data is 'Price' (Last).
        # Cost = Spread/2 + Slippage. Total Penalty = Spread + 2*Slippage per round trip.
        # Let's just apply "Entry Penalty" and "Exit Penalty".
        # Penalty = (Spread_Pts / 2) + Slippage_Pts.
        
        penalty = (spread_pts / 2.0) + slippage_pts
        
        real_entry_price = raw_entry_tick + penalty if direction == 1 else raw_entry_tick - penalty
        
        # Adjust Targets/Stops based on REAL entry
        current_sl_price = real_entry_price - stop_pts if direction == 1 else real_entry_price + stop_pts
        tp1_price = real_entry_price + tp1_pts if direction == 1 and tp1_pts else None
        target_price = real_entry_price + target_pts if direction == 1 else real_entry_price - target_pts
        
        tp1_hit = False
        tp1_pnl = 0.0
        runner_pnl = 0.0
        exit_time = entry_time # Default
        
        # Handle numpy datetime arithmetic
        if isinstance(entry_time, np.datetime64):
            end_time_limit = entry_time + np.timedelta64(4, 'h')
        else:
            end_time_limit = entry_time + timedelta(hours=4)
            
        end_idx = self.tick_times.searchsorted(end_time_limit)
        
        prices = self.tick_bids[idx:end_idx] if direction == 1 else self.tick_asks[idx:end_idx]
        times = self.tick_times[idx:end_idx]
        
        for i in range(len(prices)):
            price = prices[i] # Bid for Long Exit, Ask for Short Exit
            current_time = times[i]
            exit_time = current_time 
            
            # EXIT CHECK (With Slippage on Exit?)
            # Usually SL is Market Order -> Slippage.
            # TP is Limit Order -> No Slippage (optimistic) or Standard Fill.
            # Let's apply Penalty to SL exits, but maybe not TP if it's Limit.
            # User asked for "1 tick slippage". 
            # Conservative: Apply penalty to ALL exits.
            
            # 1. Check TP1
            if tp1_price and not tp1_hit:
                # Limit Order Logic: Price touches level?
                # If Limit, we get the Price.
                hit_tp1 = (price >= tp1_price) if direction == 1 else (price <= tp1_price)
                if hit_tp1:
                    tp1_hit = True
                    tp1_pnl = tp1_pts # Limit fill = exact pts
                    if move_to_be_after_tp1:
                        # Move SL to Break Even (Real Entry)
                        current_sl_price = real_entry_price
                        
            # 2. Check SL
            hit_sl = (price <= current_sl_price) if direction == 1 else (price >= current_sl_price)
            if hit_sl:
                # Market Exit -> Apply Penalty
                exit_price = price - penalty if direction == 1 else price + penalty
                loss = (exit_price - real_entry_price) if direction == 1 else (real_entry_price - exit_price)
                runner_pnl = loss
                break
                
            # 3. Check Target (TP2)
            hit_target = (price >= target_price) if direction == 1 else (price <= target_price)
            if hit_target:
                # Limit Exit -> Exact Pts
                runner_pnl = target_pts
                break
                
        else:
            # End of loop force close (Market Exit)
            last_price = prices[-1] if len(prices) > 0 else real_entry_price
            exit_price = last_price - penalty if direction == 1 else last_price + penalty
            runner_pnl = (exit_price - real_entry_price) if direction == 1 else (real_entry_price - exit_price)
            
        if tp1_hit:
            total_pnl = (tp1_pnl * tp1_pct) + (runner_pnl * (1.0 - tp1_pct))
            reason = "TP1_HIT_RUNNER_CLOSED"
        else:
            total_pnl = runner_pnl
            reason = "SL_OR_TP2"
            
        return total_pnl, reason, exit_time

    def backtest_signals(self, signals, direction=1, stop_pts=10, target_pts=50, 
                         tp1_pts=None, tp1_pct=0.5, move_to_be=True,
                         slippage_pts=0.0, spread_pts=0.0):
        """
        Backtest signals with COST simulation.
        """
        start_time = time.time()
        trades = []
        total_pnl = 0.0
        wins = 0
        
        is_dict_list = len(signals) > 0 and isinstance(signals[0], dict)
        
        if len(self.tick_times) > 0:
            # DEBUG
            # print(f"DEBUG: Tick Type: {type(self.tick_times[0])} val: {self.tick_times[0]}")
            pass

        for sig in signals:
            if is_dict_list:
                sig_time = sig['time']
            else:
                sig_time = sig
            
            # Ensure sig_time is compatible with self.tick_times (numpy array)
            if isinstance(sig_time, pd.Timestamp):
                sig_time = sig_time.to_datetime64()
            
            fill_price, fill_time = self.find_fill_price(sig_time, direction)
            if fill_price is None:
                continue
                
            current_target_pts = target_pts
            if is_dict_list and 'target_price' in sig:
                target_price_level = sig['target_price']
                if direction == 1:
                    if target_price_level > fill_price:
                        current_target_pts = target_price_level - fill_price
                else:
                    if target_price_level < fill_price:
                        current_target_pts = fill_price - target_price_level
                        
            pnl, reason, exit_time = self.find_exit_price_with_partials(
                fill_time, direction, stop_pts, current_target_pts, 
                tp1_pts, tp1_pct, move_to_be
            )
            
            trades.append({
                'entry_time': fill_time,
                'exit_time': exit_time,
                'pnl': pnl,
                'reason': reason
            })
            total_pnl += pnl
            if pnl > 0:
                wins += 1
                
        num_trades = len(trades)
        pf = 0.0
        if num_trades > 0:
            gross_profit = sum([t['pnl'] for t in trades if t['pnl'] > 0])
            gross_loss = abs(sum([t['pnl'] for t in trades if t['pnl'] < 0]))
            pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
            wr = (wins / num_trades) * 100
        else:
            wr = 0.0
            
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return {
            "pnl": total_pnl,
            "trades": num_trades,
            "pf": pf,
            "wr": wr,
            "trade_list": trades,
            "trades_df": trades_df
        }
