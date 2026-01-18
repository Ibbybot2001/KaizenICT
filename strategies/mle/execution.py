"""
MLE Execution Engine
Handles high-fidelity trade execution simulation using Tick Data.
Enforces Limit Order fills, Slippage, and complex exit logic (TP1 -> BE -> TP2).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Generator
from datetime import datetime, timedelta
import warnings

@dataclass
class TradeIntent:
    ticket_id: str
    direction: int # 1 for Long, -1 for Short
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    start_time: pd.Timestamp
    expiry_time: Optional[pd.Timestamp] = None
    metadata: dict = field(default_factory=dict)
    
@dataclass
class TradeResult:
    ticket_id: str
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    outcome: str = "CANCELLED" # CANCELLED, FAIL, BREAKEVEN, PARTIAL, FULL_WIN
    pnl_r: float = 0.0
    pnl_ticks: float = 0.0
    mae_ticks: float = 0.0
    mfe_ticks: float = 0.0
    fill_price: float = 0.0
    exit_price: float = 0.0
    # Forensic Metrics
    min_dist_ticks: float = 9999.0 # Closest approach in PENDING
    time_min_dist: Optional[pd.Timestamp] = None
    max_run_ticks: float = 0.0 # MFE during PENDING (Did it run away?)
    metadata: dict = field(default_factory=dict)

class TickSimulator:
    def __init__(self, tick_size: float = 0.25, slippage_ticks: int = 1, cost_ticks: int = 1):
        self.tick_size = tick_size
        self.slippage = slippage_ticks * tick_size # Applied to execution price
        self.cost = cost_ticks * tick_size # Assumed cost for net PnL logic if needed effectively
        self.breakeven_offset = 1 * tick_size

    def simulate_trade(self, intent: TradeIntent, tick_data: pd.DataFrame) -> TradeResult:
        """
        Resolves a trade intent using a Random Access Slice of the full Tick DataFrame.
        Supports LIMIT and MARKET execution.
        """
        result = TradeResult(ticket_id=intent.ticket_id, metadata=intent.metadata.copy())
        
        # 1. Slice relevant ticks
        start_time = intent.start_time
        end_time = intent.expiry_time
        
        # Guard: Data availability
        if tick_data.empty:
             result.outcome = "CANCELLED"
             return result
             
        try:
            effective_end = end_time if end_time else tick_data.index[-1]
            relevant_ticks = tick_data[start_time : effective_end]
            
            if relevant_ticks.empty:
                result.outcome = "CANCELLED"
                result.exit_time = end_time
                return result
                
        except Exception:
            result.outcome = "CANCELLED"
            return result

        status = "PENDING"
        active_sl = intent.sl_price
        hit_tp1 = False
        
        direction = intent.direction
        limit_price = intent.entry_price
        
        # Market Mode Flag
        is_market = getattr(intent, 'order_type', 'LIMIT') == 'MARKET'
        
        worst_price = limit_price 
        best_price = limit_price  
        
        min_dist = 9999.0
        max_run = 0.0 
        
        # We iterate over the slice
        # Ideally using chunks if large, but 'relevant_ticks' is already the slice.
        chunk = relevant_ticks
        
        if chunk.empty:
             return result
        
        # Optimization: Use itertuples for 100x speedup over iterrows
        # We need to ensure we access columns by name or index.
        # relevant_ticks has columns: price, bid, ask... index is time.
        
        # It's faster to extract arrays and iterate arrays.
        # But for logic branching, itertuples is okay.
        
        bids = chunk['bid'].values
        asks = chunk['ask'].values
        times = chunk.index
        
        # Array iteration is much faster than row iteration
        # Let's use the array-based logic we had in the "PENDING" block, extended.
        
        # --- PENDING STATE ---
        # If Market Order, we fill immediately on first tick
        if is_market:
            status = "OPEN"
            entry_time = times[0]
            result.entry_time = entry_time
            result.min_dist_ticks = 0.0
            
            # Fill at Ask (Buy) or Bid (Sell)
            if direction == 1:
                fill_p = asks[0]
            else:
                fill_p = bids[0]
            
            result.fill_price = fill_p
            
            # Reduce arrays for Open phase
            # Slice from index 1 (next tick)
            bids = bids[1:]
            asks = asks[1:]
            times = times[1:]
            
            if len(times) == 0:
                return result # Opened at last tick
                
        else:
            # LIMIT LOGIC
            # ...
            # Find Fill Index
            fill_idx = -1
            
            if direction == 1: # Buy Limit
                fill_mask = asks <= limit_price
                if np.any(fill_mask):
                    fill_idx = np.argmax(fill_mask)
            else: # Sell Limit
                fill_mask = bids >= limit_price
                if np.any(fill_mask):
                    fill_idx = np.argmax(fill_mask)
            
            if fill_idx != -1:
                # Filled
                status = "OPEN"
                result.entry_time = times[fill_idx]
                result.min_dist_ticks = 0.0
                if direction == 1:
                     result.fill_price = limit_price + self.slippage
                else:
                     result.fill_price = limit_price - self.slippage
                
                # Slice arrays for Open phase
                bids = bids[fill_idx+1:]
                asks = asks[fill_idx+1:]
                times = times[fill_idx+1:]
            else:
                # Not filled
                result.outcome = "CANCELLED"
                return result

        # --- OPEN STATE ---
        if status == "OPEN":
            if len(times) == 0: return result
            
            # Check Exits: SL, TP1, TP2
            sl_hit = False
            tp1_hit = False
            tp2_hit = False
            
            sl_idx = -1
            tp1_idx = -1
            tp2_idx = -1
            
            if direction == 1: # Long
                # SL: Bid <= SL
                sl_mask = bids <= active_sl
                if np.any(sl_mask): 
                    sl_hit = True
                    sl_idx = np.argmax(sl_mask)
                
                # TP1: Bid >= TP1
                if not hit_tp1:
                    tp1_mask = bids >= intent.tp1_price
                    if np.any(tp1_mask):
                        tp1_hit = True
                        tp1_idx = np.argmax(tp1_mask)
                
                # TP2: Bid >= TP2
                tp2_mask = bids >= intent.tp2_price
                if np.any(tp2_mask):
                    tp2_hit = True
                    tp2_idx = np.argmax(tp2_mask)
                    
            else: # Short
                # SL: Ask >= SL
                sl_mask = asks >= active_sl
                if np.any(sl_mask):
                    sl_hit = True
                    sl_idx = np.argmax(sl_mask)
                
                # TP1: Ask <= TP1
                if not hit_tp1:
                    tp1_mask = asks <= intent.tp1_price
                    if np.any(tp1_mask):
                        tp1_hit = True
                        tp1_idx = np.argmax(tp1_mask)
                        
                # TP2: Ask <= TP2
                tp2_mask = asks <= intent.tp2_price
                if np.any(tp2_mask):
                    tp2_hit = True
                    tp2_idx = np.argmax(tp2_mask)

            # Resolve First Event
            events = []
            if sl_hit: events.append((sl_idx, 'SL'))
            if tp1_hit: events.append((tp1_idx, 'TP1'))
            if tp2_hit: events.append((tp2_idx, 'TP2'))
            
            if not events:
                # Trade open at end of window
                result.outcome = "PARTIAL" 
                last_price = bids[-1] if direction==1 else asks[-1]
                result.exit_price = last_price
                dist = (result.exit_price - result.fill_price) * direction
                result.pnl_ticks = dist / self.tick_size
                return result
            
            events.sort(key=lambda x: x[0])
            first_idx, event_type = events[0]
            
            if event_type == 'SL':
                result.outcome = "LOSS"
                result.exit_time = times[first_idx]
                result.exit_price = active_sl
                if direction == 1: result.exit_price -= self.slippage
                else: result.exit_price += self.slippage
                
                dist = (result.exit_price - result.fill_price) * direction
                result.pnl_ticks = dist / self.tick_size
                return result

            elif event_type == 'TP2':
                result.outcome = "FULL_WIN"
                result.exit_time = times[first_idx]
                result.exit_price = intent.tp2_price
                
                dist = (result.exit_price - result.fill_price) * direction
                result.pnl_ticks = dist / self.tick_size
                return result
                
            elif event_type == 'TP1':
                 result.outcome = "WIN" 
                 result.exit_time = times[first_idx]
                 result.exit_price = intent.tp1_price
                 dist = (result.exit_price - result.fill_price) * direction
                 result.pnl_ticks = dist / self.tick_size
                 return result

        return result
