import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from .config import TICK_SIZE, COMMISSION_PER_CONTRACT, SLIPPAGE_TICKS, CONTRACT_MULTIPLIER

class OrderType:
    LIMIT = 'LIMIT'
    MARKET = 'MARKET'
    STOP = 'STOP'

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str # 'LIMIT', 'STOP', 'MARKET'
    price: float
    qty: int
    sl: float = None
    tp: float = None
    status: str = 'PENDING' # PENDING, FILLED, CANCELED

@dataclass
class Trade:
    id: str
    entry_time: pd.Timestamp
    entry_price: float
    side: str
    qty: int
    sl: float = None
    tp: float = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = 'OPEN'

class Backtester:
    def __init__(self, data: pd.DataFrame, strategy=None):
        self.data = data
        self.strategy = strategy
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.equity = 100000.0  # Starting Balance
        self.current_time = None
        
    @property
    def position_size(self):
        size = 0
        for trade in self.trades:
            if trade.status == 'OPEN':
                if trade.side == 'BUY':
                    size += trade.qty
                else:
                    size -= trade.qty
        return size

    def run(self):
        """
        Main Event Loop. Iterates bar by bar.
        """
        print(f"Starting Backtest on {len(self.data)} bars...")
        
        # Pre-calculate indicators (Swings, FVGs) for the whole dataset slightly cheated
        # BUT we must only access them if their index <= current_time in the loop.
        # Ideally, we pass the "slice" to the strategy.
        
        # For performance, we'll iterate with index
        # ict_library functions return Series aligned with index. 
        
        if self.strategy:
            self.strategy.on_start(self.data)
            
        for i in range(len(self.data)):
            # 1. Update Time
            bar = self.data.iloc[i]
            self.current_time = self.data.index[i]
            
            # 2. Process Pending Orders (Limits / Stops)
            self._process_orders(bar)
            
            # 3. Check Active Positions (SL / TP)
            self._check_positions(bar)
            
            # 4. Strategy Logic (Generate Signals)
            if self.strategy:
                # We pass the CURRENT index 'i' to the strategy.
                # The strategy is responsible for looking back safely (i-1, i-2...)
                self.strategy.on_bar(i, bar)
        
        # Force-close any remaining open trades at the final bar's close
        final_bar = self.data.iloc[-1]
        final_time = self.data.index[-1]
        for trade in self.trades:
            if trade.status == 'OPEN':
                self._close_trade(trade, final_bar['close'], final_time)
                
        print(f"Backtest Complete. Final Equity: {self.equity:.2f}")

    def place_order(self, price, size, side, order_type, sl=None, tp=None):
        order = Order(
            id=f"ord_{len(self.orders)}",
            symbol="MNQ",
            side=side.upper(),
            order_type=order_type,
            price=price,
            qty=size,
            sl=sl,
            tp=tp
        )
        self.orders.append(order)
        # print(f"[{self.current_time}] Placed {order_type} {side} @ {price}")
        return order

    def place_limit_order(self, side, price, sl, tp, qty=1):
        return self.place_order(price, qty, side, 'LIMIT', sl, tp)

    def _process_orders(self, bar):
        """
        Check if pending orders are filled by current bar OHLC.
        """
        for order in self.orders:
            if order.status != 'PENDING':
                continue
                
            # execution logic
            filled = False
            fill_price = order.price
            
            if order.order_type == 'LIMIT':
                if order.side == 'BUY':
                    # If Low <= Limit, we buy.
                    if bar['low'] <= order.price:
                        filled = True
                        # Assume fill at limit unless gap
                        fill_price = order.price 
                        if bar['open'] < order.price: 
                            # If opened below limit, we fill at Open (Gap down)
                            fill_price = bar['open']
                            
                elif order.side == 'SELL':
                    if bar['high'] >= order.price:
                        filled = True
                        fill_price = order.price
                        if bar['open'] > order.price:
                            fill_price = bar['open']
            
            elif order.order_type == 'MARKET':
                filled = True
                fill_price = bar['open'] # Market orders execute at Open of the NEXT bar? 
                # OR if placed during the bar, they execute at Close of that bar?
                # In our on_bar logic, we place orders based on data up to Close.
                # So they should fill at OPEN of NEXT bar.
                # But here `bar` is the CURRENT bar being iterated.
                # If we placed the order at index i, and we process orders at i+1...
                # Actually, `self._process_orders(bar)` is called BEFORE `strategy.on_bar`.
                # So orders placed at `i` are pending. Loop moves to `i+1`. `process_orders` sees the order.
                # It fills at `i+1` Open. Correct.
            
            if filled:
                self._execute_trade(order, fill_price)

    def _execute_trade(self, order, price):
        order.status = 'FILLED'
        # Slippage
        if order.side == 'BUY':
            price += SLIPPAGE_TICKS * TICK_SIZE
        else:
            price -= SLIPPAGE_TICKS * TICK_SIZE
            
        trade = Trade(
            id=f"trd_{len(self.trades)}",
            entry_time=self.current_time,
            entry_price=price,
            side=order.side,
            qty=order.qty,
            sl=order.sl,
            tp=order.tp
        )
        self.trades.append(trade)
        # print(f"[{self.current_time}] FILLED {order.side} @ {price}")

    def _check_positions(self, bar):
        """
        Check SL/TP for open trades.
        """
        for trade in self.trades:
            if trade.status != 'OPEN':
                continue
            
            exit_price = None
            
            if trade.side == 'BUY':
                # Check SL (Low hits SL)
                if trade.sl and bar['low'] <= trade.sl:
                    exit_price = trade.sl
                    # Slippage on STOP
                    exit_price -= SLIPPAGE_TICKS * TICK_SIZE
                    
                # Check TP (High hits TP)
                elif trade.tp and bar['high'] >= trade.tp:
                    exit_price = trade.tp
                    # Limit exit usually usually no slippage or positive. We assume 0.
            
            elif trade.side == 'SELL':
                # Check SL (High hits SL)
                if trade.sl and bar['high'] >= trade.sl:
                    exit_price = trade.sl
                    exit_price += SLIPPAGE_TICKS * TICK_SIZE
                    
                # Check TP (Low hits TP)
                elif trade.tp and bar['low'] <= trade.tp:
                    exit_price = trade.tp

            if exit_price:
                self._close_trade(trade, exit_price, bar.name) # bar.name is index (Timestamp)

    def _close_trade(self, trade, price, time):
        trade.exit_price = price
        trade.exit_time = time
        trade.status = 'CLOSED'
        
        # Calc PnL
        if trade.side == 'BUY':
            # (Exit - Entry) * Qty * Multiplier - Comm
            raw_pnl = (trade.exit_price - trade.entry_price) * trade.qty * CONTRACT_MULTIPLIER
        else:
            raw_pnl = (trade.entry_price - trade.exit_price) * trade.qty * CONTRACT_MULTIPLIER
            
        # PnL - Commission
        # Comm is per contract per side (open + close = 2 actions * qty)
        comm = COMMISSION_PER_CONTRACT * trade.qty * 2
        trade.pnl = raw_pnl - comm
        
        self.equity += trade.pnl
        # print(f"[{time}] CLOSED {trade.side} PnL: {trade.pnl:.2f}")

    def cancel_all_orders(self):
        for o in self.orders:
            if o.status == 'PENDING':
                o.status = 'CANCELED'
