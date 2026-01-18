"""
Fill simulator with realistic order execution logic.

RULES:
- Market orders: fill at next bar open + slippage
- Limit orders: fill only if price trades through (H/L check)
- Stop orders: fill at stop price or worse (gap handling)
- Commission tracked per contract
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

from ..constants import SLIPPAGE_TICKS, TICK_SIZE, COMMISSION_PER_CONTRACT, CONTRACT_MULTIPLIER
from .trade import Order, OrderType, Side


@dataclass
class FillResult:
    """Result of a fill attempt."""
    filled: bool
    fill_price: Optional[float] = None
    fill_time: Optional[pd.Timestamp] = None
    commission: float = 0.0
    slippage: float = 0.0
    reason: str = ""


class FillSimulator:
    """
    Simulates realistic order fills.
    
    NO LOOKAHEAD: Fill decisions use only current bar OHLC,
    which represents what was knowable at bar close.
    """
    
    def __init__(self, 
                 slippage_ticks: int = SLIPPAGE_TICKS,
                 commission_per_contract: float = COMMISSION_PER_CONTRACT):
        self.slippage_ticks = slippage_ticks
        self.slippage_points = slippage_ticks * TICK_SIZE
        self.commission_per_contract = commission_per_contract
    
    def attempt_fill(self, order: Order, bar: pd.Series) -> FillResult:
        """
        Attempt to fill an order using the given bar's OHLC.
        
        Args:
            order: The pending order
            bar: Current bar with 'open', 'high', 'low', 'close', 'name' (timestamp)
            
        Returns:
            FillResult indicating if/how the order was filled
        """
        bar_time = bar.name
        bar_open = bar['open']
        bar_high = bar['high']
        bar_low = bar['low']
        
        if order.order_type == OrderType.MARKET:
            return self._fill_market(order, bar_open, bar_time)
        elif order.order_type == OrderType.LIMIT:
            return self._fill_limit(order, bar_high, bar_low, bar_time)
        elif order.order_type == OrderType.STOP:
            return self._fill_stop(order, bar_open, bar_high, bar_low, bar_time)
        else:
            return FillResult(filled=False, reason=f"Unknown order type: {order.order_type}")
    
    def _fill_market(self, order: Order, bar_open: float, 
                     bar_time: pd.Timestamp) -> FillResult:
        """
        Market order fills at bar open + slippage.
        Slippage direction is adverse to the trade.
        """
        if order.side == Side.LONG:
            # Buy at open + slippage (worse for longs)
            fill_price = bar_open + self.slippage_points
        else:
            # Sell at open - slippage (worse for shorts)
            fill_price = bar_open - self.slippage_points
        
        commission = self.commission_per_contract * order.qty
        
        return FillResult(
            filled=True,
            fill_price=fill_price,
            fill_time=bar_time,
            commission=commission,
            slippage=self.slippage_points,
            reason="Market order filled at open + slippage"
        )
    
    def _fill_limit(self, order: Order, bar_high: float, bar_low: float,
                    bar_time: pd.Timestamp) -> FillResult:
        """
        Limit order fills only if price trades through the limit level.
        
        LONG limit: fills if bar_low <= limit_price
        SHORT limit: fills if bar_high >= limit_price
        
        Fill price is the limit price (no slippage on limit fills).
        """
        if order.side == Side.LONG:
            # Buying at limit - need price to come down to us
            if bar_low <= order.entry_price:
                return FillResult(
                    filled=True,
                    fill_price=order.entry_price,  # Limit price
                    fill_time=bar_time,
                    commission=self.commission_per_contract * order.qty,
                    slippage=0.0,
                    reason="Limit order filled - low touched limit"
                )
        else:  # SHORT
            # Selling at limit - need price to come up to us
            if bar_high >= order.entry_price:
                return FillResult(
                    filled=True,
                    fill_price=order.entry_price,  # Limit price
                    fill_time=bar_time,
                    commission=self.commission_per_contract * order.qty,
                    slippage=0.0,
                    reason="Limit order filled - high touched limit"
                )
        
        return FillResult(filled=False, reason="Limit not reached")
    
    def _fill_stop(self, order: Order, bar_open: float, bar_high: float,
                   bar_low: float, bar_time: pd.Timestamp) -> FillResult:
        """
        Stop order triggers when price reaches stop level.
        
        LONG stop (buy stop): triggers if bar_high >= stop_price
        SHORT stop (sell stop): triggers if bar_low <= stop_price
        
        Gap handling: if bar opens through stop, fill at open (worse).
        """
        if order.side == Side.LONG:
            # Buy stop - waiting for price to rise to stop
            if bar_open >= order.entry_price:
                # Gap up through stop - fill at open
                fill_price = bar_open + self.slippage_points
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_time=bar_time,
                    commission=self.commission_per_contract * order.qty,
                    slippage=abs(fill_price - order.entry_price),
                    reason="Stop order filled - gapped through"
                )
            elif bar_high >= order.entry_price:
                fill_price = order.entry_price + self.slippage_points
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_time=bar_time,
                    commission=self.commission_per_contract * order.qty,
                    slippage=self.slippage_points,
                    reason="Stop order filled - stop triggered"
                )
        else:  # SHORT
            # Sell stop - waiting for price to fall to stop
            if bar_open <= order.entry_price:
                # Gap down through stop - fill at open
                fill_price = bar_open - self.slippage_points
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_time=bar_time,
                    commission=self.commission_per_contract * order.qty,
                    slippage=abs(order.entry_price - fill_price),
                    reason="Stop order filled - gapped through"
                )
            elif bar_low <= order.entry_price:
                fill_price = order.entry_price - self.slippage_points
                return FillResult(
                    filled=True,
                    fill_price=fill_price,
                    fill_time=bar_time,
                    commission=self.commission_per_contract * order.qty,
                    slippage=self.slippage_points,
                    reason="Stop order filled - stop triggered"
                )
        
        return FillResult(filled=False, reason="Stop not triggered")
    
    def check_sl_tp(self, trade, bar: pd.Series) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Check if SL or TP is hit for an open trade.
        
        IMPORTANT: If both SL and TP could be hit in same bar,
        we assume SL is hit first (conservative).
        
        Returns:
            (is_closed, exit_price, exit_reason)
        """
        bar_high = bar['high']
        bar_low = bar['low']
        bar_open = bar['open']
        
        if trade.side == Side.LONG:
            # Check SL first (conservative)
            if bar_low <= trade.sl:
                # Could have gapped
                if bar_open <= trade.sl:
                    exit_price = bar_open - self.slippage_points  # Gap through
                else:
                    exit_price = trade.sl
                return (True, exit_price, 'SL')
            
            # Check TP
            if bar_high >= trade.tp:
                if bar_open >= trade.tp:
                    exit_price = bar_open + self.slippage_points  # Gap through (good)
                else:
                    exit_price = trade.tp
                return (True, exit_price, 'TP')
        
        else:  # SHORT
            # Check SL first
            if bar_high >= trade.sl:
                if bar_open >= trade.sl:
                    exit_price = bar_open + self.slippage_points  # Gap through
                else:
                    exit_price = trade.sl
                return (True, exit_price, 'SL')
            
            # Check TP
            if bar_low <= trade.tp:
                if bar_open <= trade.tp:
                    exit_price = bar_open - self.slippage_points  # Gap through (good)
                else:
                    exit_price = trade.tp
                return (True, exit_price, 'TP')
        
        return (False, None, None)
