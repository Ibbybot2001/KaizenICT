"""
No-Lookahead Event Engine - The simulation backbone.

NON-NEGOTIABLE RULES:
1. At bar t, only access data[0:t+1] (inclusive of current bar close)
2. Features computed via feature_builder must respect this
3. Orders execute on next bar open (market) or via H/L logic (limit/stop)
4. SL distance >= MIN_SL_POINTS enforced at order placement
5. Every event is logged for audit
"""

from typing import List, Optional, Dict, Any, Callable, Protocol
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ..constants import MIN_SL_POINTS, SLIPPAGE_TICKS, COMMISSION_PER_CONTRACT, CONTRACT_MULTIPLIER
from .trade import Order, Trade, OrderType, OrderStatus, Side, ExitReason, InvalidOrderError
from .fill_simulator import FillSimulator, FillResult
from .event_log import EventLog


class Strategy(Protocol):
    """Protocol defining strategy interface."""
    
    def on_start(self, engine: 'EventEngine') -> None:
        """Called once before simulation starts."""
        ...
    
    def on_bar(self, engine: 'EventEngine', bar_idx: int, bar: pd.Series) -> None:
        """
        Called on each bar.
        
        CRITICAL: At bar_idx, you may ONLY access engine.data.iloc[:bar_idx+1]
        Accessing future bars is CHEATING and will invalidate results.
        """
        ...
    
    def on_end(self, engine: 'EventEngine') -> None:
        """Called once after simulation ends."""
        ...


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    slippage_ticks: int = SLIPPAGE_TICKS
    commission_per_contract: float = COMMISSION_PER_CONTRACT
    min_sl_points: float = MIN_SL_POINTS
    max_open_trades: int = 1  # Maximum concurrent positions
    close_eod: bool = True  # Close positions at end of day
    eod_time: str = "16:00:00"  # End of day time (local)


class EventEngine:
    """
    Bar-by-bar event-driven simulation engine.
    
    CRITICAL: This engine enforces no-lookahead by design:
    1. Strategy.on_bar() receives only current bar index
    2. All data access must be explicit via engine.data.iloc[:bar_idx+1]
    3. Orders placed during bar N execute on bar N+1 or later
    4. SL validation rejects orders with risk < MIN_SL_POINTS
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 strategy: Optional[Strategy] = None,
                 config: Optional[SimulationConfig] = None):
        """
        Initialize the event engine.
        
        Args:
            data: OHLCV DataFrame with datetime index
            strategy: Strategy implementing on_bar()
            config: Simulation configuration
        """
        self.data = data
        self.strategy = strategy
        self.config = config or SimulationConfig()
        
        # Order and trade management
        self.pending_orders: List[Order] = []
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        
        # Simulation state
        self.current_bar_idx: int = 0
        self.current_bar: Optional[pd.Series] = None
        self.equity: float = 0.0
        self.total_commission: float = 0.0
        
        # Components
        self.fill_simulator = FillSimulator(
            slippage_ticks=self.config.slippage_ticks,
            commission_per_contract=self.config.commission_per_contract
        )
        self.event_log = EventLog()
        
        # Equity tracking
        self.equity_curve: List[float] = []
    
    # ==========================================================================
    # MAIN SIMULATION LOOP
    # ==========================================================================
    
    def run(self) -> Dict[str, Any]:
        """
        Run the simulation bar-by-bar.
        
        This is the main event loop. For each bar:
        1. Process pending orders (fills)
        2. Check open positions for SL/TP
        3. Update MAE/MFE for open trades
        4. Call strategy.on_bar()
        5. Track equity
        
        Returns:
            Dictionary with simulation results
        """
        if self.strategy:
            self.strategy.on_start(self)
        
        for idx in range(len(self.data)):
            self.current_bar_idx = idx
            self.current_bar = self.data.iloc[idx]
            self.event_log.set_bar_index(idx)
            
            bar = self.current_bar
            bar_time = bar.name
            
            # 1. Process pending orders (check for fills)
            self._process_pending_orders(bar)
            
            # 2. Check open trades for SL/TP
            self._check_open_trades(bar)
            
            # 3. Update MAE/MFE for remaining open trades
            for trade in self.open_trades:
                # Use bar high/low to update excursions
                trade.update_excursions(bar['high'])
                trade.update_excursions(bar['low'])
                trade.bars_held += 1
            
            # 4. End-of-day handling
            if self.config.close_eod:
                self._handle_eod(bar)
            
            # 5. Strategy callback
            if self.strategy:
                self.strategy.on_bar(self, idx, bar)
            
            # 6. Track equity
            unrealized = sum(self._unrealized_pnl(t, bar['close']) 
                            for t in self.open_trades)
            realized = sum(t.pnl for t in self.trades if not t.is_open)
            self.equity = realized + unrealized - self.total_commission
            self.equity_curve.append(self.equity)
        
        if self.strategy:
            self.strategy.on_end(self)
        
        return self._compile_results()
    
    # ==========================================================================
    # ORDER MANAGEMENT
    # ==========================================================================
    
    def place_order(self, 
                    side: Side,
                    order_type: OrderType,
                    entry_price: float,
                    sl: float,
                    tp: float,
                    qty: int = 1,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[Order]:
        """
        Place an order.
        
        CRITICAL: Order is validated for MIN_SL_POINTS.
        If SL distance < MIN_SL_POINTS, order is REJECTED.
        
        Args:
            side: LONG or SHORT
            order_type: MARKET, LIMIT, or STOP
            entry_price: Entry price for limit/stop orders
            sl: Stop loss price
            tp: Take profit price
            qty: Number of contracts
            metadata: Optional strategy metadata
            
        Returns:
            Order object if accepted, None if rejected
        """
        # Check max open trades
        if len(self.open_trades) >= self.config.max_open_trades:
            self.event_log.log_order_rejected(
                self.current_bar.name,
                f"Max open trades ({self.config.max_open_trades}) reached",
                {'side': side.value, 'entry_price': entry_price}
            )
            return None
        
        try:
            order = Order(
                side=side,
                order_type=order_type,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                qty=qty,
                created_time=self.current_bar.name,
                metadata=metadata or {}
            )
            
            self.pending_orders.append(order)
            self.event_log.log_order_placed(self.current_bar.name, order)
            return order
            
        except InvalidOrderError as e:
            # SL too small or other validation failure
            self.event_log.log_order_rejected(
                self.current_bar.name,
                str(e),
                {'side': side.value, 'entry_price': entry_price, 'sl': sl, 'tp': tp}
            )
            return None
    
    def place_market_order(self, side: Side, sl: float, tp: float,
                          qty: int = 1, metadata: Optional[Dict] = None) -> Optional[Order]:
        """Place a market order. Entry price is set to current close."""
        return self.place_order(
            side=side,
            order_type=OrderType.MARKET,
            entry_price=self.current_bar['close'],  # Will fill at next open
            sl=sl,
            tp=tp,
            qty=qty,
            metadata=metadata
        )
    
    def place_limit_order(self, side: Side, entry_price: float, sl: float, tp: float,
                          qty: int = 1, metadata: Optional[Dict] = None) -> Optional[Order]:
        """Place a limit order."""
        return self.place_order(
            side=side,
            order_type=OrderType.LIMIT,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            qty=qty,
            metadata=metadata
        )
    
    def cancel_all_orders(self) -> int:
        """Cancel all pending orders. Returns count cancelled."""
        count = len(self.pending_orders)
        for order in self.pending_orders:
            order.status = OrderStatus.CANCELLED
            self.event_log.log(
                self.current_bar.name,
                'ORDER_CANCELLED',
                {'order_id': order.id}
            )
        self.pending_orders.clear()
        return count
    
    def close_all_trades(self, reason: ExitReason = ExitReason.MANUAL) -> int:
        """Close all open trades at current close. Returns count closed."""
        count = 0
        for trade in self.open_trades[:]:  # Copy to allow modification
            self._close_trade(trade, self.current_bar['close'], reason)
            count += 1
        return count
    
    # ==========================================================================
    # INTERNAL PROCESSING
    # ==========================================================================
    
    def _process_pending_orders(self, bar: pd.Series) -> None:
        """Process pending orders - check for fills."""
        filled_orders = []
        
        for order in self.pending_orders:
            result = self.fill_simulator.attempt_fill(order, bar)
            
            if result.filled:
                order.status = OrderStatus.FILLED
                order.filled_time = result.fill_time
                order.filled_price = result.fill_price
                
                self.event_log.log_order_filled(
                    bar.name, order, result.fill_price, result.slippage
                )
                
                # Create trade from filled order
                trade = Trade(
                    id=order.id,
                    entry_time=result.fill_time,
                    entry_price=result.fill_price,
                    side=order.side,
                    qty=order.qty,
                    sl=order.sl,
                    tp=order.tp,
                    signal_metadata=order.metadata
                )
                
                self.trades.append(trade)
                self.open_trades.append(trade)
                self.total_commission += result.commission
                
                self.event_log.log_trade_opened(bar.name, trade)
                filled_orders.append(order)
        
        # Remove filled orders
        for order in filled_orders:
            self.pending_orders.remove(order)
    
    def _check_open_trades(self, bar: pd.Series) -> None:
        """Check open trades for SL/TP hits."""
        closed_trades = []
        
        for trade in self.open_trades:
            is_closed, exit_price, exit_reason = self.fill_simulator.check_sl_tp(trade, bar)
            
            if is_closed:
                reason = ExitReason.SL if exit_reason == 'SL' else ExitReason.TP
                self._close_trade(trade, exit_price, reason)
                closed_trades.append(trade)
        
        # Remove closed trades from open list
        for trade in closed_trades:
            self.open_trades.remove(trade)
    
    def _close_trade(self, trade: Trade, exit_price: float, 
                     reason: ExitReason) -> None:
        """Close a trade."""
        trade.close(exit_price, self.current_bar.name, reason)
        self.event_log.log_trade_closed(self.current_bar.name, trade)
    
    def _handle_eod(self, bar: pd.Series) -> None:
        """Handle end-of-day closures."""
        bar_time = bar.name
        if hasattr(bar_time, 'time'):
            time_str = bar_time.time().strftime("%H:%M:%S")
            if time_str >= self.config.eod_time:
                # Check if this is the last bar of the day
                next_idx = self.current_bar_idx + 1
                if next_idx < len(self.data):
                    next_bar_time = self.data.index[next_idx]
                    if next_bar_time.date() != bar_time.date():
                        # Next bar is a new day - close all
                        for trade in self.open_trades[:]:
                            self._close_trade(trade, bar['close'], ExitReason.EOD)
                            self.open_trades.remove(trade)
    
    def _unrealized_pnl(self, trade: Trade, current_price: float) -> float:
        """Calculate unrealized PnL for a trade."""
        if trade.side == Side.LONG:
            return (current_price - trade.entry_price) * trade.qty
        else:
            return (trade.entry_price - current_price) * trade.qty
    
    # ==========================================================================
    # DATA ACCESS - STRICT TIME DISCIPLINE
    # ==========================================================================
    
    def get_historical_data(self, lookback: int) -> pd.DataFrame:
        """
        Get historical data up to current bar.
        
        SAFE ACCESS: Returns data[max(0, current-lookback):current+1]
        Cannot access future bars by design.
        """
        start = max(0, self.current_bar_idx - lookback + 1)
        end = self.current_bar_idx + 1
        return self.data.iloc[start:end].copy()
    
    def get_past_bars(self, n: int) -> pd.DataFrame:
        """Get the last N bars including current."""
        return self.get_historical_data(n)
    
    @property
    def has_open_position(self) -> bool:
        """Check if there's an open position."""
        return len(self.open_trades) > 0
    
    @property
    def position_side(self) -> Optional[Side]:
        """Get current position side, or None if flat."""
        if not self.open_trades:
            return None
        return self.open_trades[0].side
    
    # ==========================================================================
    # RESULTS COMPILATION
    # ==========================================================================
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile simulation results."""
        closed_trades = [t for t in self.trades if not t.is_open]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'net_pnl': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_r_multiple': 0.0,
                'equity_curve': self.equity_curve,
                'event_log': self.event_log,
            }
        
        pnls = [t.pnl for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        
        return {
            'total_trades': len(closed_trades),
            'winners': len(wins),
            'losers': len(losses),
            'net_pnl': sum(pnls),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'win_rate': len(wins) / len(closed_trades) if closed_trades else 0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_pnl': np.mean(pnls),
            'avg_winner': np.mean(wins) if wins else 0,
            'avg_loser': np.mean(losses) if losses else 0,
            'avg_r_multiple': np.mean([t.r_multiple for t in closed_trades]),
            'max_r': max(t.r_multiple for t in closed_trades),
            'min_r': min(t.r_multiple for t in closed_trades),
            'total_commission': self.total_commission,
            'equity_curve': self.equity_curve,
            'trades': [t.to_dict() for t in closed_trades],
            'event_log': self.event_log,
        }
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])
