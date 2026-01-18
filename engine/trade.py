"""
Order and Trade dataclasses for the no-lookahead event engine.

HARD RULE: SL distance must be >= MIN_SL_POINTS (10 points).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import pandas as pd
import uuid

from ..constants import MIN_SL_POINTS


class OrderType(str, Enum):
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'


class OrderStatus(str, Enum):
    PENDING = 'PENDING'
    FILLED = 'FILLED'
    CANCELLED = 'CANCELLED'
    REJECTED = 'REJECTED'


class Side(str, Enum):
    LONG = 'LONG'
    SHORT = 'SHORT'


class InvalidOrderError(Exception):
    """Raised when order validation fails."""
    pass


@dataclass
class Order:
    """
    Order with strict validation.
    
    CRITICAL: SL distance must be >= MIN_SL_POINTS.
    Orders with smaller SL are REJECTED.
    """
    side: Side
    order_type: OrderType
    entry_price: float
    sl: float
    tp: float
    qty: int = 1
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_time: Optional[pd.Timestamp] = None
    filled_time: Optional[pd.Timestamp] = None
    filled_price: Optional[float] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order on creation."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate order constraints.
        
        Raises:
            InvalidOrderError: If SL distance < MIN_SL_POINTS
        """
        sl_distance = abs(self.entry_price - self.sl)
        
        if sl_distance < MIN_SL_POINTS:
            raise InvalidOrderError(
                f"SL distance {sl_distance:.2f} < minimum {MIN_SL_POINTS} points. "
                f"Entry: {self.entry_price}, SL: {self.sl}"
            )
        
        # Validate SL/TP direction based on side
        if self.side == Side.LONG:
            if self.sl >= self.entry_price:
                raise InvalidOrderError(
                    f"LONG order SL ({self.sl}) must be below entry ({self.entry_price})"
                )
            if self.tp <= self.entry_price:
                raise InvalidOrderError(
                    f"LONG order TP ({self.tp}) must be above entry ({self.entry_price})"
                )
        else:  # SHORT
            if self.sl <= self.entry_price:
                raise InvalidOrderError(
                    f"SHORT order SL ({self.sl}) must be above entry ({self.entry_price})"
                )
            if self.tp >= self.entry_price:
                raise InvalidOrderError(
                    f"SHORT order TP ({self.tp}) must be below entry ({self.entry_price})"
                )
    
    @property
    def risk_points(self) -> float:
        """Risk in points (SL distance)."""
        return abs(self.entry_price - self.sl)
    
    @property
    def reward_points(self) -> float:
        """Reward in points (TP distance)."""
        return abs(self.tp - self.entry_price)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Reward to risk ratio."""
        return self.reward_points / self.risk_points if self.risk_points > 0 else 0


class ExitReason(str, Enum):
    SL = 'SL'
    TP = 'TP'
    SIGNAL = 'SIGNAL'  # Strategy signal exit
    EOD = 'EOD'  # End of day
    MANUAL = 'MANUAL'


@dataclass
class Trade:
    """
    Complete trade record with MAE/MFE tracking.
    
    Created when an Order is filled.
    """
    id: str
    entry_time: pd.Timestamp
    entry_price: float
    side: Side
    qty: int
    sl: float
    tp: float
    
    # Exit fields (populated on close)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    
    # Performance metrics (updated during trade)
    pnl: float = 0.0
    mae: float = 0.0  # Maximum Adverse Excursion (worst unrealized loss)
    mfe: float = 0.0  # Maximum Favorable Excursion (best unrealized profit)
    
    # Tracking
    bars_held: int = 0
    high_water_mark: float = 0.0  # Best price reached
    low_water_mark: float = float('inf')  # Worst price reached
    
    # Strategy context
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_open: bool = True
    
    @property
    def risk_points(self) -> float:
        """Initial risk in points."""
        return abs(self.entry_price - self.sl)
    
    @property
    def r_multiple(self) -> float:
        """
        R-multiple: PnL expressed as multiple of initial risk.
        Positive = profit, Negative = loss.
        """
        if self.risk_points == 0:
            return 0.0
        return self.pnl / self.risk_points
    
    def update_excursions(self, current_price: float) -> None:
        """
        Update MAE/MFE based on current price.
        
        Call this on each bar while trade is open.
        """
        if self.side == Side.LONG:
            unrealized_pnl = current_price - self.entry_price
        else:
            unrealized_pnl = self.entry_price - current_price
        
        # Update MFE (best unrealized)
        if unrealized_pnl > self.mfe:
            self.mfe = unrealized_pnl
        
        # Update MAE (worst unrealized - stored as positive number)
        if unrealized_pnl < -self.mae:
            self.mae = abs(unrealized_pnl)
        
        # Track high/low water marks
        self.high_water_mark = max(self.high_water_mark, current_price)
        if current_price < self.low_water_mark:
            self.low_water_mark = current_price
    
    def close(self, exit_price: float, exit_time: pd.Timestamp, 
              exit_reason: ExitReason) -> None:
        """
        Close the trade and calculate final PnL.
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.is_open = False
        
        if self.side == Side.LONG:
            self.pnl = exit_price - self.entry_price
        else:
            self.pnl = self.entry_price - exit_price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/export."""
        return {
            'id': self.id,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'entry_price': self.entry_price,
            'side': self.side.value,
            'qty': self.qty,
            'sl': self.sl,
            'tp': self.tp,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'pnl': self.pnl,
            'mae': self.mae,
            'mfe': self.mfe,
            'r_multiple': self.r_multiple,
            'bars_held': self.bars_held,
            'risk_points': self.risk_points,
            'metadata': self.signal_metadata,
        }
