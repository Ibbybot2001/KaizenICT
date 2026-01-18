"""
Event logging for comprehensive audit trail.

Every signal, order, fill, cancel, invalidation is logged
for post-hoc analysis and lookahead detection.
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd


@dataclass
class Event:
    """Single event in the simulation."""
    timestamp: pd.Timestamp
    event_type: str
    details: Dict[str, Any]
    bar_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'bar_index': self.bar_index,
            'details': self.details
        }


class EventLog:
    """
    Comprehensive event logging for audit.
    
    Event types:
    - SIGNAL: Strategy generated a signal
    - ORDER_PLACED: Order submitted
    - ORDER_FILLED: Order executed
    - ORDER_CANCELLED: Order cancelled
    - ORDER_REJECTED: Order rejected (e.g., SL too small)
    - TRADE_OPENED: Position opened
    - TRADE_UPDATED: MAE/MFE updated
    - TRADE_CLOSED: Position closed
    - SL_HIT: Stop loss triggered
    - TP_HIT: Take profit triggered
    - INVALIDATION: Setup invalidated
    """
    
    def __init__(self):
        self.events: List[Event] = []
        self._current_bar_index: int = 0
    
    def set_bar_index(self, idx: int) -> None:
        """Update current bar index for event context."""
        self._current_bar_index = idx
    
    def log(self, timestamp: pd.Timestamp, event_type: str, 
            details: Dict[str, Any]) -> None:
        """Log an event."""
        event = Event(
            timestamp=timestamp,
            event_type=event_type,
            details=details,
            bar_index=self._current_bar_index
        )
        self.events.append(event)
    
    def log_signal(self, timestamp: pd.Timestamp, signal_type: str,
                   direction: str, metadata: Dict[str, Any]) -> None:
        """Log a strategy signal."""
        self.log(timestamp, 'SIGNAL', {
            'signal_type': signal_type,
            'direction': direction,
            **metadata
        })
    
    def log_order_placed(self, timestamp: pd.Timestamp, order) -> None:
        """Log order placement."""
        self.log(timestamp, 'ORDER_PLACED', {
            'order_id': order.id,
            'side': order.side.value,
            'type': order.order_type.value,
            'entry_price': order.entry_price,
            'sl': order.sl,
            'tp': order.tp,
            'qty': order.qty,
            'sl_distance': order.risk_points,
        })
    
    def log_order_rejected(self, timestamp: pd.Timestamp, 
                           reason: str, order_details: Dict) -> None:
        """Log order rejection."""
        self.log(timestamp, 'ORDER_REJECTED', {
            'reason': reason,
            **order_details
        })
    
    def log_order_filled(self, timestamp: pd.Timestamp, order,
                         fill_price: float, slippage: float) -> None:
        """Log order fill."""
        self.log(timestamp, 'ORDER_FILLED', {
            'order_id': order.id,
            'fill_price': fill_price,
            'slippage': slippage,
        })
    
    def log_trade_opened(self, timestamp: pd.Timestamp, trade) -> None:
        """Log trade opening."""
        self.log(timestamp, 'TRADE_OPENED', trade.to_dict())
    
    def log_trade_closed(self, timestamp: pd.Timestamp, trade) -> None:
        """Log trade closing."""
        self.log(timestamp, 'TRADE_CLOSED', trade.to_dict())
    
    def log_invalidation(self, timestamp: pd.Timestamp, 
                         reason: str, context: Dict[str, Any]) -> None:
        """Log setup invalidation."""
        self.log(timestamp, 'INVALIDATION', {
            'reason': reason,
            **context
        })
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert events to DataFrame."""
        if not self.events:
            return pd.DataFrame()
        
        records = [e.to_dict() for e in self.events]
        return pd.DataFrame(records)
    
    def save_jsonl(self, path: Path) -> None:
        """Save events as JSON lines for easy parsing."""
        with open(path, 'w') as f:
            for event in self.events:
                f.write(json.dumps(event.to_dict()) + '\n')
    
    def get_events_by_type(self, event_type: str) -> List[Event]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def summary(self) -> Dict[str, int]:
        """Get event type counts."""
        counts = {}
        for event in self.events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
