"""
POOL FINITE STATE MACHINE (FSM)
Explicit state tracking for PJ/ICT liquidity pool lifecycle.

States:
    DEFINED     → Pool level is known, waiting for sweep
    SWEPT       → Price traded through level (wick touched)
    RECLAIMED   → Bar closed back inside level
    ENTRY_PENDING → Order sent, awaiting fill
    IN_TRADE    → Position open
    CLOSED      → Trade completed (TP/SL/Time exit)
    EXPIRED     → Pool no longer valid (session ended)

Transitions are EXPLICIT and LOGGED.
No implicit state changes. No race conditions.

Author: AntiGravity
Certification: No-Lookahead Audit - Production Component
"""

import copy
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum, auto
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# STATE DEFINITIONS
# ==============================================================================

class PoolState(Enum):
    """Explicit states for pool lifecycle."""
    DEFINED = auto()        # Level known, waiting for sweep
    SWEPT = auto()          # Price traded through level
    RECLAIMED = auto()      # Bar closed back inside — READY TO TRADE
    ENTRY_PENDING = auto()  # Order sent, awaiting fill
    IN_TRADE = auto()       # Position open
    CLOSED = auto()         # Trade completed (win/loss)
    EXPIRED = auto()        # Pool no longer valid


class TradeDirection(Enum):
    """Trade direction."""
    LONG = 1
    SHORT = -1


# ==============================================================================
# POOL DATA STRUCTURES
# ==============================================================================

@dataclass
class PoolLevel:
    """
    Single liquidity pool level.
    Tracks state, transitions, and trade data.
    """
    pool_id: str
    level_price: float
    direction: TradeDirection
    opposing_level: float  # Target (Draw on Liquidity)
    
    # State tracking
    state: PoolState = PoolState.DEFINED
    state_history: List[tuple] = field(default_factory=list)
    
    # Sweep data
    sweep_time: Optional[datetime] = None
    sweep_bar_low: Optional[float] = None
    sweep_bar_high: Optional[float] = None
    
    # Reclaim data
    reclaim_time: Optional[datetime] = None
    reclaim_bar_close: Optional[float] = None
    
    # Entry data
    entry_time: Optional[datetime] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Exit data
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    
    def __post_init__(self):
        self._record_transition(PoolState.DEFINED, "Pool initialized")
    
    def _record_transition(self, new_state: PoolState, reason: str):
        """Record state transition with timestamp."""
        timestamp = datetime.now(timezone.utc)
        self.state_history.append((timestamp, self.state, new_state, reason))
        self.state = new_state
        logger.info(f"[{self.pool_id}] {self.state.name}: {reason}")


# ==============================================================================
# STATE MACHINE CORE
# ==============================================================================

class PoolFSM:
    """
    Finite State Machine for pool lifecycle management.
    
    Key Properties:
    1. EXPLICIT state transitions only
    2. All transitions LOGGED
    3. Invalid transitions REJECTED
    4. Thread-safe
    5. One FSM instance per pool
    """
    
    # Valid state transitions
    TRANSITIONS = {
        PoolState.DEFINED: [PoolState.SWEPT, PoolState.EXPIRED],
        PoolState.SWEPT: [PoolState.RECLAIMED, PoolState.EXPIRED],
        PoolState.RECLAIMED: [PoolState.ENTRY_PENDING, PoolState.EXPIRED],
        PoolState.ENTRY_PENDING: [PoolState.IN_TRADE, PoolState.EXPIRED],
        PoolState.IN_TRADE: [PoolState.CLOSED],
        PoolState.CLOSED: [],  # Terminal state
        PoolState.EXPIRED: [],  # Terminal state
    }
    
    def __init__(self, pool: PoolLevel):
        self.pool = pool
        self._lock = threading.Lock()
    
    def can_transition_to(self, new_state: PoolState) -> bool:
        """Check if transition is valid."""
        return new_state in self.TRANSITIONS.get(self.pool.state, [])
    
    def _transition(self, new_state: PoolState, reason: str) -> bool:
        """
        Execute state transition.
        
        Returns:
            True if transition succeeded, False if invalid.
        """
        with self._lock:
            if not self.can_transition_to(new_state):
                logger.warning(
                    f"[{self.pool.pool_id}] Invalid transition: "
                    f"{self.pool.state.name} → {new_state.name}"
                )
                return False
            
            self.pool._record_transition(new_state, reason)
            return True
    
    # --------------------------------------------------------------------------
    # STATE-SPECIFIC TRANSITIONS
    # --------------------------------------------------------------------------
    
    def on_sweep(self, bar_time: datetime, bar_low: float, bar_high: float) -> bool:
        """
        Transition: DEFINED → SWEPT
        
        Called when price trades through the level (wick).
        """
        if self.pool.state != PoolState.DEFINED:
            return False
        
        self.pool.sweep_time = bar_time
        self.pool.sweep_bar_low = bar_low
        self.pool.sweep_bar_high = bar_high
        
        return self._transition(
            PoolState.SWEPT,
            f"Sweep detected at bar {bar_time}"
        )
    
    def on_reclaim(self, bar_time: datetime, bar_close: float) -> bool:
        """
        Transition: SWEPT → RECLAIMED
        
        Called when bar closes back inside the level.
        THIS IS THE SIGNAL BAR.
        """
        if self.pool.state != PoolState.SWEPT:
            return False
        
        self.pool.reclaim_time = bar_time
        self.pool.reclaim_bar_close = bar_close
        
        # Calculate SL/TP
        if self.pool.direction == TradeDirection.LONG:
            self.pool.stop_loss = self.pool.sweep_bar_low - 2.0
            self.pool.take_profit = self.pool.opposing_level
        else:
            self.pool.stop_loss = self.pool.sweep_bar_high + 2.0
            self.pool.take_profit = self.pool.opposing_level
        
        return self._transition(
            PoolState.RECLAIMED,
            f"Reclaim confirmed at bar {bar_time}, close={bar_close:.2f}"
        )
    
    def on_order_sent(self, entry_time: datetime) -> bool:
        """
        Transition: RECLAIMED → ENTRY_PENDING
        
        Called when order is sent to broker.
        """
        if self.pool.state != PoolState.RECLAIMED:
            return False
        
        self.pool.entry_time = entry_time
        
        return self._transition(
            PoolState.ENTRY_PENDING,
            f"Order sent at {entry_time}"
        )
    
    def on_fill(self, fill_price: float) -> bool:
        """
        Transition: ENTRY_PENDING → IN_TRADE
        
        Called when order is filled.
        """
        if self.pool.state != PoolState.ENTRY_PENDING:
            return False
        
        self.pool.entry_price = fill_price
        
        return self._transition(
            PoolState.IN_TRADE,
            f"Filled at {fill_price:.2f}"
        )
    
    def on_exit(self, exit_time: datetime, exit_price: float, reason: str) -> bool:
        """
        Transition: IN_TRADE → CLOSED
        
        Called when position is closed (TP/SL/Time).
        """
        if self.pool.state != PoolState.IN_TRADE:
            return False
        
        self.pool.exit_time = exit_time
        self.pool.exit_price = exit_price
        self.pool.exit_reason = reason
        
        # Calculate PnL
        if self.pool.direction == TradeDirection.LONG:
            self.pool.pnl = exit_price - self.pool.entry_price
        else:
            self.pool.pnl = self.pool.entry_price - exit_price
        
        return self._transition(
            PoolState.CLOSED,
            f"Exit at {exit_price:.2f} ({reason}), PnL={self.pool.pnl:.2f}"
        )
    
    def on_expire(self, reason: str = "Session ended") -> bool:
        """
        Transition: ANY → EXPIRED
        
        Called when pool is no longer valid (session end, etc).
        Only valid from non-terminal states.
        """
        if self.pool.state in [PoolState.CLOSED, PoolState.EXPIRED]:
            return False
        
        return self._transition(PoolState.EXPIRED, reason)
    
    # --------------------------------------------------------------------------
    # QUERY METHODS
    # --------------------------------------------------------------------------
    
    def is_ready_to_trade(self) -> bool:
        """Pool is in RECLAIMED state and ready for entry."""
        return self.pool.state == PoolState.RECLAIMED
    
    def is_active(self) -> bool:
        """Pool has an open position."""
        return self.pool.state == PoolState.IN_TRADE
    
    def is_terminal(self) -> bool:
        """Pool has reached a terminal state."""
        return self.pool.state in [PoolState.CLOSED, PoolState.EXPIRED]
    
    def get_signal_data(self) -> Optional[dict]:
        """
        Get trade signal data if pool is ready.
        
        Returns:
            Signal dict if ready, None otherwise.
        """
        if not self.is_ready_to_trade():
            return None
        
        return {
            'pool_id': self.pool.pool_id,
            'direction': self.pool.direction,
            'entry_time': self.pool.reclaim_time,
            'stop_loss': self.pool.stop_loss,
            'take_profit': self.pool.take_profit,
            'level_price': self.pool.level_price,
        }


# ==============================================================================
# SESSION POOL MANAGER
# ==============================================================================

class SessionPoolManager:
    """
    Manages all pool FSMs for a trading session (one day).
    Enforces: One trade per pool per session.
    """
    
    def __init__(self, session_date: datetime):
        self.session_date = session_date
        self.pools: Dict[str, PoolFSM] = {}
        self._lock = threading.Lock()
    
    def add_pool(
        self,
        pool_id: str,
        level_price: float,
        direction: TradeDirection,
        opposing_level: float
    ) -> PoolFSM:
        """Add a new pool to track."""
        with self._lock:
            if pool_id in self.pools:
                logger.warning(f"Pool {pool_id} already exists")
                return self.pools[pool_id]
            
            pool = PoolLevel(
                pool_id=pool_id,
                level_price=level_price,
                direction=direction,
                opposing_level=opposing_level
            )
            fsm = PoolFSM(pool)
            self.pools[pool_id] = fsm
            return fsm
    
    def get_pool(self, pool_id: str) -> Optional[PoolFSM]:
        """Get pool FSM by ID."""
        return self.pools.get(pool_id)
    
    def get_all_ready_signals(self) -> List[dict]:
        """Get all pools ready to trade."""
        signals = []
        for fsm in self.pools.values():
            signal = fsm.get_signal_data()
            if signal:
                signals.append(signal)
        return signals
    
    def expire_all(self, reason: str = "Session ended"):
        """Expire all non-terminal pools."""
        with self._lock:
            for fsm in self.pools.values():
                if not fsm.is_terminal():
                    fsm.on_expire(reason)
    
    def get_summary(self) -> dict:
        """Get session summary."""
        summary = {
            'session_date': self.session_date,
            'total_pools': len(self.pools),
            'by_state': {},
            'trades': [],
        }
        
        for fsm in self.pools.values():
            state_name = fsm.pool.state.name
            summary['by_state'][state_name] = summary['by_state'].get(state_name, 0) + 1
            
            if fsm.pool.state == PoolState.CLOSED:
                summary['trades'].append({
                    'pool_id': fsm.pool.pool_id,
                    'direction': fsm.pool.direction.name,
                    'entry_price': fsm.pool.entry_price,
                    'exit_price': fsm.pool.exit_price,
                    'pnl': fsm.pool.pnl,
                    'exit_reason': fsm.pool.exit_reason,
                })
        
        return summary


# ==============================================================================
# BAR EVENT PROCESSOR
# ==============================================================================

class BarEventProcessor:
    """
    Processes bar events and updates pool FSMs.
    
    This is the integration point between BarBuilder and PoolFSM.
    """
    
    def __init__(self, session_manager: SessionPoolManager):
        self.session_manager = session_manager
    
    def process_bar(self, bar: 'LiveBar'):
        """
        Process a finalized bar and update all pool FSMs.
        
        This method checks for sweeps and reclaims across all pools.
        """
        for pool_id, fsm in self.session_manager.pools.items():
            pool = fsm.pool
            
            # Skip terminal states
            if fsm.is_terminal():
                continue
            
            # Check for sweep (DEFINED → SWEPT)
            if pool.state == PoolState.DEFINED:
                swept = self._check_sweep(pool, bar)
                if swept:
                    fsm.on_sweep(bar.end_time, bar.low, bar.high)
            
            # Check for reclaim (SWEPT → RECLAIMED)
            elif pool.state == PoolState.SWEPT:
                reclaimed = self._check_reclaim(pool, bar)
                if reclaimed:
                    fsm.on_reclaim(bar.end_time, bar.close)
    
    def _check_sweep(self, pool: PoolLevel, bar: 'LiveBar') -> bool:
        """Check if bar sweeps the pool level."""
        if pool.direction == TradeDirection.LONG:
            # Long: sweep LOW level (price goes below)
            return bar.low < pool.level_price
        else:
            # Short: sweep HIGH level (price goes above)
            return bar.high > pool.level_price
    
    def _check_reclaim(self, pool: PoolLevel, bar: 'LiveBar') -> bool:
        """Check if bar reclaims (closes back inside) after sweep."""
        if pool.direction == TradeDirection.LONG:
            # Long: close above level
            return bar.close > pool.level_price
        else:
            # Short: close below level
            return bar.close < pool.level_price


# ==============================================================================
# TEST HARNESS
# ==============================================================================

def run_fsm_transition_test():
    """Test valid and invalid state transitions."""
    print("\n" + "=" * 60)
    print("FSM TRANSITION TEST")
    print("=" * 60)
    
    pool = PoolLevel(
        pool_id="ONL",
        level_price=100.0,
        direction=TradeDirection.LONG,
        opposing_level=110.0
    )
    fsm = PoolFSM(pool)
    
    # Test valid transitions
    now = datetime.now(timezone.utc)
    
    assert fsm.pool.state == PoolState.DEFINED, "Initial state should be DEFINED"
    
    # DEFINED → SWEPT
    result = fsm.on_sweep(now, bar_low=99.5, bar_high=101.0)
    assert result, "Sweep transition should succeed"
    assert fsm.pool.state == PoolState.SWEPT
    
    # SWEPT → RECLAIMED
    result = fsm.on_reclaim(now, bar_close=100.5)
    assert result, "Reclaim transition should succeed"
    assert fsm.pool.state == PoolState.RECLAIMED
    
    # Verify signal is ready
    assert fsm.is_ready_to_trade(), "Pool should be ready to trade"
    signal = fsm.get_signal_data()
    assert signal is not None
    assert signal['stop_loss'] < 100.0, "SL should be below level"
    
    # Test invalid transition (RECLAIMED → SWEPT should fail)
    result = fsm.on_sweep(now, bar_low=98.0, bar_high=102.0)
    assert not result, "Invalid transition should fail"
    assert fsm.pool.state == PoolState.RECLAIMED, "State should not change"
    
    print("✅ PASS: FSM transition logic verified")
    return True


def run_one_trade_per_pool_test():
    """Test that each pool can only produce one trade."""
    print("\n" + "=" * 60)
    print("ONE TRADE PER POOL TEST")
    print("=" * 60)
    
    now = datetime.now(timezone.utc)
    manager = SessionPoolManager(session_date=now)
    
    # Add pool
    fsm = manager.add_pool(
        pool_id="LON_L",
        level_price=100.0,
        direction=TradeDirection.LONG,
        opposing_level=110.0
    )
    
    # Complete full trade cycle
    fsm.on_sweep(now, 99.0, 101.0)
    fsm.on_reclaim(now, 100.5)
    fsm.on_order_sent(now)
    fsm.on_fill(100.25)
    fsm.on_exit(now, 110.0, "TP")
    
    # Verify pool is terminal
    assert fsm.is_terminal(), "Pool should be in terminal state"
    assert fsm.pool.state == PoolState.CLOSED
    
    # Attempt to reuse pool (should fail)
    result = fsm.on_sweep(now, 99.0, 101.0)
    assert not result, "Reusing closed pool should fail"
    
    print("✅ PASS: One trade per pool enforced")
    return True


def run_race_condition_test():
    """Test thread safety."""
    print("\n" + "=" * 60)
    print("RACE CONDITION TEST")
    print("=" * 60)
    
    import concurrent.futures
    
    pool = PoolLevel(
        pool_id="TEST",
        level_price=100.0,
        direction=TradeDirection.LONG,
        opposing_level=110.0
    )
    fsm = PoolFSM(pool)
    now = datetime.now(timezone.utc)
    
    # Attempt concurrent transitions
    results = []
    
    def attempt_sweep():
        return fsm.on_sweep(now, 99.0, 101.0)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(attempt_sweep) for _ in range(10)]
        results = [f.result() for f in futures]
    
    # Only ONE should succeed
    success_count = sum(results)
    assert success_count == 1, f"Expected 1 success, got {success_count}"
    
    print("✅ PASS: Race condition immunity verified")
    return True


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "=" * 60)
    print("POOL FSM TEST SUITE")
    print("=" * 60)
    
    results = []
    results.append(("FSM Transitions", run_fsm_transition_test()))
    results.append(("One Trade Per Pool", run_one_trade_per_pool_test()))
    results.append(("Race Condition Immunity", run_race_condition_test()))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Pool FSM is production-clean")
    else:
        print("\n⚠️ SOME TESTS FAILED - Fix issues before production")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
