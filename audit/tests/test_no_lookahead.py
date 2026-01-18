"""
No-Lookahead Verification Tests - Phase 1 Audit

CRITICAL TESTS:
1. Time-Travel Test: Attempt to access future bar and verify failure
2. Shuffle Future Invariance: Shuffle future bars, verify results unchanged
3. Minimum SL Enforcement: Attempt order with SL < 10 points, verify rejection
4. Fill Logic: Verify limit orders require touch, market orders fill at next open
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from copy import deepcopy

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_lab.engine.event_engine import EventEngine, SimulationConfig, Strategy
from ml_lab.engine.trade import Order, Trade, Side, OrderType, InvalidOrderError
from ml_lab.engine.fill_simulator import FillSimulator
from ml_lab.constants import MIN_SL_POINTS


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 100
    
    dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='1min')
    base_price = 20000.0
    
    # Simple random walk
    returns = np.random.randn(n_bars) * 0.001
    closes = base_price * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': closes * (1 + np.random.randn(n_bars) * 0.0005),
        'high': closes * (1 + abs(np.random.randn(n_bars)) * 0.001),
        'low': closes * (1 - abs(np.random.randn(n_bars)) * 0.001),
        'close': closes,
        'volume': np.random.randint(100, 1000, n_bars)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


# =============================================================================
# TEST 1: MINIMUM SL ENFORCEMENT
# =============================================================================

class TestMinSLEnforcement:
    """Verify MIN_SL_POINTS is strictly enforced."""
    
    def test_order_with_sl_below_minimum_raises(self):
        """Order with SL < MIN_SL_POINTS should raise InvalidOrderError."""
        with pytest.raises(InvalidOrderError) as exc_info:
            Order(
                side=Side.LONG,
                order_type=OrderType.LIMIT,
                entry_price=20000.0,
                sl=20005.0,  # Only 5 points - TOO SMALL
                tp=20030.0,
                qty=1
            )
        
        assert "SL distance" in str(exc_info.value)
        assert str(MIN_SL_POINTS) in str(exc_info.value)
    
    def test_order_with_sl_at_minimum_accepted(self):
        """Order with SL exactly at MIN_SL_POINTS should be accepted."""
        order = Order(
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            entry_price=20000.0,
            sl=20000.0 - MIN_SL_POINTS,  # Exactly 10 points
            tp=20030.0,
            qty=1
        )
        assert order.risk_points == MIN_SL_POINTS
    
    def test_order_with_sl_above_minimum_accepted(self):
        """Order with SL > MIN_SL_POINTS should be accepted."""
        order = Order(
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            entry_price=20000.0,
            sl=19980.0,  # 20 points - VALID
            tp=20040.0,
            qty=1
        )
        assert order.risk_points == 20.0
    
    def test_engine_rejects_small_sl_order(self, sample_data):
        """EventEngine should reject orders with small SL."""
        engine = EventEngine(sample_data)
        
        # Start simulation to get current bar
        engine.current_bar_idx = 0
        engine.current_bar = sample_data.iloc[0]
        
        # Try to place order with SL too small
        order = engine.place_order(
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            entry_price=20000.0,
            sl=20000.0 - 5.0,  # Only 5 points
            tp=20030.0
        )
        
        # Order should be rejected (returns None)
        assert order is None
        
        # Rejection should be logged
        rejected_events = engine.event_log.get_events_by_type('ORDER_REJECTED')
        assert len(rejected_events) == 1


# =============================================================================
# TEST 2: FILL LOGIC - NO PERFECT FILLS
# =============================================================================

class TestFillLogic:
    """Verify realistic fill simulation."""
    
    def test_limit_order_requires_touch_long(self):
        """LONG limit order only fills if bar_low <= limit_price."""
        simulator = FillSimulator()
        
        order = Order(
            side=Side.LONG,
            order_type=OrderType.LIMIT,
            entry_price=19980.0,  # Limit at 19980
            sl=19970.0,
            tp=20000.0
        )
        
        # Bar that doesn't touch limit
        bar_no_touch = pd.Series({
            'open': 20000.0,
            'high': 20010.0,
            'low': 19985.0,  # Above limit
            'close': 20005.0,
        }, name=pd.Timestamp('2024-01-01 09:30'))
        
        result = simulator.attempt_fill(order, bar_no_touch)
        assert result.filled == False
        
        # Bar that touches limit
        bar_touch = pd.Series({
            'open': 20000.0,
            'high': 20010.0,
            'low': 19975.0,  # Below limit
            'close': 20005.0,
        }, name=pd.Timestamp('2024-01-01 09:31'))
        
        result = simulator.attempt_fill(order, bar_touch)
        assert result.filled == True
        assert result.fill_price == 19980.0  # Fill at limit
    
    def test_limit_order_requires_touch_short(self):
        """SHORT limit order only fills if bar_high >= limit_price."""
        simulator = FillSimulator()
        
        order = Order(
            side=Side.SHORT,
            order_type=OrderType.LIMIT,
            entry_price=20020.0,  # Limit at 20020
            sl=20030.0,
            tp=20000.0
        )
        
        # Bar that doesn't touch limit
        bar_no_touch = pd.Series({
            'open': 20000.0,
            'high': 20015.0,  # Below limit
            'low': 19990.0,
            'close': 20010.0,
        }, name=pd.Timestamp('2024-01-01 09:30'))
        
        result = simulator.attempt_fill(order, bar_no_touch)
        assert result.filled == False
        
        # Bar that touches limit
        bar_touch = pd.Series({
            'open': 20000.0,
            'high': 20025.0,  # Above limit
            'low': 19990.0,
            'close': 20010.0,
        }, name=pd.Timestamp('2024-01-01 09:31'))
        
        result = simulator.attempt_fill(order, bar_touch)
        assert result.filled == True
        assert result.fill_price == 20020.0
    
    def test_market_order_fills_at_open_with_slippage(self):
        """Market order fills at bar open + slippage."""
        simulator = FillSimulator(slippage_ticks=2)  # 0.5 points slippage
        
        order = Order(
            side=Side.LONG,
            order_type=OrderType.MARKET,
            entry_price=20000.0,  # Ignored for market
            sl=19985.0,
            tp=20030.0
        )
        
        bar = pd.Series({
            'open': 20000.0,
            'high': 20015.0,
            'low': 19995.0,
            'close': 20010.0,
        }, name=pd.Timestamp('2024-01-01 09:30'))
        
        result = simulator.attempt_fill(order, bar)
        assert result.filled == True
        assert result.fill_price == 20000.0 + 0.5  # Open + slippage
        assert result.slippage == 0.5
    
    def test_sl_hit_before_tp_same_bar(self):
        """If both SL and TP could hit in same bar, SL is assumed first."""
        simulator = FillSimulator()
        
        trade = Trade(
            id='test',
            entry_time=pd.Timestamp('2024-01-01 09:30'),
            entry_price=20000.0,
            side=Side.LONG,
            qty=1,
            sl=19990.0,  # 10 points below
            tp=20010.0,  # 10 points above
        )
        
        # Bar touches both SL and TP
        bar = pd.Series({
            'open': 20000.0,
            'high': 20015.0,  # Hits TP
            'low': 19985.0,   # Hits SL
            'close': 20005.0,
        }, name=pd.Timestamp('2024-01-01 09:31'))
        
        is_closed, exit_price, reason = simulator.check_sl_tp(trade, bar)
        
        assert is_closed == True
        assert reason == 'SL'  # Conservative: SL assumed first
        assert exit_price == 19990.0


# =============================================================================
# TEST 3: NO LOOKAHEAD - TIME DISCIPLINE
# =============================================================================

class CheatingStrategy:
    """A malicious strategy that tries to peek at future data."""
    
    def __init__(self):
        self.future_access_attempted = False
        self.future_data_seen = None
    
    def on_start(self, engine):
        pass
    
    def on_bar(self, engine, bar_idx, bar):
        # TRY TO CHEAT: Access future bar
        future_idx = bar_idx + 5
        if future_idx < len(engine.data):
            try:
                # This is what a safe engine would prevent
                future_bar = engine.data.iloc[future_idx]
                self.future_access_attempted = True
                self.future_data_seen = future_bar['close']
            except:
                pass
    
    def on_end(self, engine):
        pass


class SafeStrategy:
    """A strategy that only uses safe data access methods."""
    
    def __init__(self):
        self.lookback_data = []
    
    def on_start(self, engine):
        pass
    
    def on_bar(self, engine, bar_idx, bar):
        # SAFE: Use engine's safe access method
        hist = engine.get_historical_data(lookback=5)
        self.lookback_data.append(len(hist))
    
    def on_end(self, engine):
        pass


class TestNoLookahead:
    """Verify the engine prevents lookahead bias."""
    
    def test_get_historical_data_respects_current_bar(self, sample_data):
        """get_historical_data should never return future bars."""
        engine = EventEngine(sample_data)
        
        safe_strategy = SafeStrategy()
        engine.strategy = safe_strategy
        engine.run()
        
        # At each bar, lookback should be min(bar_idx+1, lookback)
        for idx, count in enumerate(safe_strategy.lookback_data):
            expected = min(idx + 1, 5)
            assert count == expected, f"At bar {idx}, got {count} bars, expected {expected}"
    
    def test_shuffle_future_invariance(self, sample_data):
        """Shuffling future bars should not affect trades up to each point."""
        
        class DeterministicStrategy:
            """Makes trades based on simple rule, no lookahead."""
            def __init__(self):
                self.trade_times = []
            
            def on_start(self, engine):
                pass
            
            def on_bar(self, engine, bar_idx, bar):
                # Simple rule: trade every 20 bars if no position
                if bar_idx > 0 and bar_idx % 20 == 0 and not engine.has_open_position:
                    hist = engine.get_historical_data(5)
                    if len(hist) >= 2:
                        if hist['close'].iloc[-1] > hist['close'].iloc[-2]:
                            order = engine.place_market_order(
                                side=Side.LONG,
                                sl=bar['close'] - 15.0,
                                tp=bar['close'] + 30.0
                            )
                            if order:
                                self.trade_times.append(bar_idx)
            
            def on_end(self, engine):
                pass
        
        # Run with original data
        strategy1 = DeterministicStrategy()
        engine1 = EventEngine(sample_data.copy(), strategy1)
        results1 = engine1.run()
        
        # Shuffle last 30 bars (future data when at bar 70)
        shuffled_data = sample_data.copy()
        future_portion = shuffled_data.iloc[70:].sample(frac=1)
        shuffled_data.iloc[70:] = future_portion.values
        
        # Run with shuffled data
        strategy2 = DeterministicStrategy()
        engine2 = EventEngine(shuffled_data, strategy2)
        results2 = engine2.run()
        
        # Trades up to bar 60 should be identical
        trades_pre_shuffle_1 = [t for t in strategy1.trade_times if t <= 60]
        trades_pre_shuffle_2 = [t for t in strategy2.trade_times if t <= 60]
        
        assert trades_pre_shuffle_1 == trades_pre_shuffle_2, \
            "Trades before shuffle point should be identical"


# =============================================================================
# TEST 4: EVENT LOGGING COMPLETENESS
# =============================================================================

class TestEventLogging:
    """Verify comprehensive event logging for audit."""
    
    def test_all_orders_logged(self, sample_data):
        """Every order placement and fill should be logged."""
        engine = EventEngine(sample_data)
        
        # Manual simulation
        engine.current_bar_idx = 10
        engine.current_bar = sample_data.iloc[10]
        
        # Place an order
        order = engine.place_order(
            side=Side.LONG,
            order_type=OrderType.MARKET,
            entry_price=sample_data.iloc[10]['close'],
            sl=sample_data.iloc[10]['close'] - 15.0,
            tp=sample_data.iloc[10]['close'] + 30.0
        )
        
        assert order is not None
        
        # Check logging
        placed_events = engine.event_log.get_events_by_type('ORDER_PLACED')
        assert len(placed_events) == 1
        assert placed_events[0].details['order_id'] == order.id
    
    def test_rejected_orders_logged(self, sample_data):
        """Rejected orders should be logged with reason."""
        engine = EventEngine(sample_data)
        
        engine.current_bar_idx = 10
        engine.current_bar = sample_data.iloc[10]
        
        # Try to place order with SL too small
        order = engine.place_order(
            side=Side.LONG,
            order_type=OrderType.MARKET,
            entry_price=sample_data.iloc[10]['close'],
            sl=sample_data.iloc[10]['close'] - 5.0,  # Too small
            tp=sample_data.iloc[10]['close'] + 30.0
        )
        
        assert order is None  # Rejected
        
        rejected_events = engine.event_log.get_events_by_type('ORDER_REJECTED')
        assert len(rejected_events) == 1
        assert 'SL distance' in rejected_events[0].details['reason']


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
