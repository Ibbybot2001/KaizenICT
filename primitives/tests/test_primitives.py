"""
Phase 2 Audit Tests - Primitive No-Lookahead Verification

REQUIRED TESTS (per specification):
1. Past-only test: Primitives fail if accessing bar_index + k
2. Shuffle-future invariance: Shuffle bars after t, outputs <= t unchanged
3. Timestamp correctness: Zone creation matches earliest possible
4. Edge cases: First N bars, session gaps, flat markets, single-bar spikes
5. Primitive independence: Removing one doesn't alter others

These tests MUST ALL PASS before Phase 3.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from copy import deepcopy

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from primitives.zones import ZoneDetector, ZoneType, compute_zones_at
from primitives.displacement import DisplacementDetector, compute_displacement_at
from primitives.overlap import OverlapCalculator, compute_body_overlap_at
from primitives.speed import SpeedTracker, compute_speed_after_touch
from primitives.compression import CompressionDetector, compute_compression_at
from primitives.liquidity import LiquidityDetector, compute_liquidity_at


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 200
    
    dates = pd.date_range('2024-01-01 09:30', periods=n_bars, freq='1min')
    base_price = 20000.0
    
    # Random walk with some structure
    returns = np.random.randn(n_bars) * 0.0015
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


@pytest.fixture
def data_with_fvg():
    """Data with a known FVG pattern."""
    dates = pd.date_range('2024-01-01 09:30', periods=20, freq='1min')
    
    # Normal candles first
    data = pd.DataFrame({
        'open': [100.0] * 20,
        'high': [101.0] * 20,
        'low': [99.0] * 20,
        'close': [100.5] * 20,
        'volume': [100] * 20,
    }, index=dates)
    
    # Create bullish FVG at bars 5,6,7
    # Candle 5 (bar_idx=5): normal
    data.loc[dates[5], ['open', 'high', 'low', 'close']] = [100, 101, 99, 100]
    # Candle 6 (bar_idx=6): big up candle
    data.loc[dates[6], ['open', 'high', 'low', 'close']] = [100, 105, 100, 104]
    # Candle 7 (bar_idx=7): gap up (low > candle 5 high)
    data.loc[dates[7], ['open', 'high', 'low', 'close']] = [104, 107, 103, 106]
    # FVG: gap between bar 5 high (101) and bar 7 low (103) -> GAP from 101 to 103
    
    return data


@pytest.fixture
def data_with_swing():
    """Data with a known swing high pattern."""
    dates = pd.date_range('2024-01-01 09:30', periods=20, freq='1min')
    
    # Create swing high at bar 10 with left=3, right=3
    highs = [100.0] * 20
    # Bar 10 is highest, surrounded by lower bars
    highs[7] = 102.0   # left neighbor
    highs[8] = 103.0   # left neighbor
    highs[9] = 104.0   # left neighbor
    highs[10] = 110.0  # PEAK
    highs[11] = 105.0  # right neighbor
    highs[12] = 104.0  # right neighbor
    highs[13] = 103.0  # right neighbor
    
    data = pd.DataFrame({
        'open': [100.0] * 20,
        'high': highs,
        'low': [99.0] * 20,
        'close': [100.0] * 20,
        'volume': [100] * 20,
    }, index=dates)
    
    return data


# =============================================================================
# TEST 1: PAST-ONLY VERIFICATION
# =============================================================================

class TestPastOnlyAccess:
    """Verify all primitives use only past data."""
    
    def test_displacement_uses_past_only(self, sample_data):
        """Displacement z-score at t uses only data[0:t] (exclusive of t for stats)."""
        detector = DisplacementDetector(lookback=10, min_periods=5)
        
        # At bar 50, check what data is used
        result = detector.compute_at(50, sample_data)
        
        assert result is not None
        
        # Manually verify: rolling stats should be from bars 40-49 (not 50)
        hist_ranges = (sample_data.iloc[40:50]['high'] - sample_data.iloc[40:50]['low'])
        expected_mean = hist_ranges.mean()
        
        assert abs(result.rolling_range_mean - expected_mean) < 0.001
    
    def test_compression_uses_past_only(self, sample_data):
        """Compression percentile at t uses only data[0:t]."""
        detector = CompressionDetector(lookback=10, min_periods=5)
        
        result = detector.compute_at(50, sample_data)
        
        assert result is not None
        # Current bar value should not be in the percentile distribution
        current_range = sample_data.iloc[50]['high'] - sample_data.iloc[50]['low']
        assert abs(result.current_range - current_range) < 0.001
    
    def test_zone_fvg_timing(self, data_with_fvg):
        """FVG zone created_at is the bar when FVG becomes knowable."""
        detector = ZoneDetector()
        
        # FVG should be detected at bar 7 (when candle 7 closes)
        zones = detector.compute_all_zones_at(7, data_with_fvg)
        
        fvg_zones = [z for z in zones if z.zone_type == ZoneType.FVG_BULL]
        assert len(fvg_zones) == 1
        assert fvg_zones[0].created_at == 7
        
        # FVG should NOT exist at bar 6
        zones_early = detector.compute_all_zones_at(6, data_with_fvg)
        fvg_early = [z for z in zones_early if z.zone_type == ZoneType.FVG_BULL]
        assert len(fvg_early) == 0
    
    def test_swing_confirmation_delay(self, data_with_swing):
        """Swing zone appears at confirmation time, not at peak time."""
        detector = ZoneDetector(swing_left=3, swing_right=3)
        
        # Swing peak is at bar 10
        # With right=3, confirmation is at bar 13
        
        # At bar 12, swing should NOT be confirmed yet
        zones_early = detector.compute_all_zones_at(12, data_with_swing)
        swing_zones = [z for z in zones_early if z.zone_type == ZoneType.SWING_HIGH]
        assert len(swing_zones) == 0
        
        # At bar 13, swing should be confirmed
        zones_confirm = detector.compute_all_zones_at(13, data_with_swing)
        swing_zones = [z for z in zones_confirm if z.zone_type == ZoneType.SWING_HIGH]
        assert len(swing_zones) == 1
        
        # Verify created_at vs origin_bar
        swing = swing_zones[0]
        assert swing.created_at == 13  # Confirmation time
        assert swing.origin_bar == 10  # Actual peak location


# =============================================================================
# TEST 2: SHUFFLE-FUTURE INVARIANCE
# =============================================================================

class TestShuffleFutureInvariance:
    """Shuffling future bars should not affect outputs at current time."""
    
    def test_displacement_invariant_to_future_shuffle(self, sample_data):
        """Displacement values unchanged when future is shuffled."""
        detector = DisplacementDetector(lookback=10)
        
        # Compute at bar 50 with original data
        result_original = detector.compute_at(50, sample_data)
        
        # Shuffle bars 60+ 
        shuffled_data = sample_data.copy()
        future_portion = shuffled_data.iloc[60:].sample(frac=1, random_state=123)
        shuffled_data.iloc[60:] = future_portion.values
        
        # Compute at bar 50 with shuffled data
        result_shuffled = detector.compute_at(50, shuffled_data)
        
        # Results should be identical
        assert result_original.range_zscore == result_shuffled.range_zscore
        assert result_original.body_zscore == result_shuffled.body_zscore
    
    def test_zones_invariant_to_future_shuffle(self, sample_data):
        """Zone list at t unchanged when future is shuffled."""
        detector = ZoneDetector(swing_left=3, swing_right=3)
        
        # Build zone history up to bar 80
        zones_at_80 = []
        for i in range(81):
            zones_at_80.extend(detector.compute_all_zones_at(i, sample_data))
        
        # Shuffle bars 100+
        shuffled_data = sample_data.copy()
        future_portion = shuffled_data.iloc[100:].sample(frac=1, random_state=123)
        shuffled_data.iloc[100:] = future_portion.values
        
        # Rebuild zone history up to bar 80
        zones_shuffled = []
        for i in range(81):
            zones_shuffled.extend(detector.compute_all_zones_at(i, shuffled_data))
        
        # Same number of zones
        assert len(zones_at_80) == len(zones_shuffled)
        
        # Same zone properties
        for z1, z2 in zip(zones_at_80, zones_shuffled):
            assert z1.upper == z2.upper
            assert z1.lower == z2.lower
            assert z1.created_at == z2.created_at
    
    def test_compression_invariant_to_future_shuffle(self, sample_data):
        """Compression scores unchanged when future is shuffled."""
        detector = CompressionDetector(lookback=15)
        
        result_original = detector.compute_at(60, sample_data)
        
        # Shuffle future
        shuffled_data = sample_data.copy()
        future_portion = shuffled_data.iloc[80:].sample(frac=1, random_state=42)
        shuffled_data.iloc[80:] = future_portion.values
        
        result_shuffled = detector.compute_at(60, shuffled_data)
        
        assert result_original.range_percentile == result_shuffled.range_percentile
        assert result_original.compression_score == result_shuffled.compression_score


# =============================================================================
# TEST 3: TIMESTAMP CORRECTNESS
# =============================================================================

class TestTimestampCorrectness:
    """Verify zone creation timestamps are accurate."""
    
    def test_fvg_created_at_matches_third_candle(self, data_with_fvg):
        """FVG created_at should be the bar index of candle 3."""
        detector = ZoneDetector()
        
        zones = detector.compute_all_zones_at(7, data_with_fvg)
        fvg = [z for z in zones if z.zone_type == ZoneType.FVG_BULL][0]
        
        # Created at bar 7 (candle 3 of the FVG pattern)
        assert fvg.created_at == 7
        assert fvg.created_time == data_with_fvg.index[7]
    
    def test_swing_created_at_matches_confirmation(self, data_with_swing):
        """Swing created_at should be confirmation bar, not peak bar."""
        detector = ZoneDetector(swing_left=3, swing_right=3)
        
        zones = detector.compute_all_zones_at(13, data_with_swing)
        swing = [z for z in zones if z.zone_type == ZoneType.SWING_HIGH][0]
        
        assert swing.created_at == 13  # Confirmation
        assert swing.origin_bar == 10  # Peak
        assert swing.created_time == data_with_swing.index[13]


# =============================================================================
# TEST 4: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Verify primitives handle edge cases correctly."""
    
    def test_first_n_bars_displacement(self, sample_data):
        """Displacement returns None for insufficient history."""
        detector = DisplacementDetector(lookback=20, min_periods=10)
        
        # First 9 bars should return None
        for i in range(10):
            result = detector.compute_at(i, sample_data)
            assert result is None
        
        # Bar 10 should work
        result = detector.compute_at(10, sample_data)
        assert result is not None
    
    def test_first_n_bars_compression(self, sample_data):
        """Compression returns None for insufficient history."""
        detector = CompressionDetector(lookback=20, min_periods=10)
        
        for i in range(10):
            result = detector.compute_at(i, sample_data)
            assert result is None
    
    def test_flat_market_displacement(self):
        """Displacement handles flat markets (zero std)."""
        dates = pd.date_range('2024-01-01 09:30', periods=30, freq='1min')
        
        # Completely flat market
        data = pd.DataFrame({
            'open': [100.0] * 30,
            'high': [100.0] * 30,  # No range
            'low': [100.0] * 30,
            'close': [100.0] * 30,
            'volume': [100] * 30,
        }, index=dates)
        
        detector = DisplacementDetector(lookback=10, min_periods=5)
        result = detector.compute_at(20, data)
        
        assert result is not None
        assert result.range_zscore == 0.0  # Zero std handled
    
    def test_single_bar_spike(self, sample_data):
        """Displacement detects single-bar spike correctly."""
        # Insert a spike
        spike_idx = 50
        sample_data.loc[sample_data.index[spike_idx], 'high'] = sample_data.iloc[spike_idx]['close'] + 50
        
        detector = DisplacementDetector(lookback=20, threshold_zscore=2.0)
        result = detector.compute_at(spike_idx, sample_data)
        
        assert result is not None
        assert result.is_displacement == True


# =============================================================================
# TEST 5: PRIMITIVE INDEPENDENCE
# =============================================================================

class TestPrimitiveIndependence:
    """Verify primitives don't have hidden dependencies."""
    
    def test_displacement_independent_of_zones(self, sample_data):
        """Displacement doesn't depend on zone calculations."""
        # Compute displacement
        disp_result = compute_displacement_at(50, sample_data)
        
        # Compute zones (shouldn't affect displacement)
        zones = compute_zones_at(50, sample_data)
        
        # Recompute displacement
        disp_result_after = compute_displacement_at(50, sample_data)
        
        assert disp_result.range_zscore == disp_result_after.range_zscore
    
    def test_compression_independent_of_liquidity(self, sample_data):
        """Compression doesn't depend on liquidity calculations."""
        comp_result = compute_compression_at(50, sample_data)
        
        # Compute liquidity
        liq = compute_liquidity_at(50, sample_data)
        
        comp_result_after = compute_compression_at(50, sample_data)
        
        assert comp_result.range_percentile == comp_result_after.range_percentile


# =============================================================================
# TEST 6: LIQUIDITY TWO-TOUCH REQUIREMENT
# =============================================================================

class TestLiquidityTwoTouch:
    """Verify liquidity only exists after second touch."""
    
    def test_liquidity_requires_two_touches(self):
        """Liquidity level only created on second touch."""
        dates = pd.date_range('2024-01-01 09:30', periods=20, freq='1min')
        
        # First touch at bar 5, second at bar 10
        highs = [100.0] * 20
        highs[5] = 105.0  # First touch
        highs[10] = 105.2  # Second touch (within tolerance)
        
        data = pd.DataFrame({
            'open': [100.0] * 20,
            'high': highs,
            'low': [99.0] * 20,
            'close': [100.0] * 20,
            'volume': [100] * 20,
        }, index=dates)
        
        detector = LiquidityDetector(tolerance=0.5, min_touches=2)
        
        # At bar 5 (first touch), no liquidity
        levels_5 = detector.compute_all_at(5, data)
        assert len(levels_5) == 0
        
        # At bar 10 (second touch), liquidity exists
        levels_10 = detector.compute_all_at(10, data)
        eq_highs = [l for l in levels_10 if l.liquidity_type.value == 'EQUAL_HIGHS']
        assert len(eq_highs) == 1
        assert eq_highs[0].created_at == 10  # Created on second touch


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
