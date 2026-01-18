
import torch
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Device Agnostic
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on DEVICE: {DEVICE}")

@dataclass
class BacktestStats:
    concept_id: int
    concept_name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float

class VectorizedBacktester:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.load_data()
        
    def load_data(self):
        """Load Parquet data and convert to Tensors."""
        print(f"Loading data from {self.file_path}...")
        df = pd.read_parquet(self.file_path)
        
        # Ensure sorted
        df = df.sort_index()
        
        # Extract columns
        self.times = pd.to_datetime(df.index)
        self.opens = torch.tensor(df['open'].values, dtype=torch.float32, device=DEVICE)
        self.highs = torch.tensor(df['high'].values, dtype=torch.float32, device=DEVICE)
        self.lows = torch.tensor(df['low'].values, dtype=torch.float32, device=DEVICE)
        self.closes = torch.tensor(df['close'].values, dtype=torch.float32, device=DEVICE)
        
        # Time features
        self.hours = torch.tensor(self.times.hour.values, dtype=torch.int32, device=DEVICE)
        self.minutes = torch.tensor(self.times.minute.values, dtype=torch.int32, device=DEVICE)
        self.weekdays = torch.tensor(self.times.weekday.values, dtype=torch.int32, device=DEVICE) # Mon=0
        
        self.n_bars = len(df)
        print(f"Loaded {self.n_bars} bars.")

    def run_all_concepts(self, slippage_ticks: float = 2.0, spread_ticks: float = 1.0) -> pd.DataFrame:
        """Execute all implemented concepts and return results."""
        results = []
        
        TICK_SIZE = 0.25
        cost_per_trade = (slippage_ticks + spread_ticks) * TICK_SIZE
        print(f"--- Running with Cost Per Trade: {cost_per_trade} pts ({slippage_ticks} slip + {spread_ticks} spread) ---")
        
        # --- List of Concepts to Execute ---
        concept_map = {
            1: self.concept_1_ny_orb,
            2: self.concept_2_london_fix,
            3: self.concept_3_3pm_macro,
            4: self.concept_4_friday_squeeze,
            5: self.concept_5_lunch_reversal,
            6: self.concept_6_premarket_break,
            8: self.concept_8_last_hour_momentum,
            9: self.concept_9_10am_macro,
            10: self.concept_10_midnight_retest,
            14: self.concept_14_silver_bullet_am,
            16: self.concept_16_classic_fvg,
            19: self.concept_19_power_of_3,
            23: self.concept_23_turtle_soup,
            33: self.concept_33_atr_expansion,
            34: self.concept_34_atr_contraction,
            35: self.concept_35_inside_bar_break,
            42: self.concept_42_pdh_pdl_sweep,
            51: self.concept_51_golden_cross,
            59: self.concept_59_donchian_break
        }
        
        for cid, func in concept_map.items():
            print(f"Running Concept {cid}...", end="\r")
            signals = func()
            stats = self.calculate_metrics(cid, func.__name__, signals, cost_per_trade)
            results.append(stats)
            
        print("\nDone.")
        return pd.DataFrame([vars(s) for s in results])

    def calculate_metrics(self, cid: int, name: str, signals: torch.Tensor, cost_per_trade: float) -> BacktestStats:
        """
        Vectorized PnL Calculation with Costs.
        """
        # align signal with next bar return
        next_close = torch.roll(self.closes, -1)
        # Entry assumed at 'Close' of signal bar (or Open of next). 
        # In vector backtest, Close[t] ~ Open[t+1].
        # We trade the delta of the *Next Bar*.
        price_change = next_close - self.closes
        
        trade_returns = signals * price_change
        
        # Subtract Cost
        active_mask = signals != 0
        costs = torch.zeros_like(trade_returns)
        costs[active_mask] = cost_per_trade
        
        net_returns = trade_returns - costs
        active_returns = net_returns[active_mask]
        
        n_trades = len(active_returns)
        
        if n_trades == 0:
            return BacktestStats(cid, name, 0, 0.0, 0.0, 0.0, 0.0)
            
        wins = (active_returns > 0).sum().item()
        
        win_rate = (wins / n_trades) * 100 if n_trades > 0 else 0
        
        gross_profit = active_returns[active_returns > 0].sum().item()
        gross_loss = abs(active_returns[active_returns < 0].sum().item())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        total_return = active_returns.sum().item()
        
        equity = torch.cumsum(active_returns, dim=0)
        peak = torch.cummax(equity, dim=0)[0]
        dd = equity - peak
        max_dd = dd.min().item()
        
        return BacktestStats(cid, name, n_trades, win_rate, profit_factor, total_return, max_dd)

    # --- HELPER FUNCTIONS ---
    def rolling_max(self, x: torch.Tensor, window: int) -> torch.Tensor:
        pad = window - 1
        x_padded = torch.nn.functional.pad(x, (pad, 0), value=-1e9)
        unfolded = x_padded.unfold(0, window, 1)
        return unfolded.max(dim=1)[0]
        
    def rolling_min(self, x: torch.Tensor, window: int) -> torch.Tensor:
        pad = window - 1
        x_padded = torch.nn.functional.pad(x, (pad, 0), value=1e9)
        unfolded = x_padded.unfold(0, window, 1)
        return unfolded.min(dim=1)[0]
        
    def compute_atr(self, period: int = 14) -> torch.Tensor:
        high_low = self.highs - self.lows
        high_close = (self.highs - torch.roll(self.closes, 1)).abs()
        low_close = (self.lows - torch.roll(self.closes, 1)).abs()
        tr = torch.max(high_low, torch.max(high_close, low_close))
        tr_padded = torch.nn.functional.pad(tr.view(1, 1, -1), (period-1, 0))
        weight = torch.ones(1, 1, period, device=DEVICE) / period
        atr = torch.nn.functional.conv1d(tr_padded, weight).view(-1)
        return atr

    # --- CONCEPT IMPLEMENTATIONS ---
    
    def concept_1_ny_orb(self):
        """Concept 1: NY Open ORB (15m)."""
        time_mask = (self.hours == 9) & (self.minutes == 45)
        bullish_break = (self.closes > self.opens) & time_mask
        bearish_break = (self.closes < self.opens) & time_mask
        signals = torch.zeros_like(self.closes)
        signals[bullish_break] = 1
        signals[bearish_break] = -1
        return signals

    def concept_2_london_fix(self):
        """Concept 2: London Fix Fade (11:00 AM)."""
        time_mask = (self.hours == 11) & (self.minutes == 0)
        lookback = 60
        prev_close = torch.roll(self.closes, lookback)
        signals = torch.zeros_like(self.closes)
        long_condition = (self.closes < prev_close) & time_mask 
        short_condition = (self.closes > prev_close) & time_mask
        signals[long_condition] = 1
        signals[short_condition] = -1
        return signals

    def concept_3_3pm_macro(self):
        """Concept 3: 3 PM Macro (14:50-15:10)."""
        time_mask = (self.hours == 15) & (self.minutes == 0)
        signals = torch.zeros_like(self.closes)
        prev_close = torch.roll(self.closes, 10)
        bullish = (self.closes > prev_close) & time_mask
        bearish = (self.closes < prev_close) & time_mask
        signals[bullish] = 1
        signals[bearish] = -1
        return signals
        
    def concept_4_friday_squeeze(self):
        """Concept 4: Friday Short Squeeze."""
        time_mask = (self.weekdays == 4) & (self.hours >= 14)
        week_momentum = self.closes > torch.roll(self.closes, 7200)
        green_bar = self.closes > self.opens
        long_signal = time_mask & week_momentum & green_bar
        signals = torch.zeros_like(self.closes)
        signals[long_signal] = 1
        return signals
        
    def concept_5_lunch_reversal(self):
        """Concept 5: Lunch Hour Reversal."""
        time_mask = (self.hours == 13) & (self.minutes == 0)
        price_1200 = torch.roll(self.closes, 60)
        short_condition = time_mask & (self.closes > price_1200)
        long_condition = time_mask & (self.closes < price_1200)
        signals = torch.zeros_like(self.closes)
        signals[short_condition] = -1
        signals[long_condition] = 1
        return signals

    def concept_6_premarket_break(self):
        """Concept 6: Pre-Market High Break."""
        time_mask = (self.hours == 9) & (self.minutes == 30)
        pm_highs = self.rolling_max(torch.roll(self.highs, 1), 60)
        breakout = (self.opens > pm_highs) & time_mask
        signals = torch.zeros_like(self.closes)
        signals[breakout] = 1
        return signals

    def concept_8_last_hour_momentum(self):
        """Concept 8: Last Hour Momentum."""
        time_mask = (self.hours == 15) & (self.minutes == 0)
        open_0930 = torch.roll(self.closes, 330)
        bullish = (self.closes > open_0930) & time_mask
        bearish = (self.closes < open_0930) & time_mask
        signals = torch.zeros_like(self.closes)
        signals[bullish] = 1
        signals[bearish] = -1
        return signals

    def concept_9_10am_macro(self):
        """Concept 9: 10 AM Macro (10:00-10:10). Momentum."""
        time_mask = (self.hours == 10) & (self.minutes == 0)
        
        # Momentum from 09:50
        prev_close = torch.roll(self.closes, 10)
        
        bullish = (self.closes > prev_close) & time_mask
        bearish = (self.closes < prev_close) & time_mask
        
        signals = torch.zeros_like(self.closes)
        signals[bullish] = 1
        signals[bearish] = -1
        return signals

    def concept_14_silver_bullet_am(self):
        """Concept 14: AM Silver Bullet (10:00-11:00). Breakout of 10:00 Open."""
        # 10:00 Open identified by mask/scatter.
        # Simplified: At 10:00, get Open.
        # Trade breakout during 10:00-11:00.
        
        # Find 10:00 Opens. Shift them to cover the 10:00-11:00 window?
        # Vectorized:
        # Window mask: 10:00 <= T <= 10:59
        window_mask = (self.hours == 10)
        
        # We need the 10:00 Open to be available at every bar in the window.
        # Use a "Session Open" propagation trick or standard logic.
        # Simple Logic: At any bar in window, if Close > Open_10am -> Long.
        
        # Getting Open_10am vectorized without loops:
        # 1. Create mask of 10:00 bars.
        # 2. Extract Opens.
        # 3. Use `torch.repeat_interleave` or `scatter` if fixed grid.
        # Since 1h = 60 mins exactly in clean data.
        # We can just check `Open[T - Minute]`? 
        # e.g., at 10:15, Open is at index `i - 15`.
        # Yes! `self.minutes` tells us exactly how far back the hour open is.
        # This works perfectly for reliable regular data.
        
        # Dynamic roll is hard.
        # Alternative: Just trade the 10:00 BAR MOMENTUM entering at 10:15?
        # Definition: "Silver Bullet setup forms in 10-11am".
        # Let's try: "10:15 Breakout".
        time_mask = (self.hours == 10) & (self.minutes == 15)
        
        # Compare 10:15 Close to 10:00 Open (Close[t-15])
        open_10am = torch.roll(self.closes, 15) # Approx proxy
        
        bullish = (self.closes > open_10am) & time_mask
        bearish = (self.closes < open_10am) & time_mask
        
        signals = torch.zeros_like(self.closes)
        signals[bullish] = 1
        signals[bearish] = -1
        return signals

    def concept_19_power_of_3(self):
        """Concept 19: Power of 3 (AMD). Fade the 'Manipulation' from Open."""
        # Daily Open (09:30 or 00:00). Let's Use 00:00 (Midnight Open).
        # "Judas Swing" -> Price moves away from Open, then Reverses.
        # Trigger: 10:00 AM.
        # If Price < Open (Manipulation Down) -> Buy (Distribution Up).
        
        # Assumption: 00:00 Open is approx Close of previous day.
        # Trigger at 10:00.
        time_mask = (self.hours == 10) & (self.minutes == 0)
        
        # Compare to Midnight (approx 10 hours * 60 = 600 bars ago)
        midnight_proxy = torch.roll(self.closes, 600)
        
        # If Price < Midnight -> Buy (Expect Reversal up)
        # If Price > Midnight -> Sell (Expect Reversal down)
        
        bullish = (self.closes < midnight_proxy) & time_mask
        bearish = (self.closes > midnight_proxy) & time_mask
        
        signals = torch.zeros_like(self.closes)
        signals[bullish] = 1
        signals[bearish] = -1
        return signals

    def concept_42_pdh_pdl_sweep(self):
        """Concept 42: Prev Day High/Low Sweep."""
        # Detect if High > PrevDayHigh.
        # PDH = Max of Previous 1440 bars.
        
        # At 09:45 (NY Open), check if we swept PDH/PDL.
        time_mask = (self.hours == 9) & (self.minutes == 45)
        
        # Previous 24h window
        prev_24h_high = self.rolling_max(torch.roll(self.highs, 1), 1440)
        prev_24h_low = self.rolling_min(torch.roll(self.lows, 1), 1440)
        
        # Condition: Current High > PrevHigh (Sweep).
        # Trade: Fade back in? Or Breakout?
        # ICT usually fades sweeps. "Turtle Soup".
        # Let's Fade.
        
        swept_high = (self.highs > prev_24h_high) & time_mask
        swept_low = (self.lows < prev_24h_low) & time_mask
        
        signals = torch.zeros_like(self.closes)
        signals[swept_high] = -1 # Sell the high
        signals[swept_low] = 1   # Buy the low
        
        return signals

    def concept_10_midnight_retest(self):
        """Concept 10: Midnight Open Retest."""
        # Simplified: Identify Midnight Open.
        # Check if price returns to it during 09:30-11:00.
        # Placeholder for now to fix AttributeError.
        signals = torch.zeros_like(self.closes)
        return signals

    def concept_16_classic_fvg(self):
        """Concept 16: Classic FVG."""
        high_shift_2 = torch.roll(self.highs, 2)
        low_shift_2 = torch.roll(self.lows, 2)
        valid_mask = torch.ones_like(self.closes, dtype=torch.bool)
        valid_mask[:2] = False
        bull_fvg = (self.lows > high_shift_2) & valid_mask
        bear_fvg = (self.highs < low_shift_2) & valid_mask
        signals = torch.zeros_like(self.closes)
        signals[bull_fvg] = 1
        signals[bear_fvg] = -1
        return signals
        
    def concept_23_turtle_soup(self):
        """Concept 23: Turtle Soup (Classic)."""
        window = 20
        roll_highs = self.rolling_max(torch.roll(self.highs, 1), window)
        roll_lows = self.rolling_min(torch.roll(self.lows, 1), window)
        sweep_high = (self.highs > roll_highs) & (self.closes < roll_highs)
        sweep_low = (self.lows < roll_lows) & (self.closes > roll_lows)
        signals = torch.zeros_like(self.closes)
        signals[sweep_low] = 1
        signals[sweep_high] = -1
        return signals
        
    def concept_33_atr_expansion(self):
        """Concept 33: ATR Expansion."""
        atr = self.compute_atr(14)
        bar_range = self.highs - self.lows
        expansion = bar_range > (2.0 * atr)
        bullish = expansion & (self.closes > self.opens)
        bearish = expansion & (self.closes < self.opens)
        signals = torch.zeros_like(self.closes)
        signals[bullish] = 1
        signals[bearish] = -1
        return signals

    def concept_34_atr_contraction(self):
        """Concept 34: ATR Contraction."""
        atr = self.compute_atr(14)
        bar_range = self.highs - self.lows
        contraction = bar_range < (0.5 * atr)
        squeeze_mask = contraction
        trade_mask = torch.roll(squeeze_mask, 1)
        next_close = torch.roll(self.closes, -1)
        curr_high = self.highs
        curr_low = self.lows
        prev_high = torch.roll(curr_high, 1)
        prev_low = torch.roll(curr_low, 1)
        bull_break = trade_mask & (self.closes > prev_high)
        bear_break = trade_mask & (self.closes < prev_low)
        signals = torch.zeros_like(self.closes)
        signals[bull_break] = 1
        signals[bear_break] = -1
        return signals
        
    def concept_35_inside_bar_break(self):
        """Concept 35: Inside Bar Break."""
        prev_high = torch.roll(self.highs, 1)
        prev_low = torch.roll(self.lows, 1)
        is_inside = (self.highs < prev_high) & (self.lows > prev_low)
        trade_mask = torch.roll(is_inside, 1)
        ib_high = prev_high
        ib_low = prev_low
        bull_break = trade_mask & (self.closes > ib_high)
        bear_break = trade_mask & (self.closes < ib_low)
        signals = torch.zeros_like(self.closes)
        signals[bull_break] = 1
        signals[bear_break] = -1
        return signals

    def concept_51_golden_cross(self):
        """Concept 51: Golden Cross."""
        def rolling_mean(x, window):
            x_padded = torch.nn.functional.pad(x.view(1, 1, -1), (window-1, 0))
            weight = torch.ones(1, 1, window, device=DEVICE) / window
            return torch.nn.functional.conv1d(x_padded, weight).view(-1)
        sma50 = rolling_mean(self.closes, 50)
        sma200 = rolling_mean(self.closes, 200)
        shift_sma50 = torch.roll(sma50, 1)
        shift_sma200 = torch.roll(sma200, 1)
        cross_up = (sma50 > sma200) & (shift_sma50 <= shift_sma200)
        cross_down = (sma50 < sma200) & (shift_sma50 >= shift_sma200)
        signals = torch.zeros_like(self.closes)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals
        
    def concept_59_donchian_break(self):
        """Concept 59: Donchian Break."""
        window = 20
        prev_highs = self.rolling_max(torch.roll(self.highs, 1), window)
        prev_lows = self.rolling_min(torch.roll(self.lows, 1), window)
        break_high = self.closes > prev_highs
        break_low = self.closes < prev_lows
        signals = torch.zeros_like(self.closes)
        signals[break_high] = 1
        signals[break_low] = -1
        return signals

if __name__ == "__main__":
    base_dir = Path("C:/Users/CEO/ICT reinforcement")
    data_path = base_dir / "data/GOLDEN_DATA/USTEC_2025_GOLDEN_PARQUET/USTEC_2025-01_clean_1m.parquet"
    
    if not data_path.exists():
        print(f"Data not found at {data_path}")
    else:
        runner = VectorizedBacktester(str(data_path))
        
        # Run with costs: 2 Ticks Slippage + 1 Tick Spread = 0.75 pts per trade penalty
        print(">>> STARTING STRESS TEST (Cost = 0.75 pts) <<<")
        results = runner.run_all_concepts(slippage_ticks=2.0, spread_ticks=1.0)
        
        print("\n--- GPU/Vectorized Batch Results (Stressed) ---")
        print(results.to_string())
        
        # Save
        out_path = base_dir / "output/gpu_batch_results_stressed.csv"
        results.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
