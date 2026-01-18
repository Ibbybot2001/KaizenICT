"""
Phase 5 Environment: ConceptTradingEnvV3 (Continuous)

"Continuous Edge Preservation" Edition.
- Observation Space: Box(Continuous)
- No binning. Raw features.
- Strict Constraints:
  - Friction = 0.05R (Realism)
  - Default Action = PASS
  - Forced Exit on specific conditions (optional, but keeping pure for now)
  - Added HARD Take Profit (1.5R) to match DirectPolicy training target.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict

from ml_lab.engine.trade import Trade
from ml_lab.constants import MIN_SL_POINTS

# --- Configuration ---
REWARD_ENTRY_COST = 0.0  # Removing artificial barrier for Phase 5 (Model handles selectivity)
REWARD_TIME_COST = 0.01
REWARD_FRICTION = 0.05

class ConceptTradingEnvV3(gym.Env):
    def __init__(self, df: pd.DataFrame, 
                 prim_features: pd.DataFrame, 
                 zone_features: pd.DataFrame):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        # Convert critical features to numpy for O(1) access
        self.disp_z = prim_features['disp_range_zscore'].fillna(0.0).values
        self.comp_s = prim_features['comp_score'].fillna(0.0).values
        self.is_disp = prim_features['disp_is_displacement'].fillna(0.0).values
        self.is_comp = prim_features['comp_is_compressed'].fillna(0.0).values
        
        self.dist_z = zone_features['dist_to_nearest_zone'].fillna(20.0).values
        self.age_z = zone_features['nearest_zone_age'].fillna(500.0).values
        self.is_in_zone = zone_features['inside_zone'].astype(float).values
        self.is_approaching = zone_features['approaching'].astype(float).values
        
        # Price data for execution
        self.opens = df['open'].values
        self.highs = df['high'].values
        self.lows = df['low'].values
        self.closes = df['close'].values
        
        self.n_bars = len(df)
        self.current_step = 0
        
        # Action Space: PASS, LONG, SHORT, EXIT
        self.action_space = spaces.Discrete(4)
        
        # Observation Space (8 Continuous Features)
        # 0: disp_z (z-score)
        # 1: comp_s (0-1 score)
        # 2: dist_z (signed distance -20..20)
        # 3: age_z (bars 0..500)
        # 4: bars_since_disp (latent: 0..20)
        # 5: is_disp_active (latent: 0/1)
        # 6: is_in_zone (0/1)
        # 7: is_approaching (0/1)
        self.observation_space = spaces.Box(
            low=np.array([-10, 0, -20, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, 10, 20, 1000, 50, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Tracking
        self.position = 0 
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0
        
        # Latent State Tracking
        self.active_displacement = False
        self.bars_since_disp = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.active_displacement = False
        self.bars_since_disp = 0
        return self._get_observation(), {}

    def _update_latent_state(self):
        """Track multi-bar events (displacement decay)."""
        # Using numpy array access
        is_disp_now = self.is_disp[self.current_step] > 0.5
        
        if is_disp_now:
            self.active_displacement = True
            self.bars_since_disp = 0
        elif self.active_displacement:
            self.bars_since_disp += 1
            if self.bars_since_disp > 20: # Decay after 20 bars
                self.active_displacement = False
                self.bars_since_disp = 0

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= self.n_bars:
            return np.zeros(8, dtype=np.float32)
            
        self._update_latent_state()
        
        # Construct vector
        obs = np.array([
            self.disp_z[self.current_step],
            self.comp_s[self.current_step],
            self.dist_z[self.current_step],
            self.age_z[self.current_step],
            float(self.bars_since_disp),
            float(self.active_displacement),
            self.is_in_zone[self.current_step],
            self.is_approaching[self.current_step]
        ], dtype=np.float32)
        
        return obs

    def step(self, action: int):
        if self.current_step >= self.n_bars - 1:
            return np.zeros(8, dtype=np.float32), 0.0, True, False, {}
            
        obs = self._get_observation()
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # --- Execution Logic ---
        next_open = self.opens[self.current_step + 1]
        next_low = self.lows[self.current_step + 1]
        next_high = self.highs[self.current_step + 1]
        # next_close = self.closes[self.current_step + 1]
        
        current_close = self.closes[self.current_step]
        current_open = self.opens[self.current_step]
        
        exited_this_step = False
        
        if self.position != 0:
            reward -= REWARD_TIME_COST
            
            # Check SL then TP
            if self.position == 1:
                # LONG: Check Low for SL first
                if next_low <= self.sl_price:
                    # SL Hit
                    exit_price = min(next_open, self.sl_price)
                    pnl = exit_price - self.entry_price
                    risk = self.entry_price - self.sl_price
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
                    exited_this_step = True
                elif next_high >= self.tp_price:
                    # TP Hit
                    exit_price = max(next_open, self.tp_price)
                    pnl = exit_price - self.entry_price
                    risk = self.entry_price - self.sl_price
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
                    exited_this_step = True
                    
            elif self.position == -1:
                # SHORT: Check High for SL first
                if next_high >= self.sl_price:
                    # SL Hit
                    exit_price = max(next_open, self.sl_price)
                    pnl = self.entry_price - exit_price
                    risk = self.sl_price - self.entry_price # risk is positive
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
                    exited_this_step = True
                elif next_low <= self.tp_price:
                    # TP Hit
                    exit_price = min(next_open, self.tp_price)
                    pnl = self.entry_price - exit_price
                    risk = self.sl_price - self.entry_price
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
                    exited_this_step = True

        # --- Action Processing ---
        if not exited_this_step:
            
            # Color Check (Confirmation) - Optional but good for discipline
            is_green = current_close > current_open
            is_red = current_close < current_open
            
            if action == 1: # LONG
                if self.position == 0:
                     if is_green: # Confirmation
                        self.position = 1
                        self.entry_price = next_open
                        self.sl_price = self.entry_price - MIN_SL_POINTS 
                        self.tp_price = self.entry_price + (1.5 * MIN_SL_POINTS) # 1.5R Target
                        # No Entry Cost in Phase 5
                     else:
                        reward -= 0.01 # Mild penalty
                elif self.position == -1:
                    reward -= 0.1 # Illegal reverse
                    
            elif action == 2: # SHORT
                if self.position == 0:
                    if is_red: # Confirmation
                        self.position = -1
                        self.entry_price = next_open
                        self.sl_price = self.entry_price + MIN_SL_POINTS
                        self.tp_price = self.entry_price - (1.5 * MIN_SL_POINTS) # 1.5R Target
                    else:
                        reward -= 0.01
                elif self.position == 1:
                    reward -= 0.1
                    
            elif action == 3: # EXIT
                if self.position != 0:
                    if self.position == 1:
                        pnl = next_open - self.entry_price
                        risk = self.entry_price - self.sl_price
                    else:
                        pnl = self.entry_price - next_open
                        risk = self.sl_price - self.entry_price
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
        
        self.current_step += 1
        
        if self.current_step >= self.n_bars:
            terminated = True
            
        next_obs = self._get_observation()
        
        return next_obs, reward, terminated, truncated, {}
