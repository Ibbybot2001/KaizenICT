"""
Phase 4 Environment: ConceptTradingEnvV2

"Edge Thickening" Edition.
- Sharper State Resolution:
  - bars_since_disp (0..5)
  - fine_dist_to_zone (High res near 0)
- Stricter Rules:
  - Null Regime = Forced Exit (No holding)
  - Candle Color Confirmation Mask
- Rewards (Experiment 4.2):
  - Pure PnL + Costs
  - [NEW] OUTCOME BONUS: +0.2R if trade reaches +1R MFE (One-time per trade)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from ml_lab.engine.trade import Trade
from ml_lab.constants import MIN_SL_POINTS

# --- Configuration ---
REWARD_ENTRY_COST = 0.0  # Relaxed for Exp 4.3 (True Edge Test)
REWARD_TIME_COST = 0.01  # Retain time pressure
REWARD_FRICTION = 0.05   # Retain real friction
REWARD_OUTCOME_BONUS = 0.0 # Remove bonus to trust pure PnL

# Discretization Bins (V2) - Adjusted for 0-index mapping
# np.digitize(x, bins) returns i such that bins[i-1] <= x < bins[i]
# We want 0..N-1 indices. 

# Dist Fine: [-inf, -5, -2, 0, 2, 5, inf] -> 6 buckets
# < -5 (idx 1->0)
# -5..-2 (idx 2->1)
# -2..0 (idx 3->2)
# 0..2 (idx 4->3)
# 2..5 (idx 5->4)
# > 5 (idx 6->5)
BINS_DIST_FINE   = [-float('inf'), -5, -2, 0, 2, 5, float('inf')] 

# Dist Coarse: [-inf, -10, -5, -2, 2, 5, 10, inf] -> 7 buckets
BINS_DIST_COARSE = [-float('inf'), -10, -5, -2, 2, 5, 10, float('inf')]

# Disp: [-inf, 3.0, 5.0, inf] -> 3 buckets
BINS_DISP = [-float('inf'), 3.0, 5.0, float('inf')]

# Comp: [-inf, 0.5, 0.8, inf] -> 3 buckets
BINS_COMP = [-float('inf'), 0.5, 0.8, float('inf')]

# Time: Just use direct integer value logic 0..5

class DiscreteSpace:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)

class TupleSpace:
    def __init__(self, spaces):
        self.spaces = spaces

class ConceptTradingEnvV2:
    def __init__(self, df: pd.DataFrame, 
                 prim_features: pd.DataFrame, 
                 zone_features: pd.DataFrame):
        
        self.df = df.reset_index(drop=True)
        self.prim = prim_features.reset_index(drop=True)
        self.zone = zone_features.reset_index(drop=True)
        
        self.n_bars = len(df)
        self.current_step = 0
        
        # Action Space
        self.action_space = DiscreteSpace(4) # PASS, LONG, SHORT, EXIT
        
        # Observation Space Dimensions
        # 0: Regime ID (3 states: 0, 1, 2)
        # 1: Feat1 (Disp/Comp) -> Max 3 bins
        # 2: Feat2 (Time/DistCoarse) -> Max 7 bins (Time=6, Dist=7)
        # 3: Feat3 (DistFine) -> Max 6 bins
        self.observation_space = TupleSpace((
            DiscreteSpace(3), 
            DiscreteSpace(3), 
            DiscreteSpace(7), 
            DiscreteSpace(6)
        ))
        
        # Tracking
        self.position = 0 
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.target_1r_price = 0.0 # Track if we hit 1R
        self.hit_1r = False        # Flag to ensure bonus only paid once
        
        # Latent State Tracking
        self.active_displacement = False
        self.bars_since_disp = 0
        
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.active_displacement = False
        self.bars_since_disp = 0
        self.hit_1r = False
        return self._get_observation()
        
    def _update_latent_state(self):
        """Track multi-bar events (displacement decay)."""
        p = self.prim.iloc[self.current_step]
        
        # Update Displacement State
        if p['is_displacement']:
            self.active_displacement = True
            self.bars_since_disp = 0
        elif self.active_displacement:
            self.bars_since_disp += 1
            if self.bars_since_disp > 5:
                self.active_displacement = False
                self.bars_since_disp = 0
                
    def _get_observation(self) -> Tuple[int, int, int, int]:
        if self.current_step >= self.n_bars:
            return (0, 0, 0, 0)
            
        self._update_latent_state()
        
        p = self.prim.iloc[self.current_step]
        z = self.zone.iloc[self.current_step]
        
        # 1. Post-Displacement Regime
        if self.active_displacement and abs(z['dist_to_nearest_zone']) <= 20:
             # Feat 1: Disp Intensity (3 bins)
             feat1 = np.digitize([p['disp_zscore']], BINS_DISP)[0] - 1
             
             # Feat 2: Time Decay (0..5) -> Direct mapping
             feat2 = min(self.bars_since_disp, 6) # Should be 0..5
             
             # Feat 3: Fine Dist (6 bins)
             feat3 = np.digitize([z['dist_to_nearest_zone']], BINS_DIST_FINE)[0] - 1
             
             return (1, feat1, feat2, feat3)
             
        # 2. Comp & Approaching Regime
        if p['is_compressed'] and z['approaching'] and abs(z['dist_to_nearest_zone']) <= 15:
            # Feat 1: Comp Score (3 bins)
            feat1 = np.digitize([p['comp_score']], BINS_COMP)[0] - 1
            
            # Feat 2: Coarse Dist (7 bins)
            feat2 = np.digitize([z['dist_to_nearest_zone']], BINS_DIST_COARSE)[0] - 1
            
            # Feat 3: Fine Dist (6 bins)
            feat3 = np.digitize([z['dist_to_nearest_zone']], BINS_DIST_FINE)[0] - 1
            
            return (2, feat1, feat2, feat3)
            
        # 3. Null Regime
        return (0, 0, 0, 0)

    def step(self, action: int):
        if self.current_step >= self.n_bars - 1:
            return (0, 0, 0, 0), 0.0, True, {}
            
        obs = self._get_observation()
        regime = obs[0]
        
        reward = 0.0
        done = False
        
        # --- Forced Exit Logic (Null Regime) ---
        if regime == 0 and self.position != 0:
            # Force Exit immediately
            next_open = self.df.iloc[self.current_step + 1]['open']
            
            if self.position == 1:
                pnl = next_open - self.entry_price
                risk = self.entry_price - self.sl_price
            else:
                pnl = self.entry_price - next_open
                risk = self.sl_price - self.entry_price
                
            r_mult = pnl / risk
            reward += r_mult - REWARD_FRICTION
            self.position = 0
            self.hit_1r = False
            action = 0 
            
        # --- Execution Logic & Outcome Shaping ---
        next_open = self.df.iloc[self.current_step + 1]['open']
        next_low = self.df.iloc[self.current_step + 1]['low']
        next_high = self.df.iloc[self.current_step + 1]['high']
        next_close = self.df.iloc[self.current_step + 1]['close'] # For mask
        current_close = self.df.iloc[self.current_step]['close']
        current_open = self.df.iloc[self.current_step]['open']
        
        exited_this_step = False
        
        if self.position != 0:
            reward -= REWARD_TIME_COST
            
            # Outcome Bonus Check
            if not self.hit_1r:
                if self.position == 1:
                    if next_high >= self.target_1r_price:
                        reward += REWARD_OUTCOME_BONUS
                        self.hit_1r = True
                elif self.position == -1:
                    if next_low <= self.target_1r_price:
                        reward += REWARD_OUTCOME_BONUS
                        self.hit_1r = True

            # Check SL
            if self.position == 1:
                if next_low <= self.sl_price:
                    exit_price = min(next_open, self.sl_price)
                    pnl = exit_price - self.entry_price
                    risk = self.entry_price - self.sl_price
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
                    self.hit_1r = False
                    exited_this_step = True
            elif self.position == -1:
                if next_high >= self.sl_price:
                    exit_price = max(next_open, self.sl_price)
                    pnl = self.entry_price - exit_price
                    risk = self.sl_price - self.entry_price
                    reward += (pnl / risk) - REWARD_FRICTION
                    self.position = 0
                    self.hit_1r = False
                    exited_this_step = True

        # --- Action Processing w/ Color Mask ---
        if not exited_this_step and regime != 0:
            
            is_green = current_close > current_open
            is_red = current_close < current_open
            
            if action == 1: # LONG
                if is_green:
                    if self.position == 0:
                        self.position = 1
                        self.entry_price = next_open
                        self.sl_price = self.entry_price - MIN_SL_POINTS 
                        risk_pts = self.entry_price - self.sl_price
                        self.target_1r_price = self.entry_price + risk_pts
                        self.hit_1r = False
                        reward -= REWARD_ENTRY_COST
                    elif self.position == -1:
                        reward -= 0.1 # Illegal reverse
                else:
                    reward -= 0.05 
                    
            elif action == 2: # SHORT
                if is_red:
                    if self.position == 0:
                        self.position = -1
                        self.entry_price = next_open
                        self.sl_price = self.entry_price + MIN_SL_POINTS
                        risk_pts = self.sl_price - self.entry_price
                        self.target_1r_price = self.entry_price - risk_pts
                        self.hit_1r = False
                        reward -= REWARD_ENTRY_COST
                    elif self.position == 1:
                        reward -= 0.1
                else:
                    reward -= 0.05
                    
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
                    self.hit_1r = False
        
        self.current_step += 1
        return self._get_observation(), reward, done, {}
