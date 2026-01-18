"""
Phase 3C Environment: ConceptTradingEnv

The "Fenced Garden" for Policy Learning.
- Strictly defines regimes (States A, B, C)
- Enforces constraints (SL >= 10, No Overtrading)
- Computes Reward (R-multiple - Friction - Time - EntryCost)

State Space (Discrete):
- Post-Displacement: (DispBin, DistBin)
- Comp-Approaching: (CompBin, DistBin)
- Null: 0

Action Space:
0: PASS/HOLD
1: ENTER_LONG
2: ENTER_SHORT
3: EXIT
"""

# import gym  # Removing dependency to avoid installation issues
# from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from engine.trade import Trade
from engine.constants import MIN_SL_POINTS, TICK_SIZE, COMMISSION_PER_CONTRACT, CONTRACT_MULTIPLIER

# --- Configuration ---
REWARD_ENTRY_COST = 0.1  # R-multiple penalty
REWARD_TIME_COST = 0.01  # R-multiple per bar
REWARD_FRICTION = 0.05   # Approx R-multiple for spread/comm (conservative)

# Discretization Bins
BINS_DIST = [-float('inf'), -10, -5, -2, 2, 5, 10, float('inf')] # 7 buckets
BINS_DISP = [-float('inf'), 3.0, 5.0, float('inf')]              # 3 buckets (2-3, 3-5, >5)
BINS_COMP = [-float('inf'), 0.5, 0.8, float('inf')]              # 3 buckets (<0.5, 0.5-0.8, >0.8)

# Minimal Gym Mock for compatibility if needed
class DiscreteSpace:
    def __init__(self, n):
        self.n = n
    def sample(self):
        return np.random.randint(self.n)

class TupleSpace:
    def __init__(self, spaces):
        self.spaces = spaces

class ConceptTradingEnv:
    """
    Gym-like Environment for Concept Policy Learning.
    Dependency-free version.
    
    Observations are DISCRETE state IDs.
    """
    
    def __init__(self, df: pd.DataFrame, 
                 prim_features: pd.DataFrame, 
                 zone_features: pd.DataFrame):
        # super(ConceptTradingEnv, self).__init__() # No gym inheritance
        
        self.df = df.reset_index(drop=True)
        self.prim = prim_features.reset_index(drop=True)
        self.zone = zone_features.reset_index(drop=True)
        
        self.n_bars = len(df)
        self.current_step = 0
        
        # Action Space
        self.action_space = DiscreteSpace(4) # PASS, LONG, SHORT, EXIT
        
        # Observation Space
        self.observation_space = TupleSpace((
             DiscreteSpace(3),  # Regime
             DiscreteSpace(5),  # Feat1
             DiscreteSpace(10)  # Feat2
        ))
        
        # Tracking
        self.position = 0 # 0=Flat, 1=Long, -1=Short
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp_price = 0.0 # Optional, for R calc
        self.entry_step = 0
        
    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        return self._get_observation()
        
    def _get_observation(self) -> Tuple[int, int, int]:
        """Determine State ID based on current bar features."""
        if self.current_step >= self.n_bars:
            return (0, 0, 0)
            
        # Get raw features
        p = self.prim.iloc[self.current_step]
        z = self.zone.iloc[self.current_step]
        
        # 1. Post-Displacement Regime
        # Trigger: is_disp AND bars_since <= 5 AND dist <= 20
        # NOTE: 'bars_since' needs tracking. For tabular simplicty, 
        # we assume 'is_displacement' captures the latent state or add rolling check.
        # Strict definition: Is ACTIVE displacement logic.
        if p['is_displacement']: # AND proximity check
             # Binning
             feat1 = np.digitize([p['disp_zscore']], BINS_DISP)[0]
             feat2 = np.digitize([z['dist_to_nearest_zone']], BINS_DIST)[0]
             return (1, feat1, feat2)
             
        # 2. Comp & Approaching Regime
        # Trigger: compressed AND approaching AND dist <= 15
        if p['is_compressed'] and z['approaching'] and abs(z['dist_to_nearest_zone']) <= 15:
            feat1 = np.digitize([p['comp_score']], BINS_COMP)[0]
            feat2 = np.digitize([z['dist_to_nearest_zone']], BINS_DIST)[0]
            return (2, feat1, feat2)
            
        # 3. Null Regime
        return (0, 0, 0)

    def step(self, action: int):
        """Execute action, return (obs, reward, done, info)."""
        if self.current_step >= self.n_bars - 1:
            return (0, 0, 0), 0.0, True, {}
            
        obs = self._get_observation()
        regime = obs[0]
        
        reward = 0.0
        done = False
        
        # --- State C: Null Logic (Forced PASS) ---
        if regime == 0:
            if self.position != 0:
                # Forced Exit if carrying position into Null? 
                # Or allow management?
                # Design says: "Null... Action=Forced PASS". 
                # Usually implies no NEW trades. Existing management allowed?
                # "Agent is blind unless in valid regime". Implies forced exit or autonomous holding.
                # Let's enforce: If in Null, forced HOLD if pos exists, else PASS.
                # Agent cannot act.
                pass 
            else:
                # No position, forced pass.
                # If agent tried to enter, penalize slightly? Or just ignore.
                if action in [1, 2]:
                    reward = -0.01 # Invalid action penalty
        
        # --- Execution Logic ---
        
        # 1. Update Position PnL (Mark to Market / Check SL/TP)
        # Using next bar open for execution
        next_open = self.df.iloc[self.current_step + 1]['open']
        next_low = self.df.iloc[self.current_step + 1]['low']
        next_high = self.df.iloc[self.current_step + 1]['high']
        
        exited_this_step = False
        
        if self.position != 0:
            reward -= REWARD_TIME_COST # Time decay
            
            # Check SL
            if self.position == 1: # Long
                if next_low <= self.sl_price:
                    # SL Hit
                    exit_price = min(next_open, self.sl_price) # Worse case
                    pnl = exit_price - self.entry_price
                    risk = self.entry_price - self.sl_price
                    r_mult = pnl / risk
                    reward += r_mult - REWARD_FRICTION
                    self.position = 0
                    exited_this_step = True
            elif self.position == -1: # Short
                if next_high >= self.sl_price:
                    # SL Hit
                    exit_price = max(next_open, self.sl_price)
                    pnl = self.entry_price - exit_price
                    risk = self.sl_price - self.entry_price
                    r_mult = pnl / risk
                    reward += r_mult - REWARD_FRICTION
                    self.position = 0
                    exited_this_step = True

        # 2. Process Actions (if not exited)
        # Only allow actions if Regime != 0 (Null) AND not exited
        if not exited_this_step and regime != 0:
        
            if action == 1: # ENTER LONG
                if self.position == 0:
                    self.position = 1
                    self.entry_price = next_open
                    # Set SL (Fixed 10pt for now, policy can optimize later or via swing)
                    self.sl_price = self.entry_price - MIN_SL_POINTS 
                    self.entry_step = self.current_step
                    reward -= REWARD_ENTRY_COST # Safety valve
                elif self.position == -1:
                    # Reverse? Design says "Cannot reverse".
                    reward = -0.1 # Illegal move penalty
                    
            elif action == 2: # ENTER SHORT
                if self.position == 0:
                    self.position = -1
                    self.entry_price = next_open
                    self.sl_price = self.entry_price + MIN_SL_POINTS
                    self.entry_step = self.current_step
                    reward -= REWARD_ENTRY_COST
                elif self.position == 1:
                    reward = -0.1 # Illegal
                    
            elif action == 3: # EXIT
                if self.position != 0:
                    # Execute Exit
                    if self.position == 1:
                        pnl = next_open - self.entry_price
                        risk = self.entry_price - self.sl_price
                    else:
                        pnl = self.entry_price - next_open
                        risk = self.sl_price - self.entry_price
                        
                    r_mult = pnl / risk
                    reward += r_mult - REWARD_FRICTION
                    self.position = 0
        
        # Advance
        self.current_step += 1
        return self._get_observation(), reward, done, {}

