"""
Phase 5.3: PPO Training

Trains a Continuous Policy (PPO) to optimize the edge found in Phase 5.2.
Goal: Improve Net R/Trade by refining timing.
"""
import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple

from ml_lab.ml.models.ppo import SimplePPO
from ml_lab.ml.policy.concept_env_v3 import ConceptTradingEnvV3
from ml_lab.ml.feature_builder import FeatureBuilder
from ml_lab.ml.zone_relative import build_zone_relative_features

DATA_PATH = "ml_lab/data/kaizen_1m_data_ibkr_2yr.csv"

def load_data():
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, parse_dates=['time'])
        df = df.set_index('time')
        df.columns = [c.lower() for c in df.columns]
        return df
    else: 
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

def main():
    print("PHASE 5.3: PPO OPTIMIZATION")
    
    # 1. Setup Data
    data = load_data()
    data = data.iloc[-50000:] # Use same subset to be comparable
    
    print("Building features...")
    fb = FeatureBuilder()
    prim = fb.build_feature_matrix(data)
    zone = build_zone_relative_features(data)
    
    # Train/Test Split
    split_idx = int(len(data) * 0.7)
    train_env = ConceptTradingEnvV3(data.iloc[:split_idx], prim.iloc[:split_idx], zone.iloc[:split_idx])
    test_env = ConceptTradingEnvV3(data.iloc[split_idx:], prim.iloc[split_idx:], zone.iloc[split_idx:])
    
    # 2. Setup PPO
    obs_dim = 8
    act_dim = 4
    ppo = SimplePPO(obs_dim, act_dim, lr=0.0003, gamma=0.99, K_epochs=4, eps_clip=0.2)
    
    # 3. Training Loop
    max_episodes = 200
    update_timestep = 2000 # Update policy every N timesteps
    
    timestep = 0
    
    print(f"Starting Training ({max_episodes} eps)...")
    
    for ep in range(max_episodes):
        state, _ = train_env.reset()
        ep_reward = 0
        ep_trades = 0
        
        while True:
            timestep += 1
            
            # Run Policy
            action, log_prob = ppo.select_action(state)
            
            # Execute
            next_state, reward, done, _, _ = train_env.step(action)
            
            # Save to buffer
            ppo.store(state, action, log_prob, reward, done)
            
            # Stats
            if reward != 0 and reward != -0.01: # Count specific outcome events
                 pass # Could count wins/losses
            
            state = next_state
            ep_reward += reward
            
            # Update PPO
            if timestep % update_timestep == 0:
                ppo.update()
                
            if done:
                break
        
        # Logging
        if ep % 10 == 0:
             print(f"Ep {ep}: Reward={ep_reward:.2f}")

    # 4. Save
    os.makedirs("ml_lab/runs", exist_ok=True)
    ppo.save("ml_lab/runs/ppo_v1.pth")
    print("Model saved.")
    
    # 5. OOS Eval
    print("\nRunning OOS Evaluation (PPO)...")
    state, _ = test_env.reset()
    total_reward = 0
    trades = 0
    
    while True:
        action, _ = ppo.select_action(state)
        if action != 0: trades += 1 # Count all non-PASS actions as "activity" (approx)
        
        # Note: EnvV3 `step` returns transition. 0->1 transition needed to strictly count trades?
        # Re-using logic from DirectPolicy eval for consistency.
        # Actually EnvV3 handles position internally. We just want result.
        
        next_state, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        if done: break
        
    print("="*60)
    print(f"OOS RESULTS (PPO):")
    print(f"Total Reward: {total_reward:.2f} R")
    print("="*60)
    
if __name__ == "__main__":
    main()
