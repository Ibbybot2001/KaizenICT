"""
Eval Policy Direct (Phase 5.2 Retrieval)
"""
import pandas as pd
import numpy as np
import os
from typing import Tuple
from ml_lab.engine.event_engine import EventEngine
from ml_lab.ml.feature_builder import FeatureBuilder
from ml_lab.ml.zone_relative import build_zone_relative_features
from ml_lab.ml.models.direct_policy import DirectPolicy
from ml_lab.ml.policy.concept_env_v3 import ConceptTradingEnvV3
from ml_lab.constants import MIN_SL_POINTS

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

def eval_policy(policy, env):
    print("Evaluating DirectPolicy on EnvV3...")
    obs, _ = env.reset()
    total_reward = 0.0
    trades = 0
    
    while True:
        action = policy.get_action(obs)
        if action != 0 : trades += 1
        
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done: break
        
    return total_reward, trades

def main():
    print("PHASE 5.2: EVALUATION ONLY")
    
    # 1. Load Data
    data = load_data()
    data = data.iloc[-50000:] # Same subset as training
    
    split_idx = int(len(data) * 0.7)
    
    # 2. Slice for Test (with warmup)
    warmup = 200
    test_data = data.iloc[split_idx - warmup:]
    
    # 3. Features
    print("Building features for Test Set...")
    fb = FeatureBuilder()
    prim = fb.build_feature_matrix(test_data)
    zone = build_zone_relative_features(test_data)
    
    # Remove warmup
    test_data = test_data.iloc[warmup:]
    prim = prim.iloc[warmup:]
    zone = zone.iloc[warmup:]
    
    # 4. Load Models
    print("Loading Policy...")
    policy = DirectPolicy()
    policy.load("ml_lab/runs/policy_v3_direct")
    
    # 5. Eval
    print("Running OOS Evaluation...")
    env = ConceptTradingEnvV3(test_data, prim, zone)
    
    reward, trades = eval_policy(policy, env)
    
    print("="*60)
    print(f"OOS RESULTS (DirectPolicy):")
    print(f"Total Reward: {reward:.2f} R")
    print(f"Total Trades: {trades}")
    if trades > 0:
        print(f"Avg Reward: {reward/trades:.4f} R/Trade")
    print("="*60)
    
    with open("ml_lab/runs/phase52_results.txt", "w") as f:
        f.write(f"Total Reward: {reward:.2f} R\n")
        f.write(f"Total Trades: {trades}\n")
        if trades > 0:
            f.write(f"Avg Reward: {reward/trades:.4f} R/Trade\n")

if __name__ == "__main__":
    main()
