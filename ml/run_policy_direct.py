"""
Phase 5 Track A: Direct Policy Training (XGBoost Bridge)

1. Load Data/Features.
2. Generate Labels: Did price reach 1R before SL? (Using simple lookahed for labeling).
   - Actually, using LabelGenerator "Hit TP before SL" with R=1.
3. Train DirectPolicy (Long/Short).
4. Run Eval Loop using ConceptTradingEnvV3.
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
        # Ensure proper types
        return df
    else: 
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")

def generate_labels(df: pd.DataFrame, sl_pts: float = 10.0, tp_pts: float = 10.0) -> Tuple[pd.Series, pd.Series]:
    """
    Generate target labels: 1 if TP hit before SL, 0 if SL hit or neither.
    """
    n = len(df)
    labels_long = pd.Series(index=df.index, dtype=float)
    labels_short = pd.Series(index=df.index, dtype=float)
    
    print("Generating labels (future lookahead)...")
    # Vectorized approximation or efficient loop? Loop is safer for TP/SL logic
    # Using window to speed up
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Simple window check (e.g. 60 bars)
    HORIZON = 60 
    
    for i in range(n - HORIZON):
        if i % 10000 == 0: print(f"  Labeling {i}/{n}")
        
        entry = df.iloc[i+1]['open'] # Trade at next open
        
        # Long Logic
        sl_price = entry - sl_pts
        tp_price = entry + tp_pts # 1R Target (conservative)
        
        # Check window
        win = 0
        for j in range(1, HORIZON):
            row = df.iloc[i+j] # Look at future bar
            if row['low'] <= sl_price:
                win = 0
                break
            if row['high'] >= tp_price:
                win = 1
                break
        labels_long.iloc[i] = win
        
        # Short Logic
        sl_price = entry + sl_pts
        tp_price = entry - tp_pts
        
        win = 0
        for j in range(1, HORIZON):
            row = df.iloc[i+j]
            if row['high'] >= sl_price:
                win = 0
                break
            if row['low'] <= tp_price:
                win = 1
                break
        labels_short.iloc[i] = win
            
    return labels_long, labels_short

def eval_policy(policy, env):
    print("Evaluating DirectPolicy on EnvV3...")
    obs, _ = env.reset()
    total_reward = 0.0
    trades = 0
    
    while True:
        action = policy.get_action(obs)
        if action != 0 : trades += 1 # Count entries
        
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        if done: break
        
    return total_reward, trades

def main():
    print("PHASE 5.2: DIRECT POLICY TRAINING (GBM BRIDGE)")
    
    # 1. Data
    data = load_data()
    # Use subset for dev speed? No, full data for robust model
    data = data.iloc[-50000:] 
    
    # 2. Features
    print("Building features...")
    fb = FeatureBuilder()
    prim = fb.build_feature_matrix(data)
    zone = build_zone_relative_features(data)
    
    # 3. Labels
    print("Generating labels...")
    lbl_long, lbl_short = generate_labels(data, sl_pts=10.0, tp_pts=15.0) # Aim for 1.5R to beat friction comfortably
    
    # 4. Train
    # Split Train/Test
    split_idx = int(len(data) * 0.7)
    
    policy = DirectPolicy()
    policy.train(
        data.iloc[:split_idx],
        prim.iloc[:split_idx],
        zone.iloc[:split_idx],
        lbl_long.iloc[:split_idx],
        lbl_short.iloc[:split_idx]
    )
    
    # 5. Eval
    print("Running OOS Evaluation...")
    env = ConceptTradingEnvV3(
        data.iloc[split_idx:],
        prim.iloc[split_idx:],
        zone.iloc[split_idx:]
    )
    
    reward, trades = eval_policy(policy, env)
    print("="*60)
    print(f"OOS RESULTS (DirectPolicy):")
    print(f"Total Reward: {reward:.2f} R")
    print(f"Total Trades: {trades}")
    if trades > 0:
        print(f"Avg Reward: {reward/trades:.4f} R/Trade")
    print("="*60)
    
    policy.save("ml_lab/runs/policy_v3_direct")

if __name__ == "__main__":
    main()
