"""
Evaluate Saved Policy v1
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pickle

from ml_lab.ml.zone_relative import ZoneRelativeBuilder
from ml_lab.primitives.displacement import DisplacementDetector
from ml_lab.primitives.compression import CompressionDetector
from ml_lab.ml.policy.concept_env import ConceptTradingEnv
from ml_lab.ml.policy.q_learner import QLearner

def load_data():
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    return df

def build_features(data):
    # Reuse previous logic
    print("Building features...")
    zone_builder = ZoneRelativeBuilder()
    z = zone_builder.build_features(data)
    
    p = pd.DataFrame(index=data.index)
    
    disp = DisplacementDetector(2.0)
    comp = CompressionDetector()
    
    p['is_displacement'] = False
    p['disp_zscore'] = 0.0
    p['is_compressed'] = False
    p['comp_score'] = 0.0
    
    for i in range(len(data)):
        d = disp.compute_at(i, data)
        if d: 
            p.iloc[i, p.columns.get_loc('is_displacement')] = d.is_displacement
            p.iloc[i, p.columns.get_loc('disp_zscore')] = d.range_zscore
            
        c = comp.compute_at(i, data)
        if c:
            p.iloc[i, p.columns.get_loc('is_compressed')] = c.is_compressed
            p.iloc[i, p.columns.get_loc('comp_score')] = c.compression_score
            
    return p, z

def run_episode(env, agent=None, mode='eval'):
    obs = env.reset()
    done = False
    total_reward = 0
    trades = 0
    
    while not done:
        if mode == 'random':
            action = env.action_space.sample()
        elif agent:
            # Eval mode: greedy
            action = np.argmax(agent.q_table[obs]) 
        else:
            action = 0
            
        next_obs, reward, done, _ = env.step(action)
        
        if action in [1, 2] and reward <= -0.1:
             trades += 1
             
        total_reward += reward
        obs = next_obs
        
    return total_reward, trades

def main():
    print("PHASE 3C: POLICY EVALUATION")
    
    # 1. Data
    data = load_data().iloc[:20000] # Same subset
    prim, zone = build_features(data)
    
    # 2. Env
    env = ConceptTradingEnv(data, prim, zone)
    
    # 3. Load Agent
    model_path = project_root / 'ml_lab' / 'runs' / 'policy_v1.pkl'
    print(f"Loading {model_path}...")
    
    agent = QLearner((3, 5, 10), 4)
    agent.load(model_path)
    
    # 4. Baselines
    print("\nRunning Random Baseline (10 eps)...")
    rand_rewards = []
    rand_trades = []
    for _ in range(10):
        r, t = run_episode(env, mode='random')
        rand_rewards.append(r)
        rand_trades.append(t)
        
    print(f"Random Mean Reward: {np.mean(rand_rewards):.4f}")
    print(f"Random Mean Trades: {np.mean(rand_trades):.1f}")
    
    # 5. Agent
    print("\nRunning Agent (Greedy) (10 eps)...")
    agent_rewards = []
    agent_trades = []
    for _ in range(10):
        r, t = run_episode(env, agent=agent, mode='eval')
        agent_rewards.append(r)
        agent_trades.append(t)
        
    print(f"Agent Mean Reward: {np.mean(agent_rewards):.4f}")
    print(f"Agent Mean Trades: {np.mean(agent_trades):.1f}")
    
    delta = np.mean(agent_rewards) - np.mean(rand_rewards)
    print(f"\nImprovement vs Random: {delta:+.4f}")

if __name__ == '__main__':
    main()
