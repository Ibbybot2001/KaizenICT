"""
Phase 3C: Policy Training Run

Trains a Q-Learning agent on the ConceptTradingEnv.
Compares against Baselines.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml.zone_relative import ZoneRelativeBuilder
from primitives.displacement import DisplacementDetector
from primitives.compression import CompressionDetector
from ml.envs.concept_env import ConceptTradingEnv
from ml.envs.q_learner import QLearner

def load_data():
    path = project_root / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
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

def run_episode(env, agent=None, mode='train'):
    # Mode: 'train' (explore/update), 'eval' (exploit/no-update), 'random'
    obs = env.reset()
    done = False
    total_reward = 0
    trades = 0
    
    while not done:
        if mode == 'random':
            action = env.action_space.sample()
            # Constrain random to valid entries if regime active?
            # Or strict random. Strict random likely loses money fast.
            # Let's simple-random everything.
        elif agent:
            # Mask invalid actions? Env penalizes them, but agent learns.
            # QLearner handles epsilon internally if we access that logic
            # But standard QLearner.get_action does checks
            action = agent.get_action(obs)
        else:
            action = 0 # Hold
            
        next_obs, reward, done, _ = env.step(action)
        
        if mode == 'train' and agent:
            agent.update(obs, action, reward, next_obs)
            
        if action in [1, 2] and reward <= -0.1: # Count entries (approx via cost)
             trades += 1
             
        total_reward += reward
        obs = next_obs
        
    return total_reward

def main():
    print("PHASE 3C: POLICY LEARNING")
    
    # 1. Data
    data = load_data().iloc[:20000] # 20k bars dev
    prim, zone = build_features(data)
    
    # 2. Env
    env = ConceptTradingEnv(data, prim, zone)
    
    # 3. Agent
    # Dims: Regime(3), Feat1(5), Feat2(10) -> (3, 5, 10, 4)
    agent = QLearner((3, 5, 10), 4)
    
    # 4. Baselines
    print("\nEvaluating Baselines (10 eps)...")
    rand_rewards = []
    for _ in range(10):
        rand_rewards.append(run_episode(env, mode='random'))
    print(f"Random Mean Reward: {np.mean(rand_rewards):.4f}")
    
    # 5. Training
    print("\nTraining Agent (100 eps)...")
    history = []
    
    for ep in range(100):
        r = run_episode(env, agent, mode='train')
        agent.decay_epsilon()
        history.append(r)
        
        if ep % 10 == 0:
            print(f"Ep {ep}: R={r:.4f}, eps={agent.epsilon:.3f}")
            
    # 6. Final Evaluation
    print("\nEvaluating Trained Policy (10 eps)...")
    agent.epsilon = 0 # Greedy
    eval_rewards = []
    for _ in range(10):
        eval_rewards.append(run_episode(env, agent, mode='eval'))
        
    print(f"Agent Mean Reward: {np.mean(eval_rewards):.4f}")
    
    # Compare
    delta = np.mean(eval_rewards) - np.mean(rand_rewards)
    print(f"\nImprovement vs Random: {delta:+.4f}")
    
    # Save Model
    model_path = project_root / 'output' / 'models' / 'policy_v1.pkl'
    agent.save(model_path)
    print(f"Policy saved to {model_path}")

if __name__ == '__main__':
    main()
