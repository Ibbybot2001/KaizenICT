"""
Run Policy Training (V2 - Resolution Upgrade)
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
from ml_lab.ml.policy.concept_env_v2 import ConceptTradingEnvV2
from ml_lab.ml.policy.q_learner import QLearner

def load_data():
    path = project_root / 'ml_lab' / 'data' / 'kaizen_1m_data_ibkr_2yr.csv'
    df = pd.read_csv(path, parse_dates=['time'])
    df = df.set_index('time')
    df.columns = df.columns.str.lower()
    return df

def build_features(data):
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

def run_episode(env, agent=None, mode='train', epsilon=0.1):
    obs = env.reset()
    done = False
    total_reward = 0
    trades = 0
    
    while not done:
        if mode == 'train':
            # Epsilon greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.q_table[obs])
        elif mode == 'random':
            action = env.action_space.sample()
        else: # eval
            action = np.argmax(agent.q_table[obs])
            
        # Track Intentional Trades (State Transition 0->1)
        prev_pos = env.position
        
        next_obs, reward, done, _ = env.step(action)
        
        # New trade count logic: strictly transition from 0 to non-0
        if prev_pos == 0 and env.position != 0:
            trades += 1
            
        if mode == 'train':
            agent.update(obs, action, reward, next_obs)
            
        total_reward += reward
        obs = next_obs
        
    return total_reward, trades

def main():
    print("PHASE 4.3: RELAXED CONSTRAINTS TRAINING (NO ENTRY COST)")
    
    data = load_data().iloc[:20000] 
    prim, zone = build_features(data)
    
    env = ConceptTradingEnvV2(data, prim, zone)
    
    # Obs Space V2: (Regime=3, Feat1=3, Feat2=7, Feat3=6)
    # Total States: 3 * 3 * 7 * 6 = 378
    agent = QLearner((3, 3, 7, 6), 4, alpha=0.1, gamma=0.95)
    
    print("\nTraining Agent V2 (150 eps)...")
    epsilon = 1.0
    decay = 0.97
    min_eps = 0.05
    
    for ep in range(150):
        reward, trades = run_episode(env, agent, mode='train', epsilon=epsilon)
        epsilon = max(min_eps, epsilon * decay)
        
        if ep % 10 == 0:
            print(f"Ep {ep}: R={reward:.4f}, Trades={trades}, eps={epsilon:.3f}")
            
    # Save Policy V2
    out_path = project_root / 'ml_lab' / 'runs' / 'policy_v2_resolution.pkl'
    agent.save(out_path)
    print(f"\nSaved Policy V2 to {out_path}")
    
    # Quick Eval
    print("\nEvaluating Agent V2 (Greedy)...")
    rewards = []
    total_trades = 0
    for _ in range(10):
        r, t = run_episode(env, agent, mode='eval')
        rewards.append(r)
        total_trades += t
        
    print(f"Mean Reward: {np.mean(rewards):.4f}")
    print(f"Mean Trades: {np.mean(total_trades)/10:.1f}")

if __name__ == '__main__':
    main()
