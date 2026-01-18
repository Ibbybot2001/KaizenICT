"""
Q-Learning Agent (Tabular)

Simple discrete reinforcement learning agent.
Uses a Q-table to map (Regime, Feat1, Feat2) -> Action Values.
"""

import numpy as np
import pickle
from pathlib import Path

class QLearner:
    def __init__(self, state_dims, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = np.zeros(state_dims + (n_actions,))
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.n_actions = n_actions
        
    def get_action(self, state):
        # State is tuple (r, f1, f2)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions) # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit
            
    def update(self, state, action, reward, next_state):
        old_val = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        
        # Q-Learning update rule
        new_val = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_val
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
