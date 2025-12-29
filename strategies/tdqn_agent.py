from .base_strategy import BaseStrategy
from .data_augmentation import DataAugmentation
import pandas as pd
import numpy as np
import os
import random
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# --- Custom Trading Env for TDQN ---
class TDQNTradingEnv(gym.Env):
    """
    Custom Trading Environment for TDQN Agent.
    Supports training on multiple trajectories (e.g. artificial slices).
    On reset, picks a random trajectory.
    """
    def __init__(self, data_source, initial_balance=10000):
        super(TDQNTradingEnv, self).__init__()
        
        # data_source can be a single DF or a list of DFs
        self.trajectories = data_source if isinstance(data_source, list) else [data_source]
        self.initial_balance = initial_balance
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 10 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Current episode state
        self.df = None
        self.current_step = 0
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random trajectory for this episode
        self.df = random.choice(self.trajectories)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        
        return self._next_observation(), {}
        
    def _next_observation(self):
        obs = np.zeros(10, dtype=np.float32)
        if self.current_step < len(self.df):
            row = self.df.iloc[self.current_step]
            # Simple Normalization
            # Features: Close, Volume, Open, High, Low, Balance, Shares, NetWorth
            # We strictly need relative inputs for generalization.
            # Using % change or log returns is better, but obeying simplistic Paper 1 idea:
            
            c = row['close']
            obs[0] = c / 1000.0 
            obs[1] = row['volume'] / 1e6 if 'volume' in row else 0
            obs[2] = self.shares_held
            obs[3] = self.balance / 10000.0
            # ... fill others as needed
            
        return obs
        
    def step(self, action):
        # Execute Action
        current_price = self.df.iloc[self.current_step]['close']
        
        # 1=Buy, 2=Sell
        if action == 1:
            # Invest 10%
            cost = self.balance * 0.1
            if cost > current_price:
                n_shares = int(cost / current_price)
                self.balance -= n_shares * current_price
                self.shares_held += n_shares
        elif action == 2:
            # Sell all
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        
        # Update Net Worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Reward: Change in Net Worth (Profit)
        # Scaled for stability
        prev_net = (self.initial_balance) # Simplification, ideally track prev step
        # Let's just reward total return at end? or step return?
        # SB3 likes dense rewards.
        reward = (self.net_worth - self.initial_balance) / self.initial_balance
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        
        return self._next_observation(), reward, terminated, False, {}


class TDQNAgent(BaseStrategy):
    """
    RL Agent Strategy using Stable Baselines 3 DQN.
    """
    def __init__(self, model_path="models/tdqn_model"):
        super().__init__("TDQN (AI Agent)")
        self.model_path = model_path
        self.model = None
            
    def train(self, data_dict, total_timesteps=10000):
        print("Training TDQN Agent with Data Augmentation...")
        
        # Generate Artificial Trajectories from ALL available assets
        all_trajectories = []
        for symbol, df in data_dict.items():
            # Paper 1 suggests splitting history into windows
            trajs = DataAugmentation.generate_trajectories(df, window_size=252, step_size=60)
            all_trajectories.extend(trajs)
            
        print(f"Training on {len(all_trajectories)} augmented trajectories.")
        
        env = DummyVecEnv([lambda: TDQNTradingEnv(all_trajectories)])
        
        self.model = DQN("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def generate_signal(self, current_date, data_dict, current_positions, portfolio_value):
        """
        Inference: Uses the model to predict action.
        """
        # If no model, random or hold
        if self.model is None:
            # Try load
            try:
                self.model = DQN.load(self.model_path)
            except:
                return [] # No model yet
        
        orders = []
        # For each asset, construct observations and predict
        for symbol, df in data_dict.items():
            # Get data locally
            # Construct state (must match Env state def)
            obs = np.zeros(10, dtype=np.float32) 
            # fill obs ...
            
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Map action to Order
            # 0=Hold, 1=Buy, 2=Sell
            if action == 1:
                orders.append({'symbol': symbol, 'action': 'buy', 'cash_amount': portfolio_value * 0.1})
            elif action == 2:
                orders.append({'symbol': symbol, 'action': 'sell'})
                
        return orders
