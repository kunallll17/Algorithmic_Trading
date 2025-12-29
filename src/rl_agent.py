"""
Reinforcement Learning module for algorithmic trading.
Uses Gymnasium and Stable Baselines 3.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

class TradingEnv(gym.Env):
    """
    Custom Trading Environment that follows gymnasium interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission=0.001):
        super(TradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: 
        # [Close Price, Volume, RSI, MACD, Signal, Hist, SMA_20, SMA_50, Balance, Position, Net Worth]
        # We assume the dataframe already has technical indicators
        # But for robustness, we'll select specific columns or add them
        
        # Let's ensure we have basic features. If not, we should probably calculate them in preprocessor.
        # For now, we'll try to use: Close, Volume, Open, High, Low
        # We need a fixed size observation
        # For simplicity, let's use:
        # 1. Close Price (normalized)
        # 2. Volume (normalized by max)
        # 3. Balance (normalized by initial)
        # 4. Position (0 or 1, or fractional)
        # 5. Profit/Loss
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        
        self.current_step = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get the current price
        current_price = self.df.iloc[self.current_step]['close']
        volume = self.df.iloc[self.current_step]['volume']
        
        obs = np.array([
            current_price / 200, # crude normalization
            volume / 1000000,
            self.balance / self.initial_balance,
            self.shares_held,
            (self.net_worth - self.initial_balance) / self.initial_balance, 
            self.cost_basis / 200 # crude normalization
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            terminated = True
        else:
            terminated = False
            
        reward = self.net_worth - self.initial_balance # Simple reward: total profit
        # A better reward might be step-based return or Sharpe Ratio
        # But for a simple demo, net profit is okay.
        
        # Ideally, we want step-change in net worth as reward
        # reward = self.net_worth - self.prev_net_worth
        
        obs = self._next_observation()
        info = {'net_worth': self.net_worth}
        
        return obs, reward, terminated, False, info

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # 0: Hold
        # 1: Buy
        # 2: Sell
        
        if action == 1: # Buy
            # Buy with 10% of balance
            amount_to_invest = self.balance * 0.1
            if amount_to_invest > current_price:
                shares = int(amount_to_invest / current_price)
                cost = shares * current_price
                comm = cost * self.commission
                
                self.balance -= (cost + comm)
                self.shares_held += shares
                self.cost_basis = current_price # Simplified
                
        elif action == 2: # Sell
            # Sell all
            if self.shares_held > 0:
                sale_value = self.shares_held * current_price
                comm = sale_value * self.commission
                
                self.balance += (sale_value - comm)
                self.shares_held = 0
                self.cost_basis = 0
                
        # Update net worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def render(self, mode='human', close=False):
        price = self.df.iloc[self.current_step]['close']
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares: {self.shares_held}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Price: {price}')


def train_rl_model(df, symbol, total_timesteps=10000):
    """
    Trains a PPO agent on the provided DataFrame.
    """
    print(f"Training RL Agent for {symbol}...")
    
    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(df)])
    
    # Instantiate the agent
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    
    # Save the agent
    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_{symbol}"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def evaluate_model(model, df):
    """
    Evaluates the trained model.
    """
    env = DummyVecEnv([lambda: TradingEnv(df)])
    obs = env.reset()
    
    total_rewards = 0
    done = False
    
    print("Evaluating model...")
    step_count = 0
    max_steps = len(df) - 1
    
    while not done and step_count < max_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_rewards += rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        step_count += 1
        
    return total_rewards
