import pandas as pd
import numpy as np

class DataAugmentation:
    """
    Implements 'Artificial Trajectory Generation' from Paper 1.
    Splits long financial time series into shorter, overlapping episodes 
    to provide more training data for the RL agent.
    """
    
    @staticmethod
    def generate_trajectories(df, window_size=252, step_size=20):
        """
        Slices a dataframe into multiple overlapping windows.
        
        Args:
            df (pd.DataFrame): The history source.
            window_size (int): Length of each trajectory (e.g., 1 year).
            step_size (int): How many distinct days to shift for next window.
            
        Returns:
            list of pd.DataFrame: The slices.
        """
        trajectories = []
        
        if len(df) < window_size:
            return [df]
            
        for i in range(0, len(df) - window_size + 1, step_size):
            slice_df = df.iloc[i : i + window_size].copy()
            # Normalize? Paper suggests normalizing each trajectory independently
            # so the agent learns patterns, not absolute price levels.
            # We'll leave normalization to the Environment/Agent logic, 
            # here we just slice.
            trajectories.append(slice_df)
            
        print(f"Generated {len(trajectories)} trajectories from data length {len(df)}")
        return trajectories
