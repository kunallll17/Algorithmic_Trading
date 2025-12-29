"""
Data preprocessing module for calculating technical indicators.
"""
import pandas as pd
import ta
from typing import Optional


class Preprocessor:
    """Handles data preprocessing and technical indicator calculation."""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list = [20, 50, 200]) -> pd.DataFrame:
        """Add moving averages to the dataframe."""
        df = df.copy()
        for period in periods:
            df[f"MA{period}"] = df["close"].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add RSI (Relative Strength Index) indicator."""
        df = df.copy()
        rsi_indicator = ta.momentum.RSIIndicator(close=df["close"], window=window)
        df["RSI"] = rsi_indicator.rsi()
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicator."""
        df = df.copy()
        macd_indicator = ta.trend.MACD(
            close=df["close"],
            window_fast=fast,
            window_slow=slow,
            window_sign=signal
        )
        df["MACD"] = macd_indicator.macd()
        df["MACD_signal"] = macd_indicator.macd_signal()
        df["MACD_diff"] = macd_indicator.macd_diff()
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Add Bollinger Bands."""
        df = df.copy()
        bb_indicator = ta.volatility.BollingerBands(
            close=df["close"],
            window=window,
            window_dev=num_std
        )
        df["BB_high"] = bb_indicator.bollinger_hband()
        df["BB_low"] = bb_indicator.bollinger_lband()
        df["BB_mid"] = bb_indicator.bollinger_mavg()
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        df = df.copy()
        df["Volume_MA"] = df["volume"].rolling(window=20).mean()
        return df
    
    @staticmethod
    def preprocess_data(
        df: pd.DataFrame,
        indicators: Optional[list] = None,
        dropna: bool = True
    ) -> pd.DataFrame:
        """
        Apply all preprocessing steps to the dataframe.
        
        Args:
            df: Input dataframe with OHLCV data
            indicators: List of indicators to add. If None, adds all.
            dropna: Whether to drop rows with NaN values
        
        Returns:
            Preprocessed dataframe
        """
        df = df.copy()
        
        if indicators is None:
            indicators = ["MA", "RSI", "MACD", "BB", "Volume"]
        
        if "MA" in indicators:
            df = Preprocessor.add_moving_averages(df)
        
        if "RSI" in indicators:
            df = Preprocessor.add_rsi(df)
        
        if "MACD" in indicators:
            df = Preprocessor.add_macd(df)
        
        if "BB" in indicators:
            df = Preprocessor.add_bollinger_bands(df)
        
        if "Volume" in indicators:
            df = Preprocessor.add_volume_indicators(df)
        
        if dropna:
            df = df.dropna()
        
        return df




