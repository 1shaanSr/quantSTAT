import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_vwap(data):
    return ((data['Close'] * data['Volume']).cumsum() / 
            data['Volume'].cumsum())

def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def add_indicators(df):
    """Add all technical indicators to the dataframe"""
    df['RSI'] = calculate_rsi(df)
    df['VWAP'] = calculate_vwap(df)
    df['ATR'] = calculate_atr(df)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    return df