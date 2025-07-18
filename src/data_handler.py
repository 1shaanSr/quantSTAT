import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def add_indicators(df):
    """Add all technical indicators to the dataframe"""
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # VWAP calculation
    df['VWAP'] = ((df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum())
    
    # ATR calculation
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Trading signals
    df['Strong_Buy'] = (
        (df['RSI'] < 30) & 
        (df['Close'] < df['VWAP']) & 
        (df['RSI'].shift(1) >= 30)
    )
    
    df['Strong_Sell'] = (
        (df['RSI'] > 70) & 
        (df['Close'] > df['VWAP']) & 
        (df['RSI'].shift(1) <= 70)
    )
    
    return df

class DataHandler:
    @staticmethod
    def fetch_market_data(symbol, days_back=30, interval='5m'):
        try:
            end = datetime.now()
            start = end - timedelta(days=days_back)
            
            df = yf.download(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                interval=interval,
                progress=False,
                auto_adjust=False
            )
            
            if df.empty:
                return None
                
            return add_indicators(df)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None