import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from .indicators import add_indicators

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