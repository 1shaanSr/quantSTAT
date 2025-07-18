import pytz

class Config:
    EASTERN_TZ = pytz.timezone("US/Eastern")
    BASE_URL = "https://paper-api.alpaca.markets"
    API_VERSION = 'v2'
    
    COLORS = {
        'primary': '#00D4FF',
        'profit': '#00FF88',
        'loss': '#FF4444',
        'neutral': '#FFFFFF',
        'accent': '#FFD700',
        'grid': '#333333'
    }
    
    STRATEGY = {
        'default_symbol': 'SPY',
        'risk_pct': 0.01,
        'refresh_interval': 60,
        'rsi_period': 14,
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'atr_period': 14,
        'profit_factor': 1.5
    }