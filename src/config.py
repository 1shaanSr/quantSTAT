"""
Configuration Management for QuantRSI Trading Platform
-----------------------------------------------------
Centralized configuration handling for all trading parameters,
API settings, and risk management rules.

Author: Professional Trading Systems
Date: July 2025
"""

import os
import pytz
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class TradingConfig:
    """Core trading configuration parameters."""
    
    # Account & Risk Management
    initial_balance: float = 10000.0
    risk_per_trade: float = 0.02
    max_positions: int = 3
    max_daily_trades: int = 10
    
    # Statistical Arbitrage Parameters
    lookback_period: int = 15
    entry_zscore: float = 0.7
    exit_zscore: float = 0.1
    stop_loss_zscore: float = 1.5
    profit_target_zscore: float = 0.05
    
    # Pair Selection Criteria
    min_correlation: float = 0.3
    max_correlation: float = 0.95
    cointegration_pvalue_threshold: float = 0.05
    
    # Position Management
    min_holding_period: int = 1  # days
    max_holding_period: int = 3  # days
    position_sizing_method: str = "fixed_risk"
    
    # Market Data
    data_source: str = "alpaca"
    trading_hours_only: bool = True
    market_timezone: str = "America/New_York"

@dataclass
class APIConfig:
    """API connection configuration."""
    
    # Alpaca API Settings
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: str = "https://paper-api.alpaca.markets"
    
    # Connection Settings
    timeout_seconds: int = 30
    retry_attempts: int = 3
    rate_limit_buffer: float = 0.1

class Config:
    """Legacy configuration class for backward compatibility."""
    
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

class ConfigManager:
    """
    Centralized configuration management system.
    Handles loading from environment variables, config files, and defaults.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file
        self.trading_config = TradingConfig()
        self.api_config = APIConfig()
        
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from multiple sources with priority order."""
        
        # 1. Load from config file if provided
        if self.config_file and os.path.exists(self.config_file):
            self._load_from_file()
        
        # 2. Override with environment variables
        self._load_from_environment()
        
        # 3. Validate configuration
        self._validate_configuration()
    
    def _load_from_file(self) -> None:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update trading config
            if 'trading' in config_data:
                for key, value in config_data['trading'].items():
                    if hasattr(self.trading_config, key):
                        setattr(self.trading_config, key, value)
            
            # Update API config
            if 'api' in config_data:
                for key, value in config_data['api'].items():
                    if hasattr(self.api_config, key):
                        setattr(self.api_config, key, value)
                        
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _load_from_environment(self) -> None:
        """Load sensitive configuration from environment variables."""
        
        # API credentials (never store in code/config files)
        self.api_config.api_key = os.getenv('ALPACA_API_KEY')
        self.api_config.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Optional environment overrides
        if os.getenv('ALPACA_BASE_URL'):
            self.api_config.base_url = os.getenv('ALPACA_BASE_URL')
        
        if os.getenv('INITIAL_BALANCE'):
            try:
                self.trading_config.initial_balance = float(os.getenv('INITIAL_BALANCE'))
            except ValueError:
                pass
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters for safety."""
        
        # Risk management validation
        if self.trading_config.risk_per_trade > 0.1:
            print("Warning: Risk per trade > 10% is extremely aggressive")
        
        if self.trading_config.max_positions > 10:
            print("Warning: More than 10 concurrent positions may be difficult to manage")
    
    def get_trading_config(self) -> TradingConfig:
        """Get current trading configuration."""
        return self.trading_config
    
    def get_api_config(self) -> APIConfig:
        """Get current API configuration."""
        return self.api_config
    
    def print_configuration(self) -> None:
        """Print current configuration (excluding sensitive data)."""
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION")
        print("="*50)
        
        print("\nTrading Parameters:")
        print(f"  Initial Balance: ${self.trading_config.initial_balance:,.2f}")
        print(f"  Risk Per Trade: {self.trading_config.risk_per_trade:.1%}")
        print(f"  Max Positions: {self.trading_config.max_positions}")
        
        print("\nStatistical Arbitrage:")
        print(f"  Entry Z-Score: ±{self.trading_config.entry_zscore}")
        print(f"  Exit Z-Score: ±{self.trading_config.exit_zscore}")
        print(f"  Stop Loss Z-Score: ±{self.trading_config.stop_loss_zscore}")
        
        print("="*50)

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager

def get_trading_config() -> TradingConfig:
    """Get current trading configuration."""
    return config_manager.get_trading_config()

def get_api_config() -> APIConfig:
    """Get current API configuration."""
    return config_manager.get_api_config()