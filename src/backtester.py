from .data_handler import DataHandler
from .config import Config
from datetime import datetime
import pandas as pd

class Backtester:
    def __init__(self, api_handler):
        self.api = api_handler.api
        self.data_handler = DataHandler()
        self.risk_pct = Config.STRATEGY['risk_pct']

    def run(self, symbol, days):
        print(f"\n=== Backtesting {symbol} for {days} days ===")
        
        try:
            # Get account info for initial capital
            account = self.api.get_account()
            starting_capital = float(account.buying_power)
            capital = starting_capital
            
            # Get historical data
            df = self.data_handler.fetch_market_data(symbol, days)
            if df is None or df.empty:
                print("No data available for backtest")
                return
            
            # Run backtest
            results = self._run_simulation(df, capital)
            
            # Print results
            self._print_results(results, starting_capital)
            
        except Exception as e:
            print(f"Backtest error: {e}")

    def _run_simulation(self, df, capital):
        # Implementation of your backtest logic here
        return {
            'trades': [],
            'final_capital': capital,
            'win_rate': 0,
            'profit_factor': 0
        }

    def _print_results(self, results, starting_capital):
        print("\n=== Backtest Results ===")
        print(f"Starting Capital: ${starting_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Return: {((results['final_capital']/starting_capital)-1)*100:.2f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")