from .data_handler import DataHandler
from .config import Config
from datetime import datetime
import pandas as pd
import numpy as np

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
            
            # Debug: Check dataframe structure
            print(df.head())
            print("Columns:", df.columns)
            
            # Flatten columns if multi-indexed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() for col in df.columns.values]
                print("Flattened Columns:", df.columns)

            # Detect close column dynamically
            close_col = None
            for col in df.columns:
                if 'close' in col.lower():
                    close_col = col
                    break

            if close_col is None:
                raise Exception("No close column found in dataframe")

            # Run backtest
            results = self._run_simulation(df, capital, close_col)
            
            # Print results
            self._print_results(results, starting_capital)
            
        except Exception as e:
            print(f"Backtest error: {e}")

    def _run_simulation(self, df, capital, close_col):
        """
        Simple backtest:
        - Buy if yesterday closed higher than the day before
        - Sell if yesterday closed lower than the day before
        """
        position = 0
        trades = []

        df['Prev_Close'] = df[close_col].shift(1)
        df['Signal'] = np.where(df[close_col] > df['Prev_Close'], 1, -1)
        df = df.dropna()

        for idx, row in df.iterrows():
            price = row[close_col]
            signal = row['Signal']

            if signal == 1 and position == 0:
                # Buy full capital worth
                position = capital / price
                trades.append({'Date': idx, 'Type': 'BUY', 'Price': price})
                capital = 0
            elif signal == -1 and position > 0:
                # Sell all
                capital = position * price
                trades.append({'Date': idx, 'Type': 'SELL', 'Price': price})
                position = 0

        # Close open position at end
        if position > 0:
            final_price = df[close_col].iloc[-1]
            capital = position * final_price
            trades.append({'Date': df.index[-1], 'Type': 'SELL', 'Price': final_price})

        # Calculate metrics
        trade_df = pd.DataFrame(trades)
        buy_prices = trade_df[trade_df['Type'] == 'BUY']['Price'].values
        sell_prices = trade_df[trade_df['Type'] == 'SELL']['Price'].values

        num_trades = min(len(buy_prices), len(sell_prices))
        if num_trades == 0:
            win_rate = 0
            profit_factor = 0
        else:
            profits = sell_prices[:num_trades] - buy_prices[:num_trades]
            wins = profits[profits > 0].sum()
            losses = -profits[profits < 0].sum()
            win_rate = (profits > 0).sum() / num_trades * 100
            profit_factor = wins / losses if losses != 0 else np.inf

        return {
            'trades': trades,
            'final_capital': capital,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

    def _print_results(self, results, starting_capital):
        print("\n=== Backtest Results ===")
        print(f"Starting Capital: ${starting_capital:,.2f}")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Return: {((results['final_capital']/starting_capital)-1)*100:.2f}%")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
