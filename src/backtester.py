import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Backtester:
    """
    Simple backtesting class compatible with your main.py and data model.
    Usage: backtester.run(symbol, days)
    """
    def __init__(self, api_handler):
        self.api = api_handler.api if hasattr(api_handler, 'api') else api_handler
        self.initial_balance = 10000
        self.current_balance = 10000
        self.risk_per_trade = 0.02
        self.max_positions = 1
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.drawdown = []

    def run(self, symbol, days):
        print(f"\n=== Backtesting {symbol} for {days} days ===")
        try:
            # Get account info for initial capital
            if hasattr(self.api, 'get_account'):
                account = self.api.get_account()
                self.initial_balance = float(getattr(account, 'buying_power', 10000))
            else:
                print("Using default balance of $10,000")
                self.initial_balance = 10000
            self.current_balance = self.initial_balance
            self.positions = []
            self.trades = []
            self.equity_curve = []
            self.drawdown = []

            # Get historical data (replace with your real data handler if available)
            df = self._create_sample_data(symbol, days * 5)
            if df is None or df.empty:
                print("No data available for backtest")
                return

            # Add indicators
            df = self._add_indicators(df)

            # Run simulation
            results = self._run_backtest(df)
            self._print_results(results)
        except Exception as e:
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()

    def _create_sample_data(self, symbol="SPY", days=252):
        # Simple sample data generator
        np.random.seed(hash(symbol) % 1000)
        dates = pd.date_range(datetime.now() - timedelta(days=int(days * 1.4)), periods=days, freq='B')
        prices = np.cumprod(1 + np.random.normal(0.0008, 0.02, days)) * 100
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': prices * (1 + np.random.uniform(0.001, 0.02, days)),
            'Low': prices * (1 - np.random.uniform(0.001, 0.02, days)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, days)
        })
        return df

    def _add_indicators(self, df):
        # Add basic indicators
        df['RSI'] = self._calculate_rsi(df)
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['ATR'] = self._calculate_atr(df)
        return df

    def _calculate_rsi(self, df, period=14):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _run_backtest(self, df):
        # Simple backtest loop
        trades = []
        position = None
        for i in range(15, len(df)):
            price = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            prev_rsi = df['RSI'].iloc[i-1]
            vwap = df['VWAP'].iloc[i]
            atr = df['ATR'].iloc[i]
            
            # Skip if any values are NaN
            if pd.isna(rsi) or pd.isna(prev_rsi) or pd.isna(vwap) or pd.isna(atr):
                continue
                
            # Entry logic
            if not position:
                if prev_rsi < 30 and rsi >= 30 and price < vwap:
                    side = 'buy'
                elif prev_rsi > 70 and rsi <= 70 and price > vwap:
                    side = 'sell'
                else:
                    continue
                risk_amount = self.current_balance * self.risk_per_trade
                qty = int(risk_amount // (atr if atr else 1))
                if qty < 1:
                    continue
                entry = price
                stop = entry - atr if side == 'buy' else entry + atr
                target = entry + 1.5 * atr if side == 'buy' else entry - 1.5 * atr
                position = {
                    'side': side,
                    'qty': qty,
                    'entry': entry,
                    'stop': stop,
                    'target': target
                }
            else:
                if position['side'] == 'buy':
                    if price <= position['stop'] or price >= position['target']:
                        pnl = (price - position['entry']) * position['qty']
                        trades.append(pnl)
                        self.current_balance += pnl
                        position = None
                elif position['side'] == 'sell':
                    if price >= position['stop'] or price <= position['target']:
                        pnl = (position['entry'] - price) * position['qty']
                        trades.append(pnl)
                        self.current_balance += pnl
                        position = None
        
        # Results
        total_trades = len(trades)
        win_trades = [t for t in trades if t > 0]
        loss_trades = [t for t in trades if t <= 0]
        win_rate = len(win_trades) / total_trades * 100 if total_trades else 0
        total_pnl = sum(trades)
        return_pct = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'return_pct': return_pct,
            'final_balance': self.current_balance
        }

    def _print_results(self, results):
        print("\n=== Backtest Results ===")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total P&L: ${results['total_pnl']:,.2f}")