import alpaca_trade_api as tradeapi
import getpass
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import yfinance as yf
import pytz
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TradingDashboard:
    def execute_trade(self):
        print("\n=== Manual Trade Execution ===")
        print("Select trade type:")
        print("1. Buy (go long)")
        print("2. Sell (close owned shares)")
        print("3. Short (sell to open)")
        print("4. Exit")
        trade_type = input("Enter choice (1-4): ").strip()
        if trade_type not in ("1", "2", "3"):
            print("Trade cancelled.")
            return
        symbol = input("Enter symbol to trade: ").strip().upper()
        if not symbol:
            print("Invalid input. Trade cancelled.")
            return
        # Option 1: Buy (go long)
        if trade_type == "1":
            qty = input("Enter quantity to BUY: ").strip()
            if not qty:
                print("Invalid input. Trade cancelled.")
                return
            try:
                qty = int(qty)
            except ValueError:
                print("Quantity must be an integer.")
                return
            confirm = input(f"Confirm BUY (long) {qty} shares of {symbol}? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Trade cancelled.")
                return
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Order submitted! ID: {order.id}")
            except Exception as e:
                print(f"Trade error: {e}")
            return
        # Option 2: Sell (close owned shares)
        elif trade_type == "2":
            try:
                positions = self.api.list_positions()
                owned = next((p for p in positions if p.symbol.upper() == symbol and p.side == 'long'), None)
                if not owned:
                    print(f"No owned (long) shares of {symbol} to sell.")
                    return
                max_qty = int(float(owned.qty))
                print(f"You own {max_qty} shares of {symbol}.")
                qty = input(f"Enter quantity to SELL (max {max_qty}): ").strip()
                try:
                    qty = int(qty)
                except ValueError:
                    print("Quantity must be an integer.")
                    return
                if qty < 1 or qty > max_qty:
                    print("Invalid quantity.")
                    return
                confirm = input(f"Confirm SELL (close) {qty} shares of {symbol}? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Trade cancelled.")
                    return
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Order submitted! ID: {order.id}")
            except Exception as e:
                print(f"Trade error: {e}")
            return
        # Option 3: Short (sell to open) or exit short (buy to cover)
        elif trade_type == "3":
            try:
                positions = self.api.list_positions()
                short_pos = next((p for p in positions if p.symbol.upper() == symbol and p.side == 'short'), None)
                if short_pos:
                    max_qty = int(float(short_pos.qty))
                    print(f"You have an open SHORT of {max_qty} shares of {symbol}.")
                    close_short = input(f"Do you want to BUY TO COVER (exit short) these shares? (y/n): ").strip().lower()
                    if close_short == 'y':
                        qty = input(f"Enter quantity to BUY TO COVER (max {max_qty}): ").strip()
                        try:
                            qty = int(qty)
                        except ValueError:
                            print("Quantity must be an integer.")
                            return
                        if qty < 1 or qty > max_qty:
                            print("Invalid quantity.")
                            return
                        confirm = input(f"Confirm BUY TO COVER {qty} shares of {symbol}? (y/n): ").strip().lower()
                        if confirm != 'y':
                            print("Trade cancelled.")
                            return
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side="buy",
                            type="market",
                            time_in_force="gtc"
                        )
                        print(f"Order submitted! ID: {order.id}")
                        return
                # If no open short, allow to open new short
                qty = input("Enter quantity to SHORT (sell to open): ").strip()
                if not qty:
                    print("Invalid input. Trade cancelled.")
                    return
                try:
                    qty = int(qty)
                except ValueError:
                    print("Quantity must be an integer.")
                    return
                confirm = input(f"Confirm SHORT (sell to open) {qty} shares of {symbol}? (y/n): ").strip().lower()
                if confirm != 'y':
                    print("Trade cancelled.")
                    return
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Order submitted! ID: {order.id}")
            except Exception as e:
                print(f"Trade error: {e}")
            return
    def fetch_positions(self):
        """Fetch current positions with enhanced data"""
        try:
            positions = self.api.list_positions()
            if not positions:
                return pd.DataFrame()
            data = []
            for pos in positions:
                data.append({
                    'Symbol': pos.symbol,
                    'Qty': float(pos.qty),
                    'Market Value': float(pos.market_value),
                    'Entry Price': float(pos.avg_entry_price),
                    'Current Price': float(pos.current_price),
                    'Unrealized P&L': float(pos.unrealized_pl),
                    'P&L %': float(pos.unrealized_plpc) * 100,
                    'Side': pos.side
                })
            df = pd.DataFrame(data)
            return df.sort_values('Market Value', key=abs, ascending=False)
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return pd.DataFrame()

    def fetch_market_data(self, symbols, days_back=1):
        """Fetch enhanced market data"""
        if not symbols:
            return {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        data = {}
        print(f"Fetching market data for {len(symbols)} symbols...")
        for symbol in symbols:
            try:
                df = yf.download(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='5m',
                    progress=False,
                    auto_adjust=False
                )
                if not df.empty:
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                    data[symbol] = df
                    print(f"SUCCESS: {symbol} - {len(df)} data points retrieved")
                else:
                    print(f"WARNING: {symbol} - No data available")
            except Exception as e:
                print(f"ERROR: {symbol} - {e}")
        return data

    def create_enhanced_dashboard(self, save_png=False):
        """Create professional dashboard with multiple panels"""
        positions_df = self.fetch_positions()
        account_info = self.get_account_info()
        print("\n" + "="*60)
        print("PORTFOLIO DASHBOARD")
        print("="*60)
        if account_info:
            print(f"Account Status: {account_info['status']}")
            print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
            print(f"Cash: ${account_info['cash']:,.2f}")
            print(f"Buying Power: ${account_info['buying_power']:,.2f}")
        if positions_df.empty:
            print("\nNo current positions")
            return
        print(f"\nCURRENT POSITIONS ({len(positions_df)})")
        print("-" * 80)
        for _, pos in positions_df.iterrows():
            status = "PROFIT" if pos['Unrealized P&L'] > 0 else "LOSS"
            print(f"{pos['Symbol']:<6} | {pos['Qty']:>8.0f} | ${pos['Market Value']:>10,.2f} | "
                  f"{status:<6} ${pos['Unrealized P&L']:>8,.2f} ({pos['P&L %']:>6.1f}%)")
        symbols = positions_df['Symbol'].tolist()[:6]
        market_data = self.fetch_market_data(symbols)
        if market_data:
            self.create_professional_charts(positions_df, market_data, account_info, save_png=save_png)

    def create_professional_charts(self, positions_df, market_data, account_info, save_png=False):
        """Create professional, business-appropriate charts"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('TRADING DASHBOARD', fontsize=24, fontweight='bold', y=0.98)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        # 1. Portfolio Pie Chart
        ax1 = fig.add_subplot(gs[0, :2])
        if not positions_df.empty:
            sizes = positions_df['Market Value'].abs()
            colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=positions_df['Symbol'],
                autopct=lambda pct: f'${sizes.iloc[int(pct*len(sizes)/100)]:.0f}\n({pct:.1f}%)',
                colors=colors,
                startangle=90,
                textprops={'fontsize': 10, 'fontweight': 'bold'}
            )
            ax1.set_title('Portfolio Allocation', fontsize=16, fontweight='bold', pad=20)
        # 2. P&L Bar Chart
        ax2 = fig.add_subplot(gs[0, 2:])
        if not positions_df.empty:
            colors = [self.colors['profit'] if x > 0 else self.colors['loss'] 
                     for x in positions_df['Unrealized P&L']]
            bars = ax2.bar(positions_df['Symbol'], positions_df['Unrealized P&L'], 
                          color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax2.set_title('Unrealized P&L', fontsize=16, fontweight='bold')
            ax2.set_ylabel('P&L ($)', fontsize=12)
            ax2.grid(True, alpha=0.3, color=self.colors['grid'])
            ax2.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, positions_df['Unrealized P&L']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${value:.0f}', ha='center', 
                        va='bottom' if height > 0 else 'top', 
                        fontsize=10, fontweight='bold')
        # 3. Price Charts for top positions
        chart_positions = 0
        for i, (symbol, df) in enumerate(market_data.items()):
            if chart_positions >= 4:
                break
            row = 1 + (chart_positions // 2)
            col = (chart_positions % 2) * 2
            ax = fig.add_subplot(gs[row, col:col+2])
            for j in range(len(df)):
                row_data = df.iloc[j]
                open_price = float(row_data['Open'])
                high_price = float(row_data['High'])
                low_price = float(row_data['Low'])
                close_price = float(row_data['Close'])
                color = self.colors['profit'] if close_price > open_price else self.colors['loss']
                ax.plot([j, j], [low_price, high_price], color=color, linewidth=1, alpha=0.7)
                height = abs(close_price - open_price)
                bottom = min(open_price, close_price)
                rect = Rectangle((j-0.3, bottom), 0.6, height, 
                               facecolor=color, alpha=0.8, edgecolor='white')
                ax.add_patch(rect)
            if 'SMA_20' in df.columns:
                ax.plot(range(len(df)), df['SMA_20'], 
                       color=self.colors['accent'], linewidth=2, alpha=0.8, label='SMA 20')
            position_info = positions_df[positions_df['Symbol'] == symbol]
            if not position_info.empty:
                entry_price = position_info.iloc[0]['Entry Price']
                ax.axhline(y=entry_price, color=self.colors['primary'], 
                          linestyle='--', linewidth=2, alpha=0.8, 
                          label=f'Entry: ${entry_price:.2f}')
            ax.set_title(f'{symbol} - 5 Minute Chart', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=10)
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            ax.legend(loc='upper left', fontsize=8)
            if len(df) > 0:
                step = max(1, len(df) // 10)
                ax.set_xticks(range(0, len(df), step))
                ax.set_xticklabels([df.index[i].strftime('%H:%M') 
                                  for i in range(0, len(df), step)], rotation=45)
            chart_positions += 1
        fig.text(0.99, 0.01, f'Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='right', va='bottom', fontsize=10, alpha=0.7)
        plt.tight_layout()
        if save_png:
            plt.savefig('trading_dashboard.png', dpi=300, bbox_inches='tight', 
                       facecolor='#1a1a1a', edgecolor='none')
            print(f"\nDashboard saved as 'trading_dashboard.png'")
        plt.show()
    def __init__(self):
        self.api = None
        self.eastern = pytz.timezone("US/Eastern")
        # Professional styling
        plt.style.use('dark_background')
        self.setup_colors()

        # Automated strategy state
        self.automated_active = False
        self.automated_log = []
        self.strategy_positions = {}
        self.last_signal = None
        self.strategy_symbol = "SPY"  # Default symbol for strategy
        self.strategy_qty = 0
        self.strategy_entry = None
        self.strategy_side = None
        self.strategy_stop = None
        self.strategy_target = None
        self.strategy_order_id = None
        self.strategy_risk_pct = 0.01  # 1% risk per trade
        self.strategy_refresh_interval = 60  # seconds

    def setup_colors(self):
        """Define professional color scheme"""
        self.colors = {
            'primary': '#00D4FF',
            'profit': '#00FF88',
            'loss': '#FF4444',
            'neutral': '#FFFFFF',
            'accent': '#FFD700',
            'grid': '#333333'
        }

    # =====================
    # Automated Strategy
    # =====================
    def run_automated_strategy(self):
        """Main loop for automated RSI/VWAP/ATR strategy with exit option"""
        print("\n=== Automated Intraday RSI/VWAP/ATR Strategy ===")
        print(f"Symbol: {self.strategy_symbol} | Risk per trade: {self.strategy_risk_pct*100:.1f}% of equity")
        self.automated_active = True
        self.automated_log = []
        self.strategy_positions = {}
        self.last_signal = None
        self.strategy_order_id = None
        self.strategy_entry = None
        self.strategy_side = None
        self.strategy_stop = None
        self.strategy_target = None
        try:
            while self.automated_active:
                now = datetime.now(self.eastern)
                if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
                    print("Market is closed. Waiting for open...")
                    user_input = input("Type 'exit' to return to menu or press Enter to keep waiting: ").strip().lower()
                    if user_input == 'exit':
                        print("Exiting to main menu...")
                        break
                    time.sleep(60)
                    continue
                # Fetch latest 5-min data
                df = self.fetch_intraday_data(self.strategy_symbol, interval='5m', lookback=30)
                if df is None or df.empty:
                    print("No data. Retrying...")
                    user_input = input("Type 'exit' to return to menu or press Enter to keep retrying: ").strip().lower()
                    if user_input == 'exit':
                        print("Exiting to main menu...")
                        break
                    time.sleep(self.strategy_refresh_interval)
                    continue
                # Calculate indicators
                df = self.calculate_indicators(df)
                # Generate signal
                signal, rsi, vwap, atr = self.generate_strategy_signal(df)
                # Monitor open position
                self.monitor_strategy_position(df, atr)
                # If no open position, act on signal
                if self.strategy_order_id is None and signal:
                    self.execute_strategy_trade(signal, df, atr, vwap)
                # Periodic log flush
                self.flush_strategy_log()
                # Allow user to exit after each loop
                user_input = input("Type 'exit' to return to menu or press Enter to continue: ").strip().lower()
                if user_input == 'exit':
                    print("Exiting to main menu...")
                    break
                time.sleep(self.strategy_refresh_interval)
        except KeyboardInterrupt:
            print("\nAutomated strategy stopped by user.")
        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            self.automated_active = False
            self.flush_strategy_log()

    def fetch_intraday_data(self, symbol, interval='5m', lookback=30):
        """Fetch recent intraday bars for the symbol"""
        try:
            end = datetime.now()
            start = end - timedelta(minutes=lookback*5+30)
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
            df = df.tail(lookback)
            return df
        except Exception as e:
            print(f"Data fetch error: {e}")
            return None

    def calculate_indicators(self, df):
        """Add RSI(14), VWAP, ATR(14) columns to df"""
        # RSI(14)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=14).mean()
        avg_loss = loss.rolling(window=14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        # VWAP (anchored from daily open)
        vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['VWAP'] = vwap
        # ATR(14)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=14).mean()
        return df

    def generate_strategy_signal(self, df):
        """Return ('buy' or 'sell' or None), latest RSI, VWAP, ATR"""
        if len(df) < 15:
            return None, None, None, None
        rsi = df['RSI'].iloc[-1]
        prev_rsi = df['RSI'].iloc[-2]
        price = df['Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        # Buy: RSI crosses above 30, price below VWAP
        if prev_rsi < 30 and rsi >= 30 and price < vwap:
            return 'buy', rsi, vwap, atr
        # Sell: RSI crosses below 70, price above VWAP
        if prev_rsi > 70 and rsi <= 70 and price > vwap:
            return 'sell', rsi, vwap, atr
        return None, rsi, vwap, atr

    def execute_strategy_trade(self, signal, df, atr, vwap):
        """Place order based on signal, set stops/targets, log"""
        try:
            account = self.get_account_info()
            equity = account.get('equity', 0)
            price = df['Close'].iloc[-1]
            risk_amt = equity * self.strategy_risk_pct
            qty = int(risk_amt // (atr if atr else 1))
            if qty < 1:
                print("Position size too small. Skipping.")
                return
            side = signal
            stop = price - atr if side == 'buy' else price + atr
            target = price + 1.5*atr if side == 'buy' else price - 1.5*atr
            # Place order
            print(f"Placing {side.upper()} order: {qty} shares at ${price:.2f}")
            order = self.api.submit_order(
                symbol=self.strategy_symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            self.strategy_order_id = order.id
            self.strategy_entry = price
            self.strategy_qty = qty
            self.strategy_side = side
            self.strategy_stop = stop
            self.strategy_target = target
            self.log_strategy_event(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | SIGNAL: {side.upper()} | Entry: ${price:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f} | Qty: {qty}")
        except Exception as e:
            print(f"Order error: {e}")
            self.log_strategy_event(f"Order error: {e}")

    def monitor_strategy_position(self, df, atr):
        """Check open position for stop/target exit, close if needed"""
        if self.strategy_order_id is None:
            return
        try:
            order = self.api.get_order(self.strategy_order_id)
            if order.status in ['filled', 'partially_filled']:
                price = df['Close'].iloc[-1]
                if self.strategy_side == 'buy':
                    if price <= self.strategy_stop:
                        self.close_strategy_position('stop', price)
                    elif price >= self.strategy_target:
                        self.close_strategy_position('target', price)
                elif self.strategy_side == 'sell':
                    if price >= self.strategy_stop:
                        self.close_strategy_position('stop', price)
                    elif price <= self.strategy_target:
                        self.close_strategy_position('target', price)
        except Exception as e:
            print(f"Monitor error: {e}")

    def close_strategy_position(self, reason, exit_price):
        """Close open position and log result"""
        try:
            side = 'sell' if self.strategy_side == 'buy' else 'buy'
            print(f"Closing position ({reason.upper()}) at ${exit_price:.2f}")
            order = self.api.submit_order(
                symbol=self.strategy_symbol,
                qty=self.strategy_qty,
                side=side,
                type="market",
                time_in_force="gtc"
            )
            pnl = (exit_price - self.strategy_entry) * self.strategy_qty
            if self.strategy_side == 'sell':
                pnl = -pnl
            self.log_strategy_event(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | EXIT: {reason.upper()} | Exit: ${exit_price:.2f} | P&L: ${pnl:.2f}")
        except Exception as e:
            print(f"Close error: {e}")
            self.log_strategy_event(f"Close error: {e}")
        finally:
            self.strategy_order_id = None
            self.strategy_entry = None
            self.strategy_qty = 0
            self.strategy_side = None
            self.strategy_stop = None
            self.strategy_target = None

    def log_strategy_event(self, msg):
        print(msg)
        self.automated_log.append(msg)
        with open('trading_log.txt', 'a') as f:
            f.write(msg + '\n')

    def flush_strategy_log(self):
        if self.automated_log:
            with open('trading_log.txt', 'a') as f:
                for line in self.automated_log:
                    f.write(line + '\n')
            self.automated_log = []

    # =====================
    # Basic Backtesting
    # =====================
    def backtest_strategy(self, symbol=None, days=5):
        """Run basic backtest of strategy on historical data"""
        symbol = symbol or self.strategy_symbol
        print(f"\n=== Backtesting {symbol} for {days} days ===")
        try:
            df = yf.download(symbol, period=f'{days}d', interval='5m', progress=False, auto_adjust=False)
        except Exception as e:
            print(f"Error fetching data for backtest: {e}")
            return
        if df is None or df.empty:
            print("No data for backtest. Market may be closed or symbol is invalid.")
            return
        df = self.calculate_indicators(df)
        position = None
        entry = 0
        stop = 0
        target = 0
        side = None
        equity = 100000
        risk_pct = self.strategy_risk_pct
        atr = 0
        trades = []
        for i in range(15, len(df)):
            price = float(df['Close'].iloc[i])
            atr = float(df['ATR'].iloc[i]) if not pd.isna(df['ATR'].iloc[i]) else 0
            vwap = float(df['VWAP'].iloc[i]) if not pd.isna(df['VWAP'].iloc[i]) else 0
            rsi = float(df['RSI'].iloc[i]) if not pd.isna(df['RSI'].iloc[i]) else 0
            prev_rsi = float(df['RSI'].iloc[i-1]) if not pd.isna(df['RSI'].iloc[i-1]) else 0
            if not position:
                if prev_rsi < 30 and rsi >= 30 and price < vwap:
                    side = 'buy'
                elif prev_rsi > 70 and rsi <= 70 and price > vwap:
                    side = 'sell'
                else:
                    continue
                risk_amount = equity * risk_pct
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
                        reason = 'stop' if price <= position['stop'] else 'target'
                        trades.append({
                            'entry': position['entry'],
                            'exit': price,
                            'side': 'buy',
                            'pnl': pnl,
                            'reason': reason
                        })
                        position = None
                elif position['side'] == 'sell':
                    if price >= position['stop'] or price <= position['target']:
                        pnl = (position['entry'] - price) * position['qty']
                        reason = 'stop' if price >= position['stop'] else 'target'
                        trades.append({
                            'entry': position['entry'],
                            'exit': price,
                            'side': 'sell',
                            'pnl': pnl,
                            'reason': reason
                        })
                        position = None
        # Results
        if not trades:
            print("No trades were made during the backtest period. This may be due to market conditions or lack of data.")
            return
        total_pnl = sum(t['pnl'] for t in trades)
        print(f"Backtest completed: {len(trades)} trades | Total P&L: ${total_pnl:.2f}")
        for t in trades:
            print(f"{t['side'].upper()} | Entry: ${t['entry']:.2f} | Exit: ${t['exit']:.2f} | P&L: ${t['pnl']:.2f} | {t['reason']}")

    # ...existing code for Alpaca setup, manual trade, dashboard, etc...

    def setup_alpaca(self):
        print("=== Alpaca Trading Setup ===")
        API_KEY = input("Enter your Alpaca API Key ID: ")
        API_SECRET = getpass.getpass("Enter your Alpaca API Secret Key: ")
        BASE_URL = "https://paper-api.alpaca.markets"
        try:
            self.api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
            account = self.api.get_account()
            print(f"Connected successfully! Account status: {account.status}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def get_account_info(self):
        try:
            account = self.api.get_account()
            return {
                'status': account.status,
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'day_trade_buying_power': float(account.day_trade_buying_power)
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    # ...existing code for manual trade, dashboard, etc...


def main():
    dashboard = TradingDashboard()
    print("PROFESSIONAL TRADING DASHBOARD")
    print("=" * 50)
    if not dashboard.setup_alpaca():
        return
    while True:
        print("\nSELECT MODE:")
        print("1. Execute Trade")
        print("2. View Dashboard")
        print("3. Refresh Data")
        print("4. Exit")
        print("5. Automated Intraday Strategy")
        print("6. Backtest Strategy")
        choice = input("\nEnter choice (1-6): ").strip()
        if choice == '1':
            dashboard.execute_trade()
        elif choice == '2':
            dashboard.create_enhanced_dashboard()
        elif choice == '3':
            print("Refreshing data...")
            dashboard.create_enhanced_dashboard()
        elif choice == '4':
            print("Goodbye!")
            break
        elif choice == '5':
            dashboard.run_automated_strategy()
        elif choice == '6':
            symbol = input("Enter symbol for backtest (default SPY): ").strip().upper() or "SPY"
            days = input("Enter number of days to backtest (default 5): ").strip()
            try:
                days = int(days)
            except:
                days = 5
            dashboard.backtest_strategy(symbol, days)
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
