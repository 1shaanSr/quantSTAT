import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import yfinance as yf
import time
import threading
from datetime import datetime, timedelta, time as dt_time
import numpy as np
import warnings
from .trade_executor import TradeExecutor
warnings.filterwarnings('ignore')

class Dashboard:
    def __init__(self, api_handler):
        self.api = api_handler.api
        self.trade_executor = TradeExecutor(api_handler)
        plt.style.use('dark_background')
        plt.ion()
        
        self.colors = {
            'profit': '#00C851',
            'loss': '#FF4444',
            'neutral': '#FFFFFF',
            'accent': '#1E88E5',
            'background': '#0D1421',
            'panel': '#1A2332',
            'text': '#E8EAF0',
            'grid': '#2A3441',
            'candle_up': '#00C851',
            'candle_down': '#FF4444',
            'volume': '#64B5F6',
            'high_marker': '#FF9800',
            'low_marker': '#4CAF50'
        }
        
        self.running = False
        self.windows = {}
        self.last_positions_df = None
        self.last_account = None

    def _is_market_open(self):
        """Check if market is open (simplified)"""
        now = datetime.now()
        weekday = now.weekday()
        current_time = now.time()
        
        if weekday >= 5:  # Weekend
            return False
        
        # Market hours: 9:30 AM to 4:00 PM ET
        return dt_time(9, 30) <= current_time <= dt_time(16, 0)

    def _get_stock_data(self, symbol):
        """Get comprehensive stock data with proper OHLCV"""
        try:
            # Get current date and ensure we don't request future data
            now = datetime.now()
            
            # For demonstration purposes, if we're in the future (2025), use historical data
            if now.year >= 2025:
                # Use a safe historical period that definitely has data
                end = datetime(2024, 12, 20)  # Use end of 2024
                start = end - timedelta(days=90)  # 3 months of data
                
                try:
                    data = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
                    if not data.empty:
                        processed_data = self._process_yfinance_data(data)
                        if processed_data is not None:
                            print(f"Using historical data from {start.date()} to {end.date()}")
                            return processed_data
                except Exception as e:
                    print(f"Error fetching historical data: {e}")
            
            if self._is_market_open():
                # Live data - get recent trading data
                end = now
                start = end - timedelta(days=5)  # Look back 5 days to ensure data
                
                # Try 15-minute intervals first
                try:
                    data = yf.download(symbol, start=start, end=end, interval='15m', progress=False)
                    if not data.empty:
                        return self._process_yfinance_data(data)
                except Exception:
                    pass
                
                # Fallback to 1-hour intervals
                try:
                    data = yf.download(symbol, start=start, end=end, interval='1h', progress=False)
                    if not data.empty:
                        return self._process_yfinance_data(data)
                except Exception:
                    pass
            
            # Market closed or live data failed - get the most recent available data
            end = now
            
            # Try different time ranges to find data
            for days_back in [1, 2, 3, 5, 7, 14, 30]:
                start = end - timedelta(days=days_back)
                
                try:
                    # Try daily data first (most reliable)
                    data = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
                    if not data.empty:
                        return self._process_yfinance_data(data)
                except Exception:
                    continue
                    
                try:
                    # Try hourly data
                    data = yf.download(symbol, start=start, end=end, interval='1h', progress=False)
                    if not data.empty:
                        return self._process_yfinance_data(data)
                except Exception:
                    continue
            
            # Final fallback - try a simple daily download
            try:
                start = now - timedelta(days=90)  # 3 months back
                data = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
                if not data.empty:
                    return self._process_yfinance_data(data)
            except Exception:
                pass
                
            print(f"No data available for {symbol}")
            return None
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def _process_yfinance_data(self, data):
        """Process yfinance data to ensure proper format"""
        try:
            if data is None:
                return None
                
            # Check if data is empty
            if hasattr(data, 'empty') and data.empty:
                return None
                
            if len(data) == 0:
                return None
            
            # Handle multi-level columns from yfinance
            if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
                # Flatten multi-level columns
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                print(f"Missing required columns. Available: {list(data.columns)}")
                return None
            
            # Select only the columns we need
            data = data[required_columns].copy()
            
            # Remove any rows with NaN values
            data = data.dropna()
            
            # Ensure we have at least some data
            if len(data) == 0:
                return None
                
            return data
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return None

    def _plot_candlesticks(self, ax, data, symbol):
        """Plot professional candlestick chart"""
        if len(data) == 0:
            return
            
        # Calculate bar width based on time interval
        if len(data) > 1:
            time_diff = data.index[1] - data.index[0]
            width = time_diff.total_seconds() / (24 * 3600) * 0.6  # 60% of interval
        else:
            width = 0.02
        
        for timestamp, row in data.iterrows():
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine color
            color = self.colors['candle_up'] if close_price >= open_price else self.colors['candle_down']
            
            # Draw candlestick
            # High-low line (wick)
            ax.plot([timestamp, timestamp], [low_price, high_price], 
                   color=color, linewidth=1, alpha=0.8)
            
            # Open-close body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle((mdates.date2num(timestamp) - width/2, body_bottom),
                           width, body_height,
                           facecolor=color, edgecolor=color,
                           alpha=0.8 if body_height > 0 else 0.3)
            ax.add_patch(rect)

    def _find_significant_levels(self, data, window=10):
        """Find significant high and low levels"""
        if len(data) < window * 2:
            return [], []
        
        # Find local highs and lows
        highs = []
        lows = []
        
        for i in range(window, len(data) - window):
            # Check if this is a local high
            current_high = data['High'].iloc[i]
            surrounding_highs = data['High'].iloc[i-window:i+window+1]
            if current_high == surrounding_highs.max():
                highs.append((data.index[i], current_high))
            
            # Check if this is a local low
            current_low = data['Low'].iloc[i]
            surrounding_lows = data['Low'].iloc[i-window:i+window+1]
            if current_low == surrounding_lows.min():
                lows.append((data.index[i], current_low))
        
        # Get the most recent and significant levels
        highs = sorted(highs, key=lambda x: x[1], reverse=True)[:3]  # Top 3 highs
        lows = sorted(lows, key=lambda x: x[1])[:3]  # Bottom 3 lows
        
        return highs, lows

    def _create_overview_window(self, positions_df, account):
        """Create professional portfolio overview window"""
        if 'overview' in self.windows:
            plt.close(self.windows['overview'])
            
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Create professional layout
        ax1 = plt.subplot(2, 2, 1)  # Account summary
        ax2 = plt.subplot(2, 2, 2)  # P&L bar chart
        ax3 = plt.subplot(2, 2, 3)  # Portfolio allocation pie
        ax4 = plt.subplot(2, 2, 4)  # Performance metrics
        
        is_live = self._is_market_open()
        market_status = "🟢 MARKET OPEN" if is_live else "🔴 MARKET CLOSED"
        data_status = "LIVE DATA" if is_live else "SHOWING LAST AVAILABLE DATA"
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        fig.suptitle(f'Portfolio Dashboard | {market_status} | {data_status} | {timestamp}', 
                    fontsize=18, color=self.colors['accent'], fontweight='bold')
        
        # Account Summary (top left) - enhanced with data status
        ax1.axis('off')
        total_pl = positions_df['P&L'].sum()
        equity = float(account.equity)
        cash = float(account.cash)
        day_pl = total_pl  # Simplified - you could calculate daily P&L separately
        day_pl_pct = (day_pl / equity) * 100 if equity > 0 else 0
        
        # Format numbers professionally
        equity_str = f"${equity:,.2f}" if equity < 1000000 else f"${equity/1000000:.2f}M"
        cash_str = f"${cash:,.2f}" if cash < 1000000 else f"${cash/1000000:.2f}M"
        
        summary_text = f"""ACCOUNT SUMMARY
{'='*25}
Total Equity: {equity_str}
Available Cash: {cash_str}
Buying Power: ${float(account.buying_power):,.2f}

DAILY PERFORMANCE
{'='*25}
Day P&L: ${day_pl:,.2f}
Day P&L %: {day_pl_pct:+.2f}%
Total Positions: {len(positions_df)}

MARKET STATUS
{'='*25}
Status: {market_status[2:]}
Data: {data_status}
Portfolio Value: {equity_str}"""
        
        # Add control instructions to the overview window
        control_text = """DASHBOARD CONTROLS
{'='*25}
Type in terminal:
  'r' or 'refresh' - Update data
  'q' or 'quit' - Exit dashboard
  
Auto-refresh: DISABLED
Manual control: ENABLED"""
        
        ax1.text(0.05, 0.02, control_text, transform=ax1.transAxes, fontsize=9,
                color=self.colors['accent'], verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'], 
                         alpha=0.8, edgecolor=self.colors['accent']))
        
        ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=12,
                color=self.colors['text'], verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=self.colors['panel'], 
                         alpha=0.9, edgecolor=self.colors['grid']))
        
        # P&L Bar Chart (top right)
        symbols = positions_df['Symbol'].tolist()
        pl_values = positions_df['P&L'].tolist()
        pl_percentages = positions_df['P&L %'].tolist()
        
        colors = [self.colors['profit'] if pl >= 0 else self.colors['loss'] for pl in pl_values]
        
        bars = ax2.barh(symbols, pl_values, color=colors, alpha=0.8, height=0.6)
        ax2.axvline(x=0, color=self.colors['neutral'], linestyle='-', alpha=0.7, linewidth=2)
        ax2.set_title('Position P&L', color=self.colors['text'], fontsize=14, fontweight='bold')
        ax2.set_xlabel('Unrealized P&L ($)', color=self.colors['text'])
        
        # Add value and percentage labels
        for bar, value, pct in zip(bars, pl_values, pl_percentages):
            width = bar.get_width()
            label_x = width + (abs(width) * 0.1 if width != 0 else 50)
            ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'${value:.0f}\n({pct:+.1f}%)', 
                    ha='left' if width >= 0 else 'right', va='center',
                    color=self.colors['text'], fontweight='bold', fontsize=9)
        
        # Portfolio Allocation Pie (bottom left)
        market_values = positions_df['Market Value'].tolist()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(symbols)))  # Professional color scheme
        
        # Fixed: Removed unused variables by assigning to _ (throwaway variables)
        _, _, _ = ax3.pie(market_values, labels=symbols, autopct='%1.1f%%', 
                         startangle=90, colors=colors_pie,
                         textprops={'color': self.colors['text'], 'fontsize': 10})
        ax3.set_title('Portfolio Allocation', color=self.colors['text'], 
                     fontsize=14, fontweight='bold')
        
        # Performance Metrics (bottom right)
        ax4.axis('off')
        
        # Calculate additional metrics
        total_value = positions_df['Market Value'].sum()
        largest_position = positions_df.loc[positions_df['Market Value'].idxmax()]
        best_performer = positions_df.loc[positions_df['P&L %'].idxmax()]
        worst_performer = positions_df.loc[positions_df['P&L %'].idxmin()]
        
        metrics_text = f"""PORTFOLIO METRICS
{'='*25}
Total Market Value: ${total_value:,.2f}
Total Unrealized P&L: ${total_pl:,.2f}
Average Position Size: ${total_value/len(positions_df):,.2f}

TOP PERFORMERS
{'='*25}
Largest Position:
  {largest_position['Symbol']} (${largest_position['Market Value']:,.2f})

Best Performer:
  {best_performer['Symbol']} ({best_performer['P&L %']:+.1f}%)

Worst Performer:
  {worst_performer['Symbol']} ({worst_performer['P&L %']:+.1f}%)

RISK METRICS
{'='*25}
Win Rate: {len(positions_df[positions_df['P&L'] > 0])/len(positions_df)*100:.1f}%
Positions in Profit: {len(positions_df[positions_df['P&L'] > 0])}
Positions in Loss: {len(positions_df[positions_df['P&L'] < 0])}"""
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
                color=self.colors['text'], verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=self.colors['panel'], 
                         alpha=0.9, edgecolor=self.colors['grid']))
        
        # Style all axes professionally
        for ax in [ax2]:  # Only ax2 has actual plot elements
            ax.set_facecolor(self.colors['background'])
            ax.grid(True, color=self.colors['grid'], alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(colors=self.colors['text'], labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])
        
        ax3.set_facecolor(self.colors['background'])
        
        plt.tight_layout()
        self.windows['overview'] = fig
        plt.show(block=False)

    def _create_stock_window(self, symbol, position):
        """Create professional individual stock window with candlesticks"""
        data = self._get_stock_data(symbol)
        
        # Properly check if data is available
        if data is None or (hasattr(data, 'empty') and data.empty) or len(data) == 0:
            print(f"No data available for {symbol}")
            return
            
        window_key = f'stock_{symbol}'
        if window_key in self.windows:
            plt.close(self.windows[window_key])
        
        # Create figure with subplots for price and volume
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor(self.colors['background'])
        
        # Price chart (main, top 70%)
        ax1 = plt.subplot(3, 1, (1, 2))
        # Volume chart (bottom 30%)
        ax2 = plt.subplot(3, 1, 3)
        
        is_live = self._is_market_open()
        market_status = "🟢 LIVE DATA" if is_live else "🔴 HISTORICAL DATA"
        
        # Add data freshness indicator
        last_data_time = data.index[-1].strftime('%m/%d %H:%M') if hasattr(data.index[-1], 'strftime') else str(data.index[-1])
        
        # Calculate P&L color
        pl_color = self.colors['profit'] if position['P&L'] >= 0 else self.colors['loss']
        
        current_price = data['Close'].iloc[-1]
        daily_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        change_symbol = "+" if daily_change >= 0 else ""
        
        # Enhanced title with data status
        title_text = f'{symbol} - ${current_price:.2f} ({change_symbol}{daily_change:.2f}%) | {market_status}'
        if not is_live:
            title_text += f' | Last Update: {last_data_time}'
            
        fig.suptitle(title_text, fontsize=16, color=pl_color, fontweight='bold')
        
        # Plot candlesticks
        self._plot_candlesticks(ax1, data, symbol)
        
        # Find and mark significant levels
        highs, lows = self._find_significant_levels(data)
        
        # Mark significant highs
        for timestamp_high, price_high in highs:
            ax1.scatter(timestamp_high, price_high, color=self.colors['high_marker'], 
                       s=80, marker='^', zorder=10, edgecolors='white', linewidths=1)
            ax1.annotate(f'${price_high:.2f}', 
                        (timestamp_high, price_high),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=9, color=self.colors['high_marker'],
                        fontweight='bold')
        
        # Mark significant lows
        for timestamp_low, price_low in lows:
            ax1.scatter(timestamp_low, price_low, color=self.colors['low_marker'], 
                       s=80, marker='v', zorder=10, edgecolors='white', linewidths=1)
            ax1.annotate(f'${price_low:.2f}', 
                        (timestamp_low, price_low),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=9, color=self.colors['low_marker'],
                        fontweight='bold')
        
        # Entry price line
        entry_price = position['Avg Entry']
        ax1.axhline(y=entry_price, color=self.colors['accent'], linestyle='--', 
                   alpha=0.8, linewidth=2, label=f'Entry: ${entry_price:.2f}')
        
        # Add moving average for context
        if len(data) > 20:
            ma20 = data['Close'].rolling(window=min(20, len(data)//2)).mean()
            ax1.plot(data.index, ma20, color=self.colors['accent'], 
                    alpha=0.6, linewidth=1.5, label=f'MA{min(20, len(data)//2)}')
        
        # Enhanced info panel with data status
        data_status_text = "LIVE" if is_live else f"HISTORICAL (Last: {last_data_time})"
        data_interval = "Intraday" if len(data) > 50 or is_live else "Daily"
        
        info_text = f"""POSITION DETAILS
Current Price: ${current_price:.2f}
Entry Price: ${entry_price:.2f}
Quantity: {position['Qty']:,} shares
Market Value: ${position['Market Value']:,.2f}

PERFORMANCE
Unrealized P&L: ${position['P&L']:.2f}
P&L Percentage: {position['P&L %']:+.2f}%
Daily Change: {change_symbol}{daily_change:.2f}%

DATA STATUS
Status: {data_status_text}
Interval: {data_interval}
Points: {len(data)} bars

RANGE INFO
High: ${data['High'].max():.2f}
Low: ${data['Low'].min():.2f}
Volume: {data['Volume'].iloc[-1]:,.0f}"""
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                color=self.colors['text'], verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=self.colors['panel'], 
                         alpha=0.9, edgecolor=self.colors['grid']))
        
        # Calculate bar width for volume chart
        if len(data) > 1:
            time_diff = data.index[1] - data.index[0]
            if hasattr(time_diff, 'total_seconds'):
                width = time_diff.total_seconds() / (24 * 3600) * 0.8  # 80% of interval
            else:
                width = 0.8
        else:
            width = 0.8
        
        # Volume bars with corrected color calculation
        volume_colors = [self.colors['candle_up'] if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                        else self.colors['candle_down'] for i in range(len(data))]
        
        # Plot volume bars
        ax2.bar(data.index, data['Volume'], color=volume_colors, alpha=0.7, width=width)
        
        # Volume moving average
        if len(data) > 10:
            vol_ma = data['Volume'].rolling(window=min(10, len(data)//2)).mean()
            ax2.plot(data.index, vol_ma, color=self.colors['volume'], 
                    alpha=0.8, linewidth=2, label=f'Vol MA{min(10, len(data)//2)}')
        
        # Add watermark for historical data
        if not is_live:
            ax1.text(0.5, 0.5, 'HISTORICAL DATA\nMARKET CLOSED', 
                    transform=ax1.transAxes, fontsize=20, color=self.colors['grid'],
                    alpha=0.3, ha='center', va='center', rotation=45, fontweight='bold')
            ax2.text(0.5, 0.5, 'HISTORICAL', 
                    transform=ax2.transAxes, fontsize=16, color=self.colors['grid'],
                    alpha=0.3, ha='center', va='center', rotation=45, fontweight='bold')
        
        # Style the axes
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.colors['background'])
            ax.grid(True, color=self.colors['grid'], alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(colors=self.colors['text'], labelsize=10)
            ax.spines['top'].set_color(self.colors['grid'])
            ax.spines['bottom'].set_color(self.colors['grid'])
            ax.spines['left'].set_color(self.colors['grid'])
            ax.spines['right'].set_color(self.colors['grid'])
        
        # Format x-axis for time - improved for historical data
        if is_live:
            # Live data formatting (intraday)
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            # Historical data formatting
            if len(data) > 50:  # Intraday historical data
                # Show fewer labels to avoid crowding
                ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            else:  # Daily historical data
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        ax1.set_ylabel('Price ($)', color=self.colors['text'], fontweight='bold')
        ax2.set_ylabel('Volume', color=self.colors['text'], fontweight='bold')
        ax2.set_xlabel('Time', color=self.colors['text'], fontweight='bold')
        
        # Add legends
        ax1.legend(loc='upper left', framealpha=0.8, facecolor=self.colors['panel'])
        if len(data) > 10:  # Only show volume legend if we have moving average
            ax2.legend(loc='upper right', framealpha=0.8, facecolor=self.colors['panel'])
        
        # Add control panel to stock window
        control_text_stock = "Press 'r' in terminal to refresh | 'q' to quit"
        ax2.text(0.02, 0.02, control_text_stock, transform=ax2.transAxes, fontsize=9,
                color=self.colors['accent'], verticalalignment='bottom', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['background'], 
                         alpha=0.8, edgecolor=self.colors['accent']))
        
        plt.tight_layout()
        self.windows[window_key] = fig
        plt.show(block=False)

    def _fetch_data_once(self):
        """Fetch data once (no background thread)"""
        try:
            positions = self.api.list_positions()
            account = self.api.get_account()
            
            if not positions:
                print("No positions found")
                return None, None
            
            positions_df = pd.DataFrame([{
                'Symbol': p.symbol,
                'Qty': int(float(p.qty)),
                'Avg Entry': float(p.avg_entry_price),
                'Current': float(p.current_price),
                'P&L': float(p.unrealized_pl),
                'P&L %': float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0.0,
                'Market Value': float(p.market_value) if p.market_value else 0.0
            } for p in positions])
            
            return positions_df, account
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, None

    def _refresh_dashboard(self):
        """Refresh dashboard data and charts"""
        print("Refreshing dashboard data...")
        positions_df, account = self._fetch_data_once()
        
        if positions_df is None or account is None:
            print("Failed to fetch data")
            return False
        
        # Store the data
        self.last_positions_df = positions_df
        self.last_account = account
        
        # Close existing windows to prevent buildup
        for window in list(self.windows.values()):
            plt.close(window)
        self.windows.clear()
        
        # Create overview window
        self._create_overview_window(positions_df, account)
        
        # Create individual stock windows
        for _, position in positions_df.iterrows():
            self._create_stock_window(position['Symbol'], position)
        
        print(f"Dashboard updated at {datetime.now().strftime('%H:%M:%S')} - {len(positions_df)} positions")
        plt.pause(0.1)  # Small pause to allow rendering
        return True

    def start_dashboard(self):
        """Start the dashboard with manual control"""
        self.running = True
        
        market_status = "🟢 OPEN" if self._is_market_open() else "🔴 CLOSED"
        print(f"\n{'='*50}")
        print(f"MANUAL CONTROL PORTFOLIO DASHBOARD")
        print(f"{'='*50}")
        print(f"Market Status: {market_status}")
        print(f"Mode: Manual refresh (no auto-updates)")
        print(f"{'='*50}")
        print("\nCONTROLS:")
        print("  'r' or 'refresh' - Update dashboard data")
        print("  'q' or 'quit' - Exit dashboard")
        print(f"{'='*50}")
        
        # Initial data load
        if not self._refresh_dashboard():
            print("Failed to load initial data")
            return
        
        # Manual control loop
        try:
            while self.running:
                try:
                    print("\nDashboard Commands:")
                    print("r = refresh dashboard")
                    print("b = buy stock")
                    print("s = sell stock (close long positions)")
                    print("h = short stock") 
                    print("c = cover short (close short positions)")
                    print("q = quit dashboard")
                    command = input("\nEnter command: ").strip().lower()
                    
                    if command in ['q', 'quit', 'exit']:
                        print("Exiting dashboard...")
                        break
                    elif command in ['r', 'refresh', 'update']:
                        if not self._refresh_dashboard():
                            print("Refresh failed, but dashboard remains active")
                    elif command in ['b', 'buy']:
                        self._handle_dashboard_buy()
                    elif command in ['s', 'sell']:
                        self._handle_dashboard_sell()
                    elif command in ['h', 'short']:
                        self._handle_dashboard_short()
                    elif command in ['c', 'cover']:
                        self._handle_dashboard_cover()
                    elif command == '':
                        continue
                    else:
                        print("Unknown command. Use commands above.")
                        
                except EOFError:
                    # Handle Ctrl+D
                    print("\nReceived EOF, exiting dashboard...")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print("\nReceived interrupt, exiting dashboard...")
                    break
                    
        except Exception as e:
            print(f"Dashboard error: {e}")
        finally:
            self.stop_dashboard()

    def stop_dashboard(self):
        """Stop the dashboard"""
        self.running = False
        for window in list(self.windows.values()):
            plt.close(window)
        self.windows.clear()
        print("Dashboard stopped. All windows closed.")

    def create_enhanced_dashboard(self, refresh_interval=None):
        """Backward compatibility method - ignore refresh_interval now"""
        print("Note: Auto-refresh disabled. Using manual control mode.")
        return self.start_dashboard()

    def _handle_dashboard_buy(self):
        """Handle buy command from dashboard"""
        symbol = input("Enter symbol to BUY: ").strip().upper()
        if not symbol:
            print("Invalid input. Trade cancelled.")
            return
        
        qty = input(f"Enter quantity to BUY {symbol}: ").strip()
        if not qty:
            print("Invalid input. Trade cancelled.")
            return
        try:
            qty = int(qty)
        except ValueError:
            print("Quantity must be an integer.")
            return
        
        confirm = input(f"Confirm BUY {qty} shares of {symbol}? (y/n): ").strip().lower()
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

    def _handle_dashboard_sell(self):
        """Handle sell command from dashboard"""
        symbol = input("Enter symbol to SELL: ").strip().upper()
        if not symbol:
            print("Invalid input. Trade cancelled.")
            return
        
        try:
            positions = self.api.list_positions()
            position = next((p for p in positions if p.symbol.upper() == symbol and float(p.qty) > 0), None)
            if not position:
                print(f"No long position found for {symbol}")
                return
            
            max_qty = int(float(position.qty))
            print(f"You own {max_qty} shares of {symbol}")
            
            qty = input(f"Enter quantity to SELL (max {max_qty}): ").strip()
            try:
                qty = int(qty)
                if qty > max_qty:
                    print("Cannot sell more shares than owned.")
                    return
            except ValueError:
                print("Quantity must be an integer.")
                return
            
            confirm = input(f"Confirm SELL {qty} shares of {symbol}? (y/n): ").strip().lower()
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

    def _handle_dashboard_short(self):
        """Handle short command from dashboard"""
        symbol = input("Enter symbol to SHORT: ").strip().upper()
        if not symbol:
            print("Invalid input. Trade cancelled.")
            return
        
        qty = input(f"Enter quantity to SHORT {symbol}: ").strip()
        if not qty:
            print("Invalid input. Trade cancelled.")
            return
        try:
            qty = int(qty)
        except ValueError:
            print("Quantity must be an integer.")
            return
        
        confirm = input(f"Confirm SHORT {qty} shares of {symbol}? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Trade cancelled.")
            return
        
        try:
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

    def _handle_dashboard_cover(self):
        """Handle cover command from dashboard"""
        symbol = input("Enter symbol to COVER: ").strip().upper()
        if not symbol:
            print("Invalid input. Trade cancelled.")
            return
        
        try:
            positions = self.api.list_positions()
            position = next((p for p in positions if p.symbol.upper() == symbol and float(p.qty) < 0), None)
            if not position:
                print(f"No short position found for {symbol}")
                return
            
            max_qty = abs(int(float(position.qty)))
            print(f"You are short {max_qty} shares of {symbol}")
            
            qty = input(f"Enter quantity to COVER (max {max_qty}): ").strip()
            try:
                qty = int(qty)
                if qty > max_qty:
                    print("Cannot cover more shares than shorted.")
                    return
            except ValueError:
                print("Quantity must be an integer.")
                return
            
            confirm = input(f"Confirm COVER {qty} shares of {symbol}? (y/n): ").strip().lower()
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
        except Exception as e:
            print(f"Trade error: {e}")