import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import yfinance as yf
import time
import queue
import threading
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Dashboard:
    def __init__(self, api_handler):
        self.api = api_handler.api
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
        self.update_queue = queue.Queue()
        self.windows = {}

    def _is_market_open(self):
        """Check if market is open (simplified)"""
        now = datetime.now()
        weekday = now.weekday()
        current_time = now.time()
        
        if weekday >= 5:  # Weekend
            return False
        
        # Market hours: 9:30 AM to 4:00 PM ET
        return time(9, 30) <= current_time <= time(16, 0)

    def _get_stock_data(self, symbol):
        """Get comprehensive stock data with proper OHLCV"""
        try:
            if self._is_market_open():
                # Live data - get intraday with 15min intervals for better candlesticks
                end = datetime.now()
                start = end - timedelta(days=2)
                data = yf.download(symbol, start=start, end=end, interval='15m', progress=False)
                
                # Also get today's data with 5min for more recent updates
                today_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
                today_data = yf.download(symbol, start=today_start, end=end, interval='5m', progress=False)
                
                return today_data if not today_data.empty else data
            else:
                # Market closed - get recent daily data for better visualization
                end = datetime.now()
                start = end - timedelta(days=30)  # 30 days for context
                data = yf.download(symbol, start=start, end=end, interval='1d', progress=False)
                
            if data.empty:
                return None
                
            # Ensure we have OHLCV columns
            if len(data.columns) > 5:  # Multi-level columns from yfinance
                data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return data
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
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
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
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
        
        market_status = "🟢 MARKET OPEN" if self._is_market_open() else "🔴 MARKET CLOSED"
        timestamp = datetime.now().strftime("%H:%M:%S")
        fig.suptitle(f'Portfolio Dashboard | {market_status} | {timestamp}', 
                    fontsize=18, color=self.colors['accent'], fontweight='bold')
        
        # Account Summary (top left)
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

ACCOUNT STATUS
{'='*25}
Portfolio Value: {equity_str}
Market Status: {market_status[2:]}"""
        
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
        for i, (bar, value, pct) in enumerate(zip(bars, pl_values, pl_percentages)):
            width = bar.get_width()
            label_x = width + (abs(width) * 0.1 if width != 0 else 50)
            ax2.text(label_x, bar.get_y() + bar.get_height()/2,
                    f'${value:.0f}\n({pct:+.1f}%)', 
                    ha='left' if width >= 0 else 'right', va='center',
                    color=self.colors['text'], fontweight='bold', fontsize=9)
        
        # Portfolio Allocation Pie (bottom left)
        market_values = positions_df['Market Value'].tolist()
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(symbols)))  # Professional color scheme
        
        wedges, texts, autotexts = ax3.pie(market_values, labels=symbols, autopct='%1.1f%%', 
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
        if data is None:
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
        
        market_status = "🟢 LIVE" if self._is_market_open() else "🔴 CLOSED"
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Calculate P&L color
        pl_color = self.colors['profit'] if position['P&L'] >= 0 else self.colors['loss']
        
        current_price = data['Close'].iloc[-1]
        daily_change = ((current_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
        change_symbol = "+" if daily_change >= 0 else ""
        
        fig.suptitle(f'{symbol} - ${current_price:.2f} ({change_symbol}{daily_change:.2f}%) | {market_status}', 
                    fontsize=16, color=pl_color, fontweight='bold')
        
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
        
        # Professional info panel
        info_text = f"""POSITION DETAILS
Current Price: ${current_price:.2f}
Entry Price: ${entry_price:.2f}
Quantity: {position['Qty']:,} shares
Market Value: ${position['Market Value']:,.2f}

PERFORMANCE
Unrealized P&L: ${position['P&L']:.2f}
P&L Percentage: {position['P&L %']:+.2f}%
Daily Change: {change_symbol}{daily_change:.2f}%

TODAY'S RANGE
High: ${data['High'].max():.2f}
Low: ${data['Low'].min():.2f}
Volume: {data['Volume'].iloc[-1]:,.0f}"""
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                color=self.colors['text'], verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=self.colors['panel'], 
                         alpha=0.9, edgecolor=self.colors['grid']))
        
        # Volume bars
        volume_colors = [self.colors['candle_up'] if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                        else self.colors['candle_down'] for i in range(len(data))]
        
        bars = ax2.bar(data.index, data['Volume'], color=volume_colors, alpha=0.7, width=width)
        
        # Volume moving average
        if len(data) > 10:
            vol_ma = data['Volume'].rolling(window=min(10, len(data)//2)).mean()
            ax2.plot(data.index, vol_ma, color=self.colors['volume'], 
                    alpha=0.8, linewidth=2, label=f'Vol MA{min(10, len(data)//2)}')
        
        # Style the axes
        for ax in [ax1, ax2]:
            ax.set_facecolor(self.colors['background'])
            ax.grid(True, color=self.colors['grid'], alpha=0.3, linestyle='-', linewidth=0.5)
            ax.tick_params(colors=self.colors['text'], labelsize=10)
            ax.spines['top'].set_color(self.colors['grid'])
            ax.spines['bottom'].set_color(self.colors['grid'])
            ax.spines['left'].set_color(self.colors['grid'])
            ax.spines['right'].set_color(self.colors['grid'])
        
        # Format x-axis for time
        if self._is_market_open():
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(data)//10)))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        ax1.set_ylabel('Price ($)', color=self.colors['text'], fontweight='bold')
        ax2.set_ylabel('Volume', color=self.colors['text'], fontweight='bold')
        ax2.set_xlabel('Time', color=self.colors['text'], fontweight='bold')
        
        # Add legends
        ax1.legend(loc='upper left', framealpha=0.8, facecolor=self.colors['panel'])
        ax2.legend(loc='upper right', framealpha=0.8, facecolor=self.colors['panel'])
        
        plt.tight_layout()
        self.windows[window_key] = fig
        plt.show(block=False)

    def _fetch_data(self):
        """Fetch all data in background"""
        while self.running:
            try:
                positions = self.api.list_positions()
                account = self.api.get_account()
                
                if not positions:
                    time.sleep(30)
                    continue
                
                positions_df = pd.DataFrame([{
                    'Symbol': p.symbol,
                    'Qty': int(float(p.qty)),
                    'Avg Entry': float(p.avg_entry_price),
                    'Current': float(p.current_price),
                    'P&L': float(p.unrealized_pl),
                    'P&L %': float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0.0,
                    'Market Value': float(p.market_value) if p.market_value else 0.0
                } for p in positions])
                
                self.update_queue.put({
                    'positions': positions_df,
                    'account': account
                })
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)

    def start_dashboard(self):
        """Start the dashboard"""
        self.running = True
        
        market_status = "🟢 OPEN" if self._is_market_open() else "🔴 CLOSED"
        print(f"Starting Simple Portfolio Dashboard")
        print(f"Market Status: {market_status}")
        print("="*40)
        
        # Start data fetching thread
        data_thread = threading.Thread(target=self._fetch_data)
        data_thread.daemon = True
        data_thread.start()
        
        try:
            while self.running:
                try:
                    # Get latest data
                    data = self.update_queue.get(timeout=1)
                    positions_df = data['positions']
                    account = data['account']
                    
                    # Create overview window
                    self._create_overview_window(positions_df, account)
                    
                    # Create individual stock windows
                    for _, position in positions_df.iterrows():
                        self._create_stock_window(position['Symbol'], position)
                    
                    print(f"Updated at {datetime.now().strftime('%H:%M:%S')} - {len(positions_df)} positions")
                    
                    plt.pause(0.1)
                    
                except queue.Empty:
                    plt.pause(0.1)
                    continue
                    
        except KeyboardInterrupt:
            self.stop_dashboard()

    def stop_dashboard(self):
        """Stop the dashboard"""
        self.running = False
        for window in self.windows.values():
            plt.close(window)
        print("Dashboard stopped")

    def create_enhanced_dashboard(self, refresh_interval=30):
        """Backward compatibility method"""
        return self.start_dashboard()