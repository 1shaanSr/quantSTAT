import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import time
import queue
import threading
import platform
import sys

class Dashboard:
    def __init__(self, api_handler):
        self.api = api_handler.api
        self._setup_matplotlib()
        self.colors = {
            'primary': '#00D4FF',
            'profit': '#00FF88',
            'loss': '#FF4444',
            'neutral': '#FFFFFF',
            'accent': '#FFD700',
            'grid': '#333333',
            'background': '#1E1E1E',
            'panel': '#2E2E2E',
            'text': '#E0E0E0',
            'volume': '#4A90E2',
            'pending': '#FFA500'
        }
        
        self.windows = {}
        self.refresh_interval = 30
        self.running = False
        self.update_queue = queue.Queue()
        self.data_thread = None

    def _setup_matplotlib(self):
        try:
            if platform.system() == 'Darwin':
                import matplotlib
                matplotlib.use('TkAgg')
            elif platform.system() == 'Windows':
                import matplotlib
                matplotlib.use('Qt5Agg')
            else:
                import matplotlib
                matplotlib.use('TkAgg')
            
            plt.style.use('dark_background')
            plt.ion()
            
        except ImportError:
            plt.style.use('dark_background')
            plt.ion()

    def start_live_dashboard(self, refresh_interval=30):
        self.refresh_interval = refresh_interval
        self.running = True
        
        print("🚀 Starting Portfolio Dashboard")
        print(f"Platform: {platform.system()}")
        print("="*50)
        
        self.data_thread = threading.Thread(target=self._data_fetcher)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        if platform.system() == 'Darwin':
            self._run_mac_loop()
        else:
            self._run_main_loop()

    def stop_dashboard(self):
        self.running = False
        if self.data_thread:
            self.data_thread.join(timeout=1)
        print("\n👋 Dashboard stopped")

    def create_enhanced_dashboard(self, refresh_interval=30):
        return self.start_live_dashboard(refresh_interval)

    def _data_fetcher(self):
        while self.running:
            try:
                positions = self.api.list_positions()
                account = self.api.get_account()
                orders = self.api.list_orders(status='open')
                
                if not positions:
                    time.sleep(self.refresh_interval)
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
                
                symbols = positions_df['Symbol'].tolist()
                market_data = self._fetch_market_data(symbols)
                
                pending_orders = []
                for order in orders:
                    if hasattr(order, 'status') and order.status in ['new', 'partially_filled', 'pending_new']:
                        pending_orders.append({
                            'Symbol': order.symbol,
                            'Side': order.side,
                            'Qty': int(float(order.qty)),
                            'Price': float(order.limit_price) if order.limit_price else 'Market',
                            'Status': order.status.replace('_', ' ').title(),
                            'Order Type': order.order_type.upper(),
                            'Time': order.submitted_at
                        })
                
                self.update_queue.put({
                    'positions': positions_df,
                    'account': account,
                    'market_data': market_data,
                    'pending_orders': pending_orders
                })
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                print(f"❌ Error fetching data: {e}")
                time.sleep(5)

    def _run_mac_loop(self):
        last_update = 0
        try:
            while self.running:
                current_time = time.time()
                
                if not self.update_queue.empty():
                    try:
                        data = self.update_queue.get_nowait()
                        
                        positions_df = data['positions']
                        account = data['account']
                        market_data = data['market_data']
                        pending_orders = data['pending_orders']
                        
                        symbols = positions_df['Symbol'].tolist()
                        
                        for symbol in symbols:
                            if symbol in market_data:
                                position_data = positions_df[positions_df['Symbol'] == symbol].iloc[0]
                                self._create_stock_window(symbol, market_data[symbol], position_data)
                        
                        self._create_metrics_window(positions_df, account, pending_orders)
                        
                        print(f"📊 Updated at {datetime.now().strftime('%H:%M:%S')}")
                        print(f"📈 {len(symbols)} stocks + metrics | 📋 {len(pending_orders)} pending orders")
                        
                        last_update = current_time
                        
                    except queue.Empty:
                        pass
                
                plt.pause(0.1)
                        
        except KeyboardInterrupt:
            self.stop_dashboard()

    def _run_main_loop(self):
        try:
            while self.running:
                try:
                    data = self.update_queue.get(timeout=1)
                    
                    positions_df = data['positions']
                    account = data['account']
                    market_data = data['market_data']
                    pending_orders = data['pending_orders']
                    
                    symbols = positions_df['Symbol'].tolist()
                    
                    for symbol in symbols:
                        if symbol in market_data:
                            position_data = positions_df[positions_df['Symbol'] == symbol].iloc[0]
                            self._create_stock_window(symbol, market_data[symbol], position_data)
                    
                    self._create_metrics_window(positions_df, account, pending_orders)
                    
                    print(f"📊 Updated at {datetime.now().strftime('%H:%M:%S')}")
                    print(f"📈 {len(symbols)} stocks + metrics | 📋 {len(pending_orders)} pending orders")
                    
                    plt.pause(0.1)
                    
                except queue.Empty:
                    plt.pause(0.1)
                    continue
                except Exception as e:
                    print(f"❌ Error updating dashboard: {e}")
                    time.sleep(1)
                    
        except KeyboardInterrupt:
            self.stop_dashboard()

    def _fetch_market_data(self, symbols):
        market_data = {}
        end = datetime.now()
        start = end - timedelta(days=1)
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, start=start, end=end, interval='5m', progress=False)
                if not df.empty:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    market_data[symbol] = df
                    
            except Exception as e:
                print(f"⚠️  Error fetching {symbol}: {e}")
                continue
                
        return market_data

    def _create_stock_window(self, symbol, data, position):
        window_key = f'stock_{symbol}'
        
        if window_key in self.windows:
            try:
                plt.close(self.windows[window_key])
            except:
                pass
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            fig.patch.set_facecolor(self.colors['background'])
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            fig.suptitle(f'📈 {symbol} - Live Chart | Updated: {timestamp}', 
                        fontsize=16, color=self.colors['accent'])
            
            if not data.empty:
                color = self.colors['profit'] if position['P&L'] >= 0 else self.colors['loss']
                ax1.plot(data.index, data['close'], color=color, linewidth=2, label='Price')
                
                ax1.axhline(y=position['Avg Entry'], color=self.colors['accent'], 
                           linestyle='--', alpha=0.7, linewidth=2, label='Entry Price')
                
                current_price = data['close'].iloc[-1]
                ax1.text(0.02, 0.98, f'Current: ${current_price:.2f}', 
                        transform=ax1.transAxes, fontsize=14, fontweight='bold',
                        color=color, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['panel'], alpha=0.9))
                
                ax1.text(0.02, 0.85, f'Entry: ${position["Avg Entry"]:.2f}', 
                        transform=ax1.transAxes, fontsize=12, color=self.colors['accent'],
                        verticalalignment='top')
                
                ax1.text(0.02, 0.75, f'Qty: {position["Qty"]}', 
                        transform=ax1.transAxes, fontsize=12, color=self.colors['text'],
                        verticalalignment='top')
                
                ax1.text(0.98, 0.98, f'P&L: ${position["P&L"]:.2f}', 
                        transform=ax1.transAxes, fontsize=14, color=color, fontweight='bold',
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['panel'], alpha=0.9))
                
                ax1.text(0.98, 0.85, f'P&L %: {position["P&L %"]:.2f}%', 
                        transform=ax1.transAxes, fontsize=12, color=color,
                        verticalalignment='top', horizontalalignment='right')
                
                ax1.text(0.98, 0.75, f'Value: ${position["Market Value"]:.2f}', 
                        transform=ax1.transAxes, fontsize=12, color=self.colors['text'],
                        verticalalignment='top', horizontalalignment='right')
                
                ax2.bar(data.index, data['volume'], alpha=0.6, color=self.colors['volume'])
                ax2.set_ylabel('Volume', color=self.colors['volume'])
                ax2.tick_params(axis='y', labelcolor=self.colors['volume'])
            
            ax1.set_facecolor(self.colors['panel'])
            ax1.grid(True, color=self.colors['grid'], alpha=0.3)
            ax1.tick_params(colors=self.colors['text'])
            ax1.legend(loc='upper left', framealpha=0.7)
            ax1.set_ylabel('Price ($)', color=self.colors['text'])
            
            ax2.set_facecolor(self.colors['panel'])
            ax2.grid(True, color=self.colors['grid'], alpha=0.3)
            ax2.tick_params(colors=self.colors['text'])
            
            ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            self.windows[window_key] = fig
            
            if platform.system() == 'Windows':
                plt.show(block=False)
                plt.draw()
            else:
                plt.show(block=False)
                
        except Exception as e:
            print(f"❌ Error creating stock window for {symbol}: {e}")

    def _create_metrics_window(self, positions_df, account, pending_orders):
        if 'metrics' in self.windows:
            try:
                plt.close(self.windows['metrics'])
            except:
                pass
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            fig.patch.set_facecolor(self.colors['background'])
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            fig.suptitle(f'📊 Portfolio Metrics | Updated: {timestamp}', 
                        fontsize=16, color=self.colors['accent'])
            
            ax1.axis('off')
            equity = float(account.equity)
            buying_power = float(account.buying_power)
            cash = float(account.cash)
            total_pl = positions_df['P&L'].sum()
            total_pl_pct = (total_pl / equity) * 100 if equity > 0 else 0
            
            metrics_text = f"""
            Account Value: ${equity:,.2f}
            Buying Power: ${buying_power:,.2f}
            Cash: ${cash:,.2f}
            
            Total P&L: ${total_pl:,.2f}
            Total P&L %: {total_pl_pct:+.2f}%
            
            Positions: {len(positions_df)}
            Pending Orders: {len(pending_orders)}
            """
            
            ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes, fontsize=14,
                    color=self.colors['text'], verticalalignment='top', fontfamily='monospace')
            ax1.set_title('Account Overview', color=self.colors['neutral'], fontsize=14)
            
            symbols = positions_df['Symbol'].tolist()
            pl_values = positions_df['P&L'].tolist()
            colors = [self.colors['profit'] if pl >= 0 else self.colors['loss'] for pl in pl_values]
            
            bars = ax2.bar(symbols, pl_values, color=colors, alpha=0.8)
            ax2.axhline(y=0, color=self.colors['neutral'], linestyle='-', alpha=0.3)
            ax2.set_title('P&L by Position', color=self.colors['neutral'], fontsize=14)
            ax2.set_ylabel('P&L ($)', color=self.colors['text'])
            ax2.tick_params(colors=self.colors['text'])
            ax2.set_facecolor(self.colors['panel'])
            ax2.grid(True, color=self.colors['grid'], alpha=0.3)
            
            for bar, value in zip(bars, pl_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${value:.0f}', ha='center', va='bottom' if height >= 0 else 'top',
                        color=self.colors['text'], fontsize=10, weight='bold')
            
            market_values = positions_df['Market Value'].tolist()
            ax3.pie(market_values, labels=symbols, autopct='%1.1f%%', startangle=90,
                   colors=[self.colors['primary'], self.colors['profit'], self.colors['accent'], 
                          self.colors['volume'], self.colors['loss']][:len(symbols)],
                   textprops={'color': self.colors['text']})
            ax3.set_title('Position Allocation', color=self.colors['neutral'], fontsize=14)
            
            ax4.axis('off')
            
            if pending_orders:
                orders_text = "PENDING ORDERS:\n" + "="*20 + "\n"
                for order in pending_orders[:5]:
                    side_color = "🟢" if order['Side'] == 'buy' else "🔴"
                    price_str = f"${order['Price']:.2f}" if order['Price'] != 'Market' else 'Market'
                    orders_text += f"{side_color} {order['Symbol']} {order['Side'].upper()} {order['Qty']} @ {price_str}\n"
                    orders_text += f"   Status: {order['Status']}\n\n"
                
                if len(pending_orders) > 5:
                    orders_text += f"... and {len(pending_orders) - 5} more orders"
            else:
                orders_text = "No pending orders"
            
            ax4.text(0.1, 0.9, orders_text, transform=ax4.transAxes, fontsize=11,
                    color=self.colors['text'], verticalalignment='top', fontfamily='monospace')
            ax4.set_title('Open Orders', color=self.colors['neutral'], fontsize=14)
            
            for ax in [ax2]:
                ax.set_facecolor(self.colors['panel'])
                ax.tick_params(colors=self.colors['text'])
            
            plt.tight_layout()
            self.windows['metrics'] = fig
            
            if platform.system() == 'Windows':
                plt.show(block=False)
                plt.draw()
            else:
                plt.show(block=False)
                
        except Exception as e:
            print(f"❌ Error creating metrics window: {e}")