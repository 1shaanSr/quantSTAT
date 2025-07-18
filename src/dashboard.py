import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

class Dashboard:
    def __init__(self, api_handler):
        self.api = api_handler.api
        plt.style.use('dark_background')
        self.colors = {
            'primary': '#00D4FF',
            'profit': '#00FF88',
            'loss': '#FF4444',
            'neutral': '#FFFFFF',
            'accent': '#FFD700',
            'grid': '#333333'
        }

    def create_enhanced_dashboard(self, save_png=False):
        try:
            positions = self.api.list_positions()
            positions_df = pd.DataFrame([{
                'Symbol': p.symbol,
                'Qty': int(float(p.qty)),
                'Avg Entry': float(p.avg_entry_price),
                'Current': float(p.current_price),
                'P&L': float(p.unrealized_pl),
                'P&L %': float(p.unrealized_plpc) * 100
            } for p in positions])
            
            account = self.api.get_account()
            
            print("\n" + "="*60)
            print("PORTFOLIO DASHBOARD")
            print("="*60)
            print(f"Account Value: ${float(account.equity):,.2f}")
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            
            if positions_df.empty:
                print("\nNo current positions")
                return
            
            print("\nCurrent Positions:")
            for _, pos in positions_df.iterrows():
                pl_color = '\033[92m' if pos['P&L'] >= 0 else '\033[91m'
                print(f"\n{pos['Symbol']}:")
                print(f"Qty: {pos['Qty']}")
                print(f"Avg Entry: ${pos['Avg Entry']:.2f}")
                print(f"Current: ${pos['Current']:.2f}")
                print(f"P&L: {pl_color}${pos['P&L']:.2f} ({pos['P&L %']:.2f}%)\033[0m")
            
            # Get symbols for chart creation
            symbols = positions_df['Symbol'].tolist()[:6]  # Limit to 6 symbols
            market_data = self._fetch_market_data(symbols)
            
            if market_data:
                self._create_charts(positions_df, market_data, account, save_png)
                
        except Exception as e:
            print(f"Error creating dashboard: {e}")

    def _fetch_market_data(self, symbols, days_back=1):
        market_data = {}
        end = datetime.now()
        start = end - timedelta(days=days_back)
        
        try:
            for symbol in symbols:
                bars = self.api.get_bars(
                    symbol,
                    '5Min',
                    start.strftime('%Y-%m-%d'),
                    end.strftime('%Y-%m-%d'),
                    adjustment='raw'
                )
                if bars:
                    market_data[symbol] = pd.DataFrame([{
                        'timestamp': b.t,
                        'open': b.o,
                        'high': b.h,
                        'low': b.l,
                        'close': b.c,
                        'volume': b.v
                    } for b in bars])
            return market_data
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def _create_charts(self, positions_df, market_data, account_info, save_png):
        n_positions = len(market_data)
        if n_positions == 0:
            return
            
        rows = (n_positions + 1) // 2
        fig, axs = plt.subplots(rows, 2, figsize=(15, 5*rows))
        fig.patch.set_facecolor('#1E1E1E')
        
        if rows == 1:
            axs = axs.reshape(1, -1)
            
        for idx, (symbol, data) in enumerate(market_data.items()):
            row = idx // 2
            col = idx % 2
            ax = axs[row, col]
            
            position = positions_df[positions_df['Symbol'] == symbol].iloc[0]
            is_profit = position['P&L'] >= 0
            color = self.colors['profit'] if is_profit else self.colors['loss']
            
            ax.plot(data.index, data['close'], color=color, linewidth=2)
            ax.set_title(f"{symbol} (${position['Current']:.2f})", color=self.colors['neutral'])
            ax.grid(True, color=self.colors['grid'], alpha=0.3)
            ax.tick_params(colors=self.colors['neutral'])
            
            # Add entry price line
            ax.axhline(y=position['Avg Entry'], color=self.colors['accent'], 
                      linestyle='--', alpha=0.5)
            
            # Customize appearance
            ax.set_facecolor('#2E2E2E')
            for spine in ax.spines.values():
                spine.set_color(self.colors['grid'])
                
        # Remove empty subplots if odd number of positions
        if n_positions % 2 == 1:
            axs[-1, -1].remove()
            
        plt.tight_layout()
        if save_png:
            plt.savefig('dashboard.png', facecolor='#1E1E1E', bbox_inches='tight')
        else:
            plt.show()