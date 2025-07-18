from datetime import datetime
import time
from .config import Config
from .data_handler import DataHandler

class Strategy:
    def __init__(self, api_handler):
        self.api = api_handler.api
        self.eastern = Config.EASTERN_TZ
        self.symbol = Config.STRATEGY['default_symbol']
        self.risk_pct = Config.STRATEGY['risk_pct']
        self.refresh_interval = Config.STRATEGY['refresh_interval']
        self.data_handler = DataHandler()
        self._init_state()

    def _init_state(self):
        self.active = False
        self.position = None
        self.order_id = None

    def run_automated_strategy(self):
        print(f"\n=== Starting Automated Strategy on {self.symbol} ===")
        self.active = True
        
        try:
            while self.active:
                now = datetime.now(self.eastern)
                
                # Check market hours
                if not self._is_market_open(now):
                    print("Market is closed. Waiting...")
                    self._check_exit()
                    time.sleep(60)
                    continue

                # Get market data
                df = self.data_handler.fetch_market_data(self.symbol)
                if df is None or df.empty:
                    print("No data available. Retrying...")
                    self._check_exit()
                    time.sleep(self.refresh_interval)
                    continue

                # Check positions and generate signals
                self._process_data(df)
                
                self._check_exit()
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\nStrategy stopped by user.")
        finally:
            self.active = False

    def _is_market_open(self, now):
        return (
            (now.hour > 9 or (now.hour == 9 and now.minute >= 30))
            and now.hour < 16
        )

    def _check_exit(self):
        user_input = input("Type 'exit' to stop or press Enter to continue: ").strip().lower()
        if user_input == 'exit':
            self.active = False

    def _process_data(self, df):
        # Implementation of your strategy logic here
        pass