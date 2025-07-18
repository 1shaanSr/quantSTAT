from datetime import datetime
import time
import pytz
from data_handler import DataHandler

class Strategy:
    def __init__(self, api_handler):
        self.api = api_handler.api if hasattr(api_handler, 'api') else api_handler
        self.eastern = pytz.timezone("US/Eastern")
        self.symbol = "SPY"
        self.risk_pct = 0.01
        self.refresh_interval = 60
        self.data_handler = DataHandler()
        self._init_state()

    def _init_state(self):
        self.active = False
        self.position = None
        self.order_id = None

    def run_automated_strategy(self):
        print(f"\n=== Starting Automated Strategy on {self.symbol} ===")
        print("Strategy is running in demo mode...")
        print("This would connect to live market data and execute trades.")
        
        # Simple demo implementation
        for i in range(5):
            print(f"Strategy cycle {i+1}/5...")
            print(f"Checking signals for {self.symbol}...")
            time.sleep(2)
            
            user_input = input("Type 'exit' to stop or press Enter to continue: ").strip().lower()
            if user_input == 'exit':
                break
        
        print("Strategy execution completed.")

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
        # Simple strategy implementation
        print(f"Processing data for {self.symbol}...")
        if len(df) > 0:
            latest = df.iloc[-1]
            print(f"Latest price: ${latest['Close']:.2f}")
            if 'RSI' in latest:
                print(f"RSI: {latest['RSI']:.2f}")
