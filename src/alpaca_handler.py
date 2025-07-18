import alpaca_trade_api as tradeapi
import getpass

class AlpacaHandler:
    def __init__(self):
        self.api = None

    def setup(self):
        print("=== Alpaca Trading Setup ===")
        API_KEY = input("Enter your Alpaca API Key ID: ")
        API_SECRET = getpass.getpass("Enter your Alpaca API Secret Key: ")
        
        try:
            self.api = tradeapi.REST(API_KEY, API_SECRET, "https://paper-api.alpaca.markets", api_version='v2')
            account = self.api.get_account()
            print(f"Connected successfully! Account status: {account.status}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def get_account_info(self):
        try:
            account = self.api.get_account()
            print("DEBUG: Alpaca account buying_power =", account.buying_power)
            return {
                'status': account.status,
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'day_trade_buying_power': float(getattr(account, 'day_trade_buying_power', account.buying_power))
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}