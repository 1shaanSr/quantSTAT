import getpass

class AlpacaHandler:
    def __init__(self):
        self.api = None

    def setup(self):
        """
        Connect to Alpaca for live/paper trading. This is only required for
        menu options 1 (Execute Trade) and 2 (Dashboard), which need a live
        account. Backtesting (option 4) needs no live connection at all, so
        typing 'skip' here still leaves the app usable.
        """
        print("=== Alpaca Trading Setup ===")
        print("(Live trading and dashboard require an Alpaca account. Type 'skip' to")
        print(" go straight to the menu -- backtesting works without a connection.)")
        API_KEY = input("Enter your Alpaca API Key ID (or 'skip'): ")
        if API_KEY.strip().lower() == 'skip':
            print("Skipping Alpaca connection. Live trading and dashboard will be unavailable.")
            return True

        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            print("alpaca-trade-api is not installed (see requirements-live.txt).")
            print("Continuing without a live connection -- backtesting still works.")
            return True

        API_SECRET = getpass.getpass("Enter your Alpaca API Secret Key: ")
        try:
            self.api = tradeapi.REST(API_KEY, API_SECRET, "https://paper-api.alpaca.markets", api_version='v2')
            account = self.api.get_account()
            print(f"Connected successfully! Account status: {account.status}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Continuing without a live connection -- backtesting still works.")
            return True

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