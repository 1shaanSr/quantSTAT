class TradeExecutor:
    def __init__(self, api_handler):
        self.api = api_handler.api

    def execute_trade(self):
        print("\n=== Manual Trade Execution ===")
        while True:
            print("\nSelect trade type:")
            print("1. Buy (go long)")
            print("2. Sell (close owned shares)")
            print("3. Short (sell to open)")
            print("4. Exit short (buy to cover)")
            print("5. Exit to main menu")
            trade_type = input("Enter choice (1-5): ").strip()
            
            if trade_type == "5":
                print("Returning to main menu.")
                return

            symbol = input("Enter symbol to trade: ").strip().upper()
            if not symbol:
                print("Invalid input. Trade cancelled.")
                continue

            if trade_type == "1":
                self._handle_buy(symbol)
            elif trade_type == "2":
                self._handle_sell(symbol)
            elif trade_type == "3":
                self._handle_short(symbol)
            elif trade_type == "4":
                self._handle_cover(symbol)
            else:
                print("Invalid choice. Please select a valid option.")

    def _handle_buy(self, symbol):
        qty = input("Enter quantity to BUY: ").strip()
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

    def _handle_sell(self, symbol):
        try:
            positions = self.api.list_positions()
            position = next((p for p in positions if p.symbol.upper() == symbol), None)
            if not position:
                print(f"No position found for {symbol}")
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

    def _handle_short(self, symbol):
        qty = input("Enter quantity to SHORT: ").strip()
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

    def _handle_cover(self, symbol):
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