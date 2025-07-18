"""
QuantRSI Trading Dashboard
-------------------------
A professional algorithmic trading platform that implements RSI-based
strategies with real-time market data and backtesting capabilities.

Author: [Your Name]
Date: July 2025
"""

from typing import Optional
from alpaca_handler import AlpacaHandler
from trade_executor import TradeExecutor
from dashboard import Dashboard
from strategy import Strategy
from backtester import Backtester

def main() -> None:
    """
    Main entry point for the QuantRSI trading platform.
    Handles user interface and high-level program flow.
    """
    try:
        api_handler = AlpacaHandler()
        if not api_handler.setup():
            return

        trade_exec = TradeExecutor(api_handler)
        dashboard = Dashboard(api_handler)
        strategy = Strategy(api_handler)
        backtester = Backtester(api_handler)

        print("PROFESSIONAL TRADING DASHBOARD")
        print("=" * 50)

        while True:
            try:
                _display_menu()
                choice = input("\nEnter choice (1-6): ").strip()
                
                if not _handle_menu_choice(choice, trade_exec, dashboard, strategy, backtester):
                    break
                    
            except Exception as e:
                print(f"Error in menu option: {e}")
                input("Press Enter to continue...")
                
    except Exception as e:
        print(f"Critical error: {e}")
        input("Press Enter to exit...")

def _display_menu() -> None:
    """Display the main menu options."""
    print("\nSELECT MODE:")
    print("1. Execute Trade")
    print("2. View Dashboard")
    print("3. Refresh Data")
    print("4. Exit")
    print("5. Automated Intraday Strategy")
    print("6. Backtest Strategy")

def _handle_menu_choice(
    choice: str,
    trade_exec: TradeExecutor,
    dashboard: Dashboard,
    strategy: Strategy,
    backtester: Backtester
) -> bool:
    """
    Handle user menu selection.
    
    Args:
        choice: User's menu selection
        trade_exec: Trading execution instance
        dashboard: Dashboard visualization instance
        strategy: Strategy implementation instance
        backtester: Backtesting engine instance
    
    Returns:
        bool: False if program should exit, True otherwise
    """
    if choice == '1':
        trade_exec.execute_trade()
    elif choice == '2':
        dashboard.create_enhanced_dashboard()
    elif choice == '3':
        print("Refreshing data...")
        dashboard.create_enhanced_dashboard()
    elif choice == '4':
        print("Goodbye!")
        return False
    elif choice == '5':
        strategy.run_automated_strategy()
    elif choice == '6':
        _handle_backtest(backtester)
    else:
        print("Invalid choice. Please try again.")
    return True

def _handle_backtest(backtester: Backtester) -> None:
    """Handle backtest parameter input and execution."""
    symbol = input("Enter symbol for backtest (default SPY): ").strip().upper() or "SPY"
    days = input("Enter number of days to backtest (default 5): ").strip()
    try:
        days = int(days)
    except ValueError:
        days = 5
    backtester.run(symbol, days)

if __name__ == "__main__":
    main()