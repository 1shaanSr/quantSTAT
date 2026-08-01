"""
QuantRSI Trading Dashboard
-------------------------
A professional algorithmic trading platform that implements RSI-based
strategies with real-time market data and backtesting capabilities.

Author: [Your Name]
Date: July 2025
"""

from typing import Optional
from src.alpaca_handler import AlpacaHandler
from src.trade_executor import TradeExecutor
from src.dashboard import Dashboard
from src.strategy import Strategy
from src.backtester import StatisticalArbitrageBacktester

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
        backtester = StatisticalArbitrageBacktester(api_handler)

        print("PROFESSIONAL TRADING DASHBOARD")
        print("=" * 50)

        while True:
            try:
                _display_menu()
                choice = input("\nEnter choice (1-4): ").strip()
                
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
    print("3. Exit")
    print("4. Statistical Arbitrage Backtesting")

def _handle_menu_choice(
    choice: str,
    trade_exec: TradeExecutor,
    dashboard: Dashboard,
    strategy: Strategy,
    backtester: StatisticalArbitrageBacktester
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
        if trade_exec.api is None:
            print("No live Alpaca connection -- restart and enter API credentials to trade.")
        else:
            trade_exec.execute_trade()
    elif choice == '2':
        if dashboard.api is None:
            print("No live Alpaca connection -- restart and enter API credentials for the dashboard.")
        else:
            dashboard.create_enhanced_dashboard()
    elif choice == '3':
        print("Goodbye!")
        return False
    elif choice == '4':
        _handle_backtest(backtester)
    else:
        print("Invalid choice. Please try again.")
    return True

def _handle_backtest(backtester: StatisticalArbitrageBacktester) -> None:
    """
    Handle backtest parameter input and execution. This runs the full
    portfolio-level statistical arbitrage engine (see TECHNICAL_DOCS.md) --
    `symbol` only highlights a pair of interest in the report, and `days`
    only trims the reported window to the trailing N out-of-sample days.
    """
    symbol = input("Highlight a symbol in the report (optional, e.g. SPY): ").strip().upper() or None
    days_raw = input("Trim report to trailing N out-of-sample days (blank = full test window): ").strip()
    days = int(days_raw) if days_raw.isdigit() else None
    retune_raw = input("Re-run hyperparameter tuning instead of using locked defaults? (y/N): ").strip().lower()
    retune = retune_raw == 'y'
    backtester.run(symbol=symbol, days=days, retune=retune)

if __name__ == "__main__":
    main()