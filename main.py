"""
QuantSTAT: Market-Neutral Statistical Arbitrage
------------------------------------------------
Engle-Granger cointegration screening, a Kalman-filtered dynamic hedge
ratio, and a locked train/test walk-forward split, run over real market
data. See README.md for methodology and TECHNICAL_DOCS.md for the full
derivation.
"""

from src.backtester import StatisticalArbitrageBacktester


def main() -> None:
    print("QuantSTAT -- Market-Neutral Statistical Arbitrage")
    print("=" * 55)

    symbol = input("Highlight a symbol in the report (optional, e.g. SPY): ").strip().upper() or None
    days_raw = input("Trim report to trailing N out-of-sample days (blank = full test window): ").strip()
    days = int(days_raw) if days_raw.isdigit() else None
    retune = input("Re-run hyperparameter tuning instead of using locked defaults? (y/N): ").strip().lower() == 'y'
    sensitivity = input("Also run market-beta and hyperparameter sensitivity analysis? (y/N): ").strip().lower() == 'y'

    backtester = StatisticalArbitrageBacktester()
    backtester.run(symbol=symbol, days=days, retune=retune, sensitivity=sensitivity)


if __name__ == "__main__":
    main()
