"""
Liquid large-cap US equity universe. Every ticker here is chosen for
genuine tradability -- a lesson learned the hard way in this project's
history (an earlier version explored dual-class share arbitrage and found
spectacular-looking backtest results that turned out to be entirely a
liquidity artifact: one share class trading only 100-5,000 shares/day,
producing stale, non-executable closing prices). All 77 names below trade
well above $10M/day in average dollar volume.
"""

TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'CRM', 'ADBE',
    'CSCO', 'INTC', 'AMD', 'TXN', 'QCOM', 'IBM',
    # Consumer discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'BKNG', 'DIS',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
    # Industrials
    'BA', 'CAT', 'HON', 'UPS', 'GE', 'LMT', 'RTX', 'DE', 'MMM', 'UNP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Consumer staples
    'PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'MO', 'PM',
    # Communication services
    'VZ', 'T', 'CMCSA', 'NFLX',
    # Utilities / real estate
    'NEE', 'DUK', 'SO', 'AMT', 'PLD',
]
