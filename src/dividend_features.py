"""
Dividend-based factors: genuinely point-in-time-safe, unlike financial-
statement ratios. A dividend payment is a historical fact fixed on its
ex-dividend date and never restated -- fundamentally different from a
`.info` snapshot or a quarterly-financials pull that only returns the last
few periods as currently known (verified: this data source's quarterly
financials only go back ~5 quarters and annual statements only ~5 years,
far too shallow for this project's walk-forward validation).

Coverage is real but partial: growth/tech non-payers (e.g. Amazon, Tesla,
Netflix, AMD, and some others before they initiated dividends) get an
honest 0, not a fabricated value.

Tested (see TECHNICAL_DOCS.md section 4.7): adding these factors did not
improve the ML predictor's formation-period information coefficient.
Provided here as opt-in, documented, reproducible code, not because it
changed the production decision.
"""
import numpy as np
import pandas as pd
import yfinance as yf


def fetch_dividend_history(tickers, start, end):
    div_by_ticker = {}
    for t in tickers:
        try:
            div = yf.Ticker(t).dividends
            div.index = div.index.tz_localize(None)
            div = div[(div.index >= start) & (div.index <= end)]
            div_by_ticker[t] = div
        except Exception:
            div_by_ticker[t] = pd.Series(dtype=float)
    return div_by_ticker


def build_dividend_features(close: pd.DataFrame, div_by_ticker: dict):
    dates = close.index
    trailing_yield = pd.DataFrame(0.0, index=dates, columns=close.columns)
    div_growth = pd.DataFrame(0.0, index=dates, columns=close.columns)

    for ticker in close.columns:
        div = div_by_ticker.get(ticker, pd.Series(dtype=float))
        if len(div) == 0:
            continue
        div_daily = div.reindex(dates, fill_value=0.0)
        trailing_sum = div_daily.rolling(252, min_periods=1).sum()
        trailing_yield[ticker] = trailing_sum / close[ticker]

        prior_year_sum = trailing_sum.shift(252)
        growth = (trailing_sum - prior_year_sum) / prior_year_sum.replace(0, np.nan)
        div_growth[ticker] = growth.fillna(0.0)

    return trailing_yield, div_growth
