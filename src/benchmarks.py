"""
Benchmark comparison: contextualizes the risk-parity result against naive
alternatives, plus two well-known historical crisis sub-periods (COVID
crash, 2022 rate-hike bear market) to show WHEN risk parity's defensive
positioning actually earns its keep. The crisis dates are fixed,
independently-documented historical events chosen before looking at any
result here, not selected after the fact to flatter the comparison.

Two different conventions are used deliberately, for two different
questions:

- Test-window comparison: each strategy starts fresh with $100k cash at
  the first test-window rebalance date, matching exactly how this
  project's headline result is defined everywhere else.
- Crisis sub-periods: each strategy runs CONTINUOUSLY from the start of
  available history and is sliced at the crisis window, representing what
  an already-invested, ongoingly-rebalanced portfolio would have
  experienced. A "fresh start exactly at the crisis boundary" convention
  was tried first and rejected: it turned out to be highly sensitive to
  which arbitrary date the 20-day rebalance grid happened to land on
  relative to the crisis onset (whether the portfolio's single entry
  trade landed a few days before or after the crash began materially
  changed the result) -- not a property of the strategy, an artifact of
  the measurement convention. A continuously-run, already-invested
  portfolio is both more realistic and not sensitive to that artifact.
"""
import numpy as np
import pandas as pd
import yfinance as yf

BENCHMARK_TICKERS = ['SPY', 'IEF']

CRISIS_PERIODS = {
    'COVID crash': ('2020-02-14', '2020-04-15'),
    '2022 rate-hike bear market': ('2022-01-03', '2022-10-14'),
}


def fetch_benchmark_prices(start, end):
    raw = yf.download(BENCHMARK_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    return raw['Close']


def run_fixed_weight_portfolio(prices: pd.DataFrame, target_weights: dict, rebalance_dates,
                                cost_bps=10, capital=100_000.0):
    """Rebalance to fixed target weights at each rebalance date, hold between."""
    returns = prices.pct_change()
    dates = prices.index
    tickers = list(target_weights.keys())
    target = pd.Series(target_weights)

    daily_pnl = pd.Series(0.0, index=dates)
    current_weights = pd.Series(0.0, index=tickers)
    rebal_set = set(rebalance_dates)

    for i, date in enumerate(dates):
        if i > 0:
            day_ret = returns.iloc[i][tickers].fillna(0.0)
            daily_pnl.iloc[i] += (current_weights * day_ret * capital).sum()
        if date in rebal_set:
            turnover = (target - current_weights).abs().sum()
            daily_pnl.iloc[i] -= (cost_bps / 10000.0) * turnover * capital
            current_weights = target.copy()

    return daily_pnl


def build_benchmark_curves(close: pd.DataFrame, bench_prices: pd.DataFrame, start, end,
                            cost_bps=10, capital=100_000.0, rebalance_every=20):
    """
    Equal-weight 1/N (same universe), SPY buy-and-hold, 60/40 SPY/IEF --
    each started FRESH with `capital` at the first rebalance date within
    [start, end]. Use for the test-window headline comparison.
    """
    rebalance_dates = close.loc[start:end].index[::rebalance_every]

    n = len(close.columns)
    eq_weights = {t: 1.0 / n for t in close.columns}
    pnl_eq = run_fixed_weight_portfolio(close, eq_weights, rebalance_dates, cost_bps, capital)
    eq_curve = (capital + pnl_eq.cumsum()).loc[start:end]

    pnl_spy = run_fixed_weight_portfolio(bench_prices[['SPY']], {'SPY': 1.0}, rebalance_dates, cost_bps, capital)
    spy_curve = (capital + pnl_spy.cumsum()).loc[start:end]

    pnl_6040 = run_fixed_weight_portfolio(bench_prices[['SPY', 'IEF']], {'SPY': 0.6, 'IEF': 0.4},
                                           rebalance_dates, cost_bps, capital)
    curve_6040 = (capital + pnl_6040.cumsum()).loc[start:end]

    return {'Equal-weight 1/N': eq_curve, 'SPY buy-and-hold': spy_curve, '60/40 SPY/IEF': curve_6040}


def build_continuous_benchmark_curves(close: pd.DataFrame, bench_prices: pd.DataFrame,
                                       cost_bps=10, capital=100_000.0, rebalance_every=20):
    """
    Same three benchmarks, but run ONCE continuously across the entire
    available history (close.index[0] to close.index[-1]) with the
    standard rebalance cadence. Callers slice whatever sub-window they
    need (e.g. a crisis period) from the returned curves -- this
    represents an already-invested, ongoingly-managed portfolio, not one
    that happens to start fresh exactly at the window boundary.
    """
    start, end = close.index[0], close.index[-1]
    rebalance_dates = close.index[::rebalance_every]

    n = len(close.columns)
    eq_weights = {t: 1.0 / n for t in close.columns}
    pnl_eq = run_fixed_weight_portfolio(close, eq_weights, rebalance_dates, cost_bps, capital)
    eq_curve = capital + pnl_eq.cumsum()

    pnl_spy = run_fixed_weight_portfolio(bench_prices[['SPY']], {'SPY': 1.0}, rebalance_dates, cost_bps, capital)
    spy_curve = capital + pnl_spy.cumsum()

    pnl_6040 = run_fixed_weight_portfolio(bench_prices[['SPY', 'IEF']], {'SPY': 0.6, 'IEF': 0.4},
                                           rebalance_dates, cost_bps, capital)
    curve_6040 = capital + pnl_6040.cumsum()

    return {'Equal-weight 1/N': eq_curve, 'SPY buy-and-hold': spy_curve, '60/40 SPY/IEF': curve_6040}


def crisis_period_return_dd(curve: pd.Series, start, end):
    sub = curve.loc[start:end]
    if len(sub) < 2:
        return None
    ret = sub.iloc[-1] / sub.iloc[0] - 1
    dd = (sub / sub.cummax() - 1).min()
    return ret, dd
