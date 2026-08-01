# QuantSTAT: Market-Neutral Statistical Arbitrage (Pairs Trading)

A systematic pairs-trading research pipeline: Engle-Granger cointegration
screening across an economically-grouped universe, a Kalman-filtered
(recursive Bayesian) dynamic hedge ratio in place of a static OLS estimate,
precision-weighted position sizing, and a genuine train/test walk-forward
split so every reported number is out-of-sample.

## Why this exists

An earlier version of this project reported backtest results (20-60%
annual returns, Sharpe 1.8-2.4, Calmar 4.5-11.2) that turned out to be
generated against **synthetic price data** fabricated by the backtester
itself, using a hardcoded mean-reverting process engineered to produce
frequent, profitable-looking trades. None of those numbers reflected real
market behavior. This version replaces that pipeline end to end: real
market data, a proper cointegration test, no look-ahead bias, transaction
costs, and a locked train/test split. See `TECHNICAL_DOCS.md` for the full
derivation, including approaches that were tried and rejected because they
didn't hold up out-of-sample.

## Methodology

1. **Universe**: ~100 liquid US-listed tickers grouped into buckets with an
   a priori economic rationale (sector ETFs, same-industry stock pairs,
   commodity-tracking ETF pairs, credit/rates ETFs, etc.) -- see
   `src/universe.py`. Pairs are only tested within a bucket, not across the
   full cross-product, to avoid blind data mining. Country-vs-country
   equity ETF pairs are deliberately excluded (see comment in
   `src/universe.py` for why).
2. **Formation / test split**: the real daily price history (2018-2026) is
   split 65% formation (in-sample) / 35% test (out-of-sample). Formation
   data is used for cointegration screening and hyperparameter tuning; test
   data is never touched until final evaluation.
3. **Pair screening** (`src/cointegration.py`): Engle-Granger cointegration
   test (p<0.05) plus an Ornstein-Uhlenbeck half-life filter (2-60 trading
   days), computed only on the formation window.
4. **Hyperparameter tuning** (`src/pairs_engine.py:tune_hyperparams`): a
   small grid search over entry/exit z-score thresholds, evaluated via
   rolling folds strictly inside the formation window.
5. **Signal**: a Kalman filter (`src/kalman_hedge.py`) recursively estimates
   a time-varying hedge ratio; the trading signal is an EWMA-smoothed
   version of the filter's normalized innovation (standardized one-step-
   ahead prediction error). This replaces a fixed-window OLS hedge ratio,
   which lags regime changes.
6. **Position sizing**: scaled by the filter's posterior precision on the
   hedge ratio (`KalmanHedge.precision_weight`) -- smaller positions when
   the relationship is poorly identified, larger when well identified.
7. **Portfolio construction**: validated pairs share one capital pool
   (rather than static per-pair silos that leave capital idle when a pair
   has no open signal), with a cap on total concurrent gross notional.

## Verified out-of-sample results

Test window: 2023-09-26 to 2026-07-31 (714 trading days, 2.83 years),
never used in pair selection or hyperparameter tuning.

| Metric | Value |
|---|---|
| Annualized return (CAGR) | +0.98% |
| Annualized volatility | 1.39% |
| Sharpe (rf=0%) | 0.71 |
| Sharpe (rf=4.5% T-bill) | -2.46 |
| Sortino (rf=0%) | 0.37 |
| Max drawdown | -1.41% (peak 2026-06-23 -> trough 2026-07-29) |
| Calmar | 0.70 |
| Trades | 21, across 11 validated pairs |

The two Sharpe figures reflect a genuine ambiguity in how to benchmark a
market-neutral book, not a computation error -- see `TECHNICAL_DOCS.md` for
which one is appropriate for which claim. These numbers are modest by
design: they are what survives a real cointegration test, real transaction
costs, and an untouched test set, on liquid, heavily-arbitraged
instruments. That is the honest ceiling for a simple version of this
strategy today, not a limitation of the code.

## Installation

```bash
pip install -r requirements.txt
```

Live/paper trade execution via Alpaca is optional and kept in a separate
file because it has an unresolvable dependency conflict with `yfinance`
(see `requirements-live.txt` for details) -- install it in its own
virtualenv only if you need live execution.

## Running

```bash
python main.py
```

Type `skip` at the Alpaca prompt to go straight to the menu -- backtesting
(option 4) needs no live account. Option 4 downloads the universe, screens
for cointegration, and runs the full portfolio backtest on real data.

## Architecture

```
src/
  universe.py       # economically-grouped candidate pair universe
  cointegration.py  # Engle-Granger + half-life screening
  kalman_hedge.py    # recursive Bayesian hedge-ratio filter
  metrics.py         # Sharpe/Sortino/MaxDD/Calmar from an equity curve
  pairs_engine.py    # formation/test split, tuning, pooled-portfolio simulation
  backtester.py      # orchestrates the pipeline for main.py
  data_handler.py    # yfinance + RSI/VWAP/ATR indicators (used by strategy.py/dashboard.py)
  alpaca_handler.py  # optional live/paper trading connection
  trade_executor.py  # manual order entry via Alpaca
  dashboard.py        # portfolio visualization via Alpaca
  strategy.py         # simple RSI demo loop (separate from the pairs engine)
```

## Disclaimer

For educational and research purposes. Past performance, in-sample or
out-of-sample, does not guarantee future results.
