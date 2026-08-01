# QuantSTAT: Market-Neutral Statistical Arbitrage -- Project Summary

## What this is

A systematic pairs-trading research pipeline built around a real,
falsifiable question: does a classical statistical-arbitrage strategy
(Engle-Granger cointegration, dynamic hedge ratio, mean-reversion signal)
show a genuine, tradeable edge on liquid US equities/ETFs today, once
tested honestly?

## What changed from the original version

The original implementation reported 20-60% annual returns and a Calmar
ratio up to 11.2. Those numbers came from a backtester that generated
**synthetic price data** engineered to mean-revert, never touched real
market history, and never actually computed Sharpe, drawdown, or Calmar
in code -- the numbers were written directly into the documentation as
prose. Full details in `TECHNICAL_DOCS.md`.

This version replaces the entire pipeline:

- Real daily price data (~8 years, ~100 tickers) via `yfinance`.
- A real Engle-Granger cointegration screen with an Ornstein-Uhlenbeck
  half-life filter, run only on an in-sample formation window.
- A Kalman filter (recursive Bayesian estimator) for a time-varying hedge
  ratio, replacing a static/full-sample OLS fit -- eliminating the
  original's look-ahead bias by construction, since the filter only ever
  uses information available up to the current day.
- Position sizing driven by the filter's own posterior uncertainty on the
  hedge ratio (smaller size when poorly identified, larger when precise)
  -- a genuine use of Bayesian uncertainty quantification in the risk
  management step, not just in the point estimate.
- Transaction costs, a locked hyperparameter search that never sees the
  test data, and a true out-of-sample test window.

## Honest results

Out-of-sample (2023-09-26 to 2026-07-31, 2.83 years, never used in pair
selection or tuning): **CAGR +0.98%, Sharpe (rf=0%) 0.71, max drawdown
-1.41%, Calmar 0.70**, across 21 trades on 11 validated pairs.

These are modest, not spectacular -- and that is the point. They are what
a disciplined, walk-forward-validated implementation of this strategy
actually produces on liquid, heavily-arbitraged instruments, consistent
with the academic literature documenting the decay of classical
pairs-trading returns as the strategy became crowded over the past two
decades. `TECHNICAL_DOCS.md` documents two intermediate approaches that
were tried and rejected specifically because they looked better in-sample
but failed to generalize out-of-sample -- that record is part of the work,
not a footnote.

## What this demonstrates

- Correct implementation of Engle-Granger cointegration testing and
  Ornstein-Uhlenbeck half-life estimation, not just a name-check of the
  method.
- A recursive Bayesian (Kalman filter) estimator applied where it is
  actually appropriate -- tracking a relationship that is independently
  verified to exist (via cointegration) and explicitly evolving under
  uncertainty -- rather than as unjustified added complexity.
- A disciplined train/test methodology: every reported number comes from
  a window that pair selection and hyperparameter tuning never saw, and
  every rejected alternative is documented rather than silently dropped.
- Honest reporting of a weak-but-real edge over pretending a stronger one
  exists.

## Architecture

See `README.md` for the module layout (`src/universe.py`,
`src/cointegration.py`, `src/kalman_hedge.py`, `src/metrics.py`,
`src/pairs_engine.py`, `src/backtester.py`).

## Disclaimer

For educational and research purposes. Not investment advice. Past
performance, in-sample or out-of-sample, does not guarantee future
results.
