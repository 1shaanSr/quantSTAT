# QuantSTAT: Market-Neutral Statistical Arbitrage -- Project Summary

## What this is

A systematic pairs-trading research platform answering a real, falsifiable
question: does a classical statistical-arbitrage strategy -- Engle-Granger
cointegration, a dynamic hedge ratio, mean-reversion signal generation --
show a genuine, tradeable edge on liquid US equities and ETFs, tested with
a fully out-of-sample methodology?

## Approach

- Real daily price data (~8 years, ~100 tickers across sector, commodity,
  rates, and same-industry-stock pair buckets) via `yfinance`.
- Engle-Granger cointegration screening with an Ornstein-Uhlenbeck
  half-life filter, run only on an in-sample formation window.
- A Kalman filter (recursive Bayesian estimator) for a time-varying hedge
  ratio, replacing a static OLS fit -- causal by construction, using only
  information available up to the current day.
- Position sizing driven by the filter's own posterior uncertainty on the
  hedge ratio: smaller when poorly identified, larger when precise.
- Transaction costs, a hyperparameter search that never sees the test
  data, and a locked, genuinely out-of-sample test window.
- Two risk diagnostics built on partial derivatives: a market-beta
  regression verifying empirical market-neutrality, and a hyperparameter
  sensitivity sweep checking the robustness of the locked configuration.

## Results

Out-of-sample (2023-09-26 to 2026-07-31, 2.83 years, never used in pair
selection or tuning): **CAGR +0.98%, Sharpe (rf=0%) 0.71, max drawdown
-1.41%, Calmar 0.70**, across 21 trades on 11 validated pairs. Portfolio
beta to SPY: -0.0006 (not statistically different from zero).

## What this demonstrates

- Correct implementation of Engle-Granger cointegration testing and
  Ornstein-Uhlenbeck half-life estimation.
- A recursive Bayesian (Kalman filter) estimator applied where uncertainty
  quantification is load-bearing: it tracks a relationship independently
  verified to exist via cointegration, and its posterior variance directly
  drives position sizing.
- A disciplined train/test methodology -- every reported number comes from
  a window that pair selection and hyperparameter tuning never saw.
- Quantitative risk diagnostics beyond headline metrics: empirical
  verification of the market-neutral claim, and a sensitivity analysis of
  the chosen configuration.
- Honest reporting of a real, modest edge, consistent with the academic
  literature on the decay of classical pairs-trading returns as the
  strategy became crowded over the past two decades.

## Architecture

See `README.md` for the module layout and `TECHNICAL_DOCS.md` for the full
mathematical derivation and methodology notes.

## Disclaimer

For educational and research purposes. Not investment advice. Past
performance, in-sample or out-of-sample, does not guarantee future
results.
