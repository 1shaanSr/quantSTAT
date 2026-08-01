# QuantSTAT: Market-Neutral Statistical Arbitrage

A systematic pairs-trading research platform: Engle-Granger cointegration
screening across an economically-grouped universe, a Kalman-filtered
(recursive Bayesian) dynamic hedge ratio, precision-weighted position
sizing, and a locked train/test walk-forward split so every reported
number is genuinely out-of-sample.

## Methodology

1. **Universe** (`src/universe.py`): ~100 liquid US-listed tickers grouped
   into buckets with an a priori economic rationale -- sector ETFs,
   same-industry stock pairs (e.g. Mastercard/Visa, Goldman Sachs/Morgan
   Stanley), commodity-tracking ETF pairs, credit/rates ETFs. Pairs are
   only tested within a bucket, never across the full cross-product, to
   keep every candidate pair economically grounded rather than a product
   of blind data mining.
2. **Formation / test split**: real daily price history is split 65%
   formation (in-sample) / 35% test (out-of-sample). All pair selection
   and hyperparameter tuning happens on the formation window; the test
   window is untouched until final evaluation.
3. **Pair screening** (`src/cointegration.py`): Engle-Granger cointegration
   test (p < 0.05) plus an Ornstein-Uhlenbeck half-life filter (2-60
   trading days), computed only on the formation window.
4. **Hyperparameter tuning** (`src/pairs_engine.py`): a grid search over
   entry/exit z-score thresholds, evaluated via rolling folds strictly
   inside the formation window.
5. **Signal** (`src/kalman_hedge.py`): a Kalman filter recursively
   estimates a time-varying hedge ratio in place of a static OLS fit. The
   trading signal is an EWMA-smoothed version of the filter's normalized
   innovation (standardized one-step-ahead prediction error) -- causal by
   construction, using only information available up to the current day.
6. **Position sizing**: scaled by the filter's posterior precision on the
   hedge ratio -- smaller positions when the relationship is poorly
   identified, larger when well identified.
7. **Portfolio construction**: validated pairs share one pooled capital
   base with a cap on total concurrent gross notional, rather than static
   per-pair silos.
8. **Risk diagnostics** (`src/risk_analysis.py`): a market-beta regression
   verifies the book is empirically market-neutral, not just neutral by
   construction; a hyperparameter sensitivity sweep checks whether the
   locked configuration sits on a stable region of the search space.

## Out-of-sample results

Test window: 2023-09-26 to 2026-07-31 (714 trading days, 2.83 years),
never used in pair selection or hyperparameter tuning.

| Metric | Value |
|---|---|
| Annualized return (CAGR) | +0.98% |
| Annualized volatility | 1.39% |
| Sharpe (rf=0%) | 0.71 |
| Sharpe (rf=4.5% T-bill) | -2.46 |
| Sortino (rf=0%) | 0.37 |
| Max drawdown | -1.41% |
| Calmar | 0.70 |
| Trades | 21, across 11 validated pairs |
| Beta to SPY | -0.0006 (t-stat -0.18, not significant) |

The two Sharpe figures reflect a genuine choice in how to benchmark a
market-neutral book -- see `TECHNICAL_DOCS.md` for which applies to which
claim. Full derivation, including the exact hyperparameters and the
diagnostics above, is in `TECHNICAL_DOCS.md`.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

Prompts let you highlight a symbol in the report, trim the report to a
trailing window, re-run hyperparameter tuning from scratch (reproducing
the locked configuration), and run the market-beta/sensitivity
diagnostics. All flags are also available directly:

```python
from src.backtester import StatisticalArbitrageBacktester
bt = StatisticalArbitrageBacktester()
bt.run(sensitivity=True)
```

## Architecture

```
src/
  universe.py        # economically-grouped candidate pair universe
  cointegration.py    # Engle-Granger + half-life screening
  kalman_hedge.py       # recursive Bayesian hedge-ratio filter
  metrics.py             # Sharpe/Sortino/MaxDD/Calmar from an equity curve
  pairs_engine.py         # formation/test split, tuning, pooled-portfolio simulation
  risk_analysis.py         # market-beta exposure + hyperparameter sensitivity
  allocation.py             # minimum-variance capital allocation (opt-in; see TECHNICAL_DOCS.md 3.2)
  backtester.py               # orchestrates the pipeline
```

## Disclaimer

For educational and research purposes. Not investment advice. Past
performance, in-sample or out-of-sample, does not guarantee future
results.
