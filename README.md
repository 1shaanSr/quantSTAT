# Risk-Parity Portfolio with an ML Factor-Tilt Overlay

A systematic portfolio construction platform: a walk-forward gradient-
boosting model predicts short-horizon stock returns from causal price/
volume factors, evaluated honestly via information coefficient rather than
an inflated trading Sharpe, and a risk-parity allocation across a liquid
large-cap universe forms the diversified base -- with the ML signal blended
in only to the extent formation-period evidence says it actually helps.

## Methodology

1. **Universe** (`src/universe.py`): 77 liquid large-cap US equities across
   all major sectors, each trading well above $10M/day in average dollar
   volume -- every name here is genuinely tradable at meaningful size.
2. **Factors** (`src/features.py`): 12-1 month momentum, 1-month momentum,
   1- and 2-week reversal, 21-day realized volatility, volume trend, and
   distance from the 252-day high. Deliberately price/volume-derived only
   -- fundamental ratios (P/E, ROE) from a source like a live `.info` call
   are current-day snapshots, not point-in-time historical series, and
   using them for historical training features would silently leak
   today's known fundamentals into the past.
3. **ML predictor** (`src/ml_predictor.py`): a conservatively-regularized
   gradient boosting model, retrained at every 20-trading-day rebalance
   using only data whose labels are already fully realized (an expanding
   walk-forward window), predicting each stock's forward 20-day return.
   Evaluated via the Spearman rank correlation between predictions and
   realized returns (information coefficient), the standard honest metric
   for a return-prediction model.
4. **Portfolio construction** (`src/risk_parity.py`): a long-only,
   equal-risk-contribution (risk-parity) base allocation, solved via
   constrained optimization over a shrinkage-regularized covariance
   matrix. The ML signal, when used, tilts these weights by a confidence-
   scaled multiplier, capped so no single high-scoring name can dominate
   the book.
5. **Formation / test discipline**: the walk-forward split, ML tilt
   strength, and every other design choice were selected using only a
   formation window; every headline number below comes from a test window
   neither the model nor the tilt decision ever saw.

## Out-of-sample results

Test window: 2023-04-26 to 2026-07-07 (41 rebalance periods, ~3.2 years),
never used to select the tilt strength or any other configuration choice.

| Metric | Value |
|---|---|
| Annualized return (CAGR) | +14.08% |
| Annualized volatility | 9.27% |
| Sharpe (rf=0%) | 1.47 |
| Sharpe (rf=4.5% T-bill) | 1.00 |
| Sortino (rf=0%) | 2.02 |
| Max drawdown | -11.07% |
| Calmar | 1.27 |
| ML tilt strength used | 0.0 (pure risk parity) |

The ML predictor's true out-of-sample information coefficient is +0.048
(t-stat 1.65) -- a modest, not-fully-significant signal, roughly in the
range considered a genuinely useful factor at the individual-stock level
in the quantitative equity literature. Formation-period evidence found
that acting on this signal (tilting risk-parity weights toward it) did
not improve portfolio Sharpe, so the locked configuration uses pure risk
parity with no tilt -- the ML component's honest, disciplined finding is
that it doesn't currently earn its way into the book, not that it was
force-fit in. Full derivation, including cost sensitivity and year-by-year
consistency, in `TECHNICAL_DOCS.md`.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Architecture

```
src/
  universe.py           # liquid large-cap ticker list
  features.py            # causal, point-in-time-safe factor construction
  ml_predictor.py          # walk-forward gradient boosting + IC evaluation
  risk_parity.py             # risk-parity solver + confidence-scaled tilt
  portfolio_backtest.py        # rebalance/hold/cost simulation engine
  metrics.py                     # Sharpe/Sortino/MaxDD/Calmar from an equity curve
  backtester.py                    # orchestrates the pipeline
```

## Disclaimer

For educational and research purposes. Not investment advice. Past
performance, in-sample or out-of-sample, does not guarantee future
results.
