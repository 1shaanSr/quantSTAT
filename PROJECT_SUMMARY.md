# Risk-Parity Portfolio with an ML Factor-Tilt Overlay -- Project Summary

## What this is

A systematic portfolio construction platform combining two distinct,
honestly-evaluated components: a walk-forward machine learning model that
predicts short-horizon stock returns from causal price/volume factors, and
a risk-parity allocation framework that decides how much to actually act
on those predictions.

## Approach

- **Liquid universe**: 77 large-cap US equities, each individually
  verified to trade well above $10M/day -- a constraint imposed after this
  project's history found that ignoring liquidity can manufacture a
  spectacular-looking but entirely fake backtest result (see below).
- **Causal factor engineering**: momentum, reversal, volatility, and
  volume-trend factors computed from price/volume data only, deliberately
  avoiding fundamental-ratio features that can't be trusted point-in-time.
- **Walk-forward ML prediction**: a conservatively regularized gradient
  boosting model, retrained at every rebalance using only fully-realized
  historical labels, evaluated by information coefficient (the standard,
  honest metric for a return-prediction model) rather than an inflated
  trading Sharpe.
- **Risk-parity portfolio construction**: a long-only, equal-risk-
  contribution allocation solved via constrained optimization -- genuine
  diversification math, not a forecast.
- **Confidence-scaled blending**: the ML signal tilts the risk-parity base
  only in proportion to formation-period evidence that it actually helps;
  formation testing found it currently doesn't, so the locked
  configuration reports that honestly rather than forcing it in.

## Results

Out-of-sample (2023-04-26 to 2026-07-07, ~3.2 years, tilt strength and all
other configuration chosen only from formation data): **CAGR +14.08%,
Sharpe (rf=0%) 1.47, max drawdown -11.07%, Calmar 1.27**. Every year in
the test window was individually profitable, and the result is materially
insensitive to transaction costs (Sharpe 1.48 at 5bps down to 1.43 at
50bps) -- both signs of a genuinely diversified result rather than a
concentrated or fragile one.

## Project history

This project replaced an earlier market-neutral pairs-trading strategy
(Engle-Granger cointegration, Kalman-filtered hedge ratios) that itself
achieved a real, validated 0.71 Sharpe. Four independently-tested
follow-on ideas -- alternative capital allocation, a wider pair universe,
structurally-enforced dual-class share arbitrage, and classic
cross-sectional reversal -- were all rigorously walk-forward tested and
none beat that baseline once genuine out-of-sample discipline was applied.
One (dual-class arbitrage) briefly appeared to reach Sharpe 3.4 before
being traced to an illiquid-security artifact: one share class traded
only ~100-300 shares/day, meaning its "price" in the backtest was a stale,
non-executable print, not a real trade. That specific lesson -- verify
liquidity before trusting any result -- is why this project's universe is
restricted to names with $10M/day+ average volume from the start, and why
portfolio construction (a structurally more reliable source of edge than
price prediction) is the core of this project rather than a fallback.

## What this demonstrates

- Rigorous machine learning practice for financial time series: avoiding
  both look-ahead bias (point-in-time-safe features) and label-overlap
  leakage (non-overlapping sample dates), proper expanding-window
  validation, and honest evaluation via information coefficient rather
  than an overclaimed trading return.
- Correct implementation of risk-parity portfolio optimization, including
  covariance shrinkage and an explicit, disclosed discussion of what
  diversification benefit it does and doesn't provide within a single
  asset class.
- A disciplined decision process: the ML tilt was built, evaluated
  honestly, and NOT adopted because formation-only evidence didn't support
  it -- a real demonstration of resisting the temptation to force a
  sophisticated-looking component into the final result.
- A documented history of testing multiple approaches, catching a
  liquidity-driven false positive before it became the headline result,
  and pivoting the project's core thesis in response rather than chasing
  a bigger number.

## Architecture

See `README.md` for the module layout and `TECHNICAL_DOCS.md` for the full
mathematical derivation, the formation-only tilt-strength selection, and
the complete project history.

## Disclaimer

For educational and research purposes. Not investment advice. Past
performance, in-sample or out-of-sample, does not guarantee future
results.
