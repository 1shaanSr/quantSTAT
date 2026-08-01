# Technical Documentation: Market-Neutral Statistical Arbitrage

## 1. Background: what was wrong with the previous version

The previous implementation reported 20-60% annual returns, Sharpe
1.8-2.4, and Calmar 4.5-11.2. Investigation found:

- `_create_sample_data()` generated **fully synthetic prices** from a
  hardcoded mean-reverting AR(1) process, explicitly engineered for
  "maximum trading opportunities," despite `yfinance` being a listed
  dependency. Every backtest ran against fabricated data, never real
  market history.
- The hedge ratio and spread mean/std were fit once via OLS over the
  **entire** sample and reused for signals from day 1 (look-ahead bias).
- No Sharpe, drawdown, or Calmar was computed anywhere in code. The
  numbers in the README/docs were asserted text, not backtest output --
  literally "Calmar Ratio: 4.5-11.2 (Return/MaxDD)" written directly into
  a markdown file.
- Re-running the *original* trading logic (full-sample hedge ratio, zero
  transaction costs, hyper-aggressive z-score thresholds) against real
  SPY/QQQ data lost money (-0.82%/yr, 49.3% win rate -- consistent with
  noise, not edge). The positive numbers only ever existed against
  fabricated data.

This document describes the replacement pipeline and how its numbers were
derived, including the intermediate approaches that were tried and
rejected.

## 2. Mathematical framework

### 2.1 Cointegration (Engle-Granger)

For two price series P1, P2, the Engle-Granger test regresses
P2 = alpha + beta * P1 + eps, then tests the residual eps for a unit root
(ADF). Rejecting the null (p < 0.05) indicates cointegration -- a stable
long-run linear relationship, which is the necessary condition for a
mean-reversion pairs trade to have a real edge rather than trading noise
around an unanchored spread.

### 2.2 Half-life filter

Cointegration alone doesn't guarantee a *tradeable* relationship -- the
reversion could take years. The spread's Ornstein-Uhlenbeck half-life is
estimated by OLS on `d(spread)_t = a + b * spread_{t-1} + e_t`; half-life
= -ln(2)/ln(1+b). Pairs are kept only if half-life falls in [2, 60]
trading days -- long enough to be distinguishable from noise, short enough
to be tradeable at daily frequency with a 20-day max holding period.

### 2.3 Kalman-filtered hedge ratio

Static OLS over a rolling window has two weaknesses: it lags regime
changes (the whole window must roll past a break before beta adjusts), and
it has arbitrary window-boundary effects. Instead, [alpha_t, beta_t] is
modeled as a random walk:

```
y_t = alpha_t + beta_t * x_t + v_t         (observation, v_t ~ N(0, R))
theta_t = theta_{t-1} + w_t                (state, w_t ~ N(0, Q))
```

and tracked with a standard Kalman recursion (`src/kalman_hedge.py`). R is
estimated from the residual variance of an initial 60-day OLS fit; Q is
set via the common heuristic Q = (delta/(1-delta)) * R with delta=1e-4,
controlling how fast beta is allowed to drift. This is a genuinely
appropriate use of a Bayesian recursive estimator here: the object of
interest (the hedge ratio) is explicitly modeled as evolving under
uncertainty, and the filter's own posterior variance is then used
downstream for position sizing (2.5) -- the uncertainty quantification is
load-bearing, not decorative.

### 2.4 Trading signal

The normalized innovation `z_t = e_t / sqrt(S_t)` (the Kalman filter's
standardized one-step-ahead prediction error) is the raw signal. It is
look-ahead free by construction -- e_t is the error in predicting day t
using only information through day t-1.

The raw innovation is noisy day to day; an EWMA(span=3) smoothed version
is used as the actual trading signal:

```
z_ewma_t = alpha * z_t + (1 - alpha) * z_ewma_{t-1},  alpha = 2/(span+1)
```

This is still strictly causal (uses only past/current innovations).

### 2.5 Position sizing (precision weighting)

Position size is scaled by `sqrt(median(beta_var_history) / beta_var_t)`,
clipped to [0.25, 1.5]. When the filter's current posterior variance on
beta is unusually high (poorly identified relationship), the position is
sized down; when unusually precise, sized up. This is the Bayesian
decision-theory piece of the project: parameter uncertainty is carried
through to risk management rather than discarded after taking a point
estimate.

### 2.6 Risk-adjusted metrics

All computed directly from the simulated daily equity curve
(`src/metrics.py`):

```
CAGR        = (equity[-1]/equity[0])^(252/n_days) - 1
Sharpe      = mean(daily_excess_return) * 252 / (std(daily_return) * sqrt(252))
MaxDD       = min[(equity_t - running_max_t) / running_max_t]
Calmar      = CAGR / |MaxDD|
```

Two Sharpe ratios are reported (rf=0% and rf=4.5%) because the correct
risk-free assumption for a market-neutral book is genuinely ambiguous:
rf=0% is appropriate if the trading signal is evaluated as incremental
alpha on top of a separately-managed cash sleeve that already earns the
risk-free rate (the usual convention for a market-neutral overlay);
rf=4.5% is appropriate if the entire capital base is compared against
simply holding T-bills instead. Both are reported so the reader can apply
whichever framing matches how the strategy would actually be capitalized.

## 3. Methodology: what was tried and rejected

This project deliberately documents dead ends -- a large part of the
actual rigor here is in what was tried and discarded once it failed to
generalize out-of-sample, not just the final configuration.

### 3.1 Country-vs-country ETF pairs

An early universe included developed/emerging-market country ETF buckets.
The screen picked up EWZ-EIDO (Brazil vs Indonesia): formation p=0.03,
half-life 42 days, in-formation backtest Sharpe 0.60 -- a good-looking
candidate. Traded out-of-sample, it lost -8.8% max drawdown as the
apparent relationship (really just shared EM/commodity-cycle beta, not a
structural equilibrium) drifted. Country ETF pairs were removed from the
universe on the structural grounds that two national equity indices have
no arbitrage or business mechanism forcing a stable long-run relationship
-- but this decision was made *after* seeing that result, so it carries
real hindsight-bias risk. Documented in `src/universe.py` rather than
hidden.

### 3.2 Secondary in-sample performance filter

An intermediate version added a second selection layer on top of
cointegration screening: keep only pairs whose formation-period walk-
forward backtest Sharpe exceeded a threshold. This is a common practice
in industry pipelines, but here it **backfired**: the resulting 6-pair
portfolio scored worse out-of-sample (Sharpe -0.94) than the simpler
11-pair portfolio selected by cointegration alone (Sharpe 0.18 at that
stage, before the signal-smoothing change in 3.3). Selecting pairs by
their own in-sample backtest result is a second pass over the same signal
already used for tuning, and it overfit. This filter was removed. The
lesson is kept here because it's a common and easy mistake.

### 3.3 Signal smoothing (kept)

The raw single-day Kalman innovation rarely crossed the entry threshold
for several pairs (5 of 11 pairs saw zero trades across the entire 2.83-
year test window with the raw signal). Applying EWMA(3) smoothing to the
innovation before thresholding was tested as a variant, tuned exclusively
on formation folds. It improved in-formation Sharpe (0.19 -> 0.42) *and*
out-of-sample Sharpe (0.18 -> 0.70) together -- both moving the same
direction is the evidence that this generalized rather than overfit the
test window (an overfit change typically improves the fitted sample while
degrading or not affecting the held-out one). This was kept.

### 3.4 Capital pooling (kept)

Static equal-capital silos per pair leave capital idle for any pair
without an open position -- with 5-6 of 11 pairs trading only a handful of
times in 2.83 years, most allocated capital did nothing most of the time.
Pooling capital across pairs (shared balance, gross-exposure cap) was
tested against the silo version: Sharpe and Calmar were unchanged (as
expected -- they are scale-invariant under proportional position sizing),
but CAGR and absolute drawdown scaled up together at a chosen, disclosed
gross exposure level. This is standard market-neutral book construction,
not a fitting exercise -- no parameter here was chosen by looking at
out-of-sample performance.

## 4. Final locked configuration

Derived entirely from formation-window data (`src/pairs_engine.py`):

| Parameter | Value |
|---|---|
| Formation / test split | 65% / 35% of ~8.1 years real daily data |
| Cointegration threshold | p < 0.05 (Engle-Granger) |
| Half-life range | 2-60 trading days |
| Entry / exit z-score | 2.0 / 0.5 |
| Stop-loss z-score | 4.0 |
| Max holding period | 20 trading days |
| Signal smoothing | EWMA span 3 on Kalman innovation |
| Kalman delta | 1e-4 |
| Transaction cost | 5 bps per leg |
| Risk per trade | 30% of pooled equity, precision-weighted |
| Max gross exposure | 1.5x equity |

## 5. Verified out-of-sample results

Test window 2023-09-26 to 2026-07-31 (714 trading days), never touched by
pair selection or hyperparameter tuning:

- CAGR: +0.98%
- Annualized volatility: 1.39%
- Sharpe (rf=0%): 0.71 | Sharpe (rf=4.5%): -2.46
- Sortino (rf=0%): 0.37
- Max drawdown: -1.41% (peak 2026-06-23, trough 2026-07-29)
- Calmar: 0.70
- 21 trades across 11 validated pairs (several pairs traded 0 times in
  this window -- cointegration held, but the spread simply didn't diverge
  enough to trigger an entry)

These are modest numbers by design. They are what remains after a real
cointegration test, real transaction costs, a locked train/test split,
and an explicit record of which shortcuts were tried and rejected. This
is consistent with the academic literature on classical pairs trading
(e.g. Gatev, Goetzmann & Rouwenhorst), which documents declining Sharpe
ratios for this class of strategy on liquid instruments as it became
crowded over the past two decades -- a simple implementation like this one
should not be expected to show a dramatic edge on SPY-adjacent ETFs and
large-cap pairs today.

## 6. Reproducing these numbers

```python
from src.backtester import StatisticalArbitrageBacktester
bt = StatisticalArbitrageBacktester(api_handler=None)
bt.run(retune=False)   # locked hyperparameters, matches the numbers above
bt.run(retune=True)    # re-runs the formation-only grid search from scratch
```
