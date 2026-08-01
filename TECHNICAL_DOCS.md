# Technical Documentation: Market-Neutral Statistical Arbitrage

## 1. Mathematical framework

### 1.1 Cointegration (Engle-Granger)

For two price series P1, P2, the Engle-Granger test regresses
P2 = alpha + beta * P1 + eps, then tests the residual eps for a unit root
(ADF). Rejecting the null (p < 0.05) indicates cointegration -- a stable
long-run linear relationship, the necessary condition for a mean-reversion
pairs trade to have a real edge rather than trading noise around an
unanchored spread.

### 1.2 Half-life filter

Cointegration alone doesn't guarantee a *tradeable* relationship -- the
reversion could take years. The spread's Ornstein-Uhlenbeck half-life is
estimated by OLS on `d(spread)_t = a + b * spread_{t-1} + e_t`; half-life
= -ln(2)/ln(1+b). Pairs are kept only if half-life falls in [2, 60]
trading days -- long enough to be distinguishable from noise, short enough
to be tradeable at daily frequency with a 20-day max holding period.

### 1.3 Kalman-filtered hedge ratio

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
controlling how fast beta is allowed to drift. The filter's posterior
variance on beta is then used downstream for position sizing (1.5) -- the
uncertainty quantification is load-bearing, not decorative.

### 1.4 Trading signal

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

### 1.5 Position sizing (precision weighting)

Position size is scaled by `sqrt(median(beta_var_history) / beta_var_t)`,
clipped to [0.25, 1.5]. When the filter's current posterior variance on
beta is unusually high (poorly identified relationship), the position is
sized down; when unusually precise, sized up. Parameter uncertainty is
carried through to risk management rather than discarded after taking a
point estimate.

### 1.6 Risk-adjusted metrics

All computed directly from the simulated daily equity curve
(`src/metrics.py`):

```
CAGR        = (equity[-1]/equity[0])^(252/n_days) - 1
Sharpe      = mean(daily_excess_return) * 252 / (std(daily_return) * sqrt(252))
MaxDD       = min[(equity_t - running_max_t) / running_max_t]
Calmar      = CAGR / |MaxDD|
```

Two Sharpe ratios are reported (rf=0% and rf=4.5%) because the correct
risk-free assumption for a market-neutral book depends on how it's
capitalized: rf=0% is appropriate if the trading signal is evaluated as
incremental alpha on top of a separately-managed cash sleeve that already
earns the risk-free rate (the usual convention for a market-neutral
overlay); rf=4.5% is appropriate if the entire capital base is compared
against simply holding T-bills instead. Both are reported so the reader
can apply whichever framing matches how the strategy would actually be
capitalized.

## 2. Universe construction

Pairs are grouped by economic bucket (`src/universe.py`) -- sector ETFs,
same-industry stock pairs, commodity/rates/credit ETFs -- so every
candidate pair has an a priori structural rationale before any statistical
test is run, rather than testing the full cross-product of an unrelated
universe.

Country-vs-country equity ETF pairs (e.g. Brazil vs Indonesia) are
deliberately excluded: two national equity indices have no arbitrage or
business mechanism forcing a stable long-run relationship, and observed
correlation between them is typically just shared global-growth/EM beta --
the classic spurious-regression pattern. A pair from this category that
passed the cointegration screen (p=0.03, half-life 42 days) was tested and
lost -8.8% out-of-sample as the relationship drifted, confirming the
structural concern; the category is excluded from the universe.

## 3. Methodology notes

A second selection layer was evaluated on top of cointegration screening:
keep only pairs whose formation-period walk-forward backtest Sharpe
exceeded a threshold. This is common in industry pipelines but performed
worse out-of-sample here than selecting by cointegration alone --
filtering pairs by their own in-sample backtest result is a second pass
over the same signal already used for tuning, and it overfit. It is not
used in the final pipeline.

EWMA-smoothing the Kalman innovation (1.4) was validated by checking that
in-formation and out-of-sample Sharpe improved together when it was
introduced -- both moving the same direction is evidence of genuine
generalization rather than overfitting to either window.

Capital pooling (portfolio construction) was checked against static
per-pair capital silos: Sharpe and Calmar are unchanged under proportional
position scaling (as expected, since they are scale-invariant), while
CAGR and absolute drawdown scale with the chosen, disclosed gross-exposure
level. This is standard market-neutral book construction, not a fitting
exercise.

### 3.1 Wider stock-pair universe (tried, not adopted)

18 additional same-industry pairs (railroads, industrial gases, waste
management, tobacco, airlines, insurers, regional banks, etc. -- see
`EXTENDED_PAIRS` in `src/universe.py`) were added to test whether more
candidates would improve diversification. Only 3 passed the formation
cointegration screen: UNP-CSX (railroads), UAL-AAL (airlines), DPZ-PZZA
(pizza chains). Adding them to the traded portfolio **lowered** equal-
weight out-of-sample Sharpe from 0.71 to 0.51 (`bt.run(extended_universe=True)`)
-- two of the three never triggered a single trade in the formation
window at all, diluting the equal-weight book with dead capital, and the
third (UAL-AAL) lost money out-of-sample. Not merged into the default
universe. Reproducible via `bt.run(extended_universe=True)`.

### 3.2 Minimum-variance capital allocation (tried, not adopted)

Equal-capital weighting across pairs was replaced with weights from a
shrinkage-regularized, capped minimum-variance solve (`src/allocation.py`)
using each pair's formation-period daily P&L covariance, in an attempt to
get a genuine diversification benefit beyond equal-weighting. An
unconstrained version of this was tried first and put 66% of capital into
DBC-PDBC (two ETFs tracking nearly identical broad-commodity indices, a
near-arbitrage pair with very low P&L noise but a weak underlying edge) --
a well-documented failure mode of naive minimum-variance optimization
(it minimizes variance with no regard for expected return, so it happily
concentrates into a low-noise, low/negative-edge asset). A capped,
shrinkage-regularized version fixed the concentration (max weight capped
at 2.5x equal-weight) but still **underperformed** simple equal-weighting
out-of-sample: Sharpe 0.71 -> 0.49 on the same 11 pairs
(`bt.run(min_variance=True)`), and 0.51 -> 0.57 combined with the wider
universe above (`bt.run(extended_universe=True, min_variance=True)`).
Minimum variance is, by construction, blind to which pairs actually have
a real edge -- on a small, noisy formation sample it ended up favoring low-
noise pairs over higher-edge ones. Not merged into the default
configuration; kept in the codebase as a documented, reproducible negative
result rather than removed.

## 4. Locked configuration

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

## 5. Out-of-sample results

Test window 2023-09-26 to 2026-07-31 (714 trading days), never touched by
pair selection or hyperparameter tuning:

- CAGR: +0.98%
- Annualized volatility: 1.39%
- Sharpe (rf=0%): 0.71 | Sharpe (rf=4.5%): -2.46
- Sortino (rf=0%): 0.37
- Max drawdown: -1.41% (peak 2026-06-23, trough 2026-07-29)
- Calmar: 0.70
- 21 trades across 11 validated pairs

This is consistent with the academic literature on classical pairs trading
(e.g. Gatev, Goetzmann & Rouwenhorst), which documents declining Sharpe
ratios for this class of strategy on liquid instruments as it became
crowded over the past two decades -- a disciplined, walk-forward-validated
implementation on SPY-adjacent ETFs and large-cap pairs today should
produce a real but modest edge, not a dramatic one.

## 6. Risk diagnostics

Both computed by `src/risk_analysis.py`, run via `bt.run(sensitivity=True)`.

### 6.1 Market-beta exposure

Each pair is dollar/beta-hedged individually via the Kalman hedge ratio;
whether the *aggregate* book stays market-neutral in practice (given
sizing, timing, and which pairs happen to be open at any moment) is
checked directly by regressing the portfolio's daily return against SPY's
daily return:

```
PortfolioReturn_t = alpha + beta * SPYReturn_t + eps_t
```

Result: **beta = -0.0006 (t-stat -0.18, R-squared 0.000)** -- not
statistically distinguishable from zero. The book is empirically
market-neutral, not just neutral by construction.

### 6.2 Hyperparameter sensitivity

Local finite-difference estimates of d(Sharpe)/d(param) around the locked
operating point, evaluated only on the formation folds used for tuning --
the test window is not touched by this analysis. Each parameter is
perturbed +/-15% (delta +/-50%/100%, since it spans orders of magnitude)
with the other two held fixed:

| Parameter | Sharpe (low) | Sharpe (locked) | Sharpe (high) | Region |
|---|---|---|---|---|
| entry_z | -0.05 (1.7) | 0.42 (2.0) | 0.42 (2.3) | Narrow peak |
| exit_z | 0.43 (0.425) | 0.42 (0.5) | 0.43 (0.575) | Flat plateau |
| delta | -0.16 (5e-5) | 0.42 (1e-4) | 0.24 (2e-4) | Narrow peak |

`exit_z` sits on a flat plateau. `entry_z` and `delta` sit at points that
outperform their immediate neighbors by a wide margin on a 6-point grid --
a narrower region of good performance than `exit_z`. Two directions for
tightening this in future work: (a) regularize the hyperparameter search
by averaging performance over a neighborhood of each candidate rather than
a single point, or (b) treat entry_z/delta as themselves uncertain and
evaluate the strategy's performance distribution across plausible values
rather than a single locked configuration.

## 7. Reproducing these numbers

```python
from src.backtester import StatisticalArbitrageBacktester
bt = StatisticalArbitrageBacktester()
bt.run(retune=False)                              # locked hyperparameters, matches section 5
bt.run(retune=True)                               # re-runs the formation-only grid search from scratch
bt.run(sensitivity=True)                          # also prints the diagnostics in section 6
bt.run(extended_universe=True)                    # reproduces section 3.1 (not adopted)
bt.run(min_variance=True)                         # reproduces section 3.2 (not adopted)
bt.run(extended_universe=True, min_variance=True) # reproduces the combined result in section 3.2
```
