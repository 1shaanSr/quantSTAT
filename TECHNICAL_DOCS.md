# Technical Documentation: Risk-Parity Portfolio with an ML Factor-Tilt Overlay

## 1. Mathematical framework

### 1.1 Factor construction

All factors are computed causally from OHLCV data only (`src/features.py`):

- `mom_12_1`: cumulative return from t-252 to t-21 (skips the most recent
  month, the classic Jegadeesh-Titman momentum definition -- skipping
  avoids contaminating momentum with short-term reversal).
- `mom_1m`: trailing 21-day return.
- `rev_1w`, `rev_2w`: trailing 5- and 10-day returns (short-term reversal).
- `vol_21d`: trailing 21-day realized volatility, annualized.
- `vol_trend`: 21-day average volume relative to 63-day average volume.
- `dist_from_high`: current price relative to the trailing 252-day high
  (anchoring/momentum factor, George & Hwang 2004).

Fundamental ratios (P/E, ROE, etc.) were deliberately excluded. A live
data source's fundamentals endpoint returns the CURRENT value only, not a
point-in-time historical series -- using it to build historical training
features would mean a sample dated 2019 is built with 2026-vintage
fundamental data, a severe and easy-to-miss form of look-ahead bias. Price
and volume are the only data honestly available point-in-time from this
data source, so the factor set is restricted to what can be trusted.

### 1.2 Walk-forward ML predictor

`src/ml_predictor.py` retrains a `HistGradientBoostingRegressor` at every
rebalance date, using an expanding window of only the training samples
whose forward-return labels are already fully realized as of that date.
Two leakage traps are avoided by construction:

1. **Overlapping labels**: a forward-20-day return label at date t shares
   19 of its 20 days with the label at date t+1. Training on adjacent
   daily rows would let near-duplicate label information leak across
   nominally different samples. Fixed by sampling training/prediction
   dates only every 20 trading days (`forward_days` apart) -- labels never
   overlap.
2. **Training on future information**: at each rebalance date, only prior
   periods are used for training, never the current or future period.

Hyperparameters (`max_depth=3, max_iter=50, learning_rate=0.05,
min_samples_leaf=30, l2_regularization=1.0`) are fixed by convention, not
searched. Return prediction has a very low signal-to-noise ratio, and this
project's history (see section 3) has repeatedly found that aggressively
tuned configurations turn out to be fragile, sample-specific peaks that
don't generalize -- a heavily-tuned complex model here would very likely
fit formation noise rather than real signal.

### 1.3 Evaluation: information coefficient, not trading Sharpe

The model is evaluated by the Spearman rank correlation between its
predictions and realized forward returns at each rebalance date (the
"information coefficient," standard in quantitative equity research --
see Grinold & Kahn, *Active Portfolio Management*). This is a more honest
metric for a prediction model than backing into a trading Sharpe, since it
directly measures forecasting skill without conflating it with position
sizing or portfolio construction choices.

An IC in the 0.02-0.05 range is generally considered a genuinely useful
signal at the individual-stock level in the professional literature (IC of
0.05 is often cited as a strong result for a single simple factor) --
context for judging whether the 0.048 test-period result below is
reasonable or suspicious.

### 1.4 Risk parity

`src/risk_parity.py` solves for long-only weights where every asset
contributes equally to total portfolio variance:

```
minimize   sum_i (RC_i - 1/N)^2
subject to sum(w) = 1, w >= 0
where      RC_i = w_i * (Sigma w)_i / (w' Sigma w)
```

solved via SLSQP over a shrinkage-regularized covariance matrix (`Sigma =
(1-s)*Sigma_sample + s*diag(Sigma_sample)`, s=0.3) -- a raw sample
covariance over 77 names from limited trailing history is noisy, the same
lesson applied to portfolio construction throughout this project's
history.

**Scope note**: classic risk parity (e.g. the "All Weather" style) is
usually applied ACROSS asset classes with genuinely different risk drivers
(equities, bonds, commodities), where the diversification benefit is large
because the assets are only weakly correlated. Applied within a single
equity universe, as here, all 77 names still share a common market-beta
factor, so the benefit is real but more modest than the classic multi-
asset-class use -- it captures a genuine low-volatility tilt (favoring
lower-vol/lower-covariance names) rather than true cross-asset-class
diversification. Disclosed here rather than implied to be more than it is.

### 1.5 ML tilt (Black-Litterman-style, confidence-scaled)

When used, the ML signal tilts risk-parity weights:

```
tilted_w_i = base_w_i * exp(tilt_strength * z_i)
```

where `z_i` is the cross-sectionally standardized ML prediction for name
i, renormalized to sum to 1 and capped at `max_tilt_multiple`x (default
3x) its risk-parity base weight so a single high-scoring name can't
dominate the book. `tilt_strength=0` reproduces pure risk parity exactly.

## 2. Formation-only tilt-strength decision

`tilt_strength` was chosen via 3 rolling formation folds, evaluating
portfolio Sharpe at tilt strengths from 0.0 to 3.0:

| tilt_strength | avg formation Sharpe |
|---|---|
| **0.0** | **0.913** |
| 0.25 | 0.897 |
| 0.5 | 0.881 |
| 1.0 | 0.866 |
| 1.5 | 0.859 |
| 2.0 | 0.866 |
| 3.0 | 0.881 |

Every non-zero tilt strength underperformed pure risk parity in formation.
This is consistent with the formation-period IC (below) being
indistinguishable from zero -- the model genuinely had no demonstrated
skill during formation, so tilting toward its predictions correctly hurt
performance. **Locked: tilt_strength=0.0.**

Purely for transparency (not used to make this decision), the same sweep
on the test period shows the same pattern -- Sharpe is flat-to-declining
as tilt strength increases (1.47 at 0.0, 1.43 at 1.0, 1.37 at 2.0), CAGR
rises slightly as the tilt takes more concentrated bets. The formation-
only decision and the test-period-informational check agree, which is
reassuring evidence the formation-based selection process is sound --
though it was not, and should not be, the basis for the decision itself.

## 3. Project history: why this replaced a pairs-trading strategy

This repository previously implemented a market-neutral statistical
arbitrage strategy (Engle-Granger cointegration pairs, Kalman-filtered
hedge ratios). That approach was rigorously validated end to end and
achieved a genuine 0.71 Sharpe / 0.70 Calmar out-of-sample -- but repeated
attempts to improve on it (minimum-variance capital allocation, a wider
pair universe, structurally-enforced dual-class share arbitrage, and
classic cross-sectional short-term reversal) all either failed to beat
that baseline or, in one case, appeared to reach Sharpe 3.4 before being
traced to a liquidity artifact (one leg of the pair trading only ~100-300
shares/day, producing stale, non-executable closing prices -- a lesson
directly informing the strict $10M/day liquidity bar on this project's
universe). Across four independently-tested strategy families, price-
prediction and pairwise-relationship approaches on liquid public daily-bar
equity data consistently topped out at or below Sharpe ~0.7-1.0 once
properly validated -- consistent with the well-documented decay of
classical statistical arbitrage effects as they became crowded over the
past two decades (see e.g. Khandani & Lo on the 2007 quant quake).

This project reflects a deliberate pivot in approach rather than another
attempt at the same thing: instead of trying to predict which stocks will
outperform (the exercise that kept failing), it leads with portfolio
construction -- risk parity reliably improves risk-adjusted return through
genuine diversification math, not through forecasting skill, which is a
structurally different and more dependable source of edge. The ML
component is still built and evaluated rigorously (honest IC, proper
walk-forward validation), but is not required to carry the result, and
this document reports plainly that it currently doesn't.

## 4. Reproducing these numbers

```python
from src.backtester import RiskParityMLBacktester
bt = RiskParityMLBacktester()
bt.run()                        # locked config: tilt_strength=0.0, matches section above
bt.run(tilt_strength=1.0)       # informational -- not the locked/recommended configuration
```
