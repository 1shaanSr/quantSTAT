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
Dividend history is the one legitimate exception (section 4.7).

Two further defaults, adopted after being validated as genuine
improvements (section 4.1):

- **Cross-sectional rank normalization**: every feature is converted to
  its percentile rank across the universe at each date, instead of its
  raw value. Removes regime-dependent scale effects -- what "21-day vol"
  means is very different in a calm market vs. a crisis.
- **Market-relative labels**: the prediction target is a stock's forward
  return MINUS the cross-sectional (universe) average forward return that
  day, not the raw return. Raw returns are dominated by common market-wide
  moves that have nothing to do with stock-picking skill; stripping that
  out isolates the part of the return actually related to which stock was
  picked.

Neither changed the tilt-strength decision (section 2), but both are
standard, principled practice in cross-sectional equity ML and were kept
as the default regardless.

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
context for judging whether the test-period IC result below (~0.05-0.06,
t-stat ~2) is reasonable or suspicious.

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

## 4. Attempts to strengthen the ML signal

Section 2 found that with the original price/volume factors and raw
forward-return label, the ML predictor's formation-period IC was
indistinguishable from zero and no tilt strength helped. Rather than
accept that at face value, seven methodologically distinct, principled
attempts were made to find genuine, formation-validated skill before
concluding the locked configuration (tilt_strength=0) is correct. All
seven are documented here -- including the two that were kept as opt-in
code (4.1, adopted as the new default; 4.7, available but not default) --
because "we tried this and it didn't hold up" is real information, and a
project that only shows what worked isn't credible.

### 4.1 Market-relative labels + rank-normalized features (adopted as default)

Described in section 1.1. Formation IC went from ~-0.006 to ~-0.017 (still
statistically indistinguishable from zero either way); test-period IC rose
from ~0.048 to ~0.057 (t-stat ~2.0-2.1). Adopted as the default because
both changes are principled, standard practice regardless of the outcome
-- not because they flipped the tilt decision, which they didn't.

### 4.2 Rolling vs. expanding training windows

Rolling 20- and 40-period training windows were tested against the
expanding-window default, hypothesizing that stale early-history data
might be diluting the signal. Formation IC remained negative in every
variant (-0.022 to -0.040), no better than expanding. Not adopted.

### 4.3 Ridge regression vs. gradient boosting

A linear Ridge model was tested as a more heavily regularized alternative
to gradient boosting, in case the tree-based model was overfitting
formation noise. Formation IC remained negative for Ridge too (-0.015 to
-0.045 across window variants) -- ruling out "wrong model class" as the
explanation. Not adopted (gradient boosting kept as the more flexible
default, since the choice of model didn't matter here anyway).

### 4.4 Longer prediction horizon (60-day / quarterly) -- caught a false positive

Motivated by the classical momentum literature (Jegadeesh-Titman), which
finds its effects concentrated at 3-12 month horizons, not 20 days.
Formation IC at forward_days=60 was +0.106 (t-stat 2.31, 19 periods), and
manual inspection ruled out a single-outlier artifact (median IC 0.112;
still +0.086 excluding the single best period). This looked like a real
finding.

It wasn't tested in isolation, though -- it emerged from checking 5
different horizons (5, 10, 20, 40, 60 days), which is itself a multiple-
testing risk: test enough variants and one looks significant by chance.
Rather than adopt it on a single underpowered sample (19 formation
periods), it was checked against genuinely fresh data never used anywhere
else in this project: the full 2009-2015 history (73 of the 77 tickers
existed and had liquid trading back to 2009), entirely prior to and
non-overlapping with the 2016-2026 window used everywhere else.

**The signal did not replicate**: mean IC -0.049, t-stat -1.25, only 38%
of periods positive -- worse than a coin flip. This is the expected
behavior of a false positive found via testing multiple horizons: it
doesn't hold up on data it wasn't fit to, even loosely. Not adopted.
This is arguably the most important negative result in this document,
since the replication check is exactly the kind of scrutiny that
distinguishes a real finding from a data-mined one.

### 4.5 Adaptive tilt sized by trailing realized IC (`src/adaptive_tilt.py`, kept as opt-in)

A more sophisticated design than a single static tilt_strength: at each
rebalance date, scale the tilt by the model's own trailing K-period
realized IC (using only IC values from periods whose labels are already
known -- fully causal). Trade the signal harder when it's recently been
working, back off to pure risk parity when it hasn't. This is standard
performance-based signal weighting, not a fitting exercise.

Tested across a grid of `ic_lookback` (5/10/15/20) and `reference_ic`
(0.03/0.05/0.08) on formation data: every combination still underperformed
static pure risk parity (0.83-0.86 avg formation Sharpe vs. 0.872
baseline). Checked informationally on the test period too (not used for
the decision): Sharpe 1.480 vs. 1.470 for pure risk parity -- a negligible
difference either way. Not adopted as the default, but kept as
documented, reproducible code (`bt.run(adaptive_tilt=True)`) since it is
a legitimately more sophisticated mechanism that simply had nothing to
work with here.

### 4.6 Fundamental/alternative data feasibility check

Before building anything, this data source's actual historical depth was
checked directly: quarterly financials returned only ~5 quarters of
history, annual income statements only ~5 years, and earnings-date history
only ~3 years -- all far short of the 8+ years this project's walk-forward
methodology requires. Attempting to use them would force a choice between
an underpowered backtest or silently using current/restated data for
historical dates (the exact look-ahead trap documented in section 1.1).
Neither is acceptable, so financial-statement fundamentals were ruled out
on data-availability grounds before any model was built with them.

### 4.7 Dividend-based factors (`src/dividend_features.py`, kept as opt-in)

Dividend payment history is the one genuinely point-in-time-safe
alternative data available: a dividend is a historical fact fixed on its
ex-date and never restated, unlike a financial-statement pull. Trailing
252-day dividend yield and year-over-year dividend growth were added as
two more rank-normalized features (68-70 of 77 tickers have real dividend
history; non-payers like Amazon and Tesla get an honest 0, not a
fabricated value).

Formation IC with these added: -0.008 (t-stat -0.35) -- no better than
without them, and the informational test-period IC actually fell (from
~0.057 to -0.024), suggesting the extra features diluted rather than
strengthened the existing weak signal. Not adopted as the default, but
kept as documented, reproducible code (`bt.run(include_dividends=True)`).

### 4.8 Conclusion

Seven methodologically distinct, principled attempts -- including one that
correctly caught and rejected its own false positive via out-of-period
replication -- found no formation-validated, replicable ML edge on this
universe at this frequency with data genuinely available here. This is a
substantive finding, not a failure to search hard enough: it means that
with honest point-in-time data and disciplined validation, there currently
isn't an exploitable signal to act on. The locked configuration
(tilt_strength=0.0, pure risk parity) remains correct.

## 5. Verification: benchmark comparison and statistical significance

Run by default (`verify=True`) after the headline result. Both checks
apply to the LOCKED pure risk-parity strategy specifically (tilt=0),
regardless of what tilt experiment was requested in the same `run()` call
-- the verification section always characterizes the actual production
configuration.

### 5.1 Benchmark comparison (`src/benchmarks.py`)

The headline Sharpe/Calmar mean little without context. Three naive
alternatives are computed over the IDENTICAL test window (same dates, same
transaction cost assumption):

- **Equal-weight 1/N** across the same 77-name universe -- isolates
  whether risk parity's smarter weighting beats naive weighting on
  identical assets.
- **SPY buy-and-hold** -- the classic "did you beat the market" bar.
- **60/40 SPY/IEF** -- the traditional balanced-portfolio standard.

On the test window (2023-04-26 to 2026-07-07, a fairly calm bull market
with no severe equity crash): risk parity roughly ties or slightly trails
all three (Sharpe ~1.47 vs. 1.46-1.62 for the alternatives). Reported
plainly rather than hidden -- risk parity is not designed to maximize
upside participation in a calm bull run, and this period didn't test the
part of its value proposition that would show up.

To evaluate that part directly, the same four strategies are compared over
two independently-documented historical crisis windows (chosen for their
historical significance -- COVID market crash, 2022 rate-hike bear market
-- not selected after looking at any result here).

**A methodology correction made before shipping this section**: an earlier
version of this comparison computed the crisis-period benchmarks with a
"fresh start exactly at the crisis boundary" convention, but risk parity
with a "continuously running since 2016" convention -- an apples-to-oranges
mix that inflated risk parity's apparent crisis advantage (the fresh-start
benchmarks looked artificially worse because a single cold-start entry
trade landed at an arbitrary point in the crash, not because of anything
about the strategy itself). The "fresh start exactly at a crisis boundary"
convention was also tested for risk parity alone and rejected on the same
grounds: it was highly sensitive to which specific date the 20-day
rebalance grid happened to land on relative to the crisis onset -- a
measurement artifact, not a property of the strategy. All four strategies
below use the same, consistent convention: continuously managed from the
start of available history (2016), sliced at each crisis window. See
`src/benchmarks.py` for both conventions and why each is used where it is.

| Period | Risk parity | Equal-weight 1/N | SPY | 60/40 SPY/IEF |
|---|---|---|---|---|
| COVID crash (2020-02-14 to 2020-04-15) | -7.74% | -7.66% | -9.08% | -4.01% |
| 2022 bear market (2022-01-03 to 2022-10-14) | -8.50% | -8.60% | -12.28% | -12.94% |

The honest finding, once measured consistently: risk parity essentially
**ties equal-weight** in both crises -- on this single 77-stock universe,
most of the downside protection comes from being broadly diversified
across many names at all, not specifically from the risk-parity weighting
scheme over equal weighting. It does modestly beat SPY and 60/40 in the
2022 bear market (both stocks and traditional bonds fell that year). But
60/40 clearly wins during COVID specifically -- real bonds provide a
genuine cross-asset-class hedge (a "flight to quality") that a pure-equity
book, risk-parity-weighted or not, cannot structurally replicate. This is
a more modest and more honest finding than "risk parity dramatically
protects you in every crisis," and is consistent with the scope note in
section 1.4 about what within-equity risk parity does and doesn't provide.

### 5.2 Statistical significance (`src/significance.py`)

A single Sharpe point estimate from a ~3-year, 41-rebalance-period sample
has real sampling uncertainty. A moving-block bootstrap (resampling
contiguous 20-day blocks rather than individual days, to preserve
volatility clustering and autocorrelation that an i.i.d. bootstrap would
destroy) generates 5,000 resampled return paths from the test-period daily
returns and recomputes Sharpe/Calmar on each:

- Sharpe: point estimate 1.47, 95% CI **[0.43, 2.69]**, P(Sharpe > 0) =
  99.8%.
- Calmar: point estimate 1.27, 95% CI [0.26, 4.32], P(Calmar > 0) = 99.8%.

The CI is wide (reflecting real uncertainty from a modest sample) but
excludes zero comfortably -- the result is statistically distinguishable
from a no-skill outcome, not just a point estimate asserted without a
sense of its own uncertainty. Checked for robustness across block sizes
5/10/20/40 days; the CI bounds move by less than 0.2 in either direction,
indicating the result isn't an artifact of the specific block-size choice.

## 6. Reproducing these numbers

```python
from src.backtester import RiskParityMLBacktester
bt = RiskParityMLBacktester()
bt.run()                                # locked config: tilt_strength=0.0, matches section 2
bt.run(tilt_strength=1.0)               # informational -- not the locked/recommended configuration
bt.run(adaptive_tilt=True)              # reproduces section 4.5 (not adopted)
bt.run(include_dividends=True)          # reproduces section 4.7 (not adopted)
bt.run(verify=False)                    # skip benchmark comparison + bootstrap (faster)
```
