"""
Block bootstrap confidence intervals for Sharpe/Calmar. A naive iid
bootstrap on daily returns would understate uncertainty -- financial
returns have volatility clustering and short-horizon autocorrelation, so
resampling individual days independently destroys real dependence
structure. The moving-block bootstrap resamples contiguous blocks instead,
preserving that structure. Answers directly: is the reported Sharpe
statistically distinguishable from a no-skill (Sharpe=0) outcome, or could
it plausibly be a lucky draw from a ~3-year sample?
"""
import numpy as np
import pandas as pd


def moving_block_bootstrap_returns(returns: np.ndarray, block_size=20, n_resamples=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    resampled = np.empty((n_resamples, n_blocks * block_size))
    for r in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        blocks = [returns[s:s + block_size] for s in starts]
        resampled[r] = np.concatenate(blocks)
    return resampled[:, :n]


def bootstrap_sharpe_calmar(equity: pd.Series, block_size=20, n_resamples=5000, rf_annual=0.0, seed=42):
    daily_returns = equity.pct_change().dropna().values
    resampled_returns = moving_block_bootstrap_returns(daily_returns, block_size=block_size,
                                                          n_resamples=n_resamples, seed=seed)
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    sharpes = np.empty(n_resamples)
    calmars = np.empty(n_resamples)
    for r in range(n_resamples):
        rets = resampled_returns[r]
        ann_vol = rets.std() * np.sqrt(252)
        sharpes[r] = ((rets.mean() - rf_daily) * 252) / ann_vol if ann_vol > 0 else np.nan

        path = 100_000.0 * np.cumprod(1 + rets)
        running_max = np.maximum.accumulate(path)
        dd = (path - running_max) / running_max
        max_dd = dd.min()
        years = len(rets) / 252
        total_ret = path[-1] / path[0] - 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
        calmars[r] = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return sharpes, calmars


def summarize_bootstrap(values: np.ndarray, label: str, point_estimate: float):
    values = values[~np.isnan(values)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    frac_positive = (values > 0).mean()
    print(f"{label}: point estimate {point_estimate:.2f}, bootstrap mean {values.mean():.2f}, "
          f"95% CI [{lo:.2f}, {hi:.2f}], P(> 0) = {frac_positive*100:.1f}%")
    return dict(point=point_estimate, mean=values.mean(), ci_low=lo, ci_high=hi, frac_positive=frac_positive)
