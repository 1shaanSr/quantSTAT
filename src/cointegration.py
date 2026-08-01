import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

def half_life(spread: pd.Series):
    """
    Ornstein-Uhlenbeck mean-reversion half-life: fit
        d(spread)_t = a + b * spread_{t-1} + e_t
    via OLS. If b < 0 (mean-reverting), half-life = -ln(2)/ln(1+b) in periods.
    Returns +inf if the fit implies no mean reversion (b >= 0).
    """
    s_lag = spread.shift(1).dropna()
    s = spread.loc[s_lag.index]
    d = s - s_lag
    X = np.column_stack([np.ones(len(s_lag)), s_lag.values])
    coef, *_ = np.linalg.lstsq(X, d.values, rcond=None)
    b = coef[1]
    if b >= 0:
        return np.inf
    return -np.log(2) / np.log(1 + b)

def screen_pairs(prices: pd.DataFrame, pairs, pvalue_thresh=0.05,
                  min_corr=0.5, min_halflife=2, max_halflife=60):
    """
    Engle-Granger cointegration test + OU half-life filter. `prices` should
    be the FORMATION (in-sample) window only -- the caller is responsible
    for never passing test-period data here, or the resulting pair
    selection is not out-of-sample-safe.
    """
    results = []
    for a, b, bucket in pairs:
        if a not in prices.columns or b not in prices.columns:
            continue
        pa, pb = prices[a], prices[b]
        corr = pa.corr(pb)
        if abs(corr) < min_corr:
            continue
        score, pvalue, _ = coint(pa, pb)
        if pvalue >= pvalue_thresh:
            continue
        X = np.column_stack([np.ones(len(pa)), pa.values])
        coef, *_ = np.linalg.lstsq(X, pb.values, rcond=None)
        beta = coef[1]
        spread = pb - beta * pa
        hl = half_life(spread)
        if not (min_halflife <= hl <= max_halflife):
            continue
        results.append({
            'a': a, 'b': b, 'bucket': bucket, 'pvalue': pvalue,
            'corr': corr, 'beta': beta, 'half_life': hl
        })
    results.sort(key=lambda r: r['pvalue'])
    return results
