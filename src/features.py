"""
Causal, point-in-time-safe factor construction.

Price/volume factors only by default -- fundamental data (P/E, ROE, etc.)
from a source like a live `.info` call is a CURRENT snapshot, not a
point-in-time historical series, and using it to build historical training
features would silently leak today's known fundamentals into past dates (a
severe form of look-ahead bias). Dividend history is the one exception
used here (`src/dividend_features.py`, opt-in): a dividend payment is a
historical fact fixed on its ex-date and never restated, so it's honestly
usable point-in-time in a way financial-statement data isn't.

Two choices below (rank-normalized features, market-relative labels) were
adopted as the default after being validated as a genuine methodological
improvement -- see TECHNICAL_DOCS.md section 4 for the full investigation,
including six other legitimate attempts to strengthen the ML signal that
did NOT hold up and were correctly not adopted.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = ['mom_12_1', 'mom_1m', 'rev_1w', 'rev_2w', 'vol_21d', 'vol_trend', 'dist_from_high']
FEATURE_COLS_WITH_DIVIDENDS = FEATURE_COLS + ['div_yield', 'div_growth']


def build_features(close: pd.DataFrame, volume: pd.DataFrame, div_yield=None, div_growth=None):
    """
    Returns a dict of factor DataFrames (same shape/index/columns as close),
    each value at (date, ticker) computed using only data through that date.
    `div_yield`/`div_growth`, if provided (see src/dividend_features.py),
    are included as additional factors.
    """
    returns = close.pct_change()
    features = {}

    # Momentum 12-1: cumulative return from t-252 to t-21, skipping the most
    # recent month (classic Jegadeesh-Titman momentum definition -- skipping
    # the last month avoids contaminating momentum with short-term reversal).
    features['mom_12_1'] = close.shift(21) / close.shift(252) - 1

    # Medium-term momentum (1-month).
    features['mom_1m'] = close.pct_change(21)

    # Short-term reversal signals.
    features['rev_1w'] = close.pct_change(5)
    features['rev_2w'] = close.pct_change(10)

    # Trailing realized volatility (21-day, annualized).
    features['vol_21d'] = returns.rolling(21).std() * np.sqrt(252)

    # Volume trend: recent average volume relative to longer trailing average.
    features['vol_trend'] = volume.rolling(21).mean() / volume.rolling(63).mean() - 1

    # Distance from 252-day high (anchoring/momentum factor, George & Hwang 2004).
    features['dist_from_high'] = close / close.rolling(252).max() - 1

    if div_yield is not None:
        features['div_yield'] = div_yield
    if div_growth is not None:
        features['div_growth'] = div_growth

    return features


def build_feature_panel(close: pd.DataFrame, volume: pd.DataFrame, forward_days=20,
                         div_yield=None, div_growth=None, rank_normalize=True, market_relative_label=True):
    """
    Long-format panel: one row per (date, ticker), columns = features + a
    forward-return LABEL. The label is NaN for the last `forward_days` rows
    of each ticker (not yet known) -- prediction-only use, never training.

    `rank_normalize=True` (default): each feature is converted to its
    cross-sectional percentile rank at each date, removing regime-dependent
    scale effects (what "21-day vol" means is very different in a calm
    market vs. a crisis).
    `market_relative_label=True` (default): the label is a stock's forward
    return MINUS the cross-sectional (universe) average that day, isolating
    stock-specific return from common market-wide moves.
    """
    features = build_features(close, volume, div_yield=div_yield, div_growth=div_growth)
    if rank_normalize:
        features = {name: feat.rank(axis=1, pct=True) for name, feat in features.items()}

    raw_forward_return = close.shift(-forward_days) / close - 1
    if market_relative_label:
        market_avg = raw_forward_return.mean(axis=1)
        label_return = raw_forward_return.sub(market_avg, axis=0)
    else:
        label_return = raw_forward_return

    frames = []
    for ticker in close.columns:
        df = pd.DataFrame({name: feat[ticker] for name, feat in features.items()})
        df['label'] = label_return[ticker]
        df['ticker'] = ticker
        df['date'] = close.index
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
