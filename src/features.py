"""
Causal, point-in-time-safe factor construction from OHLCV data only.

Deliberately price/volume-derived only -- fundamental data (P/E, ROE, etc.)
from a source like yfinance's .info is a CURRENT snapshot, not a
point-in-time historical series. Using it to build historical training
features would silently leak today's known fundamentals into past dates
(a severe form of look-ahead bias). Rather than build a fragile
fundamentals pipeline on data that can't be trusted point-in-time, this
sticks to factors honestly computable from historical price/volume alone.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = ['mom_12_1', 'mom_1m', 'rev_1w', 'rev_2w', 'vol_21d', 'vol_trend', 'dist_from_high']


def build_features(close: pd.DataFrame, volume: pd.DataFrame):
    """
    Returns a dict of factor DataFrames (same shape/index/columns as close),
    each value at (date, ticker) computed using only data through that date.
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

    return features


def build_feature_panel(close: pd.DataFrame, volume: pd.DataFrame, forward_days=20):
    """
    Long-format panel: one row per (date, ticker), columns = features + a
    forward-return LABEL (return from t to t+forward_days). The label is
    NaN for the last `forward_days` rows of each ticker (label not yet
    known) -- those rows are for prediction-only use, never training.
    """
    features = build_features(close, volume)
    forward_return = close.shift(-forward_days) / close - 1

    frames = []
    for ticker in close.columns:
        df = pd.DataFrame({name: feat[ticker] for name, feat in features.items()})
        df['label'] = forward_return[ticker]
        df['ticker'] = ticker
        df['date'] = close.index
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
