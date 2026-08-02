"""
Portfolio-level backtest engine: at each rebalance date, compute risk-parity
weights from trailing covariance (using only data strictly before that
date), optionally tilt using that date's walk-forward ML predictions, hold
until the next rebalance date, accumulate daily P&L with transaction costs.
`tilt_strength=0` reproduces the pure risk-parity baseline exactly.
"""
import numpy as np
import pandas as pd
from src.risk_parity import risk_parity_weights, tilt_weights


def run_portfolio(close: pd.DataFrame, pred_df: pd.DataFrame, rebalance_dates,
                   cov_lookback=126, tilt_strength=0.0, max_tilt_multiple=3.0,
                   cost_bps=10, capital=100_000.0, shrinkage=0.3):
    """
    `tilt_strength` may be a constant float, or a {date: float} schedule
    (e.g. from src.adaptive_tilt.build_adaptive_tilt_schedule) to vary the
    tilt over time based on the model's own trailing track record.
    """
    returns = close.pct_change()
    dates = close.index
    pred_by_date = ({d: g.set_index('ticker')['pred'] for d, g in pred_df.groupby('date')}
                     if pred_df is not None and len(pred_df) else {})
    tilt_schedule = tilt_strength if isinstance(tilt_strength, dict) else None

    daily_pnl = pd.Series(0.0, index=dates)
    current_weights = pd.Series(0.0, index=close.columns)
    weight_history = {}
    rebal_set = set(tilt_schedule.keys()) if tilt_schedule is not None else set(rebalance_dates)

    for i, date in enumerate(dates):
        if i > 0:
            day_ret = returns.iloc[i].fillna(0.0)
            daily_pnl.iloc[i] += (current_weights * day_ret * capital).sum()

        if date in rebal_set and i >= cov_lookback:
            trailing = returns.iloc[i - cov_lookback:i].dropna(axis=1, how='any')
            if len(trailing.columns) < 10:
                continue
            base_w = risk_parity_weights(trailing, shrinkage=shrinkage)

            current_tilt = tilt_schedule[date] if tilt_schedule is not None else tilt_strength
            if current_tilt > 0 and date in pred_by_date:
                scores = pred_by_date[date]
                scores = (scores - scores.mean()) / (scores.std() + 1e-9)
                new_w = tilt_weights(base_w, scores, tilt_strength=current_tilt, max_tilt_multiple=max_tilt_multiple)
            else:
                new_w = base_w

            new_w_full = pd.Series(0.0, index=close.columns)
            new_w_full.loc[new_w.index] = new_w

            turnover = (new_w_full - current_weights).abs().sum()
            daily_pnl.iloc[i] -= (cost_bps / 10000.0) * turnover * capital

            current_weights = new_w_full
            weight_history[date] = new_w_full

    return daily_pnl, weight_history
