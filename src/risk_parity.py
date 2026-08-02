"""
Risk-parity portfolio construction: each asset contributes equally to total
portfolio variance (not equal dollar weight -- a low-vol stock gets a
larger dollar weight than a high-vol stock, so no single name dominates
portfolio risk). Solved via scipy as a proper constrained optimization
(minimize dispersion of risk contributions), with shrinkage-toward-diagonal
covariance regularization -- a raw sample covariance over ~77 names from
limited history is noisy, the same lesson applied to portfolio construction
throughout this project's history.

Note on scope: classic risk parity (e.g. the "All Weather" style) is
usually applied ACROSS asset classes with genuinely different risk drivers
(equities, bonds, commodities), where the diversification benefit is
large because the assets are only weakly correlated. Applied within a
single equity universe, as here, all names still share a common market-beta
factor, so the diversification benefit is real but more modest than the
classic multi-asset-class use -- it captures a genuine low-volatility tilt
(systematically favoring lower-vol/lower-covariance names) rather than
true cross-asset-class diversification. Disclosed here rather than
implied to be more than it is.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def shrunk_covariance(returns: pd.DataFrame, shrinkage=0.3):
    sigma = returns.cov().values
    diag_only = np.diag(np.diag(sigma))
    return shrinkage * diag_only + (1 - shrinkage) * sigma


def risk_parity_weights(returns: pd.DataFrame, shrinkage=0.3):
    """Long-only equal-risk-contribution weights over `returns.columns`."""
    sigma = shrunk_covariance(returns, shrinkage=shrinkage)
    n = sigma.shape[0]

    def risk_contributions(w):
        port_var = w @ sigma @ w
        marginal = sigma @ w
        return w * marginal / port_var if port_var > 0 else np.zeros(n)

    def objective(w):
        rc = risk_contributions(w)
        target = 1.0 / n
        return np.sum((rc - target) ** 2)

    w0 = np.full(n, 1.0 / n)
    bounds = [(1e-6, 1.0) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-14})
    w = result.x if result.success else w0
    w = np.clip(w, 0, None)
    return pd.Series(w / w.sum(), index=returns.columns)


def tilt_weights(base_weights: pd.Series, scores: pd.Series, tilt_strength=0.5, max_tilt_multiple=3.0):
    """
    Tilt risk-parity base weights toward names with higher ML-predicted
    scores, scaled by `tilt_strength` (0 = pure risk parity). `scores`
    should be cross-sectionally standardized (z-scores) already. Tilted
    weight for asset i = base_weight_i * exp(tilt_strength * score_i),
    renormalized, capped at `max_tilt_multiple`x its base weight so a
    single strong-score name can't dominate the book.
    """
    scores = scores.reindex(base_weights.index).fillna(0.0)
    raw_tilt = base_weights * np.exp(tilt_strength * scores)
    capped = np.minimum(raw_tilt, base_weights * max_tilt_multiple)
    return capped / capped.sum()
