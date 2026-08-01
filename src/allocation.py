"""
Minimum-variance capital allocation across validated pairs, replacing
equal-capital (1/N) weighting.

Rationale for minimum-variance over full mean-variance (Markowitz)
optimization: mean-variance weights are highly sensitive to noisy
expected-return estimates, and this project already has a documented
lesson (TECHNICAL_DOCS.md section 3) about a formation-period-performance
based pair filter overfitting and hurting out-of-sample results. Minimum-
variance weighting only requires a covariance estimate, not a return
estimate, and is well-documented in the literature (DeMiguel, Garlappi &
Uppal 2009) as more robust out-of-sample than full Markowitz optimization
specifically because of this.

Two safeguards are applied because an unconstrained min-variance solve on
this data is known to misbehave (verified empirically, not hypothetically
-- an early version of this allocator put 66% of capital into DBC-PDBC,
two ETFs that track nearly identical broad-commodity indices and trade
frequently with very low P&L noise, i.e. a near-arbitrage basis trade
rather than a genuine differentiated statistical-arbitrage bet, while
zeroing out two pairs that were actively and independently profitable in
formation). This is a well-documented failure mode of naive min-variance
optimization: it chases the lowest-noise asset rather than true
diversification.

  1. Shrinkage toward the diagonal, controlling how much off-diagonal
     (correlation) structure is trusted given sparse per-pair P&L series.
  2. A hard cap on any single pair's weight (as a multiple of equal-weight),
     solved as a proper constrained QP (minimize w'Sigma w s.t. sum(w)=1,
     0<=w<=cap) via scipy -- a standard, disclosed portfolio-construction
     constraint, not a fit to any observed result.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def shrunk_covariance(pnl_matrix: pd.DataFrame, shrinkage=0.5):
    """
    Linear shrinkage toward the diagonal: Sigma_shrunk = (1-s)*Sigma + s*diag(Sigma).
    A simplified, fixed-intensity version of Ledoit-Wolf shrinkage -- chosen
    over the fully data-driven Ledoit-Wolf estimator for transparency (one
    disclosed knob) given the sparse per-pair P&L series here.
    """
    sigma = pnl_matrix.cov().values
    diag_only = np.diag(np.diag(sigma))
    return shrinkage * diag_only + (1 - shrinkage) * sigma


def _solve_capped_min_variance(sigma, cap):
    n = sigma.shape[0]
    w0 = np.full(n, 1.0 / n)

    def objective(w):
        return w @ sigma @ w

    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bounds = [(0.0, cap) for _ in range(n)]

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 500, 'ftol': 1e-12})
    if not result.success or np.any(result.x < -1e-6):
        return w0  # fall back to equal weight if the solver fails
    return np.clip(result.x, 0, cap)


def min_variance_weights(pnl_matrix: pd.DataFrame, shrinkage=0.5, min_active_days=3, weight_cap_multiple=2.5):
    """
    Solve for long-only, capped minimum-variance weights over the pairs
    whose columns are in `pnl_matrix` (each column = one pair's formation-
    period daily P&L at a reference position size). Pairs with fewer than
    `min_active_days` nonzero P&L days in formation have no meaningful
    variance/covariance estimate and are given a floor weight instead of
    letting a near-zero-variance column dominate or a solver treat them as
    identically zero-risk.
    """
    active = (pnl_matrix != 0).sum(axis=0)
    tradeable = active[active >= min_active_days].index.tolist()
    illiquid = [c for c in pnl_matrix.columns if c not in tradeable]

    weights = pd.Series(0.0, index=pnl_matrix.columns)

    if len(tradeable) >= 2:
        sub = pnl_matrix[tradeable]
        sigma = shrunk_covariance(sub, shrinkage=shrinkage)
        cap = weight_cap_multiple / len(tradeable)
        raw = _solve_capped_min_variance(sigma, cap)
        weights.loc[tradeable] = raw
    elif len(tradeable) == 1:
        weights.loc[tradeable[0]] = 1.0

    # Reserve a modest floor allocation for pairs that rarely/never traded
    # in formation -- cointegration held, but there's no formation evidence
    # to weight them by, so they get equal, capped shares of a small pool
    # rather than zero (excluding them outright would just be a second,
    # undisclosed selection filter of exactly the kind already flagged as
    # a mistake elsewhere in this project).
    floor_pool = 0.10 if illiquid else 0.0
    if illiquid:
        weights.loc[tradeable] *= (1 - floor_pool)
        weights.loc[illiquid] = floor_pool / len(illiquid)

    return weights
