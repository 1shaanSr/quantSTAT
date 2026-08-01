"""
Two supplementary diagnostics, both built on a partial derivative:

1. Market-beta exposure: regressing the portfolio's daily return against a
   market proxy's (SPY) daily return gives beta = d(PortfolioReturn) /
   d(MarketReturn) -- the realized sensitivity of the book to the broad
   market. This project is described as "market-neutral"; that claim has
   an empirical implication (beta ~ 0) which is checked directly here
   rather than assumed from the dollar-neutral construction of each pair.

2. Hyperparameter sensitivity: local finite-difference estimates of
   d(Sharpe)/d(param) around the locked operating point (entry_z, exit_z,
   Kalman delta), evaluated ONLY on formation-period folds -- the same
   data used for tuning, never the test window. This checks whether the
   grid-search optimum sits on a stable plateau or a fragile, isolated
   peak; a peak that vanishes under a small perturbation is a classic
   overfitting signature.
"""
import numpy as np
import pandas as pd


def market_beta_exposure(portfolio_equity: pd.Series, market_prices: pd.Series, rf_annual=0.045):
    """
    OLS regression: portfolio_daily_return = alpha + beta * market_daily_return + eps.
    beta is the partial derivative of the portfolio's return with respect to
    the market's return, holding the idiosyncratic spread P&L fixed --
    i.e. the realized net directional exposure of a book that is supposed
    to be market-neutral by construction (each pair is dollar/beta-hedged
    individually, but only this regression checks whether that held up in
    aggregate, after costs, sizing, and timing effects).
    """
    port_ret = portfolio_equity.pct_change().dropna()
    mkt_ret = market_prices.pct_change().dropna()
    df = pd.concat([port_ret, mkt_ret], axis=1, join='inner')
    df.columns = ['portfolio', 'market']
    n = len(df)

    X = np.column_stack([np.ones(n), df['market'].values])
    y = df['portfolio'].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_daily, beta = coef

    y_hat = X @ coef
    resid = y - y_hat
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    dof = n - 2
    sigma2 = ss_res / dof if dof > 0 else np.nan
    X_var = np.var(df['market'].values, ddof=1) * (n - 1)
    se_beta = np.sqrt(sigma2 / X_var) if X_var > 0 else np.nan
    t_stat = beta / se_beta if se_beta and se_beta > 0 else np.nan

    corr = df['portfolio'].corr(df['market'])
    alpha_annualized = (1 + alpha_daily) ** 252 - 1

    return dict(
        n_days=n, beta=beta, alpha_daily=alpha_daily, alpha_annualized=alpha_annualized,
        r_squared=r_squared, t_stat=t_stat, se_beta=se_beta, correlation=corr,
    )


def print_market_beta(m, market_symbol="SPY"):
    print(f"\n--- Market-neutrality check: portfolio beta to {market_symbol} ---")
    print(f"d(PortfolioReturn)/d({market_symbol}Return) = beta = {m['beta']:+.4f}  "
          f"(t-stat {m['t_stat']:.2f}, se {m['se_beta']:.4f}, n={m['n_days']} days)")
    print(f"Correlation to {market_symbol}: {m['correlation']:+.3f}   R-squared: {m['r_squared']:.3f}")
    print(f"Regression alpha (annualized): {m['alpha_annualized']*100:+.2f}%")
    verdict = "not significantly different from zero" if abs(m['t_stat']) < 2 else "STATISTICALLY SIGNIFICANT non-zero exposure"
    print(f"Verdict: beta is {verdict} at conventional (|t|<2 ~ 95%) significance.")
    if abs(m['t_stat']) >= 2:
        print("         The 'market-neutral' label does not fully hold empirically -- ")
        print("         the book carries a residual, statistically real directional tilt to the market.")


def hyperparameter_sensitivity(formation, pairs, base_entry_z, base_exit_z, base_delta,
                                capital=100_000.0, risk_per_trade=0.15):
    """
    Local finite-difference sensitivity of in-formation Sharpe to each
    hyperparameter, holding the others at their locked value. Uses the
    SAME 3 rolling formation folds as tune_hyperparams -- never the test
    window. A perturbation of +/-15% is applied to each parameter in turn.
    """
    from src.pairs_engine import _simulate_pair_smoothed
    from src.metrics import compute_metrics

    fold_starts = [int(len(formation) * f) for f in (0.0, 0.15, 0.30)]
    fold_len = int(len(formation) * 0.55)

    def portfolio_sharpe(entry_z, exit_z, delta):
        fold_sharpes = []
        for fs in fold_starts:
            fold = formation.iloc[fs: fs + fold_len]
            cap_pp = capital / len(pairs)
            pnl_frames = [_simulate_pair_smoothed(fold[a], fold[b], entry_z=entry_z, exit_z=exit_z,
                                                   delta=delta, capital=cap_pp,
                                                   risk_per_trade=risk_per_trade)[0] for a, b in pairs]
            combined = pd.concat(pnl_frames, axis=1).sum(axis=1)
            equity = capital + combined.cumsum()
            m = compute_metrics(equity)
            fold_sharpes.append(m['sharpe_rf0'])
        return np.nanmean(fold_sharpes)

    base_sharpe = portfolio_sharpe(base_entry_z, base_exit_z, base_delta)

    results = {'base': {'entry_z': base_entry_z, 'exit_z': base_exit_z, 'delta': base_delta,
                         'sharpe': base_sharpe}}

    perturbations = {
        'entry_z': (base_entry_z * 0.85, base_entry_z * 1.15),
        'exit_z': (base_exit_z * 0.85, base_exit_z * 1.15),
        'delta': (base_delta * 0.5, base_delta * 2.0),  # delta spans orders of magnitude, wider sweep
    }

    for param, (lo, hi) in perturbations.items():
        kwargs_lo = dict(entry_z=base_entry_z, exit_z=base_exit_z, delta=base_delta)
        kwargs_hi = dict(entry_z=base_entry_z, exit_z=base_exit_z, delta=base_delta)
        kwargs_lo[param] = lo
        kwargs_hi[param] = hi
        sharpe_lo = portfolio_sharpe(**kwargs_lo)
        sharpe_hi = portfolio_sharpe(**kwargs_hi)
        d_param = hi - lo
        d_sharpe_d_param = (sharpe_hi - sharpe_lo) / d_param if d_param != 0 else np.nan
        results[param] = {'lo': lo, 'hi': hi, 'sharpe_lo': sharpe_lo, 'sharpe_hi': sharpe_hi,
                           'd_sharpe_d_param': d_sharpe_d_param}

    return results


def print_sensitivity(results):
    print("\n--- Hyperparameter sensitivity (formation folds only, never the test window) ---")
    base = results['base']
    print(f"Locked point: entry_z={base['entry_z']}, exit_z={base['exit_z']}, delta={base['delta']:.1e}  "
          f"-> avg in-formation Sharpe = {base['sharpe']:.3f}")
    for param in ('entry_z', 'exit_z', 'delta'):
        r = results[param]
        print(f"\n  {param}: swept [{r['lo']:.4g}, {r['hi']:.4g}] around {base[param]:.4g}")
        print(f"    Sharpe at low end:  {r['sharpe_lo']:.3f}")
        print(f"    Sharpe at locked:   {base['sharpe']:.3f}")
        print(f"    Sharpe at high end: {r['sharpe_hi']:.3f}")
        print(f"    d(Sharpe)/d({param}) ~= {r['d_sharpe_d_param']:+.4f} per unit {param}")
        swing = abs(r['sharpe_hi'] - r['sharpe_lo'])
        is_peak = base['sharpe'] > r['sharpe_lo'] and base['sharpe'] > r['sharpe_hi']
        if swing < 0.3:
            flag = "STABLE (small swing across the perturbation)"
        elif is_peak:
            flag = "FRAGILE PEAK (locked value outperforms both neighbors, but by a lot -- a knife-edge optimum, classic overfitting signature)"
        else:
            flag = "SLOPE (Sharpe moves consistently in one direction -- locked value is not even a local optimum on this axis, just where the grid search stopped)"
        print(f"    -> {flag}")
