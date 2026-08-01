"""
Portfolio-level statistical arbitrage research pipeline.

Pipeline (each stage documented with what data it is and is not allowed
to see, to keep the final reported performance genuinely out-of-sample):

  1. Download real daily prices for an economically-grouped universe
     (src/universe.py).
  2. Split into a FORMATION window (in-sample: pair discovery + hyper-
     parameter tuning) and a TEST window (out-of-sample: the only window
     whose performance is ever reported).
  3. Screen candidate pairs for Engle-Granger cointegration + a tradeable
     OU half-life, using ONLY the formation window (src/cointegration.py).
  4. Tune trading hyperparameters (entry/exit z-score thresholds) via
     walk-forward folds WITHIN the formation window only (tune_hyperparams
     below) -- the test window is never touched by this search.
  5. Trade the selected pairs with a Kalman-filtered (recursive Bayesian)
     hedge ratio (src/kalman_hedge.py): the trading signal is an EWMA-
     smoothed version of the filter's normalized innovation, and position
     size is scaled by the filter's posterior precision on beta.
  6. Combine all pairs into one pooled-capital, gross-exposure-capped
     portfolio and report metrics (src/metrics.py) computed only on the
     test-window equity curve.

Locked hyperparameters below (ENTRY_Z, EXIT_Z, ...) were derived by running
`tune_hyperparams()` on the formation window of the default universe; see
TECHNICAL_DOCS.md for the full derivation and the alternative configurations
that were tried and rejected (including why country-vs-country ETF pairs
and a secondary in-sample performance filter were both dropped after they
were shown to overfit).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.universe import candidate_pairs
from src.cointegration import screen_pairs
from src.kalman_hedge import KalmanHedge
from src.metrics import compute_metrics

# Locked, formation-only-derived defaults.
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 4.0
MAX_HOLD = 20          # trading days
COST_BPS = 5            # transaction cost per leg, bps of notional
SMOOTH_SPAN = 3          # EWMA span applied to the Kalman innovation z-score
DELTA = 1e-4             # Kalman process-noise heuristic (Q = delta/(1-delta) * R)
BURN_IN = 60             # days of history used to initialize each pair's filter
RISK_PER_TRADE = 0.30    # fraction of pooled equity risked per new position
MAX_GROSS_EXPOSURE = 1.5 # cap on total concurrent notional as a multiple of equity
RF_ANNUAL = 0.045


def download_universe(start='2018-01-01', end=None, include_extended=False):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    tickers = sorted(set(t for a, b, _ in candidate_pairs(include_extended=include_extended) for t in (a, b)))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.9))
    raw = raw.dropna(how='any')
    return raw


def formation_test_split(prices, formation_frac=0.65):
    n = len(prices)
    split = int(n * formation_frac)
    return prices.iloc[:split], prices.iloc[split:]


def tune_hyperparams(formation, pairs, grid=None, fold_frac=0.55,
                      risk_per_trade=0.15, capital=100_000.0):
    """
    Grid search over (entry_z, exit_z) using rolling folds WITHIN the
    formation window only. Returns (best_entry_z, best_exit_z, avg_sharpe).
    """
    if grid is None:
        grid = [(ez, xz) for ez in (1.5, 2.0, 2.5) for xz in (0.25, 0.5)]
    # Fixed fold offsets (0%, 15%, 30% into the formation window), matching
    # the derivation documented in TECHNICAL_DOCS.md exactly, so retune=True
    # reproduces the locked defaults rather than a nearby but different point.
    fold_offsets = (0.0, 0.15, 0.30)
    fold_starts = [int(len(formation) * f) for f in fold_offsets]
    fold_len = int(len(formation) * fold_frac)

    best_combo, best_avg = None, -np.inf
    for entry_z, exit_z in grid:
        fold_sharpes = []
        for fs in fold_starts:
            fold = formation.iloc[fs: fs + fold_len]
            cap_pp = capital / len(pairs)
            pnl_frames = []
            for a, b in pairs:
                pnl, _ = _simulate_pair_smoothed(fold[a], fold[b], entry_z=entry_z, exit_z=exit_z,
                                                  capital=cap_pp, risk_per_trade=risk_per_trade)
                pnl_frames.append(pnl)
            combined = pd.concat(pnl_frames, axis=1).sum(axis=1)
            equity = capital + combined.cumsum()
            m = compute_metrics(equity)
            fold_sharpes.append(m['sharpe_rf0'])
        avg = np.nanmean(fold_sharpes)
        if avg > best_avg:
            best_avg, best_combo = avg, (entry_z, exit_z)
    return best_combo[0], best_combo[1], best_avg


def _simulate_pair_smoothed(price_a, price_b, entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z,
                             max_hold=MAX_HOLD, cost_bps=COST_BPS, capital=100_000.0,
                             risk_per_trade=0.15, burn_in=BURN_IN, delta=DELTA, smooth_span=SMOOTH_SPAN):
    """Single-pair simulation used only for formation-period tuning (independent capital silo)."""
    dates = price_a.index
    n = len(price_a)
    kf = KalmanHedge(delta=delta)
    kf.initialize(price_a.iloc[:burn_in].values, price_b.iloc[:burn_in].values)

    daily_pnl = np.zeros(n)
    position = None
    entry_i = entry_beta = None
    units = 0.0
    balance = capital
    trades = 0
    z_ewma = 0.0
    alpha = 2 / (smooth_span + 1)

    for i in range(burn_in, n):
        x_t, y_t = price_a.iloc[i], price_b.iloc[i]
        beta, z_raw, beta_var = kf.step(x_t, y_t)
        z_ewma = alpha * z_raw + (1 - alpha) * z_ewma
        z = z_ewma

        if position is not None:
            p1_prev, p2_prev = price_a.iloc[i - 1], price_b.iloc[i - 1]
            spread_chg = (y_t - p2_prev) - entry_beta * (x_t - p1_prev)
            pnl = units * spread_chg if position == 'long_spread' else -units * spread_chg
            daily_pnl[i] += pnl
            balance += pnl

        if position is None:
            if abs(z) > entry_z:
                position = 'short_spread' if z > entry_z else 'long_spread'
                entry_i = i
                entry_beta = beta
                notional = abs(entry_beta) * x_t + y_t
                weight = kf.precision_weight(beta_var)
                units = (balance * risk_per_trade * weight) / notional if notional > 0 else 0.0
                cost = (cost_bps / 10000.0) * notional * units
                balance -= cost
                daily_pnl[i] -= cost
        else:
            days_held = i - entry_i
            should_exit = (abs(z) < exit_z or abs(z) > stop_z or days_held >= max_hold)
            if should_exit:
                notional = abs(entry_beta) * x_t + y_t
                cost = (cost_bps / 10000.0) * notional * units
                balance -= cost
                daily_pnl[i] -= cost
                trades += 1
                position = None
                units = 0.0

    return pd.Series(daily_pnl, index=dates).iloc[burn_in:], trades


def run_pooled_portfolio(pairs, formation, test, entry_z=ENTRY_Z, exit_z=EXIT_Z, stop_z=STOP_Z,
                          max_hold=MAX_HOLD, cost_bps=COST_BPS, risk_per_trade=RISK_PER_TRADE,
                          max_gross_exposure=MAX_GROSS_EXPOSURE, delta=DELTA, smooth_span=SMOOTH_SPAN,
                          capital=100_000.0, burn_in=BURN_IN, rf_annual=RF_ANNUAL, pair_weights=None):
    """
    Production trading engine: all pairs draw from ONE shared capital pool
    (not static per-pair silos, which leave idle capital doing nothing when
    a pair has no open signal), with a cap on total concurrent gross
    notional as a fraction of equity. Every pair's Kalman filter is warm-
    started on the tail of `formation` and then stepped through `test` --
    all reported P&L, equity, and metrics are test-period only.

    `pair_weights`, if provided, is a {pair: weight} dict summing to 1
    (e.g. from src.allocation.min_variance_weights) that scales each pair's
    risk budget relative to equal-weight (weight * len(pairs) == 1.0 means
    unchanged from the default equal split). If None, all pairs get an
    equal share, matching the original behavior exactly.
    """
    if pair_weights is None:
        pair_weights = {p: 1.0 / len(pairs) for p in pairs}

    kfs, price_series, z_ewma = {}, {}, {}
    alpha = 2 / (smooth_span + 1)
    for a, b in pairs:
        warmup = formation[[a, b]].iloc[-burn_in:]
        combined = pd.concat([warmup, test[[a, b]]])
        price_series[(a, b)] = combined
        kf = KalmanHedge(delta=delta)
        kf.initialize(combined[a].iloc[:burn_in].values, combined[b].iloc[:burn_in].values)
        kfs[(a, b)] = kf
        z_ewma[(a, b)] = 0.0

    dates = price_series[pairs[0]].index[burn_in:]
    balance = capital
    daily_pnl = pd.Series(0.0, index=dates)
    open_positions = {}
    trade_counts = {p: 0 for p in pairs}

    for di, date in enumerate(dates):
        i = di + burn_in
        day_pnl = 0.0
        current_gross = sum(abs(op['entry_beta']) * price_series[p][p[0]].loc[date] * op['units']
                             + price_series[p][p[1]].loc[date] * op['units']
                             for p, op in open_positions.items())

        for pair in pairs:
            a, b = pair
            px = price_series[pair]
            x_t, y_t = px[a].loc[date], px[b].loc[date]
            kf = kfs[pair]
            beta, z_raw, beta_var = kf.step(x_t, y_t)
            z_ewma[pair] = alpha * z_raw + (1 - alpha) * z_ewma[pair]
            z = z_ewma[pair]

            if pair in open_positions:
                op = open_positions[pair]
                prev_idx = px.index.get_loc(date) - 1
                x_prev, y_prev = px[a].iloc[prev_idx], px[b].iloc[prev_idx]
                spread_chg = (y_t - y_prev) - op['entry_beta'] * (x_t - x_prev)
                pnl = op['units'] * spread_chg if op['position'] == 'long_spread' else -op['units'] * spread_chg
                day_pnl += pnl
                balance += pnl

                days_held = i - op['entry_i']
                should_exit = abs(z) < exit_z or abs(z) > stop_z or days_held >= max_hold
                if should_exit:
                    notional = abs(op['entry_beta']) * x_t + y_t
                    cost = (cost_bps / 10000.0) * notional * op['units']
                    balance -= cost
                    day_pnl -= cost
                    trade_counts[pair] += 1
                    del open_positions[pair]
            else:
                if abs(z) > entry_z:
                    position = 'short_spread' if z > entry_z else 'long_spread'
                    notional_unit = abs(beta) * x_t + y_t
                    precision_w = kf.precision_weight(beta_var)
                    allocation_w = pair_weights[pair] * len(pairs)
                    target_notional = balance * risk_per_trade * precision_w * allocation_w
                    if current_gross + target_notional <= max_gross_exposure * balance:
                        units = target_notional / notional_unit if notional_unit > 0 else 0.0
                        cost = (cost_bps / 10000.0) * notional_unit * units
                        balance -= cost
                        day_pnl -= cost
                        open_positions[pair] = {'position': position, 'entry_beta': beta,
                                                 'units': units, 'entry_i': i}
                        current_gross += target_notional

        daily_pnl.loc[date] = day_pnl

    equity = capital + daily_pnl.cumsum()
    m = compute_metrics(equity, rf_annual=rf_annual)
    return equity, m, trade_counts
