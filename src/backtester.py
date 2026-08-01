import pandas as pd
from src.universe import candidate_pairs
from src.cointegration import screen_pairs
from src.metrics import compute_metrics
from src.pairs_engine import (
    download_universe, formation_test_split, tune_hyperparams, run_pooled_portfolio,
    _simulate_pair_smoothed, ENTRY_Z, EXIT_Z, STOP_Z, MAX_HOLD, COST_BPS, RISK_PER_TRADE,
    MAX_GROSS_EXPOSURE, DELTA,
)
from src.risk_analysis import (
    market_beta_exposure, print_market_beta, hyperparameter_sensitivity, print_sensitivity,
)
from src.allocation import min_variance_weights


class StatisticalArbitrageBacktester:
    """
    Portfolio-level statistical arbitrage engine.

    Scope: this is not a single-symbol backtest. The strategy trades a
    portfolio of Engle-Granger-cointegrated pairs, discovered across an
    economically-grouped universe (src/universe.py), traded with a
    Kalman-filtered (recursive Bayesian) dynamic hedge ratio and
    precision-weighted position sizing (src/kalman_hedge.py), and combined
    into one pooled-capital, gross-exposure-capped book.

    Pair selection and hyperparameter tuning happen ONLY on a formation
    (in-sample) window; every metric reported by run() is computed on a
    test (out-of-sample) window that selection/tuning never see. See
    TECHNICAL_DOCS.md for the full research derivation.
    """
    def __init__(self, initial_balance=100_000.0):
        self.initial_balance = initial_balance
        self.results = None

    def run(self, symbol=None, days=None, retune=False, sensitivity=False,
            extended_universe=False, min_variance=False):
        """
        `symbol`, if provided and present in the discovered pair list, is
        just used to highlight that pair's line in the report -- the
        strategy always trades the full validated portfolio, since a
        single-pair "backtest" is not how this strategy is meant to run
        (see TECHNICAL_DOCS.md for why the project moved away from that).
        `days`, if provided, trims the final report to the trailing N days
        of the out-of-sample test window (the walk-forward split itself is
        fixed by the available history, not by this parameter).
        `retune`, if True, re-runs the formation-only hyperparameter grid
        search instead of using the locked, documented defaults (slower,
        but reproduces exactly how those defaults were derived).
        `sensitivity`, if True, additionally runs a local finite-difference
        sensitivity sweep of Sharpe with respect to entry_z/exit_z/delta
        around the locked point, on formation folds only (slower -- similar
        cost to the tuning grid search).
        `extended_universe` and `min_variance` reproduce two things that
        were tried in pursuit of a higher Sharpe and did NOT beat the
        default -- see TECHNICAL_DOCS.md section 3 for the full writeup.
        Both default to False so the default call matches the documented,
        locked numbers.
        """
        print("\n=== Statistical Arbitrage Backtest: portfolio of cointegrated pairs (real data) ===")
        print("Downloading universe...")
        try:
            prices = download_universe(include_extended=extended_universe)
        except Exception as e:
            print(f"Data download failed: {e}")
            return None

        formation, test = formation_test_split(prices, formation_frac=0.65)
        print(f"Formation (in-sample, selection+tuning): {formation.index[0].date()} to {formation.index[-1].date()} "
              f"({len(formation)} days)")
        print(f"Test (out-of-sample, reported below):    {test.index[0].date()} to {test.index[-1].date()} "
              f"({len(test)} days)")

        print("\nScreening for cointegration (Engle-Granger + OU half-life, formation window only)...")
        pairs_all = candidate_pairs(include_extended=extended_universe)
        screened = screen_pairs(formation, pairs_all)
        if not screened:
            print("No pairs passed the cointegration screen on this formation window.")
            return None
        pairs = [(r['a'], r['b']) for r in screened]
        print(f"{len(pairs)} pairs passed (p<0.05, |corr|>=0.5, half-life in [2,60]d):")
        for r in screened:
            marker = "  <-- involves requested symbol" if symbol and symbol in (r['a'], r['b']) else ""
            print(f"  {r['a']}-{r['b']} ({r['bucket']}): p={r['pvalue']:.4f} corr={r['corr']:.3f} "
                  f"beta={r['beta']:.3f} half_life={r['half_life']:.1f}d{marker}")

        if retune:
            print("\nRe-tuning entry/exit z-score thresholds on formation-only walk-forward folds...")
            entry_z, exit_z, avg_sharpe = tune_hyperparams(formation, pairs)
            print(f"Selected entry_z={entry_z}, exit_z={exit_z} (avg in-formation Sharpe={avg_sharpe:.2f})")
        else:
            entry_z, exit_z = ENTRY_Z, EXIT_Z
            print(f"\nUsing locked, formation-derived hyperparameters: entry_z={entry_z}, exit_z={exit_z} "
                  "(pass retune=True to re-derive)")

        pair_weights = None
        if min_variance:
            print("\nComputing minimum-variance capital allocation from formation-period P&L "
                  "(see TECHNICAL_DOCS.md 3.7 -- this underperformed equal-weight when tested)...")
            pnl_frames = {}
            for a, b in pairs:
                pnl, _ = _simulate_pair_smoothed(formation[a], formation[b],
                                                  capital=self.initial_balance / len(pairs), risk_per_trade=0.15)
                pnl_frames[f"{a}-{b}"] = pnl
            pnl_matrix = pd.DataFrame(pnl_frames)
            weights_named = min_variance_weights(pnl_matrix)
            pair_weights = {(a, b): weights_named[f"{a}-{b}"] for a, b in pairs}

        print("\nRunning pooled-capital portfolio simulation on the out-of-sample test window...")
        equity, m, trade_counts = run_pooled_portfolio(
            pairs, formation, test, entry_z=entry_z, exit_z=exit_z,
            stop_z=STOP_Z, max_hold=MAX_HOLD, cost_bps=COST_BPS,
            risk_per_trade=RISK_PER_TRADE, max_gross_exposure=MAX_GROSS_EXPOSURE,
            capital=self.initial_balance, pair_weights=pair_weights,
        )

        if days:
            equity = equity.iloc[-min(days, len(equity)):]
            m = compute_metrics(equity, rf_annual=m['rf_annual'])

        self.results = {'equity': equity, 'metrics': m, 'trade_counts': trade_counts,
                         'pairs': pairs, 'screened': screened}
        self._print_results(m, trade_counts)

        beta_result = None
        if 'SPY' in prices.columns:
            spy = prices['SPY'].loc[equity.index[0]:equity.index[-1]]
            beta_result = market_beta_exposure(equity, spy)
            print_market_beta(beta_result, market_symbol="SPY")
            self.results['market_beta'] = beta_result

        if sensitivity:
            sens = hyperparameter_sensitivity(formation, pairs, entry_z, exit_z, DELTA,
                                               capital=self.initial_balance, risk_per_trade=0.15)
            print_sensitivity(sens)
            self.results['sensitivity'] = sens

        return self.results

    def _print_results(self, m, trade_counts):
        print("\n" + "=" * 60)
        print("STATISTICAL ARBITRAGE BACKTEST RESULTS (real data, out-of-sample)")
        print("=" * 60)
        print(f"Backtest window: {m['n_days']} trading days ({m['years']:.2f} years)")
        print(f"Starting capital: ${self.initial_balance:,.2f}")
        print("-" * 40)
        print(f"Total return: {m['total_return']*100:+.2f}%")
        print(f"Annualized return (CAGR): {m['ann_return']*100:+.2f}%")
        print(f"Annualized volatility: {m['ann_vol']*100:.2f}%")
        print(f"Sharpe Ratio (rf=0%):        {m['sharpe_rf0']:.2f}")
        print(f"Sharpe Ratio (rf={m['rf_annual']*100:.1f}%):     {m['sharpe_rf']:.2f}")
        print(f"Sortino Ratio (rf=0%): {m['sortino']:.2f}")
        print(f"Max Drawdown: {m['max_dd']*100:.2f}%  (peak {m['peak_date'].date()} -> trough {m['trough_date'].date()})")
        print(f"Calmar Ratio: {m['calmar']:.2f}")
        print("-" * 40)
        print(f"Total trades: {sum(trade_counts.values())}")
        for pair, n in trade_counts.items():
            print(f"  {pair[0]}-{pair[1]}: {n} trades")
        print("=" * 60)
