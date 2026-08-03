"""
Orchestrates the full pipeline: download data, build causal factors, run a
walk-forward ML return predictor with honest IC evaluation, and compare a
pure risk-parity portfolio against an ML-tilted variant -- with the tilt
strength chosen ONLY from formation-period evidence, and every headline
number computed on a test window neither the model nor the tilt decision
ever saw. See TECHNICAL_DOCS.md for the full derivation.
"""
from datetime import datetime
import pandas as pd
import yfinance as yf

from src.universe import TICKERS
from src.features import build_feature_panel, FEATURE_COLS, FEATURE_COLS_WITH_DIVIDENDS
from src.dividend_features import fetch_dividend_history, build_dividend_features
from src.ml_predictor import walk_forward_predict, summarize_ic
from src.adaptive_tilt import build_adaptive_tilt_schedule
from src.portfolio_backtest import run_portfolio
from src.metrics import compute_metrics, print_metrics
from src.benchmarks import (
    fetch_benchmark_prices, build_benchmark_curves, build_continuous_benchmark_curves,
    crisis_period_return_dd, CRISIS_PERIODS,
)
from src.significance import bootstrap_sharpe_calmar, summarize_bootstrap

FORWARD_DAYS = 20
FORMATION_FRAC = 0.65
TILT_STRENGTH = 0.0  # locked: formation-only evidence found tilting hurts Sharpe; see TECHNICAL_DOCS.md


class RiskParityMLBacktester:
    def __init__(self, initial_balance=100_000.0):
        self.initial_balance = initial_balance
        self.results = None

    def _download(self, start='2016-01-01', end=None):
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')
        raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)
        close = raw['Close'].dropna(axis=1, thresh=int(len(raw) * 0.95)).dropna(how='any')
        volume = raw['Volume'][close.columns].reindex(close.index)
        return close, volume

    def run(self, tilt_strength=None, cost_bps=10, include_dividends=False, adaptive_tilt=False, verify=True):
        """
        `include_dividends` and `adaptive_tilt` reproduce two more things
        tried in the ML-signal investigation (TECHNICAL_DOCS.md section 4)
        -- neither improved on the locked configuration, so both default
        to False. `tilt_strength` is ignored if `adaptive_tilt=True`.
        `verify=True` (default): after the headline result, runs benchmark
        comparison (equal-weight, SPY, 60/40) and a block-bootstrap
        confidence interval on the LOCKED pure-risk-parity strategy over
        its full available history -- see TECHNICAL_DOCS.md section 5.
        """
        tilt_strength = TILT_STRENGTH if tilt_strength is None else tilt_strength

        print("\n=== Risk-Parity + ML Factor Tilt Backtest (real data) ===")
        print("Downloading universe...")
        close, volume = self._download()
        print(f"Data: {close.index[0].date()} to {close.index[-1].date()} ({len(close)} days, {close.shape[1]} tickers)")

        div_yield = div_growth = None
        if include_dividends:
            print("Fetching dividend history for div_yield/div_growth factors...")
            div_by_ticker = fetch_dividend_history(close.columns.tolist(), close.index[0], close.index[-1])
            div_yield, div_growth = build_dividend_features(close, div_by_ticker)

        print("Building causal factor panel and walk-forward ML predictions...")
        panel = build_feature_panel(close, volume, forward_days=FORWARD_DAYS,
                                     div_yield=div_yield, div_growth=div_growth)
        all_dates = sorted(panel['date'].dropna().unique())
        rebalance_dates = all_dates[::FORWARD_DAYS]

        feature_cols = FEATURE_COLS_WITH_DIVIDENDS if include_dividends else FEATURE_COLS
        pred_df, ic_df = walk_forward_predict(panel, rebalance_dates, feature_cols=feature_cols)

        # Split on dates the ML model actually produced predictions for
        # (i.e. after the min_train_periods burn-in) so the formation/test
        # boundary is identical for the IC evaluation and the portfolio
        # backtest -- using the raw rebalance_dates list here would silently
        # pull the boundary earlier (into dates with no real ML history)
        # and desynchronize the two.
        usable_dates = sorted(pred_df['date'].unique())
        n = len(usable_dates)
        split = int(n * FORMATION_FRAC)
        formation_dates = usable_dates[:split]
        test_dates = usable_dates[split:]
        print(f"Formation: {pd.Timestamp(formation_dates[0]).date()} to {pd.Timestamp(formation_dates[-1]).date()} "
              f"({len(formation_dates)} rebalance periods)")
        print(f"Test (out-of-sample, reported below): {pd.Timestamp(test_dates[0]).date()} to "
              f"{pd.Timestamp(test_dates[-1]).date()} ({len(test_dates)} rebalance periods)")

        ic_formation = ic_df[ic_df['date'].isin(formation_dates)]
        ic_test = ic_df[ic_df['date'].isin(test_dates)]
        summarize_ic(ic_formation, label="ML return predictor -- formation")
        summarize_ic(ic_test, label="ML return predictor -- TRUE OOS")

        if adaptive_tilt:
            print("\nBuilding adaptive tilt schedule (trailing realized IC, causal)...")
            tilt_arg = build_adaptive_tilt_schedule(ic_df, test_dates)
            tilt_label = "adaptive (trailing-IC-scaled)"
        else:
            tilt_arg = tilt_strength
            tilt_label = str(tilt_strength)

        print(f"\nRunning risk-parity portfolio (tilt={tilt_label}) on the out-of-sample test window...")
        pnl, weight_history = run_portfolio(close, pred_df, test_dates, tilt_strength=tilt_arg,
                                             cost_bps=cost_bps, capital=self.initial_balance)
        equity = self.initial_balance + pnl.cumsum()
        equity = equity.loc[pd.Timestamp(test_dates[0]):pd.Timestamp(test_dates[-1])]
        m = compute_metrics(equity)

        self.results = {'equity': equity, 'metrics': m, 'ic_formation': ic_formation,
                         'ic_test': ic_test, 'tilt_strength': tilt_label}
        self._print_results(m, tilt_label)

        if verify:
            self._run_verification(close, test_dates, cost_bps)

        return self.results

    def _run_verification(self, close, test_dates, cost_bps):
        """
        Benchmark comparison + bootstrap significance for the LOCKED pure
        risk-parity strategy (tilt=0), regardless of what tilt_strength/
        adaptive_tilt/include_dividends the caller passed to run() -- this
        section always characterizes the actual production strategy.

        Two different, deliberate conventions (see src/benchmarks.py
        module docstring for why): the test-window comparison uses a FRESH
        start at the test boundary (matching exactly how the headline
        result is defined), while crisis sub-periods use a CONTINUOUSLY
        run curve sliced at the crisis window (representing an
        already-invested, ongoingly-managed portfolio -- rejected the
        fresh-start convention there after finding it was highly sensitive
        to which arbitrary date the rebalance grid landed on relative to
        the crisis onset, an artifact of measurement, not of the strategy).
        """
        print("\n" + "=" * 60)
        print("VERIFICATION (see TECHNICAL_DOCS.md section 5)")
        print("=" * 60)

        test_start, test_end = pd.Timestamp(test_dates[0]), pd.Timestamp(test_dates[-1])
        print(f"\nComputing pure risk parity fresh for the test window ({test_start.date()} to {test_end.date()})...")
        rp_test_pnl, _ = run_portfolio(close, None, test_dates, tilt_strength=0.0,
                                        cost_bps=cost_bps, capital=self.initial_balance)
        rp_test_equity = (self.initial_balance + rp_test_pnl.cumsum()).loc[test_start:test_end]

        print("Downloading SPY/IEF for benchmark comparison...")
        bench_prices = fetch_benchmark_prices(close.index[0], close.index[-1])
        bench_curves = build_benchmark_curves(close, bench_prices, test_start, test_end, cost_bps=cost_bps,
                                               capital=self.initial_balance)

        print(f"\n--- Benchmark comparison, identical test window ({test_start.date()} to {test_end.date()}) ---")
        rp_m = compute_metrics(rp_test_equity)
        print(f"{'Risk parity (this project)':32s} CAGR={rp_m['ann_return']*100:+7.2f}%  "
              f"Sharpe={rp_m['sharpe_rf0']:5.2f}  Calmar={rp_m['calmar']:5.2f}  MaxDD={rp_m['max_dd']*100:7.2f}%")
        for label, curve in bench_curves.items():
            bm = compute_metrics(curve)
            print(f"{label:32s} CAGR={bm['ann_return']*100:+7.2f}%  "
                  f"Sharpe={bm['sharpe_rf0']:5.2f}  Calmar={bm['calmar']:5.2f}  MaxDD={bm['max_dd']*100:7.2f}%")

        print("\n--- Crisis sub-periods: when does risk parity's defensive positioning pay off? ---")
        print("(continuously-managed curves, sliced at each crisis window -- see docstring)")
        all_rebal_dates = close.index[::20]
        rp_full_pnl, _ = run_portfolio(close, None, all_rebal_dates, tilt_strength=0.0,
                                        cost_bps=cost_bps, capital=self.initial_balance)
        rp_full_equity = self.initial_balance + rp_full_pnl.cumsum()
        continuous_bench = build_continuous_benchmark_curves(close, bench_prices, cost_bps=cost_bps,
                                                              capital=self.initial_balance)

        for crisis_name, (c_start, c_end) in CRISIS_PERIODS.items():
            c_start, c_end = pd.Timestamp(c_start), pd.Timestamp(c_end)
            print(f"\n  {crisis_name} ({c_start.date()} to {c_end.date()}):")

            ret_dd = crisis_period_return_dd(rp_full_equity, c_start, c_end)
            if ret_dd:
                print(f"    {'Risk parity':28s} return {ret_dd[0]*100:+7.2f}%  maxDD {ret_dd[1]*100:7.2f}%")
            for label, curve in continuous_bench.items():
                ret_dd = crisis_period_return_dd(curve, c_start, c_end)
                if ret_dd:
                    print(f"    {label:28s} return {ret_dd[0]*100:+7.2f}%  maxDD {ret_dd[1]*100:7.2f}%")

        print("\n--- Statistical significance: block-bootstrap 95% CI (5000 resamples, block=20 days) ---")
        print("Is the test-period Sharpe/Calmar distinguishable from a no-skill (zero) result, "
              "or could it plausibly be a lucky draw from this sample?")
        sharpes, calmars = bootstrap_sharpe_calmar(rp_test_equity, block_size=20, n_resamples=5000)
        summarize_bootstrap(sharpes, "Sharpe (rf=0%)", rp_m['sharpe_rf0'])
        summarize_bootstrap(calmars, "Calmar", rp_m['calmar'])
        print("=" * 60)

    def _print_results(self, m, tilt_strength):
        print("\n" + "=" * 60)
        print("RISK-PARITY PORTFOLIO RESULTS (real data, out-of-sample)")
        print("=" * 60)
        print(f"Tilt: {tilt_strength} ({'pure risk parity' if tilt_strength in ('0.0', '0') else 'ML-tilted'})")
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
        print("=" * 60)
