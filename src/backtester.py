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

    def run(self, tilt_strength=None, cost_bps=10, include_dividends=False, adaptive_tilt=False):
        """
        `include_dividends` and `adaptive_tilt` reproduce two more things
        tried in the ML-signal investigation (TECHNICAL_DOCS.md section 4)
        -- neither improved on the locked configuration, so both default
        to False. `tilt_strength` is ignored if `adaptive_tilt=True`.
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
        return self.results

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
