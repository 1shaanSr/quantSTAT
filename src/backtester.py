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
from src.features import build_feature_panel
from src.ml_predictor import walk_forward_predict, summarize_ic
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

    def run(self, tilt_strength=None, cost_bps=10):
        tilt_strength = TILT_STRENGTH if tilt_strength is None else tilt_strength

        print("\n=== Risk-Parity + ML Factor Tilt Backtest (real data) ===")
        print("Downloading universe...")
        close, volume = self._download()
        print(f"Data: {close.index[0].date()} to {close.index[-1].date()} ({len(close)} days, {close.shape[1]} tickers)")

        print("Building causal factor panel and walk-forward ML predictions...")
        panel = build_feature_panel(close, volume, forward_days=FORWARD_DAYS)
        all_dates = sorted(panel['date'].dropna().unique())
        rebalance_dates = all_dates[::FORWARD_DAYS]

        pred_df, ic_df = walk_forward_predict(panel, rebalance_dates)

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

        print(f"\nRunning risk-parity portfolio (tilt_strength={tilt_strength}) on the out-of-sample test window...")
        pnl, weight_history = run_portfolio(close, pred_df, test_dates, tilt_strength=tilt_strength,
                                             cost_bps=cost_bps, capital=self.initial_balance)
        equity = self.initial_balance + pnl.cumsum()
        equity = equity.loc[pd.Timestamp(test_dates[0]):pd.Timestamp(test_dates[-1])]
        m = compute_metrics(equity)

        self.results = {'equity': equity, 'metrics': m, 'ic_formation': ic_formation,
                         'ic_test': ic_test, 'tilt_strength': tilt_strength}
        self._print_results(m, tilt_strength)
        return self.results

    def _print_results(self, m, tilt_strength):
        print("\n" + "=" * 60)
        print("RISK-PARITY PORTFOLIO RESULTS (real data, out-of-sample)")
        print("=" * 60)
        print(f"Tilt strength: {tilt_strength} ({'pure risk parity' if tilt_strength == 0 else 'ML-tilted'})")
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
