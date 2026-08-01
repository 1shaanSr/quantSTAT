import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

class StatisticalArbitrageBacktester:
    """
    Statistical Arbitrage Backtesting Engine (real market data)

    Identifies cointegrated pairs (Engle-Granger test) and trades mean
    reversion opportunities using a rolling-window OLS hedge ratio and
    z-score, computed using only data available up to each trading day
    (no look-ahead). Reports return, Sharpe, max drawdown, and Calmar
    computed directly from the simulated daily equity curve.
    """
    def __init__(self, api_handler):
        self.api = api_handler.api if hasattr(api_handler, 'api') else api_handler
        self.initial_balance = 10000
        self.current_balance = 10000
        self.risk_per_trade = 0.10   # fraction of capital allocated as notional per open trade
        self.max_positions = 3
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.pairs_data = {}

        # Rolling-window statistical arbitrage parameters (no look-ahead)
        self.window = 60            # trailing days used for hedge ratio + spread mean/std
        self.entry_zscore = 2.0
        self.exit_zscore = 0.5
        self.stop_loss_zscore = 3.5
        self.max_holding_period = 20  # trading days
        self.cost_bps = 5             # transaction cost per leg, in basis points of notional
        self.rf_annual = 0.045        # risk-free rate assumption for Sharpe (annualized)

        self.test_pairs = [
            ('SPY', 'QQQ'),
            ('XLF', 'XLI'),
            ('GLD', 'SLV'),
            ('USO', 'XLE'),
            ('TLT', 'IEF')
        ]

    def run(self, symbol, days):
        """
        Execute statistical arbitrage backtesting strategy on real historical data.
        `days` is the number of trailing trading days to backtest (after the
        rolling-window burn-in period, which needs an additional `self.window`
        days of history to compute the first hedge ratio).
        """
        print(f"\n=== Statistical Arbitrage Backtesting (real market data) ===")
        print(f"Primary Symbol: {symbol} | Analysis Period: {days} trading days")

        try:
            if hasattr(self.api, 'get_account'):
                account = self.api.get_account()
                self.initial_balance = float(getattr(account, 'buying_power', 10000))
            else:
                print("Using default balance of $10,000")
                self.initial_balance = 10000

            self.current_balance = self.initial_balance
            self.positions = []
            self.trades = []
            self.equity_curve = []

            candidates = [p for p in self.test_pairs if symbol in p]
            if not candidates:
                other_symbols = sorted(set(s for pair in self.test_pairs for s in pair if s != symbol))
                candidates = [(symbol, s) for s in other_symbols]

            calendar_days_needed = int((days + self.window + 40) * 1.6) + 30
            end_date = datetime.now()
            start_date = end_date - timedelta(days=calendar_days_needed)

            all_symbols = sorted(set(s for pair in candidates for s in pair))
            print(f"Downloading real price history for: {', '.join(all_symbols)}")
            raw = yf.download(all_symbols, start=start_date.strftime('%Y-%m-%d'),
                               end=end_date.strftime('%Y-%m-%d'), auto_adjust=True,
                               progress=False)
            if raw.empty:
                print("No market data returned -- check symbols/network connectivity.")
                return
            close = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
            close = close.dropna(how='any')

            if len(close) < self.window + 20:
                print(f"Insufficient real data: {len(close)} rows, need at least {self.window + 20}")
                return

            print(f"Real data range: {close.index[0].date()} to {close.index[-1].date()} ({len(close)} trading days)")

            best_pair, best_pvalue, coint_report = self._find_best_pair(close, candidates)
            if not best_pair:
                print(f"No usable pairs found for {symbol}")
                return

            symbol1, symbol2 = best_pair
            print(f"\nCointegration screen (Engle-Granger, full sample used only for pair SELECTION):")
            for line in coint_report:
                print(f"  {line}")
            print(f"\nSelected pair: {symbol1}-{symbol2} (p={best_pvalue:.4f})")
            if best_pvalue >= 0.05:
                print("WARNING: No candidate pair is cointegrated at the 5% level.")
                print("         Trading a non-cointegrated pair is not a validated market-neutral")
                print("         strategy -- results below should be read as a stress test, not a")
                print("         claim of statistical edge.")

            price1 = close[symbol1].loc[close.index[-(days + self.window):]] if len(close) > days + self.window else close[symbol1]
            price2 = close[symbol2].loc[close.index[-(days + self.window):]] if len(close) > days + self.window else close[symbol2]
            price1, price2 = price1.align(price2, join='inner')

            results = self._execute_pairs_trading(price1, price2, symbol1, symbol2)
            self._print_results(results)
            return results

        except Exception as e:
            print(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()

    def _find_best_pair(self, close, candidates):
        """
        Screen candidate pairs using the Engle-Granger cointegration test
        (statsmodels.tsa.stattools.coint) over the full downloaded sample.
        NOTE: using the full sample to SELECT which pair to trade is itself
        a form of in-sample selection -- flagged explicitly in the report.
        The hedge ratio and trading signals themselves are computed with a
        strictly rolling/trailing window (see _execute_pairs_trading), so
        the trading logic itself has no look-ahead even though pair choice does.
        """
        report = []
        results = []
        for a, b in candidates:
            if a not in close.columns or b not in close.columns:
                continue
            score, pvalue, _ = coint(close[a], close[b])
            corr = close[a].corr(close[b])
            results.append((a, b, pvalue))
            flag = "COINTEGRATED (p<0.05)" if pvalue < 0.05 else "not cointegrated"
            report.append(f"{a}-{b}: EG p-value={pvalue:.4f}  corr={corr:.3f}  [{flag}]")

        if not results:
            return None, None, report
        results.sort(key=lambda r: r[2])
        a, b, pvalue = results[0]
        return (a, b), pvalue, report

    def _execute_pairs_trading(self, price1, price2, symbol1, symbol2):
        """
        Rolling-window OLS hedge ratio and z-score: at day i, the hedge ratio
        and spread mean/std are estimated ONLY from days [i-window, i), never
        from data at or after day i. This eliminates the look-ahead bias of
        computing a single hedge ratio/spread mean+std over the entire sample
        (which the original version of this code did).
        """
        n = len(price1)
        dates = price1.index
        hedge_ratios = np.full(n, np.nan)
        zscores = np.full(n, np.nan)

        for i in range(self.window, n):
            p1w = price1.iloc[i - self.window:i].values
            p2w = price2.iloc[i - self.window:i].values
            X = np.column_stack([np.ones(self.window), p1w])
            try:
                coef, *_ = np.linalg.lstsq(X, p2w, rcond=None)
                beta = coef[1]
            except Exception:
                beta = 1.0
            hedge_ratios[i] = beta
            spread_w = p2w - beta * p1w
            mu, sigma = spread_w.mean(), spread_w.std()
            spread_today = price2.iloc[i] - beta * price1.iloc[i]
            zscores[i] = (spread_today - mu) / sigma if sigma > 0 else 0.0

        return self._simulate_pairs_trades(price1, price2, dates, hedge_ratios, zscores, symbol1, symbol2)

    def _simulate_pairs_trades(self, price1, price2, dates, hedge_ratios, zscores, symbol1, symbol2):
        n = len(price1)
        trades = []
        position = None
        entry_i = entry_beta = None
        units = 0.0
        daily_pnl = np.zeros(n)
        balance = self.initial_balance

        for i in range(self.window, n):
            z = zscores[i]
            beta = hedge_ratios[i]
            p1, p2 = price1.iloc[i], price2.iloc[i]

            if position is not None:
                p1_prev, p2_prev = price1.iloc[i - 1], price2.iloc[i - 1]
                spread_chg = (p2 - p2_prev) - entry_beta * (p1 - p1_prev)
                pnl = units * spread_chg if position == 'long_spread' else -units * spread_chg
                daily_pnl[i] += pnl
                balance += pnl

            if position is None:
                if abs(z) > self.entry_zscore:
                    position = 'short_spread' if z > self.entry_zscore else 'long_spread'
                    entry_i = i
                    entry_beta = beta
                    notional = abs(entry_beta) * p1 + p2
                    units = (balance * self.risk_per_trade) / notional if notional > 0 else 0.0
                    cost = (self.cost_bps / 10000.0) * notional * units
                    balance -= cost
                    daily_pnl[i] -= cost
            else:
                days_held = i - entry_i
                should_exit = (abs(z) < self.exit_zscore or
                               abs(z) > self.stop_loss_zscore or
                               days_held >= self.max_holding_period)
                if should_exit:
                    notional = abs(entry_beta) * p1 + p2
                    cost = (self.cost_bps / 10000.0) * notional * units
                    balance -= cost
                    daily_pnl[i] -= cost
                    exit_reason = ("stop_loss" if abs(z) > self.stop_loss_zscore
                                    else "max_holding" if days_held >= self.max_holding_period
                                    else "mean_reversion")
                    trade_pnl = (daily_pnl[entry_i + 1:i + 1].sum())
                    trades.append({
                        'entry_date': dates[entry_i], 'exit_date': dates[i],
                        'days_held': days_held, 'position_type': position,
                        'entry_z': zscores[entry_i], 'exit_z': z,
                        'pnl': trade_pnl, 'exit_reason': exit_reason
                    })
                    position = None
                    units = 0.0

        equity = self.initial_balance + np.cumsum(daily_pnl)
        equity_series = pd.Series(equity, index=dates)
        valid_equity = equity_series.iloc[self.window:]
        self.current_balance = equity[-1]
        self.equity_curve = valid_equity

        metrics = self._compute_metrics(valid_equity)

        total_trades = len(trades)
        win_rate = (sum(1 for t in trades if t['pnl'] > 0) / total_trades * 100) if total_trades else 0
        winning = [t['pnl'] for t in trades if t['pnl'] > 0]
        losing = [t['pnl'] for t in trades if t['pnl'] <= 0]

        return {
            'symbols': f"{symbol1}-{symbol2}",
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': np.mean(winning) if winning else 0,
            'avg_loss': np.mean(losing) if losing else 0,
            'trades': trades,
            'equity_curve': valid_equity,
            'final_balance': equity[-1],
            'return_pct': (equity[-1] - self.initial_balance) / self.initial_balance * 100,
            **metrics
        }

    def _compute_metrics(self, equity):
        """
        All metrics computed directly from the simulated daily equity curve
        (starting capital + cumulative realized/mark-to-market P&L).
        """
        daily_returns = equity.pct_change().dropna()
        n_days = len(equity)
        years = n_days / 252

        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        ann_vol = daily_returns.std() * np.sqrt(252)
        sharpe_rf0 = (daily_returns.mean() * 252) / ann_vol if ann_vol > 0 else float('nan')
        rf_daily = (1 + self.rf_annual) ** (1 / 252) - 1
        sharpe_rf = ((daily_returns.mean() - rf_daily) * 252) / ann_vol if ann_vol > 0 else float('nan')

        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = equity.loc[:trough_date].idxmax()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else float('nan')

        return {
            'n_days': n_days, 'years': years,
            'total_return': total_return, 'ann_return': ann_return, 'ann_vol': ann_vol,
            'sharpe_rf0': sharpe_rf0, 'sharpe_rf': sharpe_rf,
            'max_dd': max_dd, 'peak_date': peak_date, 'trough_date': trough_date,
            'calmar': calmar
        }

    def _print_results(self, results):
        print("\n" + "=" * 60)
        print("STATISTICAL ARBITRAGE BACKTEST RESULTS (real data)")
        print("=" * 60)
        if not results:
            print("No results to display")
            return

        print(f"Trading Pair: {results.get('symbols', 'N/A')}")
        print(f"Backtest window: {results['n_days']} trading days ({results['years']:.2f} years)")
        print("-" * 40)
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${results['final_balance']:,.2f}")
        print(f"Total Return: {results['return_pct']:+.2f}%")
        print("-" * 40)
        print(f"Total Trades: {results['total_trades']}")
        if results['total_trades'] > 0:
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Average Winner: ${results['avg_win']:+,.2f}")
            print(f"Average Loser: ${results['avg_loss']:+,.2f}")
        print("-" * 40)
        print("RISK-ADJUSTED METRICS (computed from the daily equity curve):")
        print(f"  Annualized Return (CAGR): {results['ann_return']*100:+.2f}%")
        print(f"  Annualized Volatility:    {results['ann_vol']*100:.2f}%")
        print(f"  Sharpe Ratio (rf=0%):        {results['sharpe_rf0']:.2f}")
        print(f"  Sharpe Ratio (rf={self.rf_annual*100:.1f}%):     {results['sharpe_rf']:.2f}")
        print(f"  Max Drawdown: {results['max_dd']*100:.2f}%  (peak {results['peak_date'].date()} -> trough {results['trough_date'].date()})")
        print(f"  Calmar Ratio: {results['calmar']:.2f}")
        print("=" * 60)
