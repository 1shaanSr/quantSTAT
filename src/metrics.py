import numpy as np
import pandas as pd

def compute_metrics(equity: pd.Series, rf_annual=0.045, periods_per_year=252):
    """
    All metrics computed directly from a daily equity curve (starting
    capital + cumulative realized/mark-to-market P&L). Two Sharpe ratios
    are reported because the correct risk-free assumption for a
    market-neutral book is genuinely ambiguous and worth showing both ways:

      rf=0%   -- appropriate if you treat this as the incremental alpha of
                 the trading signal ON TOP OF a separate cash sleeve that
                 already earns the risk-free rate (the usual convention for
                 evaluating a market-neutral overlay strategy).
      rf=4.5% -- appropriate if you treat total capital as fully committed
                 and compare against simply holding T-bills instead.
    """
    equity = equity.dropna()
    daily_returns = equity.pct_change().dropna()
    n_days = len(equity)
    years = n_days / periods_per_year

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    ann_vol = daily_returns.std() * np.sqrt(periods_per_year)
    sharpe_rf0 = (daily_returns.mean() * periods_per_year) / ann_vol if ann_vol > 0 else np.nan
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1
    sharpe_rf = ((daily_returns.mean() - rf_daily) * periods_per_year) / ann_vol if ann_vol > 0 else np.nan

    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = (daily_returns.mean() * periods_per_year) / downside_vol if downside_vol and downside_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()

    calmar = ann_return / abs(max_dd) if max_dd not in (0, np.nan) and not pd.isna(max_dd) else np.nan

    return dict(
        n_days=n_days, years=years, total_return=total_return, ann_return=ann_return,
        ann_vol=ann_vol, sharpe_rf0=sharpe_rf0, sharpe_rf=sharpe_rf, rf_annual=rf_annual,
        sortino=sortino, max_dd=max_dd, peak_date=peak_date, trough_date=trough_date, calmar=calmar
    )

def print_metrics(m, label=""):
    print(f"\n--- {label} ---")
    print(f"Period: {m['n_days']} trading days ({m['years']:.2f} yrs)")
    print(f"Total return: {m['total_return']*100:+.2f}%")
    print(f"Annualized return (CAGR): {m['ann_return']*100:+.2f}%")
    print(f"Annualized volatility: {m['ann_vol']*100:.2f}%")
    print(f"Sharpe (rf=0%): {m['sharpe_rf0']:.2f}")
    print(f"Sharpe (rf={m['rf_annual']*100:.1f}%): {m['sharpe_rf']:.2f}")
    print(f"Sortino (rf=0%): {m['sortino']:.2f}")
    print(f"Max drawdown: {m['max_dd']*100:.2f}%  (peak {m['peak_date'].date()} -> trough {m['trough_date'].date()})")
    print(f"Calmar: {m['calmar']:.2f}")
