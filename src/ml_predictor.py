"""
Walk-forward (expanding-window) gradient boosting return predictor.

Two leakage traps avoided deliberately:
1. Overlapping labels: a forward-20-day return label at date t overlaps
   with the label at date t+1 (they share 19 of 20 days). Training on
   adjacent daily rows would let the model see near-duplicate label
   information across "different" train/test splits. Fixed by sampling
   only every `forward_days` trading days -- labels never overlap.
2. Training on future data: at each rebalance date, the model is retrained
   using ONLY samples whose label is already fully realized as of that
   date (i.e., samples from `forward_days` before the rebalance date or
   earlier) -- never on partially-future information.

Deliberately conservative model (shallow trees, few boosting rounds, heavy
regularization): return prediction has a very low signal-to-noise ratio,
and this project's history has repeatedly found that aggressively-tuned
configurations turn out to be fragile, formation-fold-specific peaks that
don't generalize. A heavily-tuned complex model here would very likely fit
formation noise, not signal, so hyperparameters are fixed by convention
rather than searched.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from src.features import FEATURE_COLS


def walk_forward_predict(panel: pd.DataFrame, rebalance_dates, min_train_periods=15, feature_cols=None):
    """
    At each date in `rebalance_dates` (must be spaced `forward_days` apart,
    matching how `panel`'s labels were built), train on all prior
    non-overlapping periods whose label is fully known, then predict the
    cross-section for that date. Returns (predictions_df, ic_df).
    `feature_cols` defaults to FEATURE_COLS (price/volume only); pass
    FEATURE_COLS_WITH_DIVIDENDS to include dividend factors.
    """
    feature_cols = feature_cols or FEATURE_COLS
    panel = panel.dropna(subset=feature_cols).copy()
    predictions = []
    ic_records = []

    for idx, rebal_date in enumerate(rebalance_dates):
        if idx < min_train_periods:
            continue
        train_dates = rebalance_dates[:idx]
        train_df = panel[panel['date'].isin(train_dates)].dropna(subset=['label'])
        if len(train_df) < 200:
            continue

        predict_df = panel[panel['date'] == rebal_date]
        if len(predict_df) < 10:
            continue

        model = HistGradientBoostingRegressor(
            max_depth=3, max_iter=50, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=1.0, random_state=42
        )
        model.fit(train_df[feature_cols], train_df['label'])
        preds = model.predict(predict_df[feature_cols])

        out = predict_df[['date', 'ticker', 'label']].copy()
        out['pred'] = preds
        predictions.append(out)

        realized = out.dropna(subset=['label'])
        if len(realized) >= 10:
            ic, pval = spearmanr(realized['pred'], realized['label'])
            ic_records.append({'date': rebal_date, 'ic': ic, 'pvalue': pval, 'n': len(realized)})

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    ic_df = pd.DataFrame(ic_records)
    return pred_df, ic_df


def summarize_ic(ic_df: pd.DataFrame, label=""):
    ic_clean = ic_df.dropna(subset=['ic'])
    mean_ic = ic_clean['ic'].mean()
    std_ic = ic_clean['ic'].std()
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic_clean))) if std_ic > 0 else np.nan
    hit_rate = (ic_clean['ic'] > 0).mean()
    stats = dict(mean_ic=mean_ic, std_ic=std_ic, t_stat=t_stat, hit_rate=hit_rate, n_periods=len(ic_clean))
    print(f"\n--- IC summary: {label} ---")
    print(f"Periods: {stats['n_periods']}  Mean IC: {mean_ic:.4f}  Std IC: {std_ic:.4f}  "
          f"t-stat: {t_stat:.2f}  Positive-IC rate: {hit_rate*100:.1f}%")
    return stats
