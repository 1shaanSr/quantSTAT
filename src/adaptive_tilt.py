"""
Adaptive tilt: instead of one static tilt_strength locked from a single
formation-period average (which can wash out real regime variation), scale
the tilt at each rebalance date by the model's own TRAILING, causally-known
realized information coefficient. Trade the signal harder when it's
recently been working, back off to pure risk parity when it hasn't -- a
standard performance-based signal-weighting technique, not look-ahead (it
only ever uses IC from periods whose labels are already realized as of the
current date).

Tested (TECHNICAL_DOCS.md section 4.5): every hyperparameter combination
still underperformed static pure risk parity on formation data, so this
is not the production default. Provided as opt-in, documented,
reproducible code.
"""
import numpy as np
import pandas as pd


def build_adaptive_tilt_schedule(ic_df: pd.DataFrame, rebalance_dates, ic_lookback=10,
                                  base_tilt=1.0, reference_ic=0.05, max_multiple=2.0):
    """
    tilt_strength_t = base_tilt * clip(trailing_ic_t / reference_ic, 0, max_multiple),
    where trailing_ic_t is the mean IC over the `ic_lookback` most recent
    STRICTLY PRIOR periods. `reference_ic=0.05` follows the convention (see
    src/ml_predictor.py) that an IC around 0.05 is considered a genuinely
    useful signal at the individual-stock level in the professional
    quantitative equity literature.
    """
    ic_sorted = ic_df.sort_values('date').reset_index(drop=True)
    ic_by_date = dict(zip(ic_sorted['date'], ic_sorted['ic']))

    schedule = {}
    for date in rebalance_dates:
        prior_ics = [ic_by_date[d] for d in ic_sorted['date'] if d < date and d in ic_by_date]
        if len(prior_ics) < ic_lookback:
            schedule[date] = 0.0
            continue
        trailing_ic = np.mean(prior_ics[-ic_lookback:])
        multiple = np.clip(trailing_ic / reference_ic, 0, max_multiple)
        schedule[date] = base_tilt * multiple
    return schedule
