"""
Economically-grouped ETF/equity universe for pair discovery.

Pairs are only tested WITHIN a bucket (sector-vs-sector, same-industry
stock-vs-stock, etc.) so every candidate pair has an a priori economic
rationale. This avoids blind cross-product data mining across unrelated
assets, which inflates false-discovery risk in a cointegration screen.
"""

BUCKETS = {
    'us_sectors': ['XLF', 'XLK', 'XLE', 'XLI', 'XLY', 'XLP', 'XLV', 'XLU', 'XLB', 'XLC', 'XLRE'],
    'broad_equity_index': ['SPY', 'QQQ', 'DIA', 'IWM', 'MDY'],
    # NOTE: country-vs-country equity ETF pairs (e.g. EWZ-EIDO) were deliberately
    # excluded from this universe. Two national equity indices have no structural
    # cointegrating force -- no arbitrage or business linkage enforces a stable
    # long-run equilibrium between them. Observed correlation/cointegration
    # between country ETFs is typically just shared global-growth/EM beta, the
    # classic "spurious regression" pattern Granger warned about. An earlier
    # version of this research pipeline included a developed/emerging country
    # bucket and it picked up EWZ-EIDO (Brazil vs Indonesia): formation-period
    # p=0.03, half-life 42 days, in-formation backtest Sharpe 0.60 -- then it
    # lost -8.8% max drawdown once traded out-of-sample, as the shared-beta
    # relationship (not a true equilibrium) drifted. Removed on structural
    # grounds, though the decision was informed by seeing that result -- noted
    # here for transparency about hindsight-bias risk.
    'precious_metals': ['GLD', 'SLV', 'PPLT', 'PALL'],
    'energy': ['USO', 'UNG', 'XLE', 'XOP'],
    'broad_commodities': ['DBC', 'PDBC', 'GSG'],
    'rates_treasuries': ['TLT', 'IEF', 'SHY', 'IEI'],
    'credit': ['LQD', 'HYG', 'JNK', 'BND'],
    'financials_sub': ['XLF', 'KBE', 'KRE'],
    'tech_sub': ['QQQ', 'XLK', 'SOXX', 'SMH'],
    'homebuilders_industrials': ['XLI', 'XHB', 'ITB'],
    'factor_style': ['IVE', 'IVW', 'VTV', 'VUG', 'MTUM', 'QUAL', 'USMV', 'SPLV'],
    'reits': ['XLRE', 'VNQ', 'SCHH'],
}

# Hand-picked same-industry stock pairs: classic statistical-arbitrage
# examples with a well-documented business/economic rationale for
# co-movement (shared demand drivers, input costs, or regulatory regime).
EXPLICIT_PAIRS = [
    ('KO', 'PEP', 'beverages'),
    ('XOM', 'CVX', 'oil_majors'),
    ('JPM', 'BAC', 'money_center_banks'),
    ('MA', 'V', 'payment_networks'),
    ('HD', 'LOW', 'home_improvement'),
    ('MCD', 'YUM', 'quick_service_restaurants'),
    ('UPS', 'FDX', 'parcel_delivery'),
    ('T', 'VZ', 'telecom'),
    ('PG', 'CL', 'household_products'),
    ('WMT', 'TGT', 'big_box_retail'),
    ('GS', 'MS', 'investment_banks'),
    ('C', 'WFC', 'money_center_banks'),
    ('COP', 'EOG', 'e_and_p_oil'),
    ('SO', 'DUK', 'regulated_utilities'),
    ('CAT', 'DE', 'heavy_machinery'),
    ('LMT', 'RTX', 'defense_primes'),
    ('SBUX', 'CMG', 'restaurant_growth'),
    ('ADBE', 'CRM', 'enterprise_software'),
]

# A wider candidate set was tested (18 additional same-industry pairs:
# railroads, industrial gases, waste management, tobacco, airlines,
# insurers, regional banks, etc.). Only 3 passed the formation-window
# cointegration screen (UNP-CSX, UAL-AAL, DPZ-PZZA), and adding them
# lowered out-of-sample portfolio Sharpe (0.71 -> 0.51 equal-weighted) --
# two of the three never triggered a single trade in-sample, diluting the
# book, and the third lost money out-of-sample. Not merged into the
# default universe above; see TECHNICAL_DOCS.md section 3 for the full
# writeup and `EXTENDED_PAIRS` below to reproduce it.
EXTENDED_PAIRS = [
    ('UNP', 'CSX', 'railroads'),
    ('LIN', 'APD', 'industrial_gases'),
    ('WM', 'RSG', 'waste_management'),
    ('MO', 'PM', 'tobacco'),
    ('AMAT', 'LRCX', 'semi_equipment'),
    ('QCOM', 'AVGO', 'semiconductors'),
    ('DAL', 'UAL', 'airlines'),
    ('DAL', 'LUV', 'airlines'),
    ('UAL', 'AAL', 'airlines'),
    ('MET', 'PRU', 'life_insurance'),
    ('ROST', 'TJX', 'off_price_retail'),
    ('KEY', 'CFG', 'regional_banks'),
    ('USB', 'PNC', 'regional_banks'),
    ('PLD', 'EGP', 'industrial_reits'),
    ('CL', 'KMB', 'household_products'),
    ('DPZ', 'PZZA', 'pizza_chains'),
    ('UNH', 'ELV', 'health_insurers'),
    ('NEE', 'AEP', 'regulated_utilities'),
]

def all_tickers(include_extended=False):
    explicit = EXPLICIT_PAIRS + (EXTENDED_PAIRS if include_extended else [])
    bucket_tickers = set(t for tickers in BUCKETS.values() for t in tickers)
    explicit_tickers = set(t for a, b, _ in explicit for t in (a, b))
    return sorted(bucket_tickers | explicit_tickers)

def candidate_pairs(include_extended=False):
    """
    `include_extended=True` adds the 18 tested-and-not-adopted pairs from
    EXTENDED_PAIRS (see the comment above it) -- off by default so the
    default universe matches the documented, locked results.
    """
    pairs = []
    for bucket, tickers in BUCKETS.items():
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                pairs.append((tickers[i], tickers[j], bucket))
    explicit = EXPLICIT_PAIRS + (EXTENDED_PAIRS if include_extended else [])
    for a, b, tag in explicit:
        pairs.append((a, b, tag))
    return pairs
