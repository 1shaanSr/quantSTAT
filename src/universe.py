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

def all_tickers():
    bucket_tickers = set(t for tickers in BUCKETS.values() for t in tickers)
    explicit_tickers = set(t for a, b, _ in EXPLICIT_PAIRS for t in (a, b))
    return sorted(bucket_tickers | explicit_tickers)

def candidate_pairs():
    pairs = []
    for bucket, tickers in BUCKETS.items():
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                pairs.append((tickers[i], tickers[j], bucket))
    for a, b, tag in EXPLICIT_PAIRS:
        pairs.append((a, b, tag))
    return pairs
