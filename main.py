"""
Risk-Parity Portfolio with an ML Factor-Tilt Overlay
------------------------------------------------------
A walk-forward gradient-boosting return predictor (evaluated by honest
information coefficient, not an inflated trading Sharpe) feeding a
confidence-scaled tilt on top of a risk-parity base allocation across a
liquid large-cap universe. See README.md for methodology and
TECHNICAL_DOCS.md for the full derivation, including why the ML tilt is
NOT used in the locked configuration.
"""

from src.backtester import RiskParityMLBacktester


def main() -> None:
    print("Risk-Parity Portfolio + ML Factor Tilt")
    print("=" * 55)

    tilt_raw = input("Tilt strength (blank = locked default 0.0 = pure risk parity): ").strip()
    tilt_strength = float(tilt_raw) if tilt_raw else None
    cost_raw = input("Transaction cost in bps (blank = 10): ").strip()
    cost_bps = float(cost_raw) if cost_raw else 10

    bt = RiskParityMLBacktester()
    bt.run(tilt_strength=tilt_strength, cost_bps=cost_bps)


if __name__ == "__main__":
    main()
