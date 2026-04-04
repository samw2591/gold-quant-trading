"""
Macro Data Package
===================
Unified macro factor data pipeline for gold quantitative trading.

Data sources:
  - yfinance: DXY, VIX, Brent crude (real-time + historical)
  - FRED API: TIPS 10Y, US 2Y/10Y yields, 2-10 spread, 5Y BEI (historical)

Usage (live):
    from macro import MacroDataProvider
    provider = MacroDataProvider()
    snapshot = provider.get_snapshot()

Usage (backtest):
    from macro import MacroDataProvider
    provider = MacroDataProvider(fred_api_key="YOUR_KEY")
    provider.download_history("2015-01-01", "2026-04-01", "data/macro_history.csv")
"""

from macro.data_provider import MacroDataProvider, MacroSnapshot
from macro.regime_detector import (
    MacroRegime,
    MacroRegimeDetector,
    add_regime_column,
)

__all__ = [
    "MacroDataProvider",
    "MacroSnapshot",
    "MacroRegime",
    "MacroRegimeDetector",
    "add_regime_column",
]
