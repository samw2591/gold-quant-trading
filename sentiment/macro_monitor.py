"""
Cross-asset macro monitor — backward-compatible wrapper.

Original functionality (Brent + US10Y) has been migrated to the
unified macro data pipeline at `macro/data_provider.py`.

This module is kept as a thin wrapper so existing imports in
`gold_trader.py` continue to work without changes until the
full migration to `MacroDataProvider` is complete.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_MACRO_AVAILABLE = False
try:
    from macro.data_provider import MacroDataProvider
    _MACRO_AVAILABLE = True
except ImportError:
    pass

_YF_AVAILABLE = False
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    pass


class MacroMonitor:
    """Backward-compatible wrapper around MacroDataProvider.

    Returns the same dict format as the original implementation,
    with additional fields from the new macro pipeline.
    """

    def __init__(self, cache_ttl_seconds: int = 600):
        self._provider: Optional[MacroDataProvider] = None
        if _MACRO_AVAILABLE:
            try:
                import config
                self._provider = MacroDataProvider(
                    fred_api_key=getattr(config, 'FRED_API_KEY', ''),
                    cache_ttl=cache_ttl_seconds,
                    cache_path=str(getattr(config, 'MACRO_CACHE_PATH', None) or ''),
                )
            except Exception as e:
                logger.warning(f"[宏观监控] MacroDataProvider 初始化失败: {e}")

        self._cache_ttl = cache_ttl_seconds

    def get_cross_asset_snapshot(self) -> Optional[Dict]:
        """Return macro snapshot in the original dict format.

        Includes original fields (brent_oil_price, us10y_yield_price)
        plus new fields from MacroDataProvider (dxy, vix, tips_10y, etc).
        """
        if self._provider:
            try:
                snap = self._provider.get_snapshot()
                result = snap.to_dict()

                # Map to legacy field names for backward compat
                if snap.brent is not None:
                    result["brent_oil_price"] = snap.brent
                    result["brent_oil_prev_close"] = snap.brent_prev
                    result["brent_oil_change_pct"] = snap.brent_change_pct
                if snap.us10y is not None:
                    result["us10y_yield_price"] = snap.us10y
                    result["us10y_yield_prev_close"] = snap.us10y_prev
                    result["us10y_yield_change_pct"] = snap.us10y_change_pct

                return result
            except Exception as e:
                logger.warning(f"[宏观监控] MacroDataProvider 调用失败: {e}")

        # Fallback to direct yfinance if MacroDataProvider not available
        if not _YF_AVAILABLE:
            return None

        return self._legacy_fetch()

    def _legacy_fetch(self) -> Optional[Dict]:
        """Original implementation as ultimate fallback."""
        from datetime import datetime, timedelta

        result: Dict = {"timestamp": datetime.utcnow().isoformat()}
        brent = self._fetch_ticker("BZ=F", "brent_oil")
        us10y = self._fetch_ticker("^TNX", "us10y_yield")

        if brent is None and us10y is None:
            return None

        if brent:
            result.update(brent)
        if us10y:
            result.update(us10y)
        return result

    def _fetch_ticker(self, ticker: str, prefix: str) -> Optional[Dict]:
        from datetime import datetime, timedelta
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            data = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if data.empty or len(data) < 2:
                return None

            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                data.columns = data.columns.droplevel(1)

            latest_close = float(data["Close"].iloc[-1])
            prev_close = float(data["Close"].iloc[-2])
            change_pct = round((latest_close - prev_close) / prev_close * 100, 2)

            return {
                f"{prefix}_price": round(latest_close, 2),
                f"{prefix}_prev_close": round(prev_close, 2),
                f"{prefix}_change_pct": change_pct,
            }
        except Exception as exc:
            logger.warning(f"[宏观监控] {ticker} 获取失败: {exc}")
            return None
