"""
Cross-asset macro monitor for gold trading context.

Tracks Brent Crude Oil and US 10-Year Treasury Yield to detect
macro regime shifts (e.g. stagflation) that invert gold's normal
safe-haven behavior.

This module is OBSERVATION ONLY — it writes data to gold_daily_state.json
but does NOT influence any trading decisions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_YF_AVAILABLE = False
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    logger.warning("[宏观监控] yfinance 未安装，跨资产监控不可用")

BRENT_TICKER = "BZ=F"
US10Y_TICKER = "^TNX"


class MacroMonitor:
    """Fetches daily cross-asset data for observation / future rule building."""

    def __init__(self, cache_ttl_seconds: int = 600):
        self._cache: Dict = {}
        self._cache_ts: float = 0.0
        self._cache_ttl = cache_ttl_seconds

    def get_cross_asset_snapshot(self) -> Optional[Dict]:
        """Return latest Brent + US10Y snapshot, cached for `cache_ttl` seconds."""
        if not _YF_AVAILABLE:
            return None

        now = datetime.now().timestamp()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        try:
            snapshot = self._fetch()
            if snapshot:
                self._cache = snapshot
                self._cache_ts = now
            return snapshot
        except Exception as exc:
            logger.warning(f"[宏观监控] 数据获取失败: {exc}")
            return self._cache or None

    def _fetch(self) -> Optional[Dict]:
        result: Dict = {"timestamp": datetime.utcnow().isoformat()}

        brent = self._fetch_ticker(BRENT_TICKER, "brent_oil")
        us10y = self._fetch_ticker(US10Y_TICKER, "us10y_yield")

        if brent is None and us10y is None:
            return None

        if brent:
            result.update(brent)
        if us10y:
            result.update(us10y)

        logger.info(
            f"[宏观监控] 油价: ${result.get('brent_oil_price', 'N/A')} "
            f"({result.get('brent_oil_change_pct', 'N/A')}%) | "
            f"US10Y: {result.get('us10y_yield_price', 'N/A')}% "
            f"({result.get('us10y_yield_change_pct', 'N/A')}%)"
        )
        return result

    def _fetch_ticker(self, ticker: str, prefix: str) -> Optional[Dict]:
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            data = yf.download(
                ticker, start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if data.empty or len(data) < 2:
                logger.debug(f"[宏观监控] {ticker}: 数据不足")
                return None

            # Handle MultiIndex columns from yfinance
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
