"""
Economic calendar guard module.

Determines whether trading should be paused based on upcoming or
in-progress high-impact economic events.  Also assigns a risk level
that downstream modules use to scale position sizes.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sentiment.news_collector import NewsCollector, _ECONOMIC_CALENDAR_2026

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pause windows by impact level (seconds before/after event)
# ---------------------------------------------------------------------------
PAUSE_WINDOWS = {
    "EXTREME": {"before": 7200, "after": 7200},   # FOMC: ±2 hours
    "HIGH":    {"before": 3600, "after": 3600},    # NFP, CPI, PCE: ±1 hour
    "MEDIUM":  {"before": 1800, "after": 1800},    # GDP, PPI: ±30 minutes
}

# ---------------------------------------------------------------------------
# News-based high-risk keyword patterns (for sudden events)
# ---------------------------------------------------------------------------
BREAKING_RISK_KEYWORDS = [
    "breaking",
    "emergency",
    "surprise rate",
    "flash crash",
    "black swan",
    "war declared",
    "nuclear",
    "default",
    "bank collapse",
    "trump executive order",
]


class CalendarGuard:
    """Guards trading around high-impact economic events."""

    _NEWS_CACHE_TTL = 300  # 5 minutes

    def __init__(self, news_collector: Optional[NewsCollector] = None):
        self._collector = news_collector or NewsCollector()
        self._news_cache: List[Dict] = []
        self._news_cache_ts: float = 0

    def should_pause_trading(self) -> Tuple[bool, str]:
        """Check if trading should be paused right now.

        Returns:
            (should_pause: bool, reason: str)
        """
        now = datetime.now(timezone.utc)

        # 1. Check calendar events
        for evt in _ECONOMIC_CALENDAR_2026:
            evt_dt = evt["datetime_utc"]
            if not isinstance(evt_dt, datetime):
                continue

            impact = evt["impact"]
            window = PAUSE_WINDOWS.get(impact)
            if window is None:
                continue

            before_start = evt_dt - timedelta(seconds=window["before"])
            after_end = evt_dt + timedelta(seconds=window["after"])

            if before_start <= now <= after_end:
                reason = (
                    f"经济事件 [{evt['name']}] 即将/正在发布 "
                    f"(UTC {evt_dt.strftime('%Y-%m-%d %H:%M')}), "
                    f"影响等级: {impact}"
                )
                logger.warning(f"[日历避险] 暂停交易 — {reason}")
                return True, reason

        # 2. Check breaking news for sudden risks (cached for 5 minutes)
        try:
            if time.monotonic() - self._news_cache_ts > self._NEWS_CACHE_TTL:
                self._news_cache = self._collector.collect_gold_news()
                self._news_cache_ts = time.monotonic()
            news = self._news_cache
            for article in news[:20]:  # Only scan most recent
                title_lower = article.get("title", "").lower()
                for kw in BREAKING_RISK_KEYWORDS:
                    if kw in title_lower:
                        reason = f"突发新闻检测到高风险关键词 [{kw}]: {article['title'][:80]}"
                        logger.warning(f"[日历避险] 标记高风险 — {reason}")
                        return True, reason
        except Exception as exc:
            logger.debug(f"[日历避险] 新闻检测异常 (忽略): {exc}")

        return False, ""

    def get_risk_level(self) -> str:
        """Assess current risk level from the economic calendar.

        Returns one of: "LOW", "MEDIUM", "HIGH", "EXTREME"
        """
        now = datetime.now(timezone.utc)
        max_risk = "LOW"
        risk_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EXTREME": 3}

        for evt in _ECONOMIC_CALENDAR_2026:
            evt_dt = evt["datetime_utc"]
            if not isinstance(evt_dt, datetime):
                continue

            hours_until = (evt_dt - now).total_seconds() / 3600

            # Within 6 hours of any event, start raising risk level
            if -2 <= hours_until <= 6:
                impact = evt["impact"]
                # Map impact to risk — within the pause window is the
                # impact level itself; slightly outside is one level lower
                window = PAUSE_WINDOWS.get(impact, {"before": 1800, "after": 1800})
                in_window = (
                    evt_dt - timedelta(seconds=window["before"])
                    <= now
                    <= evt_dt + timedelta(seconds=window["after"])
                )

                if in_window:
                    risk = impact  # EXTREME, HIGH, MEDIUM
                elif hours_until <= 3:
                    # Approaching — one level below actual impact
                    downgrade = {"EXTREME": "HIGH", "HIGH": "MEDIUM", "MEDIUM": "LOW"}
                    risk = downgrade.get(impact, "LOW")
                else:
                    risk = "LOW"

                if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
                    max_risk = risk

        logger.debug(f"[日历避险] 当前风险等级: {max_risk}")
        return max_risk

    def get_next_event(self) -> Optional[Dict]:
        """Return the next upcoming economic event (or None)."""
        now = datetime.now(timezone.utc)
        closest: Optional[Dict] = None
        closest_delta = float("inf")

        for evt in _ECONOMIC_CALENDAR_2026:
            evt_dt = evt["datetime_utc"]
            if not isinstance(evt_dt, datetime):
                continue
            delta = (evt_dt - now).total_seconds()
            if 0 < delta < closest_delta:
                closest_delta = delta
                closest = evt

        return closest
