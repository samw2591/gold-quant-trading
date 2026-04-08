"""
News collector module for gold sentiment analysis.

Data sources:
  1. GDELT DOC API (primary, free, no API key, 15-min updates)
  2. Google News RSS feeds (backup)
  3. Economic calendar (manual + detection)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import feedparser
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GDELT search keywords grouped by category
# ---------------------------------------------------------------------------
# GDELT is unreachable from Singapore — disabled to avoid timeout delays.
# Set to True to re-enable if network changes.
GDELT_ENABLED = False

GOLD_QUERY = 'gold price OR gold market OR XAUUSD OR "precious metal"'
MACRO_QUERY = 'Federal Reserve OR inflation OR "US dollar" OR interest rate'
TRUMP_QUERY = 'Trump tariff OR Trump gold OR Trump trade war OR Trump economy'

GDELT_DELAY = 2

# ---------------------------------------------------------------------------
# RSS feed URLs (Google News)
# ---------------------------------------------------------------------------
RSS_FEEDS = [
    # Gold / precious metals
    "https://news.google.com/rss/search?q=gold+price+market+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=XAUUSD+gold+trading+when:6h&hl=en-US&gl=US&ceid=US:en",
    # Macro / Fed / inflation
    "https://news.google.com/rss/search?q=Federal+Reserve+interest+rate+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=inflation+data+CPI+when:6h&hl=en-US&gl=US&ceid=US:en",
    # Trump / geopolitical
    "https://news.google.com/rss/search?q=Trump+tariff+trade+war+when:6h&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Trump+economy+policy+when:6h&hl=en-US&gl=US&ceid=US:en",
]

# ---------------------------------------------------------------------------
# Request settings
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 10  # seconds
GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS = 50


class NewsCollector:
    """Collects gold-related news from multiple free sources."""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "GoldQuantTrader/1.0",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect_gold_news(self) -> List[Dict]:
        """Collect recent gold-related news from all sources.

        Returns a list of dicts: {"title", "url", "source", "published", "category"}
        """
        articles: List[Dict] = []

        # 1. GDELT (if enabled)
        if GDELT_ENABLED:
            items = self._fetch_gdelt(GOLD_QUERY)
            articles.extend(items)
            time.sleep(GDELT_DELAY)
            items = self._fetch_gdelt(MACRO_QUERY)
            articles.extend(items)

        # 2. RSS feeds (backup / supplement)
        for feed_url in RSS_FEEDS:
            items = self._fetch_rss(feed_url)
            articles.extend(items)

        # Deduplicate by title (case-insensitive)
        unique = self._deduplicate(articles)
        logger.info(f"[舆情采集] 共收集到 {len(unique)} 条去重新闻")
        return unique

    def collect_trump_posts(self) -> List[Dict]:
        """Collect Trump-related news that may impact gold markets."""
        articles: List[Dict] = []

        # GDELT (if enabled)
        if GDELT_ENABLED:
            items = self._fetch_gdelt(TRUMP_QUERY)
            for item in items:
                item["category"] = "trump"
            articles.extend(items)

        # Trump RSS (always available)
        trump_feeds = [
            f for f in RSS_FEEDS if 'Trump' in f or 'trump' in f
        ]
        for feed_url in trump_feeds:
            items = self._fetch_rss(feed_url)
            for item in items:
                item["category"] = "trump"
            articles.extend(items)

        unique = self._deduplicate(articles)
        logger.info(f"[舆情采集] 收集到 {len(unique)} 条特朗普相关新闻")
        return unique

    @staticmethod
    def _deduplicate(articles: List[Dict]) -> List[Dict]:
        seen: set = set()
        unique: List[Dict] = []
        for a in articles:
            key = a["title"].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    def get_upcoming_events(self) -> List[Dict]:
        """Return major economic events in the next 24 hours.

        Each event dict: {"name", "datetime_utc", "impact", "currency"}
        """
        now = datetime.now(timezone.utc)
        upcoming: List[Dict] = []

        for evt in _ECONOMIC_CALENDAR_2026:
            evt_dt = evt["datetime_utc"]
            if not isinstance(evt_dt, datetime):
                continue
            delta = (evt_dt - now).total_seconds()
            # Next 24 hours
            if 0 <= delta <= 86400:
                upcoming.append(evt)

        if upcoming:
            names = ", ".join(e["name"] for e in upcoming)
            logger.info(f"[经济日历] 未来24小时重大事件: {names}")
        else:
            logger.debug("[经济日历] 未来24小时无重大经济事件")

        return upcoming

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_gdelt(self, query: str) -> List[Dict]:
        """Fetch articles from GDELT DOC API for a given query."""
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": MAX_RECORDS,
            "format": "json",
        }
        try:
            resp = self._session.get(
                GDELT_BASE, params=params, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.warning(f"[舆情采集] GDELT请求失败 (query={query!r}): {exc}")
            return []
        except ValueError:
            logger.warning(f"[舆情采集] GDELT返回非JSON数据 (query={query!r})")
            return []

        raw_articles = data.get("articles", [])
        results: List[Dict] = []
        for art in raw_articles:
            results.append({
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": art.get("domain", "gdelt"),
                "published": art.get("seendate", ""),
                "category": "gold",
            })
        return results

    def _fetch_rss(self, feed_url: str) -> List[Dict]:
        """Fetch articles from an RSS feed (with timeout protection)."""
        try:
            resp = self._session.get(feed_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except requests.RequestException as exc:
            logger.warning(f"[舆情采集] RSS下载失败 ({feed_url}): {exc}")
            return []
        except Exception as exc:
            logger.warning(f"[舆情采集] RSS解析失败 ({feed_url}): {exc}")
            return []

        results: List[Dict] = []
        for entry in feed.get("entries", []):
            published = entry.get("published", "")
            results.append({
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "source": "google_news_rss",
                "published": published,
                "category": "gold",
            })
        return results


# ---------------------------------------------------------------------------
# 2026 Economic Calendar (major events only, UTC times)
# Covers key releases that significantly impact gold prices.
# ---------------------------------------------------------------------------
def _dt(month: int, day: int, hour: int = 12, minute: int = 30) -> datetime:
    """Helper to create a UTC datetime for 2026."""
    return datetime(2026, month, day, hour, minute, tzinfo=timezone.utc)


_ECONOMIC_CALENDAR_2026: List[Dict] = [
    # --- January ---
    {"name": "FOMC Minutes", "datetime_utc": _dt(1, 8, 19, 0), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Dec)", "datetime_utc": _dt(1, 14, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Dec)", "datetime_utc": _dt(1, 15, 13, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "NFP (Jan)", "datetime_utc": _dt(1, 10, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(1, 29, 19, 0), "impact": "EXTREME", "currency": "USD"},

    # --- February ---
    {"name": "NFP (Feb)", "datetime_utc": _dt(2, 7, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Jan)", "datetime_utc": _dt(2, 12, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Jan)", "datetime_utc": _dt(2, 13, 13, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "GDP (Q4 2nd)", "datetime_utc": _dt(2, 27, 13, 30), "impact": "MEDIUM", "currency": "USD"},

    # --- March ---
    {"name": "NFP (Mar)", "datetime_utc": _dt(3, 6, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Feb)", "datetime_utc": _dt(3, 11, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Feb)", "datetime_utc": _dt(3, 12, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(3, 18, 18, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "PCE Core (Feb)", "datetime_utc": _dt(3, 28, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "GDP (Q4 3rd)", "datetime_utc": _dt(3, 26, 12, 30), "impact": "MEDIUM", "currency": "USD"},

    # --- April ---
    {"name": "NFP (Apr)", "datetime_utc": _dt(4, 3, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Mar)", "datetime_utc": _dt(4, 10, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Mar)", "datetime_utc": _dt(4, 11, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "PCE Core (Mar)", "datetime_utc": _dt(4, 30, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "GDP (Q1 1st)", "datetime_utc": _dt(4, 29, 12, 30), "impact": "HIGH", "currency": "USD"},

    # --- May ---
    {"name": "NFP (May)", "datetime_utc": _dt(5, 1, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(5, 6, 18, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "CPI (Apr)", "datetime_utc": _dt(5, 13, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Apr)", "datetime_utc": _dt(5, 14, 12, 30), "impact": "MEDIUM", "currency": "USD"},

    # --- June ---
    {"name": "NFP (Jun)", "datetime_utc": _dt(6, 5, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (May)", "datetime_utc": _dt(6, 10, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (May)", "datetime_utc": _dt(6, 11, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(6, 17, 18, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "GDP (Q1 3rd)", "datetime_utc": _dt(6, 25, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "PCE Core (May)", "datetime_utc": _dt(6, 26, 12, 30), "impact": "HIGH", "currency": "USD"},

    # --- July ---
    {"name": "NFP (Jul)", "datetime_utc": _dt(7, 2, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Jun)", "datetime_utc": _dt(7, 14, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Jun)", "datetime_utc": _dt(7, 15, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(7, 29, 18, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "GDP (Q2 1st)", "datetime_utc": _dt(7, 30, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PCE Core (Jun)", "datetime_utc": _dt(7, 31, 12, 30), "impact": "HIGH", "currency": "USD"},

    # --- August ---
    {"name": "NFP (Aug)", "datetime_utc": _dt(8, 7, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Jul)", "datetime_utc": _dt(8, 12, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Jul)", "datetime_utc": _dt(8, 13, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "GDP (Q2 2nd)", "datetime_utc": _dt(8, 27, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "PCE Core (Jul)", "datetime_utc": _dt(8, 28, 12, 30), "impact": "HIGH", "currency": "USD"},

    # --- September ---
    {"name": "NFP (Sep)", "datetime_utc": _dt(9, 4, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Aug)", "datetime_utc": _dt(9, 10, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Aug)", "datetime_utc": _dt(9, 11, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(9, 16, 18, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "GDP (Q2 3rd)", "datetime_utc": _dt(9, 24, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "PCE Core (Aug)", "datetime_utc": _dt(9, 25, 12, 30), "impact": "HIGH", "currency": "USD"},

    # --- October ---
    {"name": "NFP (Oct)", "datetime_utc": _dt(10, 2, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Sep)", "datetime_utc": _dt(10, 13, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Sep)", "datetime_utc": _dt(10, 14, 12, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "GDP (Q3 1st)", "datetime_utc": _dt(10, 29, 12, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PCE Core (Sep)", "datetime_utc": _dt(10, 30, 12, 30), "impact": "HIGH", "currency": "USD"},

    # --- November ---
    {"name": "NFP (Nov)", "datetime_utc": _dt(11, 6, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(11, 4, 19, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "CPI (Oct)", "datetime_utc": _dt(11, 12, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Oct)", "datetime_utc": _dt(11, 13, 13, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "GDP (Q3 2nd)", "datetime_utc": _dt(11, 25, 13, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "PCE Core (Oct)", "datetime_utc": _dt(11, 25, 13, 30), "impact": "HIGH", "currency": "USD"},

    # --- December ---
    {"name": "NFP (Dec)", "datetime_utc": _dt(12, 4, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "CPI (Nov)", "datetime_utc": _dt(12, 10, 13, 30), "impact": "HIGH", "currency": "USD"},
    {"name": "PPI (Nov)", "datetime_utc": _dt(12, 11, 13, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "FOMC Rate Decision", "datetime_utc": _dt(12, 16, 19, 0), "impact": "EXTREME", "currency": "USD"},
    {"name": "GDP (Q3 3rd)", "datetime_utc": _dt(12, 22, 13, 30), "impact": "MEDIUM", "currency": "USD"},
    {"name": "PCE Core (Nov)", "datetime_utc": _dt(12, 23, 13, 30), "impact": "HIGH", "currency": "USD"},
]
