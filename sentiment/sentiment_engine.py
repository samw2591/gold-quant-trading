"""
Sentiment engine — main integration point for the sentiment analysis system.

Combines news collection, sentiment analysis, and calendar risk into a single
trading context that the main trading system can consume.

Uses a background thread to collect and analyze news every N seconds,
so the main trading loop is never blocked by network I/O.
"""

import logging
import threading
import time
from typing import Dict, Optional

from sentiment.news_collector import NewsCollector
from sentiment.analyzer import SentimentAnalyzer
from sentiment.calendar_guard import CalendarGuard

logger = logging.getLogger(__name__)

# Default neutral context returned when no data is available yet
_NEUTRAL_CONTEXT: Dict = {
    "sentiment": {"score": 0.0, "label": "NEUTRAL", "confidence": 0.0},
    "calendar": {"risk_level": "LOW", "pause": False, "pause_reason": "", "next_event": None},
    "news_summary": "舆情数据采集中...",
    "trade_modifier": {"allow_trading": True, "direction_bias": None, "lot_multiplier": 1.0},
}


class SentimentEngine:
    """Unified sentiment engine with background collection thread."""

    def __init__(self, update_interval: int = 300):
        """
        Args:
            update_interval: How often (seconds) the background thread refreshes
                             sentiment data.  Default 300 = 5 minutes.
        """
        self.collector = NewsCollector()
        self.analyzer = SentimentAnalyzer()
        self.calendar = CalendarGuard(news_collector=self.collector)

        self._update_interval = update_interval

        # Thread-safe result store
        self._lock = threading.Lock()
        self._latest: Dict = {}
        self._latest_ts: float = 0.0

        # Background worker
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._background_loop, daemon=True, name="SentimentWorker"
        )
        self._thread.start()
        logger.info(f"[舆情引擎] 后台线程已启动，每{update_interval}秒更新一次")

    # ------------------------------------------------------------------
    # Public API  (called from main trading thread — never blocks on I/O)
    # ------------------------------------------------------------------

    def get_trading_context(self) -> Dict:
        """Get the latest sentiment context.  Always returns instantly.

        If the background thread hasn't produced its first result yet,
        a neutral default is returned so trading is not affected.
        """
        with self._lock:
            if self._latest:
                return self._latest.copy()

        # First call before background thread finishes — return neutral
        logger.debug("[舆情引擎] 数据尚未就绪，返回中性默认值")
        return _NEUTRAL_CONTEXT.copy()

    def invalidate_cache(self):
        """Force the background thread to refresh on its next cycle."""
        with self._lock:
            self._latest = {}
            self._latest_ts = 0.0

    def stop(self):
        """Signal the background thread to stop (for clean shutdown)."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _background_loop(self):
        """Runs in a daemon thread.  Periodically fetches & analyzes news."""
        # Small initial delay so the main thread can finish setup
        time.sleep(2)

        while not self._stop_event.is_set():
            try:
                result = self._do_full_analysis()
                with self._lock:
                    self._latest = result
                    self._latest_ts = time.time()
            except Exception as exc:
                logger.warning(f"[舆情引擎] 后台分析异常: {exc}")

            # Sleep in small increments so we can respond to stop quickly
            for _ in range(self._update_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("[舆情引擎] 后台线程已退出")

    def _do_full_analysis(self) -> Dict:
        """Perform one complete analysis cycle.  May take 5-15 seconds."""
        logger.info("[舆情引擎] 开始采集舆情数据...")

        # 1. Calendar risk check (fast, local data)
        calendar_pause, pause_reason = self.calendar.should_pause_trading()
        risk_level = self.calendar.get_risk_level()
        next_event = self.calendar.get_next_event()

        calendar_info = {
            "risk_level": risk_level,
            "pause": calendar_pause,
            "pause_reason": pause_reason,
            "next_event": _format_event(next_event),
        }

        # 2. Collect news (network I/O — this is why we run in background)
        headlines = self._collect_all_headlines()

        # 3. Sentiment analysis
        sentiment = self.analyzer.get_sentiment_signal(headlines)

        # 4. Build news summary
        news_summary = self._build_summary(headlines, sentiment)

        # 5. Compute trade modifier
        trade_modifier = self._compute_trade_modifier(
            sentiment, calendar_pause, risk_level
        )

        result = {
            "sentiment": {
                "score": sentiment["score"],
                "label": sentiment["label"],
                "confidence": sentiment["confidence"],
            },
            "calendar": calendar_info,
            "news_summary": news_summary,
            "trade_modifier": trade_modifier,
        }

        logger.info(
            f"[舆情引擎] 分析完成 — "
            f"情绪: {sentiment['label']}({sentiment['score']:.2f}), "
            f"风险: {risk_level}, "
            f"允许交易: {trade_modifier['allow_trading']}, "
            f"方向偏好: {trade_modifier['direction_bias']}, "
            f"仓位系数: {trade_modifier['lot_multiplier']:.2f}"
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_all_headlines(self) -> list:
        """Collect headlines from all sources. Never raises."""
        headlines: list = []
        try:
            gold_news = self.collector.collect_gold_news()
            headlines.extend(a["title"] for a in gold_news if a.get("title"))
        except Exception as exc:
            logger.warning(f"[舆情引擎] 黄金新闻采集失败: {exc}")

        try:
            trump_news = self.collector.collect_trump_posts()
            headlines.extend(a["title"] for a in trump_news if a.get("title"))
        except Exception as exc:
            logger.warning(f"[舆情引擎] 特朗普新闻采集失败: {exc}")

        # Deduplicate while preserving order
        seen: set = set()
        unique: list = []
        for h in headlines:
            h_lower = h.lower().strip()
            if h_lower and h_lower not in seen:
                seen.add(h_lower)
                unique.append(h)

        return unique

    def _build_summary(self, headlines: list, sentiment: Dict) -> str:
        """Build a concise Chinese-language summary of the news."""
        if not headlines:
            return "当前无相关新闻数据"

        label_cn = {
            "BULLISH": "看涨",
            "BEARISH": "看跌",
            "NEUTRAL": "中性",
        }
        label = label_cn.get(sentiment["label"], "中性")
        count = len(headlines)
        score = sentiment["score"]

        # Pick up to 3 representative headlines
        samples = headlines[:3]
        sample_text = " | ".join(samples)

        return (
            f"分析{count}条新闻，整体情绪{label}(得分{score:.2f})。"
            f"代表性标题: {sample_text}"
        )

    def _compute_trade_modifier(
        self,
        sentiment: Dict,
        calendar_pause: bool,
        risk_level: str,
    ) -> Dict:
        """Decide trading adjustments based on sentiment + calendar.

        Decision logic:
          1. Calendar says pause -> allow_trading=False
          2. Use analyzer's BULLISH/BEARISH label (threshold 0.15) with confidence >= 0.3
          3. HIGH calendar risk -> lot_multiplier *= 0.5
        """
        allow_trading = True
        direction_bias: Optional[str] = None
        lot_multiplier = 1.0

        if calendar_pause:
            return {
                "allow_trading": False,
                "direction_bias": None,
                "lot_multiplier": 0.0,
            }

        label = sentiment.get("label", "NEUTRAL")
        confidence = sentiment.get("confidence", 0.0)
        if label == "BULLISH" and confidence >= 0.3:
            direction_bias = "BUY"
            lot_multiplier = 1.2
        elif label == "BEARISH" and confidence >= 0.3:
            direction_bias = "SELL"
            lot_multiplier = 1.2

        if risk_level == "HIGH":
            lot_multiplier *= 0.5
        elif risk_level == "EXTREME":
            lot_multiplier *= 0.3

        lot_multiplier = round(lot_multiplier, 2)

        return {
            "allow_trading": allow_trading,
            "direction_bias": direction_bias,
            "lot_multiplier": lot_multiplier,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _format_event(event: Optional[Dict]) -> Optional[str]:
    """Format an event dict into a readable string."""
    if event is None:
        return None
    dt = event["datetime_utc"]
    return f"{event['name']} @ {dt.strftime('%Y-%m-%d %H:%M')} UTC ({event['impact']})"
