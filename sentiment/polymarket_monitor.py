"""
Polymarket 地缘政治风险监控 (v2 — NEH API)
=============================================
通过 pizzint.watch 的 NEH (Nothing Ever Happens) API 获取经过专家筛选的
地缘政治预测市场数据，计算综合风险指数和黄金方向性影响。

v1 使用 14 个关键词搜索 Polymarket，v2 改为接入 pizzint.watch 的 curated
basket（~34 个精选市场），数据质量更高，API 调用从 14 次降为 1 次。

数据源: https://www.pizzint.watch/api/neh-index/doomsday
原始数据: Polymarket prediction markets

核心逻辑:
  - 一次 API 调用获取全部 curated 地缘市场及其实时概率
  - 按事件类型自动判断对黄金的影响方向 (bullish/bearish)
  - 加权聚合为综合风险指数 (0-100)
  - 0-30: 低风险   30-60: 中风险   60-85: 高风险   85-100: 极端风险
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

NEH_API_URL = "https://www.pizzint.watch/api/neh-index/doomsday"
REQUEST_TIMEOUT = 15

# 区域权重：中东和亚太冲突对黄金影响更大
REGION_WEIGHT = {
    "middle_east": 1.5,
    "asia": 1.3,
    "europe": 1.2,
    "americas": 1.0,
    "global": 1.0,
}

# 关键词 → 黄金影响方向和额外权重加成
# "bullish": 事件发生利多黄金（战争/冲突/危机）
# "bearish": 事件发生利空黄金（和平/降级/停火）
_BULLISH_KEYWORDS = [
    ("invade", 1.5), ("war", 1.5), ("strike", 1.3), ("attack", 1.3),
    ("military", 1.2), ("clash", 1.2), ("nuclear", 1.5), ("nato", 1.3),
    ("draft", 1.0), ("escalat", 1.3), ("blockade", 1.3), ("conflict", 1.2),
    ("ground operation", 1.2), ("insurrection", 1.0), ("capture", 1.0),
    ("sanction", 1.0), ("tariff", 0.8), ("recession", 1.0), ("shutdown", 0.8),
    ("regime fall", 1.2), ("canal", 0.8),
]

_BEARISH_KEYWORDS = [
    ("ceasefire", 1.3), ("peace", 1.3), ("de-escalat", 1.3),
    ("deal", 1.0), ("negotiat", 1.0), ("withdraw", 1.0),
]


class PolymarketMonitor:
    """从 pizzint.watch NEH API 获取地缘政治风险指数。"""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'GoldQuantTrader/2.0',
        })
        self._cache: Dict = {}
        self._cache_ts: float = 0
        self._cache_ttl: float = 300  # 5 分钟缓存

    def get_risk_index(self) -> Dict:
        """获取综合地缘风险指数。

        Returns:
            {
                "risk_index": float (0-100),
                "risk_level": str ("LOW"/"MEDIUM"/"HIGH"/"EXTREME"),
                "top_risks": [{"question": str, "probability": float, "weight": float}, ...],
                "gold_sentiment_boost": float (-0.3 to +0.3),
                "market_count": int,
                "error": str or None,
            }
        """
        now = time.time()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        raw = self._fetch_neh_markets()
        if raw is None:
            return self._error_result("NEH API 请求失败")

        markets = raw.get("markets", [])
        low_volume = raw.get("lowVolume", [])
        all_markets = markets + low_volume

        if not all_markets:
            return self._default_result()

        enriched = self._enrich_markets(all_markets)
        risk_index, top_risks, sentiment_boost = self._compute_index(enriched)
        risk_level = self._classify_risk(risk_index)

        result = {
            "risk_index": round(risk_index, 1),
            "risk_level": risk_level,
            "top_risks": top_risks[:5],
            "gold_sentiment_boost": round(sentiment_boost, 3),
            "market_count": len(all_markets),
            "error": None,
        }

        self._cache = result
        self._cache_ts = now
        logger.info(
            f"[Polymarket/NEH] {len(all_markets)} 个市场, "
            f"风险指数 {risk_index:.1f} ({risk_level}), "
            f"冠军: {top_risks[0]['question'][:50]}... ({top_risks[0]['probability']:.0f}%)"
            if top_risks else
            f"[Polymarket/NEH] {len(all_markets)} 个市场, 无高风险事件"
        )
        return result

    def _fetch_neh_markets(self) -> Optional[Dict]:
        """从 pizzint.watch NEH API 获取 curated 市场数据。"""
        try:
            resp = self._session.get(NEH_API_URL, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                logger.warning(f"[Polymarket/NEH] API 返回 {resp.status_code}")
                return None
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"[Polymarket/NEH] 请求失败: {e}")
            return None

    def _enrich_markets(self, markets: List[Dict]) -> List[Dict]:
        """为每个市场添加黄金方向性判断和权重。"""
        enriched = []
        for mkt in markets:
            label = mkt.get("label", "")
            price = mkt.get("price", 0)
            region = mkt.get("region", "global")
            volume = mkt.get("volume", 0)

            if price <= 0.01 or price >= 0.99:
                continue

            direction, kw_boost = self._classify_gold_direction(label)
            region_w = REGION_WEIGHT.get(region, 1.0)
            vol_w = min(1.5, max(0.5, (volume / 500_000) ** 0.3)) if volume > 0 else 0.5
            weight = region_w * kw_boost * vol_w

            enriched.append({
                "question": label,
                "probability": price,
                "weight": weight,
                "direction": direction,
                "region": region,
                "volume": volume,
            })
        return enriched

    @staticmethod
    def _classify_gold_direction(label: str) -> Tuple[str, float]:
        """根据市场标题判断对黄金的影响方向和权重加成。"""
        q = label.lower()

        for kw, boost in _BEARISH_KEYWORDS:
            if kw in q:
                return "yes_bearish", boost

        for kw, boost in _BULLISH_KEYWORDS:
            if kw in q:
                return "yes_bullish", boost

        return "yes_bullish", 1.0

    def _compute_index(self, markets: List[Dict]) -> Tuple[float, List[Dict], float]:
        """计算综合风险指数和黄金情绪增量。"""
        risk_contributions = []

        for mkt in markets:
            prob = mkt["probability"]
            weight = mkt["weight"]
            direction = mkt["direction"]

            if direction == "yes_bearish":
                risk_contribution = (1.0 - prob) * weight
                sentiment_contribution = -(prob * weight)
            else:
                risk_contribution = prob * weight
                sentiment_contribution = prob * weight

            risk_contributions.append({
                "question": mkt["question"][:80],
                "probability": round(prob * 100, 1),
                "weight": round(weight, 2),
                "risk_contribution": risk_contribution,
                "sentiment_contribution": sentiment_contribution,
                "direction": direction,
                "region": mkt["region"],
            })

        if not risk_contributions:
            return 0.0, [], 0.0

        total_weight = sum(r["weight"] for r in risk_contributions)
        weighted_risk = sum(r["risk_contribution"] for r in risk_contributions)
        risk_index = (weighted_risk / total_weight) * 100 if total_weight > 0 else 0

        weighted_sentiment = sum(r["sentiment_contribution"] for r in risk_contributions)
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        sentiment_boost = max(-0.3, min(0.3, avg_sentiment * 0.5))

        top_risks = sorted(
            risk_contributions,
            key=lambda x: abs(x["risk_contribution"]),
            reverse=True,
        )

        return min(100, risk_index), top_risks, sentiment_boost

    @staticmethod
    def _classify_risk(index: float) -> str:
        if index >= 85:
            return "EXTREME"
        elif index >= 60:
            return "HIGH"
        elif index >= 30:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _default_result() -> Dict:
        return {
            "risk_index": 0.0,
            "risk_level": "LOW",
            "top_risks": [],
            "gold_sentiment_boost": 0.0,
            "market_count": 0,
            "error": None,
        }

    @staticmethod
    def _error_result(msg: str) -> Dict:
        return {
            "risk_index": 0.0,
            "risk_level": "UNKNOWN",
            "top_risks": [],
            "gold_sentiment_boost": 0.0,
            "market_count": 0,
            "error": msg,
        }
