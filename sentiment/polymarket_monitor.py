"""
Polymarket 地缘政治风险监控
============================
通过 Polymarket Gamma API (免认证) 追踪高影响力地缘政治事件的
发生概率，聚合为一个 0-100 的风险指数。

类似 pizzint.watch 的 "Nothing Ever Happens Index"，但直接集成到
我们的舆情系统中。

数据源: https://gamma-api.polymarket.com/markets
文档: https://docs.polymarket.com/developers/gamma-markets-api

核心逻辑:
  - 维护一组与黄金/避险相关的 Polymarket 搜索关键词
  - 每次调用获取相关市场的最新概率
  - 按影响力加权，计算综合地缘风险指数 (0-100)
  - 0-30: 低风险   30-60: 中风险   60-85: 高风险   85-100: 极端风险
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 15

# ═══════════════════════════════════════════════════════════════
# 追踪的地缘政治主题及其对黄金的影响权重
# ═══════════════════════════════════════════════════════════════

GEOPOLITICAL_QUERIES = [
    # (搜索关键词, 黄金影响权重, 概率方向: "yes_bullish"=YES发生利多黄金, "yes_bearish"=YES发生利空)
    ("invade Iran", 3.0, "yes_bullish"),
    ("Iran war", 3.0, "yes_bullish"),
    ("Iran strike", 2.5, "yes_bullish"),
    ("China Taiwan", 3.0, "yes_bullish"),
    ("NATO", 2.5, "yes_bullish"),
    ("nuclear", 2.5, "yes_bullish"),
    ("World War", 3.0, "yes_bullish"),
    ("military action", 2.0, "yes_bullish"),
    ("government shutdown", 1.5, "yes_bullish"),
    ("recession", 2.0, "yes_bullish"),
    ("rate cut", 1.5, "yes_bullish"),
    ("ceasefire", 1.5, "yes_bearish"),
    ("tariff", 1.5, "yes_bullish"),
    ("sanctions", 1.5, "yes_bullish"),
]

SEARCH_API = f"{GAMMA_API_BASE}/public-search"


class PolymarketMonitor:
    """从 Polymarket 获取地缘政治风险指数。"""

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'GoldQuantTrader/1.0',
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
                "gold_sentiment_boost": float (-0.3 to +0.3, 可叠加到舆情分数),
                "market_count": int,
                "error": str or None,
            }
        """
        now = time.time()
        if self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        markets = self._fetch_relevant_markets()
        if markets is None:
            return self._error_result("API 请求失败")

        if not markets:
            return self._default_result()

        risk_index, top_risks, sentiment_boost = self._compute_index(markets)
        risk_level = self._classify_risk(risk_index)

        result = {
            "risk_index": round(risk_index, 1),
            "risk_level": risk_level,
            "top_risks": top_risks[:5],
            "gold_sentiment_boost": round(sentiment_boost, 3),
            "market_count": len(markets),
            "error": None,
        }

        self._cache = result
        self._cache_ts = now
        return result

    def _fetch_relevant_markets(self) -> Optional[List[Dict]]:
        """通过 /public-search 端点搜索地缘政治相关市场。"""
        all_markets = []
        seen_ids = set()

        for query, weight, direction in GEOPOLITICAL_QUERIES:
            try:
                resp = self._session.get(
                    SEARCH_API,
                    params={"q": query, "limit_per_type": "5"},
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code != 200:
                    continue

                data = resp.json()
                events = data.get("events", [])

                for event in events:
                    for mkt in event.get("markets", []):
                        # 跳过已关闭的市场
                        if mkt.get("closed"):
                            continue

                        mkt_id = mkt.get("id") or mkt.get("condition_id", "")
                        if not mkt_id or mkt_id in seen_ids:
                            continue

                        # 只保留概率在 0.02-0.98 之间的活跃市场
                        prob = self._extract_yes_probability(mkt)
                        if prob is None or prob <= 0.02 or prob >= 0.98:
                            continue

                        # 过滤已基本确定的日级市场
                        if self._is_stale_daily_market(mkt):
                            continue

                        seen_ids.add(mkt_id)
                        question = (mkt.get("question") or "").lower()
                        actual_weight, actual_dir = self._refine_weight_direction(
                            question, weight, direction
                        )
                        mkt["_weight"] = actual_weight
                        mkt["_direction"] = actual_dir
                        mkt["_query"] = query
                        all_markets.append(mkt)

            except requests.RequestException as e:
                logger.debug(f"[Polymarket] 查询 '{query}' 失败: {e}")
                continue

        # 去重: 同一主题的日级市场只保留概率最高的一个
        deduped = self._dedup_daily_markets(all_markets)
        logger.info(f"[Polymarket] 获取到 {len(all_markets)} 个市场, 去重后 {len(deduped)} 个")
        return deduped

    @staticmethod
    def _dedup_daily_markets(markets: List[Dict]) -> List[Dict]:
        """同一主题的日级市场 (April 4/5/6/7...) 只保留最接近 50% 的一个。

        这避免了 "Iran military action on April 4/5/6/7" 四个市场分别计入、
        导致单一事件被 4x 权重放大的问题。
        """
        import re

        clusters: Dict[str, List[Dict]] = {}
        non_daily = []

        date_pattern = re.compile(
            r'\b(?:on|by)\s+(\w+)\s+\d{1,2},?\s+2026'
        )

        for mkt in markets:
            question = mkt.get("question", "")
            match = date_pattern.search(question)
            if match:
                # 去掉日期部分作为聚类 key
                base = date_pattern.sub("DATE", question).strip()
                clusters.setdefault(base, []).append(mkt)
            else:
                non_daily.append(mkt)

        # 每个聚类只保留概率最接近 0.5 的市场（信息量最大）
        for base, group in clusters.items():
            if len(group) <= 1:
                non_daily.extend(group)
            else:
                best = min(
                    group,
                    key=lambda m: abs(
                        (PolymarketMonitor._static_extract_yes_prob(m) or 0.5) - 0.5
                    ),
                )
                non_daily.append(best)

        return non_daily

    def _refine_weight_direction(
        self, question: str, default_weight: float, default_dir: str
    ) -> Tuple[float, str]:
        """根据问题文本微调权重和方向。

        关键原则:
        - "yes_bullish" = 事件发生利多黄金 (战争/危机/衰退)
        - "yes_bearish" = 事件发生利空黄金 (和平/降息/好转)
        """
        q = question.lower()

        # 和平/缓和类: YES发生 = 利空黄金 (减少避险需求)
        bearish_kw = ["ceasefire", "peace deal", "de-escalat", "nuclear deal"]
        for kw in bearish_kw:
            if kw in q:
                return default_weight, "yes_bearish"

        # 冲突/危机类: YES发生 = 利多黄金 (增加避险需求)
        bullish_kw = [
            "invade", "war", "strike", "attack", "military",
            "nuclear strike", "clash", "blockade", "escalat", "conflict",
            "recession", "shutdown", "default", "tariff", "sanction",
            "withdraw from nato", "leave nato",
        ]
        for kw in bullish_kw:
            if kw in q:
                return default_weight, "yes_bullish"

        return default_weight, default_dir

    @staticmethod
    def _is_stale_daily_market(mkt: Dict) -> bool:
        """过滤掉即将到期的日级市场 (概率极端 = 已基本确定)。"""
        question = (mkt.get("question") or "").lower()
        import re
        # 匹配 "on April 2, 2026" 或 "by March 29, 2026" 等精确日期
        date_pattern = r'(?:on|by)\s+\w+\s+\d{1,2},?\s+2026'
        if re.search(date_pattern, question):
            prob = PolymarketMonitor._static_extract_yes_prob(mkt)
            if prob is not None and (prob > 0.90 or prob < 0.10):
                return True
        return False

    @staticmethod
    def _static_extract_yes_prob(mkt: Dict) -> Optional[float]:
        """静态版本的概率提取。"""
        tokens = mkt.get("tokens")
        if tokens and isinstance(tokens, list):
            for token in tokens:
                outcome = (token.get("outcome") or "").lower()
                if outcome == "yes":
                    price = token.get("price")
                    if price is not None:
                        return float(price)
        prices_str = mkt.get("outcomePrices")
        if prices_str:
            try:
                import json
                prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                if isinstance(prices, list) and len(prices) >= 1:
                    return float(prices[0])
            except (ValueError, TypeError):
                pass
        return None

    def _infer_weight_direction(self, question: str) -> Tuple[float, str]:
        """根据问题文本推断影响权重和方向。"""
        q = question.lower()

        # 利多黄金的事件（战争、危机、衰退）
        bullish_keywords = {
            "invade": 3.0, "war": 3.0, "strike": 2.5, "attack": 2.5,
            "nuclear": 3.0, "conflict": 2.5, "military": 2.0,
            "recession": 2.0, "default": 2.0, "shutdown": 1.5,
            "tariff": 1.5, "sanction": 1.5, "crisis": 2.0,
            "escalat": 2.5,
        }
        for kw, w in bullish_keywords.items():
            if kw in q:
                return w, "yes_bullish"

        # 利空黄金的事件（和平、降级）
        bearish_keywords = {
            "ceasefire": 1.5, "peace": 1.5, "de-escalat": 1.5,
            "deal": 1.0, "negotiat": 1.0,
        }
        for kw, w in bearish_keywords.items():
            if kw in q:
                return w, "yes_bearish"

        return 1.0, "yes_bullish"

    def _compute_index(self, markets: List[Dict]) -> Tuple[float, List[Dict], float]:
        """计算综合风险指数和黄金情绪增量。"""
        risk_contributions = []

        for mkt in markets:
            question = mkt.get("question") or "Unknown"
            weight = mkt.get("_weight", 1.0)
            direction = mkt.get("_direction", "yes_bullish")

            # 获取 YES 概率
            prob = self._extract_yes_probability(mkt)
            if prob is None:
                continue

            # 根据方向调整: yes_bullish → 概率越高风险越大
            # yes_bearish → 概率越高风险越小（和平是好事）
            if direction == "yes_bearish":
                risk_contribution = (1.0 - prob) * weight
                sentiment_contribution = -(prob * weight)  # 和平利空黄金
            else:
                risk_contribution = prob * weight
                sentiment_contribution = prob * weight  # 冲突利多黄金

            risk_contributions.append({
                "question": question[:80],
                "probability": round(prob * 100, 1),
                "weight": weight,
                "risk_contribution": risk_contribution,
                "sentiment_contribution": sentiment_contribution,
                "direction": direction,
            })

        if not risk_contributions:
            return 0.0, [], 0.0

        # 风险指数: 加权平均概率 × 100
        total_weight = sum(r["weight"] for r in risk_contributions)
        weighted_risk = sum(r["risk_contribution"] for r in risk_contributions)
        risk_index = (weighted_risk / total_weight) * 100 if total_weight > 0 else 0

        # 黄金情绪增量: 映射到 -0.3 ~ +0.3
        weighted_sentiment = sum(r["sentiment_contribution"] for r in risk_contributions)
        avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        sentiment_boost = max(-0.3, min(0.3, avg_sentiment * 0.5))

        # 按风险贡献排序
        top_risks = sorted(risk_contributions, key=lambda x: abs(x["risk_contribution"]), reverse=True)

        return min(100, risk_index), top_risks, sentiment_boost

    def _extract_yes_probability(self, mkt: Dict) -> Optional[float]:
        """从市场数据中提取 YES 概率。"""
        # 方式1: tokens 数组中的 price
        tokens = mkt.get("tokens")
        if tokens and isinstance(tokens, list):
            for token in tokens:
                outcome = (token.get("outcome") or "").lower()
                if outcome == "yes":
                    price = token.get("price")
                    if price is not None:
                        return float(price)

        # 方式2: outcomePrices 字符串
        prices_str = mkt.get("outcomePrices")
        if prices_str:
            try:
                import json
                prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                if isinstance(prices, list) and len(prices) >= 1:
                    return float(prices[0])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        # 方式3: bestBid
        best_bid = mkt.get("bestBid")
        if best_bid is not None:
            return float(best_bid)

        return None

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
