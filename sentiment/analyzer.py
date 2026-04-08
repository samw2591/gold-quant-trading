"""
Gold-specific sentiment analysis module v4.

v3 changes (2026-03-31):
  - Rebalanced keyword dictionary: added 33 bearish keywords (42 bull / 59 bear)
    to fix permanent BULLISH bias (was 100% positive in simulation).
  - Weight shift: FinBERT promoted to PRIMARY (50%), keywords demoted to 30%.
  - BULLISH/BEARISH threshold raised from 0.15 to 0.25 to reduce false positives.

v3.1 changes (2026-04-02):
  - Extreme FinBERT boost: when |finbert_score| > 0.30, weights shift to
    FinBERT 70% / keyword 15% / vader 15%.

v4 changes (2026-04-08):
  - CRITICAL FIX: keyword_score was ALWAYS 1.0 (10 days straight).
    Root cause: sum-based scoring inflated by high match count (~200+ matches
    across 193 headlines), then sqrt normalization still > 1.0, clipped to 1.0.
    Fix: use per-headline average score instead of total sum.
  - FinBERT weight increased to 0.60 (from 0.50), keyword reduced to 0.15
    (from 0.30). FinBERT is the only signal source with real information value.
  - Extreme FinBERT threshold lowered from 0.30 to 0.20 (triggers more often).
  - Falls back to keyword(0.30) + VADER(0.70) if FinBERT unavailable.

Scoring architecture (v4):
  Normal:   FinBERT 0.60 / keyword 0.15 / VADER 0.25
  Extreme:  FinBERT 0.80 / keyword 0.05 / VADER 0.15  (|FinBERT| > 0.20)
  Fallback: keyword 0.30 / VADER 0.70  (FinBERT unavailable)
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gold-specific keyword scoring (PRIMARY signal)
# Each keyword has a score: positive = bullish for gold, negative = bearish
# ---------------------------------------------------------------------------
GOLD_KEYWORDS = {
    # ── 利多黄金 (避险/宽松/通胀) ──  42个
    "war":              +0.30,
    "military":         +0.20,
    "iran":             +0.25,
    "strike":           +0.15,
    "attack":           +0.15,
    "conflict":         +0.20,
    "tension":          +0.20,
    "escalat":          +0.25,
    "geopolitic":       +0.20,
    "sanctions":        +0.20,
    "crisis":           +0.25,
    "safe haven":       +0.25,
    "safe-haven":       +0.25,
    "uncertainty":      +0.20,
    "recession":        +0.20,
    "inflation":        +0.15,
    "rate cut":         +0.25,
    "dovish":           +0.20,
    "debt ceiling":     +0.15,
    "tariff":           +0.15,
    "trade war":        +0.20,
    "central bank buy": +0.20,
    "gold demand":      +0.15,
    "gold surge":       +0.25,
    "gold jump":        +0.25,
    "gold rally":       +0.20,
    "gold rise":        +0.15,
    "gold gain":        +0.15,
    "gold climb":       +0.15,
    "gold soar":        +0.25,
    "gold record":      +0.20,
    "gold high":        +0.15,
    "bullion":          +0.10,
    "buy gold":         +0.20,
    "buying gold":      +0.20,
    "haven demand":     +0.20,
    "dollar weak":      +0.15,
    "dollar fall":      +0.15,
    "dollar drop":      +0.15,
    "dollar decline":   +0.15,
    "yield fall":       +0.10,
    "yield drop":       +0.10,

    # ── 利空黄金 (紧缩/强美元/风险偏好) ──  59个 (v3: 补充平衡)
    "rate hike":        -0.25,
    "hawkish":          -0.20,
    "strong dollar":    -0.20,
    "dollar strength":  -0.20,
    "dollar surge":     -0.15,
    "dollar rally":     -0.15,
    "peace deal":       -0.25,
    "peace talk":       -0.15,
    "ceasefire":        -0.20,
    "de-escalat":       -0.20,
    "negotiat":         -0.10,
    "risk on":          -0.15,
    "risk appetite":    -0.15,
    "stock rally":      -0.10,
    "equity rally":     -0.10,
    "gold fall":        -0.15,
    "gold drop":        -0.15,
    "gold decline":     -0.15,
    "gold slip":        -0.15,
    "gold dip":         -0.10,
    "gold sell":        -0.15,
    "sell gold":        -0.15,
    "gold crash":       -0.20,
    "gold plunge":      -0.20,
    "yield rise":       -0.10,
    "yield surge":      -0.10,
    # v3 新增: 利率/紧缩
    "higher for longer": -0.20,
    "rate hold":        -0.10,
    "taper":            -0.15,
    "quantitative tighten": -0.20,
    "balance sheet":    -0.10,
    "inflation expectation": -0.10,
    # v3 新增: 美元走强
    "dollar index":     -0.10,
    "dxy":              -0.10,
    "dollar demand":    -0.15,
    "dollar bid":       -0.15,
    # v3 新增: 风险偏好/股市走强
    "stock record":     -0.10,
    "stock high":       -0.10,
    "equity gain":      -0.10,
    "market rally":     -0.10,
    "risk rally":       -0.15,
    "crypto rally":     -0.10,
    # v3 新增: 黄金抛售/资金流出
    "gold outflow":     -0.15,
    "etf outflow":      -0.15,
    "gold liquidat":    -0.15,
    "profit taking":    -0.15,
    "gold pressur":     -0.10,
    "gold weaken":      -0.15,
    "gold retreat":     -0.15,
    "gold tumble":      -0.20,
    "gold shed":        -0.10,
    "gold ease":        -0.10,
    "gold pull back":   -0.15,
    "gold correct":     -0.10,
    # v3 新增: 收益率上升/债券抛售
    "yield climb":      -0.10,
    "yield jump":       -0.15,
    "bond selloff":     -0.15,
    "bond sell":        -0.10,
    "treasury sell":    -0.15,
    "real yield":       -0.10,
}

# Trump amplifier
TRUMP_AMPLIFIER = 1.3

# Subject-aware detection: headlines mentioning gold directly vs macro-only
GOLD_DIRECT_KEYWORDS = {"gold", "xau", "bullion", "precious metal", "gold price"}

# High-impact keywords get 3x weight in FinBERT/VADER scoring
HIGH_IMPACT_KEYWORDS = {
    "rate cut", "rate hike", "central bank", "fomc", "fed",
    "gold surge", "gold crash", "gold record", "safe haven",
    "war", "ceasefire", "peace deal",
}


def _is_gold_direct(headline: str) -> bool:
    """True if headline explicitly mentions gold/XAU (FinBERT score used as-is).
    False for macro-only news (FinBERT/VADER scores should be inverted for gold context)."""
    h = headline.lower()
    return any(kw in h for kw in GOLD_DIRECT_KEYWORDS)


def _headline_weight(headline: str) -> int:
    """Return scoring weight: 3 for high-impact headlines, 1 otherwise."""
    h = headline.lower()
    for kw in HIGH_IMPACT_KEYWORDS:
        if kw in h:
            return 3
    return 1

# ---------------------------------------------------------------------------
# VADER setup (lazy init)
# ---------------------------------------------------------------------------
_vader_analyzer = None


def _get_vader():
    global _vader_analyzer
    if _vader_analyzer is not None:
        return _vader_analyzer
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _vader_analyzer = SentimentIntensityAnalyzer()
        logger.info("[情绪分析] VADER模型加载成功")
        return _vader_analyzer
    except Exception as exc:
        logger.error(f"[情绪分析] VADER加载失败: {exc}")
        return None


# ---------------------------------------------------------------------------
# FinBERT setup (lazy init)
# ---------------------------------------------------------------------------
_finbert_pipeline = None
_finbert_attempted = False


def _get_finbert():
    global _finbert_pipeline, _finbert_attempted
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    if _finbert_attempted:
        return None
    _finbert_attempted = True

    import os
    if os.environ.get("GOLD_DISABLE_FINBERT", "").strip() == "1":
        logger.info("[情绪分析] FinBERT 已通过 GOLD_DISABLE_FINBERT=1 禁用 (节省~500MB内存)")
        return None

    try:
        from transformers import pipeline as hf_pipeline
        logger.info("[情绪分析] 正在加载FinBERT模型 (首次使用会自动下载)...")
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("[情绪分析] FinBERT模型加载成功")
        return _finbert_pipeline
    except Exception as exc:
        logger.warning(f"[情绪分析] FinBERT加载失败，将使用纯关键词+VADER模式: {exc}")
        return None


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------
class SentimentAnalyzer:
    """Gold-tuned sentiment analyzer v2."""

    def analyze_headlines(self, headlines: List[str]) -> Dict:
        if not headlines:
            return self._empty_result()

        kw_score = self._keyword_score(headlines)
        vader_score = self._vader_analyze(headlines)
        finbert_score = self._finbert_analyze(headlines)

        if finbert_score is not None:
            if abs(finbert_score) > 0.20:
                combined = finbert_score * 0.80 + kw_score * 0.05 + vader_score * 0.15
            else:
                combined = finbert_score * 0.60 + kw_score * 0.15 + vader_score * 0.25
            mode = "finbert+keyword+vader"
        else:
            combined = kw_score * 0.30 + vader_score * 0.70
            mode = "keyword+vader"

        combined = max(-1.0, min(1.0, combined))

        return {
            "keyword_score": round(kw_score, 4),
            "vader_score": round(vader_score, 4),
            "finbert_score": round(finbert_score, 4) if finbert_score is not None else None,
            "combined_score": round(combined, 4),
            "headline_count": len(headlines),
            "model_mode": mode,
        }

    def get_sentiment_signal(self, headlines: Optional[List[str]] = None) -> Dict:
        if not headlines:
            return {
                "score": 0.0, "label": "NEUTRAL",
                "confidence": 0.0, "details": self._empty_result(),
            }

        details = self.analyze_headlines(headlines)
        score = details["combined_score"]

        # v3: 提高阈值 0.25, 减少假阳性
        if score > 0.25:
            label = "BULLISH"
        elif score < -0.25:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        confidence = min(1.0, abs(score) * 1.5)  # v3: 缓和置信度放大

        return {
            "score": score,
            "label": label,
            "confidence": round(confidence, 4),
            "details": details,
        }

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _keyword_score(self, headlines: List[str]) -> float:
        """Gold-specific keyword scoring — per-headline average approach.

        Each headline gets a score from keyword matches, then we average across
        all headlines that had at least one match.  This prevents the old bug
        where 200+ keyword matches across 190 headlines inflated the sum so
        much that sqrt-normalisation still saturated at +1.0.
        """
        headline_scores: List[float] = []

        for headline in headlines:
            h_lower = headline.lower()
            h_score = 0.0
            h_matches = 0

            for keyword, weight in GOLD_KEYWORDS.items():
                if keyword in h_lower:
                    h_score += weight
                    h_matches += 1

            if h_matches == 0:
                continue

            if "trump" in h_lower and h_score != 0:
                h_score *= TRUMP_AMPLIFIER

            per_match_avg = h_score / h_matches
            headline_scores.append(per_match_avg)

        if not headline_scores:
            return 0.0

        avg = sum(headline_scores) / len(headline_scores)
        return max(-1.0, min(1.0, avg))

    def _vader_analyze(self, headlines: List[str]) -> float:
        vader = _get_vader()
        if vader is None:
            return 0.0
        weighted_total = 0.0
        total_weight = 0
        for h in headlines:
            score = vader.polarity_scores(h)["compound"]
            if not _is_gold_direct(h):
                score *= -1
            w = _headline_weight(h)
            weighted_total += score * w
            total_weight += w
        return weighted_total / total_weight if total_weight > 0 else 0.0

    def _finbert_analyze(self, headlines: List[str]) -> Optional[float]:
        pipe = _get_finbert()
        if pipe is None:
            return None
        try:
            batch = headlines[:50]
            results = pipe(batch, batch_size=16)
        except Exception as exc:
            logger.warning(f"[情绪分析] FinBERT推理失败: {exc}")
            return None

        weighted_total = 0.0
        total_weight = 0
        for headline, r in zip(batch, results):
            label = r["label"].lower()
            prob = r["score"]
            if label == "positive":
                score = prob
            elif label == "negative":
                score = -prob
            else:
                score = 0.0
            if not _is_gold_direct(headline):
                score *= -1
            w = _headline_weight(headline)
            weighted_total += score * w
            total_weight += w
        return weighted_total / total_weight if total_weight > 0 else 0.0

    def _empty_result(self) -> Dict:
        return {
            "keyword_score": 0.0,
            "vader_score": 0.0,
            "finbert_score": None,
            "combined_score": 0.0,
            "headline_count": 0,
            "model_mode": "none",
        }
