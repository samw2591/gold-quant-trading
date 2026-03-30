"""
Gold-specific sentiment analysis module v2.

Key change from v1:
  VADER scores are INVERTED for gold context — "war", "crisis" etc are
  negative in general sentiment but POSITIVE for gold.  Instead of relying
  on VADER's raw compound score, we now use a gold-tuned keyword scoring
  system as the PRIMARY signal, with VADER/FinBERT as secondary inputs.

Scoring architecture:
  1. Gold keyword score (weight 0.50) — domain-specific, most reliable
  2. FinBERT score      (weight 0.35) — financial context aware
  3. VADER score        (weight 0.15) — general sentiment baseline
  Falls back to keyword(0.7) + VADER(0.3) if FinBERT unavailable.
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
    # ── 强烈利好黄金 (避险/宽松/通胀) ──
    "war":              +0.30,
    "military":         +0.20,
    "iran":             +0.25,
    "strike":           +0.15,
    "attack":           +0.15,
    "conflict":         +0.20,
    "tension":          +0.20,
    "escalat":          +0.25,   # escalation, escalate, escalating
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
    
    # ── 利空黄金 (冒险/紧缩/强美元) ──
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
    "negotiat":         -0.10,   # 谈判可能利空黄金
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

        # Combine: keyword is PRIMARY
        if finbert_score is not None:
            combined = kw_score * 0.50 + finbert_score * 0.35 + vader_score * 0.15
            mode = "keyword+finbert+vader"
        else:
            combined = kw_score * 0.70 + vader_score * 0.30
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

        # 阈值降低: 0.15 即可判方向 (原来是0.3太高)
        if score > 0.15:
            label = "BULLISH"
        elif score < -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        confidence = min(1.0, abs(score) * 2)  # 放大置信度

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
        """Gold-specific keyword scoring — PRIMARY signal."""
        total_score = 0.0
        matched_count = 0
        has_trump = False
        
        for headline in headlines:
            h_lower = headline.lower()
            headline_score = 0.0
            
            if "trump" in h_lower:
                has_trump = True
            
            for keyword, weight in GOLD_KEYWORDS.items():
                if keyword in h_lower:
                    headline_score += weight
                    matched_count += 1
            
            # Trump新闻放大
            if has_trump and headline_score != 0:
                headline_score *= TRUMP_AMPLIFIER
            
            total_score += headline_score
        
        if matched_count == 0:
            return 0.0
        
        # 归一化到 [-1, 1]
        # 除以匹配数的平方根，避免新闻多时分数过大
        normalized = total_score / (matched_count ** 0.5)
        return max(-1.0, min(1.0, normalized))

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
