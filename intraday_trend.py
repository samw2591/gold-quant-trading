"""
Intraday Trend Meter
=====================
Real-time trend scoring using today's H1 bars.
Gates entries based on whether the market is trending, neutral, or choppy.

No prediction — only reacts to what has already happened today.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

import config

log = logging.getLogger(__name__)

REGIME_TRENDING = 'trending'
REGIME_NEUTRAL = 'neutral'
REGIME_CHOPPY = 'choppy'


class IntradayTrendMeter:
    """Scores the current trading day's trend strength using live H1 bars."""

    def __init__(self):
        self._score: float = 0.5
        self._regime: str = REGIME_NEUTRAL
        self._last_update_date = None
        self._today_bar_count: int = 0

    def update(self, h1_df: pd.DataFrame) -> float:
        """Recalculate trend score from the H1 dataframe.

        Extracts today's bars (UTC date) and computes a composite score.
        Returns the updated score.
        """
        if h1_df is None or len(h1_df) < 10:
            return self._score

        now_utc = datetime.now(timezone.utc)
        today = now_utc.date()

        if hasattr(h1_df.index, 'tz') and h1_df.index.tz is not None:
            today_bars = h1_df[h1_df.index.date == today]
        else:
            today_bars = h1_df[h1_df.index.date == today]

        self._today_bar_count = len(today_bars)

        if len(today_bars) < 2:
            self._score = 0.5
            self._regime = REGIME_NEUTRAL
            self._last_update_date = today
            return self._score

        self._score = self._calc_score(today_bars)
        self._regime = self._classify(self._score)
        self._last_update_date = today
        return self._score

    def _calc_score(self, today_bars: pd.DataFrame) -> float:
        """Composite trend score from today's H1 bars (0 to 1)."""
        latest = today_bars.iloc[-1]

        # 1. ADX component (weight 0.30)
        adx = float(latest.get('ADX', 20))
        if np.isnan(adx):
            adx = 20
        adx_score = min(adx / 40.0, 1.0)

        # 2. KC breakout ratio (weight 0.25)
        kc_upper = today_bars.get('KC_upper')
        kc_lower = today_bars.get('KC_lower')
        if kc_upper is not None and kc_lower is not None:
            breaks = (
                (today_bars['Close'] > kc_upper) |
                (today_bars['Close'] < kc_lower)
            ).sum()
            kc_score = min(float(breaks) / len(today_bars), 1.0)
        else:
            kc_score = 0.0

        # 3. EMA alignment consistency (weight 0.25)
        ema9 = today_bars.get('EMA9')
        ema21 = today_bars.get('EMA21')
        ema100 = today_bars.get('EMA100')
        if ema9 is not None and ema21 is not None and ema100 is not None:
            bullish = (ema9 > ema21) & (ema21 > ema100)
            bearish = (ema9 < ema21) & (ema21 < ema100)
            aligned = (bullish | bearish).sum()
            ema_score = float(aligned) / len(today_bars)
        else:
            ema_score = 0.0

        # 4. Trend intensity: |close - open| / range (weight 0.20)
        day_open = float(today_bars.iloc[0]['Open'])
        day_close = float(latest['Close'])
        day_high = float(today_bars['High'].max())
        day_low = float(today_bars['Low'].min())
        day_range = day_high - day_low
        ti = abs(day_close - day_open) / day_range if day_range > 0.01 else 0.0

        score = 0.30 * adx_score + 0.25 * kc_score + 0.25 * ema_score + 0.20 * ti
        return round(score, 3)

    def _classify(self, score: float) -> str:
        if score >= config.INTRADAY_TREND_KC_ONLY_THRESHOLD:
            return REGIME_TRENDING
        elif score >= config.INTRADAY_TREND_THRESHOLD:
            return REGIME_NEUTRAL
        else:
            return REGIME_CHOPPY

    def get_score(self) -> float:
        return self._score

    def get_regime(self) -> str:
        return self._regime

    def get_bar_count(self) -> int:
        return self._today_bar_count

    def should_allow_entry(self, timeframe: str) -> bool:
        """Decide whether to allow entries for the given timeframe.

        TRENDING: all strategies allowed
        NEUTRAL:  only H1 strategies (Keltner/ORB), skip M15 RSI
        CHOPPY:   skip all new entries
        """
        if not config.INTRADAY_TREND_ENABLED:
            return True

        if self._regime == REGIME_TRENDING:
            return True
        elif self._regime == REGIME_NEUTRAL:
            return timeframe == 'H1'
        else:
            return False

    def status_line(self) -> str:
        """One-line status for logging."""
        return (f"trend_score={self._score:.2f} ({self._regime}) "
                f"bars={self._today_bar_count}")
