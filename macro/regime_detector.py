"""
Macro regime detection for gold (XAUUSD) — rule-based, aligned with project macro framework.

Uses daily factors (VIX, DXY, TIPS, curve, breakeven inflation). For backtests, pair with
`macro_history.csv` rows; optional column `spread_2_10_chg` can be injected (see
`add_regime_column`).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from macro.data_provider import MacroSnapshot

logger = logging.getLogger(__name__)


class MacroRegime(str, Enum):
    EASING_INFLATION_UP = "easing_inflation_up"
    EASING_INFLATION_DOWN = "easing_inflation_down"
    TIGHTENING_INFLATION_UP = "tightening_inflation_up"
    TIGHTENING_INFLATION_DOWN = "tightening_inflation_down"
    RISK_OFF = "risk_off"
    LIQUIDITY_CRISIS = "liquidity_crisis"


def _finite(x: Any) -> bool:
    if x is None:
        return False
    try:
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return False
        if isinstance(x, (int, float, np.integer, np.floating)):
            return np.isfinite(float(x))
        return bool(pd.notna(x))
    except (TypeError, ValueError):
        return False


def _to_float(x: Any) -> Optional[float]:
    if not _finite(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class MacroRegimeDetector:
    """Classify macro regime from a live snapshot or a historical DataFrame row."""

    def detect_from_snapshot(self, snapshot: MacroSnapshot) -> MacroRegime:
        """Detect regime from live snapshot."""
        row = pd.Series(
            {
                "dxy": snapshot.dxy,
                "vix": snapshot.vix,
                "tips_10y": snapshot.tips_10y,
                "us10y": snapshot.us10y,
                "spread_2_10": snapshot.spread_2_10,
                "bei_5y": snapshot.bei_5y,
                "dxy_sma20": snapshot.dxy_sma20,
                "vix_percentile": snapshot.vix_percentile,
                "tips_10y_sma20": getattr(snapshot, "tips_10y_sma20", None),
                "bei_5y_sma20": getattr(snapshot, "bei_5y_sma20", None),
                "vix_regime": snapshot.vix_regime,
                "real_rate_regime": snapshot.real_rate_regime,
                "dxy_pct_change": snapshot.dxy_change_pct,
                "spread_2_10_chg": getattr(snapshot, "spread_2_10_chg", None),
            }
        )
        regime = self._detect_core(row)
        logger.info("[宏观Regime] snapshot → %s", regime.value)
        return regime

    def detect_from_row(self, row: pd.Series) -> MacroRegime:
        """Detect regime from a DataFrame row (for backtesting).

        Expected columns (subset may be present): dxy, vix, tips_10y, us10y, spread_2_10,
        bei_5y, dxy_sma20, vix_percentile, tips_10y_sma20, bei_5y_sma20, vix_regime,
        real_rate_regime, dxy_pct_change, spread_2_10_chg (optional; see add_regime_column).
        """
        regime = self._detect_core(row)
        logger.debug("[宏观Regime] row %s → %s", row.name, regime.value)
        return regime

    def _detect_core(self, row: pd.Series) -> MacroRegime:
        if self._is_liquidity_crisis(row):
            return MacroRegime.LIQUIDITY_CRISIS
        if self._is_risk_off(row):
            return MacroRegime.RISK_OFF

        monetary = self._monetary_stance(row)
        inflation = self._inflation_stance(row)

        if monetary is None and inflation is None:
            logger.debug(
                "[宏观Regime] insufficient monetary+inflation data → TIGHTENING_INFLATION_DOWN"
            )
            return MacroRegime.TIGHTENING_INFLATION_DOWN
        if monetary is None:
            monetary = "tightening"
        if inflation is None:
            inflation = "down"

        if monetary == "easing" and inflation == "up":
            return MacroRegime.EASING_INFLATION_UP
        if monetary == "easing" and inflation == "down":
            return MacroRegime.EASING_INFLATION_DOWN
        if monetary == "tightening" and inflation == "up":
            return MacroRegime.TIGHTENING_INFLATION_UP
        if monetary == "tightening" and inflation == "down":
            return MacroRegime.TIGHTENING_INFLATION_DOWN

        return MacroRegime.TIGHTENING_INFLATION_DOWN

    def _is_liquidity_crisis(self, row: pd.Series) -> bool:
        vix = _to_float(row.get("vix"))
        if vix is None or vix <= 35:
            return False

        dxy_rising = self._dxy_rising(row)
        gold_pressure = self._gold_likely_dropping_proxy(row)
        if dxy_rising and gold_pressure:
            logger.info(
                "[宏观Regime] LIQUIDITY_CRISIS: VIX=%s>35, DXY rising, USD pressure on gold",
                vix,
            )
            return True
        return False

    def _is_risk_off(self, row: pd.Series) -> bool:
        vix = _to_float(row.get("vix"))
        vr = row.get("vix_regime")
        if isinstance(vr, str) and vr.strip().lower() == "panic":
            logger.info("[宏观Regime] RISK_OFF: vix_regime=panic")
            return True
        if vix is not None and vix > 30:
            logger.info("[宏观Regime] RISK_OFF: VIX=%s>30", vix)
            return True
        return False

    @staticmethod
    def _dxy_rising(row: pd.Series) -> bool:
        chg = _to_float(row.get("dxy_pct_change"))
        if chg is not None and chg > 0:
            return True
        dxy = _to_float(row.get("dxy"))
        sma = _to_float(row.get("dxy_sma20"))
        if dxy is not None and sma is not None and dxy > sma:
            return True
        return False

    @staticmethod
    def _gold_likely_dropping_proxy(row: pd.Series) -> bool:
        """No spot gold in macro row; proxy with USD strength vs prior / SMA."""
        chg = _to_float(row.get("dxy_pct_change"))
        if chg is not None and chg > 0:
            return True
        dxy = _to_float(row.get("dxy"))
        sma = _to_float(row.get("dxy_sma20"))
        if dxy is not None and sma is not None and dxy > sma:
            return True
        return False

    def _monetary_stance(self, row: pd.Series) -> Optional[str]:
        easing_votes = 0
        tight_votes = 0

        tips = _to_float(row.get("tips_10y"))
        tips_sma = _to_float(row.get("tips_10y_sma20"))
        if tips is not None and tips_sma is not None:
            if tips < tips_sma:
                easing_votes += 1
            elif tips > tips_sma:
                tight_votes += 1

        spread = _to_float(row.get("spread_2_10"))
        spread_chg = _to_float(row.get("spread_2_10_chg"))
        if spread_chg is not None:
            if spread_chg > 0:
                easing_votes += 1
            elif spread_chg < 0:
                tight_votes += 1
        if spread is not None and spread < 0:
            tight_votes += 1

        if easing_votes == 0 and tight_votes == 0:
            rr = row.get("real_rate_regime")
            if isinstance(rr, str):
                r = rr.strip().lower()
                if r in ("negative", "low"):
                    easing_votes += 1
                elif r in ("moderate", "high"):
                    tight_votes += 1

        if easing_votes > tight_votes:
            return "easing"
        if tight_votes > easing_votes:
            return "tightening"
        return None

    @staticmethod
    def _inflation_stance(row: pd.Series) -> Optional[str]:
        bei = _to_float(row.get("bei_5y"))
        bei_sma = _to_float(row.get("bei_5y_sma20"))
        if bei is None or bei_sma is None:
            return None
        if bei > bei_sma:
            return "up"
        if bei < bei_sma:
            return "down"
        return None

    def get_strategy_weights(self, regime: MacroRegime) -> Dict[str, Union[bool, float, str]]:
        """Return strategy weight adjustments for the given regime."""
        base: Dict[str, Union[bool, float, str]] = {
            "allow_trading": True,
            "lot_multiplier": 1.0,
            "sell_enabled": True,
            "buy_enabled": True,
            "expand_sl": False,
            "regime_label": "",
            "regime_bias": "neutral",
        }

        if regime == MacroRegime.EASING_INFLATION_UP:
            base.update(
                {
                    "lot_multiplier": 1.2,
                    "sell_enabled": False,
                    "buy_enabled": True,
                    "expand_sl": False,
                    "regime_label": "宽松+通胀上升（强多黄金）",
                    "regime_bias": "bullish",
                }
            )
        elif regime == MacroRegime.EASING_INFLATION_DOWN:
            base.update(
                {
                    "lot_multiplier": 1.0,
                    "sell_enabled": True,
                    "buy_enabled": True,
                    "expand_sl": False,
                    "regime_label": "宽松+通胀下降（温和偏多）",
                    "regime_bias": "bullish",
                }
            )
        elif regime == MacroRegime.TIGHTENING_INFLATION_UP:
            base.update(
                {
                    "lot_multiplier": 0.7,
                    "sell_enabled": True,
                    "buy_enabled": True,
                    "expand_sl": False,
                    "regime_label": "紧缩+通胀上升（震荡偏空）",
                    "regime_bias": "neutral",
                }
            )
        elif regime == MacroRegime.TIGHTENING_INFLATION_DOWN:
            base.update(
                {
                    "lot_multiplier": 0.5,
                    "sell_enabled": True,
                    "buy_enabled": True,
                    "expand_sl": False,
                    "regime_label": "紧缩+通胀下降（偏空，做多谨慎）",
                    "regime_bias": "bearish",
                }
            )
        elif regime == MacroRegime.RISK_OFF:
            base.update(
                {
                    "lot_multiplier": 0.8,
                    "sell_enabled": False,
                    "buy_enabled": True,
                    "expand_sl": True,
                    "regime_label": "避险爆发（脉冲上涨，扩止损）",
                    "regime_bias": "bullish",
                }
            )
        elif regime == MacroRegime.LIQUIDITY_CRISIS:
            base.update(
                {
                    "allow_trading": False,
                    "lot_multiplier": 0.3,
                    "sell_enabled": False,
                    "buy_enabled": False,
                    "expand_sl": True,
                    "regime_label": "流动性危机（暂停交易）",
                    "regime_bias": "neutral",
                }
            )

        return base


def add_regime_column(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Add 'macro_regime' and 'regime_weights' columns to macro history DataFrame."""
    if macro_df.empty:
        logger.warning("[宏观Regime] add_regime_column: empty DataFrame")
        return macro_df

    detector = MacroRegimeDetector()
    out = macro_df.copy()

    spread_chg: Optional[pd.Series] = None
    if "spread_2_10" in out.columns:
        spread_chg = out["spread_2_10"].diff()

    regimes: list[str] = []
    weights: list[Dict[str, Union[bool, float, str]]] = []

    for i in range(len(out)):
        row = out.iloc[i].copy()
        if spread_chg is not None:
            row["spread_2_10_chg"] = spread_chg.iloc[i]
        reg = detector.detect_from_row(row)
        regimes.append(reg.value)
        weights.append(detector.get_strategy_weights(reg))

    out["macro_regime"] = regimes
    out["regime_weights"] = weights
    logger.info("[宏观Regime] classified %s rows", len(out))
    return out
