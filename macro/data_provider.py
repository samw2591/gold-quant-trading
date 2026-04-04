"""
Macro Data Provider
====================
Unified data pipeline for macro factors relevant to gold (XAUUSD) pricing.

Data sources:
  - yfinance (free, ~15min delay): DXY, VIX, Brent crude, US10Y
  - FRED API (free with key): TIPS 10Y, US2Y, US10Y, 2-10 spread, 5Y BEI

All factors are daily-frequency. For intraday use, the latest available
daily value is carried forward within the trading day.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# yfinance (lazy import for environments without it)
# ═══════════════════════════════════════════════════════════════

_YF_AVAILABLE = False
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# FRED (optional)
# ═══════════════════════════════════════════════════════════════

_FRED_AVAILABLE = False
try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════
# Ticker / Series definitions
# ═══════════════════════════════════════════════════════════════

YF_TICKERS = {
    "dxy":   "DX-Y.NYB",     # US Dollar Index
    "vix":   "^VIX",          # CBOE Volatility Index
    "brent": "BZ=F",          # Brent Crude Oil Futures
    "us10y": "^TNX",          # 10-Year Treasury Yield (×100 on yfinance)
}

FRED_SERIES = {
    "tips_10y":     "DFII10",    # 10-Year TIPS yield (real interest rate)
    "us2y":         "DGS2",      # 2-Year Treasury yield
    "us10y_fred":   "DGS10",     # 10-Year Treasury yield (official)
    "spread_2_10":  "T10Y2Y",    # 2Y-10Y spread (yield curve)
    "bei_5y":       "T5YIE",     # 5-Year breakeven inflation rate
}


@dataclass
class MacroSnapshot:
    """Point-in-time macro factor snapshot."""
    timestamp: str = ""

    # yfinance sources
    dxy: Optional[float] = None
    dxy_prev: Optional[float] = None
    dxy_change_pct: Optional[float] = None
    dxy_sma20: Optional[float] = None

    vix: Optional[float] = None
    vix_prev: Optional[float] = None
    vix_change_pct: Optional[float] = None
    vix_percentile: Optional[float] = None

    brent: Optional[float] = None
    brent_prev: Optional[float] = None
    brent_change_pct: Optional[float] = None

    us10y: Optional[float] = None
    us10y_prev: Optional[float] = None
    us10y_change_pct: Optional[float] = None

    # FRED sources
    tips_10y: Optional[float] = None
    us2y: Optional[float] = None
    spread_2_10: Optional[float] = None
    bei_5y: Optional[float] = None

    # Derived
    real_rate_regime: Optional[str] = None   # "negative" / "low" / "moderate" / "high"
    vix_regime: Optional[str] = None         # "low" / "normal" / "elevated" / "panic"

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class MacroDataProvider:
    """Fetches and caches daily macro factor data.

    Designed for two use cases:
      1. Live trading: get_snapshot() returns latest values with caching
      2. Backtesting: download_history() saves CSV for offline use
    """

    def __init__(self, fred_api_key: Optional[str] = None,
                 cache_ttl: int = 3600,
                 cache_path: Optional[str] = None):
        self._fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        self._fred: Optional['Fred'] = None
        self._cache_ttl = cache_ttl
        self._cache_path = Path(cache_path) if cache_path else None

        self._snapshot_cache: Optional[MacroSnapshot] = None
        self._cache_ts: float = 0.0

        # Historical data cache (for SMA / percentile calculations)
        self._history_cache: Optional[pd.DataFrame] = None
        self._history_cache_ts: float = 0.0

        if self._fred_api_key and _FRED_AVAILABLE:
            try:
                self._fred = Fred(api_key=self._fred_api_key)
                logger.info("[宏观数据] FRED API 已连接")
            except Exception as e:
                logger.warning(f"[宏观数据] FRED 初始化失败: {e}")

    # ── Live snapshot ─────────────────────────────────────────

    def get_snapshot(self) -> MacroSnapshot:
        """Get latest macro snapshot with caching."""
        now = datetime.now().timestamp()
        if self._snapshot_cache and (now - self._cache_ts) < self._cache_ttl:
            return self._snapshot_cache

        snapshot = self._build_snapshot()
        self._snapshot_cache = snapshot
        self._cache_ts = now

        if self._cache_path:
            self._save_snapshot_json(snapshot)

        return snapshot

    def _build_snapshot(self) -> MacroSnapshot:
        snap = MacroSnapshot(timestamp=datetime.utcnow().isoformat())

        # yfinance data (5-day window for prev close + change)
        yf_data = self._fetch_yf_recent()
        if yf_data:
            snap.dxy = yf_data.get("dxy")
            snap.dxy_prev = yf_data.get("dxy_prev")
            snap.dxy_change_pct = yf_data.get("dxy_change_pct")

            snap.vix = yf_data.get("vix")
            snap.vix_prev = yf_data.get("vix_prev")
            snap.vix_change_pct = yf_data.get("vix_change_pct")

            snap.brent = yf_data.get("brent")
            snap.brent_prev = yf_data.get("brent_prev")
            snap.brent_change_pct = yf_data.get("brent_change_pct")

            snap.us10y = yf_data.get("us10y")
            snap.us10y_prev = yf_data.get("us10y_prev")
            snap.us10y_change_pct = yf_data.get("us10y_change_pct")

        # DXY SMA20 + VIX percentile from recent history
        hist = self._get_recent_history(days=60)
        if hist is not None and not hist.empty:
            if "dxy" in hist.columns and snap.dxy is not None:
                dxy_series = hist["dxy"].dropna()
                if len(dxy_series) >= 20:
                    snap.dxy_sma20 = round(float(dxy_series.tail(20).mean()), 2)
            if "vix" in hist.columns and snap.vix is not None:
                vix_series = hist["vix"].dropna()
                if len(vix_series) >= 20:
                    snap.vix_percentile = round(
                        float((vix_series < snap.vix).sum() / len(vix_series)), 2
                    )

        # FRED data
        fred_data = self._fetch_fred_latest()
        if fred_data:
            snap.tips_10y = fred_data.get("tips_10y")
            snap.us2y = fred_data.get("us2y")
            snap.spread_2_10 = fred_data.get("spread_2_10")
            snap.bei_5y = fred_data.get("bei_5y")

        # Derived regimes
        snap.real_rate_regime = self._classify_real_rate(snap.tips_10y)
        snap.vix_regime = self._classify_vix(snap.vix)

        self._log_snapshot(snap)
        return snap

    # ── yfinance fetching ─────────────────────────────────────

    def _fetch_yf_recent(self) -> Optional[Dict]:
        """Fetch latest close + prev close for all yfinance tickers."""
        if not _YF_AVAILABLE:
            logger.debug("[宏观数据] yfinance 不可用")
            return None

        result = {}
        for key, ticker in YF_TICKERS.items():
            vals = self._fetch_single_yf(ticker, key)
            if vals:
                result.update(vals)

        return result if result else None

    def _fetch_single_yf(self, ticker: str, prefix: str) -> Optional[Dict]:
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            data = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if data.empty or len(data) < 2:
                return None

            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                data.columns = data.columns.droplevel(1)

            latest = float(data["Close"].iloc[-1])
            prev = float(data["Close"].iloc[-2])
            change = round((latest - prev) / prev * 100, 2) if prev != 0 else 0.0

            return {
                prefix: round(latest, 2),
                f"{prefix}_prev": round(prev, 2),
                f"{prefix}_change_pct": change,
            }
        except Exception as e:
            logger.debug(f"[宏观数据] yfinance {ticker} 获取失败: {e}")
            return None

    def _get_recent_history(self, days: int = 60) -> Optional[pd.DataFrame]:
        """Get recent daily history for derived indicators (SMA, percentile)."""
        now = datetime.now().timestamp()
        if (self._history_cache is not None
                and (now - self._history_cache_ts) < self._cache_ttl):
            return self._history_cache

        if not _YF_AVAILABLE:
            return None

        try:
            tickers_str = " ".join(YF_TICKERS.values())
            end = datetime.now()
            start = end - timedelta(days=days + 10)

            data = yf.download(
                tickers_str,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False, auto_adjust=True,
            )
            if data.empty:
                return None

            result = pd.DataFrame(index=data.index)
            for key, ticker in YF_TICKERS.items():
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker in data["Close"].columns:
                            result[key] = data["Close"][ticker]
                    else:
                        result[key] = data["Close"]
                except Exception:
                    pass

            self._history_cache = result
            self._history_cache_ts = now
            return result
        except Exception as e:
            logger.debug(f"[宏观数据] 历史数据获取失败: {e}")
            return None

    # ── FRED fetching ─────────────────────────────────────────

    def _fetch_fred_latest(self) -> Optional[Dict]:
        """Fetch latest values from FRED for all configured series."""
        if not self._fred:
            return None

        result = {}
        for key, series_id in FRED_SERIES.items():
            try:
                s = self._fred.get_series(series_id, observation_start=(
                    datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d"))
                if s is not None and len(s) > 0:
                    val = s.dropna().iloc[-1]
                    result[key] = round(float(val), 3)
            except Exception as e:
                logger.debug(f"[宏观数据] FRED {series_id} 获取失败: {e}")

        return result if result else None

    # ── History download (for backtesting) ────────────────────

    def download_history(self, start: str = "2015-01-01",
                         end: str = "2026-04-01",
                         save_path: str = "data/macro_history.csv") -> pd.DataFrame:
        """Download full macro history and save to CSV.

        Merges yfinance daily data (DXY, VIX, Brent, US10Y) with FRED
        daily data (TIPS, US2Y, spread, BEI) into a single DataFrame.

        Returns the merged DataFrame.
        """
        logger.info(f"[宏观数据] 下载历史: {start} → {end}")
        frames = {}

        # yfinance tickers
        if _YF_AVAILABLE:
            for key, ticker in YF_TICKERS.items():
                df = self._download_yf_series(ticker, key, start, end)
                if df is not None:
                    frames[key] = df

        # FRED series
        if self._fred:
            for key, series_id in FRED_SERIES.items():
                df = self._download_fred_series(series_id, key, start, end)
                if df is not None:
                    frames[key] = df

        if not frames:
            logger.error("[宏观数据] 没有成功下载任何数据")
            return pd.DataFrame()

        merged = self._merge_daily(frames, start, end)

        # Add derived columns
        if "dxy" in merged.columns:
            merged["dxy_sma20"] = merged["dxy"].rolling(20, min_periods=10).mean()
            merged["dxy_sma50"] = merged["dxy"].rolling(50, min_periods=25).mean()
            merged["dxy_pct_change"] = merged["dxy"].pct_change(fill_method=None) * 100

        if "vix" in merged.columns:
            merged["vix_sma20"] = merged["vix"].rolling(20, min_periods=10).mean()
            merged["vix_percentile"] = merged["vix"].rolling(252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

        if "tips_10y" in merged.columns:
            merged["tips_10y_sma20"] = merged["tips_10y"].rolling(20, min_periods=10).mean()
            merged["real_rate_regime"] = merged["tips_10y"].apply(self._classify_real_rate)

        if "vix" in merged.columns:
            merged["vix_regime"] = merged["vix"].apply(self._classify_vix)

        if "bei_5y" in merged.columns:
            merged["bei_5y_sma20"] = merged["bei_5y"].rolling(20, min_periods=10).mean()

        # Forward-fill NaN (weekends/holidays in FRED vs yfinance)
        merged = merged.ffill()

        # Save
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(save_p)
        logger.info(f"[宏观数据] 保存完成: {save_p} ({len(merged)} 行, "
                     f"{len(merged.columns)} 列)")

        # Summary
        self._print_download_summary(merged)

        return merged

    def _download_yf_series(self, ticker: str, name: str,
                            start: str, end: str) -> Optional[pd.Series]:
        try:
            data = yf.download(ticker, start=start, end=end,
                               progress=False, auto_adjust=True)
            if data.empty:
                logger.warning(f"[宏观数据] yfinance {ticker}: 无数据")
                return None

            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                data.columns = data.columns.droplevel(1)

            series = data["Close"].dropna()
            series.name = name
            logger.info(f"  {name} ({ticker}): {len(series)} 天, "
                        f"{series.index[0].date()} → {series.index[-1].date()}")
            return series
        except Exception as e:
            logger.warning(f"[宏观数据] yfinance {ticker} 下载失败: {e}")
            return None

    def _download_fred_series(self, series_id: str, name: str,
                              start: str, end: str) -> Optional[pd.Series]:
        try:
            series = self._fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            if series is None or series.empty:
                logger.warning(f"[宏观数据] FRED {series_id}: 无数据")
                return None

            series = series.dropna()
            series.name = name
            series.index = pd.DatetimeIndex(series.index)
            logger.info(f"  {name} ({series_id}): {len(series)} 天, "
                        f"{series.index[0].date()} → {series.index[-1].date()}")
            return series
        except Exception as e:
            logger.warning(f"[宏观数据] FRED {series_id} 下载失败: {e}")
            return None

    def _merge_daily(self, frames: Dict[str, pd.Series],
                     start: str, end: str) -> pd.DataFrame:
        """Merge all series onto a business-day index."""
        idx = pd.bdate_range(start=start, end=end, name="date")
        merged = pd.DataFrame(index=idx)

        for name, series in frames.items():
            s = series.copy()
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            merged[name] = s.reindex(merged.index)

        return merged

    # ── Regime classification ─────────────────────────────────

    @staticmethod
    def _classify_real_rate(tips_10y: Optional[float]) -> Optional[str]:
        """Classify TIPS 10Y into real rate regime.
        < 0%: negative (bullish gold)
        0-1%: low (neutral-bullish)
        1-2%: moderate (bearish gold)
        > 2%: high (strongly bearish gold unless risk-off overrides)
        """
        if tips_10y is None or (isinstance(tips_10y, float) and np.isnan(tips_10y)):
            return None
        if tips_10y < 0:
            return "negative"
        elif tips_10y < 1.0:
            return "low"
        elif tips_10y < 2.0:
            return "moderate"
        else:
            return "high"

    @staticmethod
    def _classify_vix(vix: Optional[float]) -> Optional[str]:
        """Classify VIX into volatility regime.
        < 15: low (risk-on, gold lacks haven premium)
        15-25: normal
        25-35: elevated (gold gains haven support)
        > 35: panic (caution: liquidity crunch may cause gold to drop first)
        """
        if vix is None or (isinstance(vix, float) and np.isnan(vix)):
            return None
        if vix < 15:
            return "low"
        elif vix < 25:
            return "normal"
        elif vix < 35:
            return "elevated"
        else:
            return "panic"

    # ── Helpers ────────────────────────────────────────────────

    def _save_snapshot_json(self, snap: MacroSnapshot):
        """Save snapshot to JSON cache file."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(snap.to_dict(), f, indent=2, default=str)
            tmp.replace(self._cache_path)
        except Exception as e:
            logger.debug(f"[宏观数据] JSON 保存失败: {e}")

    def _log_snapshot(self, snap: MacroSnapshot):
        parts = []
        if snap.dxy is not None:
            parts.append(f"DXY={snap.dxy}")
        if snap.vix is not None:
            parts.append(f"VIX={snap.vix}({snap.vix_regime})")
        if snap.tips_10y is not None:
            parts.append(f"TIPS={snap.tips_10y}%({snap.real_rate_regime})")
        if snap.brent is not None:
            parts.append(f"Brent=${snap.brent}")
        if snap.bei_5y is not None:
            parts.append(f"BEI5Y={snap.bei_5y}%")
        if snap.spread_2_10 is not None:
            parts.append(f"2-10={snap.spread_2_10}%")
        if parts:
            logger.info(f"[宏观数据] {' | '.join(parts)}")

    def _print_download_summary(self, df: pd.DataFrame):
        """Print summary of downloaded data."""
        print(f"\n  {'='*70}")
        print(f"  Macro History Download Summary")
        print(f"  {'='*70}")
        print(f"  Period: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} days)")
        print(f"  Columns: {len(df.columns)}")
        print()

        core_cols = ["dxy", "vix", "brent", "us10y", "tips_10y",
                     "us2y", "spread_2_10", "bei_5y"]
        print(f"  {'Factor':<15} {'Count':>6} {'First':>12} {'Last':>12} "
              f"{'Min':>8} {'Max':>8} {'Mean':>8}")
        print(f"  {'-'*15} {'-'*6} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

        for col in core_cols:
            if col not in df.columns:
                print(f"  {col:<15}   (not available)")
                continue
            s = df[col].dropna()
            if s.empty:
                print(f"  {col:<15}   (all NaN)")
                continue
            print(f"  {col:<15} {len(s):>6} {s.index[0].date()!s:>12} "
                  f"{s.index[-1].date()!s:>12} {s.min():>8.2f} "
                  f"{s.max():>8.2f} {s.mean():>8.2f}")

        print(f"\n  {'='*70}")


# ═══════════════════════════════════════════════════════════════
# Backtest helper: load macro CSV and align with H1 index
# ═══════════════════════════════════════════════════════════════

def load_macro_for_backtest(csv_path: str = "data/macro_history.csv") -> pd.DataFrame:
    """Load pre-downloaded macro CSV for backtest use.

    Returns a DataFrame indexed by date with all macro columns.
    To align with H1 bars: macro_row = macro_df.loc[bar_date]
    """
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    df = df.ffill()
    logger.info(f"[宏观数据] 回测数据加载: {len(df)} 天, "
                f"{len(df.columns)} 列, "
                f"{df.index[0].date()} → {df.index[-1].date()}")
    return df
