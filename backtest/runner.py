"""
Backtest Runner — Data Loading & Experiment Execution
======================================================
Eliminates the ~20 copies of "load data → prepare indicators → reset state → run"
scattered across experiment scripts.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from strategies.signals import prepare_indicators, get_orb_strategy, calc_rsi, calc_adx
import strategies.signals as signals_mod
from backtest.engine import BacktestEngine, TradeRecord
from backtest.stats import calc_stats


# ═══════════════════════════════════════════════════════════════
# Default data paths
# ═══════════════════════════════════════════════════════════════

M15_CSV_PATH = Path("data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv")
H1_CSV_PATH = Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv")


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_csv(path: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV CSV with timestamp(ms) format."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    df['is_flat'] = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])

    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]

    return df


def load_m15(csv_path: Path = M15_CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"M15 CSV not found: {csv_path}")
    df = load_csv(str(csv_path))
    print(f"  M15: {csv_path} ({len(df)} bars, {df.index[0]} -> {df.index[-1]})")
    return df


def load_h1_aligned(h1_path: Path, m15_start: pd.Timestamp) -> pd.DataFrame:
    df = load_csv(str(h1_path))
    warmup_start = m15_start - pd.Timedelta(hours=200)
    df = df[df.index >= warmup_start]
    print(f"  H1: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    return df


def add_atr_percentile(h1_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling ATR percentile column if not present."""
    if 'atr_percentile' not in h1_df.columns:
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).rank(pct=True)
        h1_df['atr_percentile'] = h1_df['atr_percentile'].fillna(0.5)
    return h1_df


def prepare_indicators_custom(df: pd.DataFrame, kc_ema=20, kc_mult=1.5, ema_trend=100) -> pd.DataFrame:
    """Recalculate indicators with custom Keltner/EMA parameters."""
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA100'] = df['Close'].ewm(span=ema_trend).mean()
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['KC_mid'] = df['Close'].ewm(span=kc_ema).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['RSI14'] = calc_rsi(df['Close'], 14)
    df['ADX'] = calc_adx(df, 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    return df


# ═══════════════════════════════════════════════════════════════
# Data bundle (load once, reuse across variants)
# ═══════════════════════════════════════════════════════════════

class DataBundle:
    """Pre-loaded and indicator-prepared data, ready for engine instantiation."""

    def __init__(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame):
        self.m15_df = m15_df
        self.h1_df = h1_df

    @classmethod
    def load_default(cls, start: str = "2015-01-01", end: Optional[str] = None) -> 'DataBundle':
        """Load with default indicators (prepare_indicators)."""
        print("\nLoading data...")
        m15_raw = load_m15()
        if start:
            m15_raw = m15_raw[m15_raw.index >= pd.Timestamp(start, tz='UTC')]
        if end:
            m15_raw = m15_raw[m15_raw.index <= pd.Timestamp(end, tz='UTC')]

        h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

        print("Preparing indicators...")
        print("  M15...", end='', flush=True)
        m15_df = prepare_indicators(m15_raw)
        print(" done")
        print("  H1...", end='', flush=True)
        h1_df = prepare_indicators(h1_raw)
        print(" done")
        h1_df = add_atr_percentile(h1_df)

        print(f"  M15: {len(m15_df)} bars, H1: {len(h1_df)} bars")
        return cls(m15_df, h1_df)

    @classmethod
    def load_custom(cls, kc_ema=20, kc_mult=1.5, ema_trend=100,
                    start: str = "2015-01-01", end: Optional[str] = None) -> 'DataBundle':
        """Load with custom indicator parameters."""
        print("\nLoading data...")
        m15_raw = load_m15()
        if start:
            m15_raw = m15_raw[m15_raw.index >= pd.Timestamp(start, tz='UTC')]
        if end:
            m15_raw = m15_raw[m15_raw.index <= pd.Timestamp(end, tz='UTC')]

        h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

        print(f"Preparing custom indicators (KC_ema={kc_ema}, KC_mult={kc_mult}, EMA={ema_trend})...")
        m15_df = prepare_indicators_custom(m15_raw, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=ema_trend)
        h1_df = prepare_indicators_custom(h1_raw, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=ema_trend)
        h1_df = add_atr_percentile(h1_df)

        print(f"  M15: {len(m15_df)} bars, H1: {len(h1_df)} bars")
        return cls(m15_df, h1_df)

    @classmethod
    def load_raw(cls, start: str = "2015-01-01", end: Optional[str] = None) -> 'DataBundle':
        """Load raw data without indicators (for custom prep later)."""
        print("\nLoading raw data...")
        m15_raw = load_m15()
        if start:
            m15_raw = m15_raw[m15_raw.index >= pd.Timestamp(start, tz='UTC')]
        if end:
            m15_raw = m15_raw[m15_raw.index <= pd.Timestamp(end, tz='UTC')]
        h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
        print(f"  M15: {len(m15_raw)} bars, H1: {len(h1_raw)} bars")
        return cls(m15_raw, h1_raw)

    def slice(self, start: str, end: str) -> 'DataBundle':
        """Return a time-windowed subset."""
        ts = pd.Timestamp(start, tz='UTC')
        te = pd.Timestamp(end, tz='UTC')
        m15 = self.m15_df[(self.m15_df.index >= ts) & (self.m15_df.index < te)]
        h1 = self.h1_df[(self.h1_df.index >= ts) & (self.h1_df.index < te)]
        return DataBundle(m15, h1)


# ═══════════════════════════════════════════════════════════════
# Run helpers
# ═══════════════════════════════════════════════════════════════

def run_variant(data: DataBundle, label: str, *, verbose: bool = True, **engine_kwargs) -> Dict:
    """Run a single backtest variant and return stats dict.

    Handles global state reset automatically.
    """
    if verbose:
        print(f"\n  [{label}]", flush=True)

    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    t0 = time.time()
    engine = BacktestEngine(data.m15_df, data.h1_df, label=label, **engine_kwargs)
    trades = engine.run()
    elapsed = time.time() - t0

    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['rsi_filtered'] = engine.rsi_filtered_count
    stats['rsi_total'] = engine.rsi_total_signals
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    stats['skipped_choppy'] = engine.skipped_choppy
    stats['skipped_neutral_m15'] = engine.skipped_neutral_m15
    stats['elapsed_s'] = round(elapsed, 1)
    stats['_trades'] = trades
    stats['_equity_curve'] = engine.equity_curve

    if verbose:
        print(f"    {stats['n']} trades (H1={engine.h1_entry_count}, M15={engine.m15_entry_count}), "
              f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, {elapsed:.0f}s")

    return stats


def run_variants(data: DataBundle, variants: List[Dict]) -> List[Dict]:
    """Run multiple variants sequentially.

    Each item in variants is a dict with 'label' + engine kwargs.
    Example:
        variants = [
            {"label": "Baseline"},
            {"label": "Trail 0.8/0.25", "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25},
        ]
    """
    results = []
    for i, v in enumerate(variants, 1):
        label = v.pop('label', f'V{i}')
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        stats = run_variant(data, label, **v)
        v['label'] = label  # restore
        results.append(stats)
    return results


def run_kfold(data: DataBundle, engine_kwargs: Dict, n_folds: int = 6,
              label_prefix: str = "") -> List[Dict]:
    """Run K-Fold cross validation with fixed time windows."""
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ][:n_folds]

    results = []
    for fold_name, start, end in folds:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        label = f"{label_prefix}{fold_name}" if label_prefix else fold_name
        stats = run_variant(fold_data, label, **engine_kwargs)
        stats['fold'] = fold_name
        stats['test_start'] = start
        stats['test_end'] = end
        results.append(stats)
    return results


# ═══════════════════════════════════════════════════════════════
# Common config presets
# ═══════════════════════════════════════════════════════════════

C12_KWARGS = {
    "trailing_activate_atr": 0.8,
    "trailing_distance_atr": 0.25,
    "sl_atr_mult": 4.5,
    "tp_atr_mult": 5.0,
    "keltner_adx_threshold": 18,
}

V3_REGIME = {
    'low': {'trail_act': 1.0, 'trail_dist': 0.35},
    'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
    'high': {'trail_act': 0.6, 'trail_dist': 0.20},
}

TRUE_BASELINE_KWARGS = {
    "trailing_activate_atr": 1.5,
    "trailing_distance_atr": 0.5,
    "sl_atr_mult": 2.5,
    "tp_atr_mult": 3.0,
    "keltner_adx_threshold": 24,
}


# ═══════════════════════════════════════════════════════════════
# JSON serialization helper
# ═══════════════════════════════════════════════════════════════

def sanitize_for_json(results: List[Dict]) -> List[Dict]:
    """Convert numpy types and drop non-serializable fields for JSON output."""
    safe = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
            else:
                sr[k] = v
        safe.append(sr)
    return safe
