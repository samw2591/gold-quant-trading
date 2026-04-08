#!/usr/bin/env python3
"""
Strategy C: Multi-TF Trend Filter (多级趋势跟随)
===================================================
Core idea: Use Daily (D1) trend as a directional filter on top of H1 signals.
Only take H1 signals that align with the higher timeframe trend.

D1 trend determination (composite):
  - EMA_fast > EMA_slow => uptrend  (e.g. EMA20 > EMA50)
  - Close > EMA200 => long-term uptrend
  - ADX > 20 on D1 => trend has conviction

Entry:
  - D1 trend = UP => only BUY signals from H1
  - D1 trend = DOWN => only SELL signals from H1
  - D1 trend = FLAT => no trades (skip)

Grid search over: D1 EMA fast/slow, trend conviction filter, SL/TP
"""
import sys, os, time, itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, BacktestEngine
from backtest.runner import C12_KWARGS, run_kfold
import strategies.signals as signals_mod

_ORIGINAL_SCAN = signals_mod.scan_all_signals

OUTPUT_FILE = "strategy_c_output.txt"


class TeeOutput:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

print("=" * 70)
print("STRATEGY C: MULTI-TF TREND FILTER")
print(f"Started: {datetime.now()}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Build D1 (Daily) data from H1
# ═══════════════════════════════════════════════════════════════

def build_d1_from_h1(h1_df):
    """Aggregate H1 bars into daily OHLC + compute D1 indicators."""
    d1 = h1_df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum',
    }).dropna(subset=['Close'])

    for span in [10, 20, 50, 200]:
        d1[f'EMA{span}'] = d1['Close'].ewm(span=span, adjust=False).mean()

    delta = d1['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d1['RSI14'] = 100 - (100 / (1 + rs))

    tr = pd.DataFrame({
        'hl': d1['High'] - d1['Low'],
        'hc': (d1['High'] - d1['Close'].shift(1)).abs(),
        'lc': (d1['Low'] - d1['Close'].shift(1)).abs(),
    }).max(axis=1)
    d1['ATR14'] = tr.rolling(14).mean()

    plus_dm = d1['High'].diff().clip(lower=0)
    minus_dm = (-d1['Low'].diff()).clip(lower=0)
    plus_dm[d1['High'].diff() < (-d1['Low'].diff())] = 0
    minus_dm[(-d1['Low'].diff()) < d1['High'].diff()] = 0

    smoothed_tr = tr.rolling(14).sum()
    smoothed_tr = smoothed_tr.replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).sum() / smoothed_tr
    minus_di = 100 * minus_dm.rolling(14).sum() / smoothed_tr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    d1['ADX14'] = dx.rolling(14).mean()

    return d1


def get_d1_trend(d1_df, bar_date, fast=20, slow=50, adx_thresh=20):
    """Determine D1 trend for a given date using only past data."""
    mask = d1_df.index.date < bar_date
    if not mask.any():
        return 'FLAT'
    row = d1_df.loc[mask].iloc[-1]

    ema_fast = row.get(f'EMA{fast}', np.nan)
    ema_slow = row.get(f'EMA{slow}', np.nan)
    ema200 = row.get('EMA200', np.nan)
    adx = row.get('ADX14', np.nan)
    close = row.get('Close', np.nan)

    if pd.isna(ema_fast) or pd.isna(ema_slow):
        return 'FLAT'

    has_conviction = True if (not pd.isna(adx) and adx > adx_thresh) else False
    ema_bullish = ema_fast > ema_slow
    above_200 = close > ema200 if not pd.isna(ema200) else True

    if ema_bullish and above_200 and has_conviction:
        return 'UP'
    elif not ema_bullish and not above_200 and has_conviction:
        return 'DOWN'
    return 'FLAT'


# ═══════════════════════════════════════════════════════════════
# Patching: filter signals by D1 trend
# ═══════════════════════════════════════════════════════════════

_d1_cache = {}

def patch_scan_with_d1_filter(d1_df, fast=20, slow=50, adx_thresh=20, allow_flat=False):
    """Monkey-patch scan_all_signals to filter by D1 trend direction."""
    _d1_cache.clear()

    def patched_scan(df, timeframe='H1', h1_adx=None):
        signals = _ORIGINAL_SCAN(df, timeframe, h1_adx=h1_adx)
        if not signals or df is None or len(df) < 1:
            return signals

        bar_date = pd.Timestamp(df.index[-1]).date()
        if bar_date not in _d1_cache:
            _d1_cache[bar_date] = get_d1_trend(d1_df, bar_date, fast, slow, adx_thresh)
        trend = _d1_cache[bar_date]

        if trend == 'FLAT' and not allow_flat:
            return []
        if trend == 'UP':
            return [s for s in signals if s['signal'] == 'BUY']
        elif trend == 'DOWN':
            return [s for s in signals if s['signal'] == 'SELL']
        return signals

    signals_mod.scan_all_signals = patched_scan


def restore_scan():
    signals_mod.scan_all_signals = _ORIGINAL_SCAN


# ═══════════════════════════════════════════════════════════════
# Run variant helper
# ═══════════════════════════════════════════════════════════════

def run_variant(data, label, verbose=True, **kwargs):
    from backtest.runner import run_variant as _rv
    return _rv(data, label, verbose=verbose, **kwargs)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
CURRENT = {**C12_KWARGS, "intraday_adaptive": True}

print(f"\n  Data: H1={len(data.h1_df):,}, M15={len(data.m15_df):,}")
print(f"  Range: {data.h1_df.index[0]} to {data.h1_df.index[-1]}")

d1_df = build_d1_from_h1(data.h1_df)
print(f"  D1 bars: {len(d1_df):,}")

print("\n--- Baseline (no D1 filter) ---")
restore_scan()
baseline = run_variant(data, "Baseline", **CURRENT)

# ═══════════════════════════════════════════════════════════════
# Grid Search
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: D1 TREND FILTER GRID SEARCH")
print("=" * 70)

FAST_EMAS = [10, 20]
SLOW_EMAS = [50, 100]
ADX_THRESHS = [15, 20, 25, 30]
ALLOW_FLATS = [False, True]

results = []
combos = list(itertools.product(FAST_EMAS, SLOW_EMAS, ADX_THRESHS, ALLOW_FLATS))
combos = [(f, s, a, af) for f, s, a, af in combos if f < s]
total = len(combos)
print(f"\n  Total combinations: {total}")

for i, (fast, slow, adx_t, allow_flat) in enumerate(combos, 1):
    label = f"D1({fast}/{slow}) ADX>{adx_t} flat={'Y' if allow_flat else 'N'}"
    _d1_cache.clear()
    patch_scan_with_d1_filter(d1_df, fast, slow, adx_t, allow_flat)
    stats = run_variant(data, label, verbose=False, **CURRENT)
    restore_scan()

    results.append({
        'label': label, 'fast': fast, 'slow': slow,
        'adx_thresh': adx_t, 'allow_flat': allow_flat,
        'n': stats['n'], 'sharpe': stats['sharpe'],
        'pnl': stats['total_pnl'], 'max_dd': stats['max_dd'],
        'win_rate': stats['win_rate'],
    })
    if i % 5 == 0 or i == total:
        print(f"  [{i}/{total}] {label}: N={stats['n']} Sharpe={stats['sharpe']:.2f} PnL=${stats['total_pnl']:,.0f}")

results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n--- All Results Ranked ---")
print(f"  {'Label':<45} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6}")
for r in results:
    print(f"  {r['label']:<45} {r['n']:>6} {r['sharpe']:>7.2f} ${r['pnl']:>9,.0f} ${r['max_dd']:>7,.0f} {r['win_rate']*100:>5.1f}%")

print(f"\n  Baseline: N={baseline['n']} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

# ═══════════════════════════════════════════════════════════════
# K-Fold on Top 3
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: 6-FOLD K-FOLD ON TOP 3 CONFIGS")
print("=" * 70)

# Baseline K-Fold for comparison
print("\n--- Baseline K-Fold ---")
restore_scan()
bl_folds = run_kfold(data, engine_kwargs=CURRENT, n_folds=6)
if bl_folds:
    bl_sharpes = [f['sharpe'] for f in bl_folds]
    print(f"  Baseline Avg Sharpe={np.mean(bl_sharpes):.2f} Std={np.std(bl_sharpes):.2f}")

for rank, r in enumerate(results[:3], 1):
    fast, slow, adx_t, allow_flat = r['fast'], r['slow'], r['adx_thresh'], r['allow_flat']
    print(f"\n--- #{rank}: {r['label']} (Full Sharpe={r['sharpe']:.2f}) ---")

    _d1_cache.clear()
    patch_scan_with_d1_filter(d1_df, fast, slow, adx_t, allow_flat)
    folds = run_kfold(data, engine_kwargs=CURRENT, n_folds=6)
    restore_scan()

    if folds:
        sharpes = [f['sharpe'] for f in folds]
        print(f"  Avg Sharpe={np.mean(sharpes):.2f} Std={np.std(sharpes):.2f}")
        print(f"  Folds: {['%.2f'%s for s in sharpes]}")
        print(f"  Positive: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")
        if bl_folds:
            delta = np.mean(sharpes) - np.mean(bl_sharpes)
            print(f"  vs Baseline: Delta Sharpe = {delta:+.3f}")

# ═══════════════════════════════════════════════════════════════
# D1 Trend Distribution
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: D1 TREND DISTRIBUTION ANALYSIS")
print("=" * 70)

if results:
    best = results[0]
    fast, slow, adx_t = best['fast'], best['slow'], best['adx_thresh']
    dates = sorted(set(data.h1_df.index.date))
    trends = {d: get_d1_trend(d1_df, d, fast, slow, adx_t) for d in dates}
    up_pct = sum(1 for v in trends.values() if v == 'UP') / len(trends)
    dn_pct = sum(1 for v in trends.values() if v == 'DOWN') / len(trends)
    fl_pct = sum(1 for v in trends.values() if v == 'FLAT') / len(trends)
    print(f"  D1 Trend ({fast}/{slow}, ADX>{adx_t}):")
    print(f"    UP:   {up_pct*100:.1f}% ({sum(1 for v in trends.values() if v == 'UP')} days)")
    print(f"    DOWN: {dn_pct*100:.1f}% ({sum(1 for v in trends.values() if v == 'DOWN')} days)")
    print(f"    FLAT: {fl_pct*100:.1f}% ({sum(1 for v in trends.values() if v == 'FLAT')} days)")

elapsed = time.time() - t_total
print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("=" * 70)
tee.close()
