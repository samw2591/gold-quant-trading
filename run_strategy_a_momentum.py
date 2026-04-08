#!/usr/bin/env python3
"""
Strategy A: Momentum Chase (动量追击)
======================================
Core idea: Enter when price moves aggressively in a short window,
WITHOUT waiting for ADX/EMA confirmation. Captures the fast moves
that Keltner misses because its indicators lag.

Entry (BUY):
  - Cumulative move over last N bars > K * ATR (upward)
  - Close > EMA100 (basic direction alignment)
Entry (SELL): mirror conditions

Exit: ATR-based SL + trailing stop (same framework as Keltner)

Grid search over: lookback_bars, atr_mult_threshold, sl_atr_mult, max_hold
Top-3 configs validated with 6-fold K-Fold.
"""
import sys, os, time, itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold
import strategies.signals as signals_mod

OUTPUT_FILE = "strategy_a_output.txt"


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
print("STRATEGY A: MOMENTUM CHASE")
print(f"Started: {datetime.now()}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# Momentum signal generator
# ═══════════════════════════════════════════════════════════════

def make_momentum_signal_func(lookback=3, atr_mult=2.0, sl_atr=4.5, tp_atr=8.0):
    """Factory: returns a signal check function for momentum chase."""
    _original_scan = signals_mod.scan_all_signals.__wrapped__ if hasattr(signals_mod.scan_all_signals, '__wrapped__') else None

    def check_momentum(df):
        if df is None or len(df) < max(lookback + 1, 15):
            return None
        latest = df.iloc[-1]
        close = float(latest['Close'])
        atr = float(latest.get('ATR', 0))
        ema100 = float(latest.get('EMA100', 0))

        if atr <= 0 or pd.isna(atr) or ema100 <= 0 or pd.isna(ema100):
            return None

        closes = df['Close'].iloc[-(lookback + 1):]
        if len(closes) < lookback + 1:
            return None

        move = float(closes.iloc[-1]) - float(closes.iloc[0])
        threshold = atr_mult * atr

        sl = round(atr * sl_atr, 2)
        sl = max(10, min(50, sl))
        tp = round(atr * tp_atr, 2)

        if move > threshold and close > ema100:
            return {
                'strategy': 'momentum_chase',
                'signal': 'BUY',
                'reason': f"MomChase BUY: move={move:.1f} > {threshold:.1f} ({lookback}bar x {atr_mult}ATR)",
                'close': close,
                'sl': sl,
                'tp': tp,
            }
        elif move < -threshold and close < ema100:
            return {
                'strategy': 'momentum_chase',
                'signal': 'SELL',
                'reason': f"MomChase SELL: move={move:.1f} < -{threshold:.1f} ({lookback}bar x {atr_mult}ATR)",
                'close': close,
                'sl': sl,
                'tp': tp,
            }
        return None

    return check_momentum


def patch_scan_with_momentum(lookback=3, atr_mult=2.0, sl_atr=4.5, tp_atr=8.0):
    """Monkey-patch scan_all_signals to also check momentum."""
    original = signals_mod._original_scan_all if hasattr(signals_mod, '_original_scan_all') else signals_mod.scan_all_signals
    signals_mod._original_scan_all = original
    check_fn = make_momentum_signal_func(lookback, atr_mult, sl_atr, tp_atr)

    def patched_scan(df, timeframe='H1', h1_adx=None):
        signals = original(df, timeframe, h1_adx=h1_adx)
        if timeframe == 'H1':
            sig = check_fn(df)
            if sig:
                signals.append(sig)
        return signals

    signals_mod.scan_all_signals = patched_scan


def restore_scan():
    if hasattr(signals_mod, '_original_scan_all'):
        signals_mod.scan_all_signals = signals_mod._original_scan_all


# ═══════════════════════════════════════════════════════════════
# Data & Baseline
# ═══════════════════════════════════════════════════════════════
t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
CURRENT = {**C12_KWARGS, "intraday_adaptive": True}

print(f"\n  Data: H1={len(data.h1_df):,}, M15={len(data.m15_df):,}")
print(f"  Range: {data.h1_df.index[0]} to {data.h1_df.index[-1]}")

print("\n--- Baseline (Keltner only) ---")
restore_scan()
baseline = run_variant(data, "Baseline", **CURRENT)

# ═══════════════════════════════════════════════════════════════
# Grid Search
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 1: PARAMETER GRID SEARCH")
print("=" * 70)

LOOKBACKS = [2, 3, 4, 5]
ATR_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
SL_ATRS = [3.0, 4.5, 6.0]
MAX_HOLDS = [8, 12, 16]

results = []
total_combos = len(LOOKBACKS) * len(ATR_MULTS) * len(SL_ATRS) * len(MAX_HOLDS)
print(f"\n  Total combinations: {total_combos}")

i = 0
for lb, am, sl_a, mh in itertools.product(LOOKBACKS, ATR_MULTS, SL_ATRS, MAX_HOLDS):
    i += 1
    label = f"lb={lb}/am={am}/sl={sl_a}/mh={mh}"
    patch_scan_with_momentum(lookback=lb, atr_mult=am, sl_atr=sl_a, tp_atr=sl_a * 1.8)

    kwargs = {
        **CURRENT,
        "keltner_max_hold_m15": mh * 4,
    }
    stats = run_variant(data, label, verbose=False, **kwargs)
    restore_scan()

    results.append({
        'label': label, 'lookback': lb, 'atr_mult': am,
        'sl_atr': sl_a, 'max_hold': mh,
        'n': stats['n'], 'sharpe': stats['sharpe'],
        'pnl': stats['total_pnl'], 'max_dd': stats['max_dd'],
        'win_rate': stats['win_rate'],
    })
    if i % 20 == 0 or i == total_combos:
        print(f"  [{i}/{total_combos}] {label}: N={stats['n']} Sharpe={stats['sharpe']:.2f} PnL=${stats['total_pnl']:,.0f}")

# Sort by Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n--- Top 10 by Sharpe ---")
print(f"  {'Label':<35} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6}")
for r in results[:10]:
    print(f"  {r['label']:<35} {r['n']:>6} {r['sharpe']:>7.2f} ${r['pnl']:>9,.0f} ${r['max_dd']:>7,.0f} {r['win_rate']*100:>5.1f}%")

print(f"\n  Baseline: N={baseline['n']} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

# ═══════════════════════════════════════════════════════════════
# K-Fold on Top 3
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: 6-FOLD K-FOLD ON TOP 3 CONFIGS")
print("=" * 70)

for rank, r in enumerate(results[:3], 1):
    lb, am, sl_a, mh = r['lookback'], r['atr_mult'], r['sl_atr'], r['max_hold']
    print(f"\n--- #{rank}: {r['label']} (Full Sharpe={r['sharpe']:.2f}) ---")

    patch_scan_with_momentum(lookback=lb, atr_mult=am, sl_atr=sl_a, tp_atr=sl_a * 1.8)
    kwargs = {**CURRENT, "keltner_max_hold_m15": mh * 4}
    folds = run_kfold(data, engine_kwargs=kwargs, n_folds=6)
    restore_scan()

    if folds:
        sharpes = [f['sharpe'] for f in folds]
        print(f"  Avg Sharpe={np.mean(sharpes):.2f} Std={np.std(sharpes):.2f}")
        print(f"  Folds: {['%.2f'%s for s in sharpes]}")
        print(f"  Positive: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

# ═══════════════════════════════════════════════════════════════
# Momentum-only analysis (without Keltner)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: MOMENTUM-ONLY (disable Keltner) - Top config")
print("=" * 70)

if results:
    best = results[0]
    lb, am, sl_a, mh = best['lookback'], best['atr_mult'], best['sl_atr'], best['max_hold']

    check_fn = make_momentum_signal_func(lb, am, sl_a, sl_a * 1.8)

    def mom_only_scan(df, timeframe='H1', h1_adx=None):
        signals = []
        if timeframe == 'H1':
            sig = check_fn(df)
            if sig:
                signals.append(sig)
        return signals

    signals_mod.scan_all_signals = mom_only_scan
    stats = run_variant(data, f"Mom-Only ({best['label']})", **CURRENT)
    restore_scan()

    print(f"  Momentum-only: N={stats['n']} Sharpe={stats['sharpe']:.2f} PnL=${stats['total_pnl']:,.0f}")
    print(f"  Keltner-only:  N={baseline['n']} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")
    print(f"  Combined:      N={best['n']} Sharpe={best['sharpe']:.2f} PnL=${best['pnl']:,.0f}")

elapsed = time.time() - t_total
print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("=" * 70)
tee.close()
