#!/usr/bin/env python3
"""
Strategy D: Trend Pullback Entry (趋势回调入场)
=================================================
Core idea: Wait for a strong trend to form, then enter on a pullback
(retracement) rather than at the breakout. Captures better risk/reward
by entering after the initial impulse wave pulls back.

Setup conditions (BUY example):
  1. Trend: Close > EMA100 (bullish bias)
  2. Impulse: Recent strong upward move (N bars cumulative move > K*ATR)
  3. Pullback: RSI drops below threshold OR price retraces to EMA support
  4. Entry: When pullback condition triggers

Mirror for SELL.

Grid search: impulse lookback/threshold, pullback RSI, pullback EMA,
             SL/TP multipliers.
"""
import sys, os, time, itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle
from backtest.runner import C12_KWARGS, run_kfold, run_variant
import strategies.signals as signals_mod

_ORIGINAL_SCAN = signals_mod.scan_all_signals

OUTPUT_FILE = "strategy_d_output.txt"


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
print("STRATEGY D: TREND PULLBACK ENTRY")
print(f"Started: {datetime.now()}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# Pullback signal generator
# ═══════════════════════════════════════════════════════════════

def make_pullback_signal_func(
    impulse_lookback=10, impulse_atr_mult=2.5,
    pullback_rsi=40, pullback_ema=21,
    sl_atr=4.5, tp_atr=8.0
):
    """Factory: returns a signal check for trend-pullback entry."""

    def check_pullback(df):
        if df is None or len(df) < max(impulse_lookback + 5, 30):
            return None

        latest = df.iloc[-1]
        close = float(latest['Close'])
        atr = float(latest.get('ATR', 0))
        ema100 = float(latest.get('EMA100', 0))
        rsi = float(latest.get('RSI2', latest.get('RSI14', np.nan)))

        ema_key = f'EMA{pullback_ema}'
        if ema_key in latest:
            ema_pb = float(latest[ema_key])
        else:
            ema_pb = float(df['Close'].ewm(span=pullback_ema, adjust=False).iloc[-1])

        if atr <= 0 or pd.isna(atr) or pd.isna(ema100):
            return None

        closes = df['Close'].values
        n = len(closes)

        peak_move = max(closes[n - impulse_lookback - 1 : n - 1]) - closes[n - impulse_lookback - 1]
        trough_move = closes[n - impulse_lookback - 1] - min(closes[n - impulse_lookback - 1 : n - 1])
        impulse_threshold = impulse_atr_mult * atr

        sl = round(atr * sl_atr, 2)
        sl = max(10, min(50, sl))
        tp = round(atr * tp_atr, 2)

        # BUY: was in uptrend + now pulling back
        if (peak_move > impulse_threshold and
            close > ema100 and
            (rsi < pullback_rsi or close <= ema_pb * 1.001)):
            return {
                'strategy': 'trend_pullback',
                'signal': 'BUY',
                'reason': f"Pullback BUY: impulse={peak_move:.1f}>{impulse_threshold:.1f}, RSI={rsi:.0f}<{pullback_rsi}, pb_ema={ema_pb:.1f}",
                'close': close,
                'sl': sl,
                'tp': tp,
            }

        # SELL: was in downtrend + now pulling back up
        if (trough_move > impulse_threshold and
            close < ema100 and
            (rsi > (100 - pullback_rsi) or close >= ema_pb * 0.999)):
            return {
                'strategy': 'trend_pullback',
                'signal': 'SELL',
                'reason': f"Pullback SELL: impulse={trough_move:.1f}>{impulse_threshold:.1f}, RSI={rsi:.0f}>{100-pullback_rsi}, pb_ema={ema_pb:.1f}",
                'close': close,
                'sl': sl,
                'tp': tp,
            }

        return None

    return check_pullback


def patch_scan_with_pullback(**params):
    """Monkey-patch scan_all_signals to include pullback entries."""
    check_fn = make_pullback_signal_func(**params)

    def patched_scan(df, timeframe='H1', h1_adx=None):
        signals = _ORIGINAL_SCAN(df, timeframe, h1_adx=h1_adx)
        if timeframe == 'H1':
            sig = check_fn(df)
            if sig:
                signals.append(sig)
        return signals

    signals_mod.scan_all_signals = patched_scan


def restore_scan():
    signals_mod.scan_all_signals = _ORIGINAL_SCAN


# ═══════════════════════════════════════════════════════════════
# Main
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
print("PART 1: PULLBACK PARAMETER GRID SEARCH")
print("=" * 70)

IMPULSE_LB = [5, 8, 10, 15]
IMPULSE_ATR = [1.5, 2.0, 2.5, 3.0]
PB_RSI = [30, 35, 40, 45]
PB_EMA = [9, 21]
SL_ATRS = [3.0, 4.5, 6.0]

results = []
combos = list(itertools.product(IMPULSE_LB, IMPULSE_ATR, PB_RSI, PB_EMA, SL_ATRS))
total = len(combos)
print(f"\n  Total combinations: {total}")

i = 0
for ilb, iam, pr, pe, sla in combos:
    i += 1
    label = f"ilb={ilb}/iam={iam}/rsi={pr}/ema={pe}/sl={sla}"
    patch_scan_with_pullback(
        impulse_lookback=ilb, impulse_atr_mult=iam,
        pullback_rsi=pr, pullback_ema=pe,
        sl_atr=sla, tp_atr=sla * 1.8,
    )
    stats = run_variant(data, label, verbose=False, **CURRENT)
    restore_scan()

    results.append({
        'label': label,
        'impulse_lb': ilb, 'impulse_atr': iam,
        'pb_rsi': pr, 'pb_ema': pe, 'sl_atr': sla,
        'n': stats['n'], 'sharpe': stats['sharpe'],
        'pnl': stats['total_pnl'], 'max_dd': stats['max_dd'],
        'win_rate': stats['win_rate'],
    })
    if i % 30 == 0 or i == total:
        print(f"  [{i}/{total}] {label}: N={stats['n']} Sharpe={stats['sharpe']:.2f} PnL=${stats['total_pnl']:,.0f}")

results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n--- Top 15 by Sharpe ---")
print(f"  {'Label':<50} {'N':>6} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6}")
for r in results[:15]:
    print(f"  {r['label']:<50} {r['n']:>6} {r['sharpe']:>7.2f} ${r['pnl']:>9,.0f} ${r['max_dd']:>7,.0f} {r['win_rate']*100:>5.1f}%")

print(f"\n  Baseline: N={baseline['n']} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")

# ═══════════════════════════════════════════════════════════════
# K-Fold on Top 3
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: 6-FOLD K-FOLD ON TOP 3 CONFIGS")
print("=" * 70)

# Baseline K-Fold
print("\n--- Baseline K-Fold ---")
restore_scan()
bl_folds = run_kfold(data, engine_kwargs=CURRENT, n_folds=6)
if bl_folds:
    bl_sharpes = [f['sharpe'] for f in bl_folds]
    print(f"  Baseline Avg Sharpe={np.mean(bl_sharpes):.2f} Std={np.std(bl_sharpes):.2f}")

for rank, r in enumerate(results[:3], 1):
    print(f"\n--- #{rank}: {r['label']} (Full Sharpe={r['sharpe']:.2f}) ---")

    patch_scan_with_pullback(
        impulse_lookback=r['impulse_lb'], impulse_atr_mult=r['impulse_atr'],
        pullback_rsi=r['pb_rsi'], pullback_ema=r['pb_ema'],
        sl_atr=r['sl_atr'], tp_atr=r['sl_atr'] * 1.8,
    )
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
# Pullback-only analysis (without Keltner)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: PULLBACK-ONLY (disable Keltner) - Top config")
print("=" * 70)

if results:
    best = results[0]
    check_fn = make_pullback_signal_func(
        impulse_lookback=best['impulse_lb'],
        impulse_atr_mult=best['impulse_atr'],
        pullback_rsi=best['pb_rsi'],
        pullback_ema=best['pb_ema'],
        sl_atr=best['sl_atr'],
        tp_atr=best['sl_atr'] * 1.8,
    )

    def pb_only_scan(df, timeframe='H1', h1_adx=None):
        signals = []
        if timeframe == 'H1':
            sig = check_fn(df)
            if sig:
                signals.append(sig)
        return signals

    signals_mod.scan_all_signals = pb_only_scan
    stats = run_variant(data, f"Pullback-Only ({best['label']})", **CURRENT)
    restore_scan()

    print(f"  Pullback-only:  N={stats['n']} Sharpe={stats['sharpe']:.2f} PnL=${stats['total_pnl']:,.0f}")
    print(f"  Keltner-only:   N={baseline['n']} Sharpe={baseline['sharpe']:.2f} PnL=${baseline['total_pnl']:,.0f}")
    print(f"  Combined:       N={best['n']} Sharpe={best['sharpe']:.2f} PnL=${best['pnl']:,.0f}")

# ═══════════════════════════════════════════════════════════════
# Overlap Analysis
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: SIGNAL OVERLAP ANALYSIS")
print("=" * 70)

if results:
    best = results[0]
    check_fn = make_pullback_signal_func(
        impulse_lookback=best['impulse_lb'],
        impulse_atr_mult=best['impulse_atr'],
        pullback_rsi=best['pb_rsi'],
        pullback_ema=best['pb_ema'],
        sl_atr=best['sl_atr'],
        tp_atr=best['sl_atr'] * 1.8,
    )

    pb_signals = 0
    kc_signals = 0
    overlaps = 0

    for idx in range(100, len(data.h1_df)):
        window = data.h1_df.iloc[max(0, idx - 100):idx + 1]
        pb_sig = check_fn(window)
        kc_sigs = _ORIGINAL_SCAN(window, 'H1')
        kc_hit = any(s.get('strategy') == 'keltner_breakout' for s in kc_sigs) if kc_sigs else False

        if pb_sig:
            pb_signals += 1
        if kc_hit:
            kc_signals += 1
        if pb_sig and kc_hit:
            overlaps += 1

    overlap_rate = overlaps / max(pb_signals, 1) * 100
    print(f"  Pullback signals: {pb_signals}")
    print(f"  Keltner signals:  {kc_signals}")
    print(f"  Overlaps:         {overlaps} ({overlap_rate:.1f}% of pullback)")
    print(f"  New unique signals from pullback: {pb_signals - overlaps}")

elapsed = time.time() - t_total
print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
print("=" * 70)
tee.close()
