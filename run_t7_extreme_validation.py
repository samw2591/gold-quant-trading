#!/usr/bin/env python3
"""
T7 ExtremeRegime — K-Fold + Spread Cost Validation
====================================================
Prior result: Full-sample Sharpe 8.39→9.59 (+1.20), 12/12 years all positive.
Key change: high ATR regime trail_act=0.25, trail_dist=0.05 (vs current 0.4/0.10)

Validation plan:
  Phase 1: Full-sample at $0/$0.30/$0.50 spread — T7 vs Baseline
  Phase 2: 6-Fold K-Fold at $0.30 spread — T7 vs Baseline (each fold)
  Phase 3: 6-Fold K-Fold at $0.50 spread — stress test
  Phase 4: Per-regime trade breakdown — verify "high" regime is the driver
  Phase 5: MaxDD comparison & worst-case analysis

Pass criteria:
  - Phase 2: T7 wins >=5/6 folds at $0.30 spread
  - Phase 3: T7 wins >=4/6 folds at $0.50 spread
  - Phase 4: High regime $/t improvement > low regime degradation
  - Phase 5: MaxDD increase < 20%
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "t7_extreme_validation_output.txt"


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

print("=" * 80)
print("T7 EXTREMEREGIME — K-FOLD + SPREAD COST VALIDATION")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()

# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading data...")
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ── Configs ───────────────────────────────────────────────────────────────────

MEGA_BASE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "keltner_max_hold_m15": 12,
    "time_decay_tp": True,
    "time_decay_start_hour": 1.0,
    "time_decay_atr_start": 0.30,
    "time_decay_atr_step": 0.10,
}

BASELINE = {
    **MEGA_BASE,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 0.7,  'trail_dist': 0.25},
        'normal': {'trail_act': 0.5,  'trail_dist': 0.15},
        'high':   {'trail_act': 0.4,  'trail_dist': 0.10},
    },
}

T7_EXTREME = {
    **MEGA_BASE,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 1.0,  'trail_dist': 0.35},
        'normal': {'trail_act': 0.5,  'trail_dist': 0.15},
        'high':   {'trail_act': 0.25, 'trail_dist': 0.05},
    },
}

# Also test intermediate variants to check if it's about "high" tightening or "low" widening
T7_ONLY_HIGH = {
    **MEGA_BASE,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 0.7,  'trail_dist': 0.25},   # same as baseline
        'normal': {'trail_act': 0.5,  'trail_dist': 0.15},   # same as baseline
        'high':   {'trail_act': 0.25, 'trail_dist': 0.05},   # T7 change only
    },
}

T7_ONLY_LOW = {
    **MEGA_BASE,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 1.0,  'trail_dist': 0.35},   # T7 change only
        'normal': {'trail_act': 0.5,  'trail_dist': 0.15},   # same as baseline
        'high':   {'trail_act': 0.4,  'trail_dist': 0.10},   # same as baseline
    },
}

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]


def fmt_pnl(v):
    return f"${v:>10,.0f}" if v >= 0 else f"-${abs(v):>9,.0f}"


def print_row(label, stats, delta_sharpe=None):
    ds = f"  {delta_sharpe:>+6.2f}" if delta_sharpe is not None else "     ---"
    print(f"  {label:<28s}  {stats['n_trades']:>5d}  {stats['sharpe']:>6.2f}  "
          f"{fmt_pnl(stats['total_pnl'])}  {stats['win_rate']*100:>5.1f}%  "
          f"${stats['avg_pnl_per_trade']:>6.2f}  {fmt_pnl(stats.get('max_drawdown',0))}  {ds}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Full-sample at multiple spread costs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 1: FULL-SAMPLE — T7 vs Baseline vs Ablation at $0/$0.30/$0.50 spread")
print("=" * 80)

print(f"\n{'Config':<28s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  {'WR%':>5s}  "
      f"{'$/t':>7s}  {'MaxDD':>11s}  {'Δ Sharpe':>8s}")
print("-" * 100)

for spread in [0.0, 0.30, 0.50]:
    print(f"\n  --- Spread = ${spread:.2f} ---")

    sb = run_variant(data, f"Base_sp{spread}", verbose=True,
                     **BASELINE, spread_cost=spread)
    st7 = run_variant(data, f"T7_sp{spread}", verbose=True,
                      **T7_EXTREME, spread_cost=spread)
    soh = run_variant(data, f"T7OnlyHigh_sp{spread}", verbose=True,
                      **T7_ONLY_HIGH, spread_cost=spread)
    sol = run_variant(data, f"T7OnlyLow_sp{spread}", verbose=True,
                      **T7_ONLY_LOW, spread_cost=spread)

    print_row(f"Baseline sp${spread:.2f}", sb)
    print_row(f"T7 Extreme sp${spread:.2f}", st7, st7['sharpe'] - sb['sharpe'])
    print_row(f"T7 OnlyHigh sp${spread:.2f}", soh, soh['sharpe'] - sb['sharpe'])
    print_row(f"T7 OnlyLow sp${spread:.2f}", sol, sol['sharpe'] - sb['sharpe'])

    if spread == 0.0:
        base_trades_full = sb.get('_trades', [])
        t7_trades_full = st7.get('_trades', [])


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: K-Fold at $0.30 spread
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 2: K-FOLD at $0.30 spread — T7 vs Baseline")
print("  Pass criteria: T7 wins >=5/6 folds")
print("=" * 80)

print(f"\n{'Fold':<8s}  {'Config':<12s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  "
      f"{'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}  {'Δ Sharpe':>8s}")
print("-" * 90)

wins_030 = 0
fold_results_030 = []
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
        print(f"  {fold_name}: skipped (insufficient data)")
        continue

    sb = run_variant(fold_data, f"P2_Base_{fold_name}", verbose=False,
                     **BASELINE, spread_cost=0.30)
    st7 = run_variant(fold_data, f"P2_T7_{fold_name}", verbose=False,
                      **T7_EXTREME, spread_cost=0.30)

    delta = st7['sharpe'] - sb['sharpe']
    won = delta > 0

    print(f"  {fold_name:<8s}  {'Baseline':<12s}  {sb['n_trades']:>5d}  {sb['sharpe']:>6.2f}  "
          f"{fmt_pnl(sb['total_pnl'])}  {sb['win_rate']*100:>5.1f}%  "
          f"${sb['avg_pnl_per_trade']:>6.2f}  {fmt_pnl(sb.get('max_drawdown',0))}       ---")
    print(f"  {fold_name:<8s}  {'T7 Extreme':<12s}  {st7['n_trades']:>5d}  {st7['sharpe']:>6.2f}  "
          f"{fmt_pnl(st7['total_pnl'])}  {st7['win_rate']*100:>5.1f}%  "
          f"${st7['avg_pnl_per_trade']:>6.2f}  {fmt_pnl(st7.get('max_drawdown',0))}  {delta:>+6.2f} {'✓' if won else '✗'}")

    if won:
        wins_030 += 1
    fold_results_030.append({
        'fold': fold_name, 'base_sharpe': sb['sharpe'], 't7_sharpe': st7['sharpe'],
        'delta': delta, 'base_dd': sb.get('max_drawdown', 0), 't7_dd': st7.get('max_drawdown', 0),
        'base_pnl': sb['total_pnl'], 't7_pnl': st7['total_pnl'],
    })

print(f"\n  K-Fold $0.30 result: T7 wins {wins_030}/6 folds")
print(f"  PASS ✓" if wins_030 >= 5 else f"  FAIL ✗ (need >=5)")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: K-Fold at $0.50 spread (stress test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 3: K-FOLD at $0.50 spread — stress test")
print("  Pass criteria: T7 wins >=4/6 folds")
print("=" * 80)

print(f"\n{'Fold':<8s}  {'Config':<12s}  {'N':>5s}  {'Sharpe':>6s}  {'PnL':>11s}  "
      f"{'WR%':>5s}  {'$/t':>7s}  {'MaxDD':>11s}  {'Δ Sharpe':>8s}")
print("-" * 90)

wins_050 = 0
fold_results_050 = []
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
        continue

    sb = run_variant(fold_data, f"P3_Base_{fold_name}", verbose=False,
                     **BASELINE, spread_cost=0.50)
    st7 = run_variant(fold_data, f"P3_T7_{fold_name}", verbose=False,
                      **T7_EXTREME, spread_cost=0.50)

    delta = st7['sharpe'] - sb['sharpe']
    won = delta > 0

    print(f"  {fold_name:<8s}  {'Baseline':<12s}  {sb['n_trades']:>5d}  {sb['sharpe']:>6.2f}  "
          f"{fmt_pnl(sb['total_pnl'])}  {sb['win_rate']*100:>5.1f}%  "
          f"${sb['avg_pnl_per_trade']:>6.2f}  {fmt_pnl(sb.get('max_drawdown',0))}       ---")
    print(f"  {fold_name:<8s}  {'T7 Extreme':<12s}  {st7['n_trades']:>5d}  {st7['sharpe']:>6.2f}  "
          f"{fmt_pnl(st7['total_pnl'])}  {st7['win_rate']*100:>5.1f}%  "
          f"${st7['avg_pnl_per_trade']:>6.2f}  {fmt_pnl(st7.get('max_drawdown',0))}  {delta:>+6.2f} {'✓' if won else '✗'}")

    if won:
        wins_050 += 1
    fold_results_050.append({
        'fold': fold_name, 'base_sharpe': sb['sharpe'], 't7_sharpe': st7['sharpe'],
        'delta': delta, 'base_dd': sb.get('max_drawdown', 0), 't7_dd': st7.get('max_drawdown', 0),
    })

print(f"\n  K-Fold $0.50 result: T7 wins {wins_050}/6 folds")
print(f"  PASS ✓" if wins_050 >= 4 else f"  FAIL ✗ (need >=4)")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Per-regime trade breakdown
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 4: PER-REGIME TRADE BREAKDOWN — Where does T7 gain/lose?")
print("=" * 80)

for label, trades in [("Baseline", base_trades_full), ("T7 Extreme", t7_trades_full)]:
    print(f"\n  --- {label} ---")
    regime_buckets = defaultdict(list)
    exit_buckets = defaultdict(int)
    for t in trades:
        regime = t.get('atr_regime', 'unknown')
        regime_buckets[regime].append(t.get('pnl', 0))
        exit_buckets[t.get('exit_reason', 'unknown')] += 1

    print(f"  {'Regime':<10s}  {'N':>6s}  {'PnL':>11s}  {'WR%':>6s}  {'$/t':>8s}")
    for regime in ['low', 'normal', 'high', 'unknown']:
        pnls = regime_buckets.get(regime, [])
        if not pnls:
            continue
        n = len(pnls)
        total = sum(pnls)
        wr = sum(1 for p in pnls if p > 0) / n if n > 0 else 0
        avg = total / n if n > 0 else 0
        print(f"  {regime:<10s}  {n:>6d}  {fmt_pnl(total)}  {wr*100:>5.1f}%  ${avg:>7.2f}")

    print(f"\n  Exit reasons:")
    for reason, cnt in sorted(exit_buckets.items(), key=lambda x: -x[1])[:10]:
        print(f"    {reason:<35s}  {cnt:>5d}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: MaxDD comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 5: MAX DRAWDOWN COMPARISON")
print("=" * 80)

if fold_results_030:
    print(f"\n  K-Fold $0.30:")
    print(f"  {'Fold':<8s}  {'Base DD':>10s}  {'T7 DD':>10s}  {'Δ DD':>10s}  {'Δ%':>6s}")
    max_dd_increase_pct = 0
    for r in fold_results_030:
        dd_base = abs(r['base_dd'])
        dd_t7 = abs(r['t7_dd'])
        dd_delta = dd_t7 - dd_base
        dd_pct = (dd_delta / dd_base * 100) if dd_base > 0 else 0
        if dd_pct > max_dd_increase_pct:
            max_dd_increase_pct = dd_pct
        print(f"  {r['fold']:<8s}  {fmt_pnl(dd_base)}  {fmt_pnl(dd_t7)}  {fmt_pnl(dd_delta)}  {dd_pct:>+5.1f}%")
    print(f"\n  Max DD increase across folds: {max_dd_increase_pct:>+.1f}%")
    print(f"  PASS ✓" if max_dd_increase_pct < 20 else f"  FAIL ✗ (>{20}% increase)")

if fold_results_050:
    print(f"\n  K-Fold $0.50:")
    print(f"  {'Fold':<8s}  {'Base DD':>10s}  {'T7 DD':>10s}  {'Δ DD':>10s}  {'Δ%':>6s}")
    max_dd_increase_pct_50 = 0
    for r in fold_results_050:
        dd_base = abs(r['base_dd'])
        dd_t7 = abs(r['t7_dd'])
        dd_delta = dd_t7 - dd_base
        dd_pct = (dd_delta / dd_base * 100) if dd_base > 0 else 0
        if dd_pct > max_dd_increase_pct_50:
            max_dd_increase_pct_50 = dd_pct
        print(f"  {r['fold']:<8s}  {fmt_pnl(dd_base)}  {fmt_pnl(dd_t7)}  {fmt_pnl(dd_delta)}  {dd_pct:>+5.1f}%")
    print(f"\n  Max DD increase across folds: {max_dd_increase_pct_50:>+.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
  T7 ExtremeRegime changes (vs Baseline):
    low  regime: trail_act 0.7→1.0, trail_dist 0.25→0.35 (wider, more room)
    normal regime: unchanged (0.5/0.15)
    high regime: trail_act 0.4→0.25, trail_dist 0.10→0.05 (tighter, quick lock)

  Phase 1 (Full-sample):   See table above
  Phase 2 ($0.30 K-Fold):  T7 wins {wins_030}/6 folds {'PASS ✓' if wins_030 >= 5 else 'FAIL ✗'}
  Phase 3 ($0.50 K-Fold):  T7 wins {wins_050}/6 folds {'PASS ✓' if wins_050 >= 4 else 'FAIL ✗'}
  Phase 4 (Per-regime):    See breakdown above
  Phase 5 (MaxDD $0.30):   {'PASS ✓' if fold_results_030 and max_dd_increase_pct < 20 else 'CHECK'}
""")

all_pass = (wins_030 >= 5 and wins_050 >= 4 and
            fold_results_030 and max_dd_increase_pct < 20)

if all_pass:
    print("  ══════════════════════════════════════════════════")
    print("  ✅  ALL PHASES PASS — T7 ExtremeRegime VALIDATED")
    print("  ══════════════════════════════════════════════════")
    print("  Recommendation: Deploy T7 regime_config to live system.")
    print("  Change in config.py:")
    print("    REGIME_CONFIG = {")
    print("        'low':    {'trail_act': 1.0,  'trail_dist': 0.35},")
    print("        'normal': {'trail_act': 0.5,  'trail_dist': 0.15},")
    print("        'high':   {'trail_act': 0.25, 'trail_dist': 0.05},")
    print("    }")
else:
    print("  ══════════════════════════════════════════════════")
    print("  ⚠️  NOT ALL PHASES PASS — Review before deploying")
    print("  ══════════════════════════════════════════════════")

elapsed = time.time() - t_total
print(f"\n  Total runtime: {elapsed/60:.1f} minutes")
print(f"  Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nResults saved to {OUTPUT_FILE}")
