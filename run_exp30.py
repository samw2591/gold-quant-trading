#!/usr/bin/env python3
"""
EXP30: Keltner 最大持仓时间优化
================================
Timeout 60 bars (15h) 是最大单项亏损源 (-$12,287)。
测试缩短到 24/32/40/48 bars 是否能减少亏损而不过多损失利润。

无点差，聚焦策略逻辑。
"""
import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant, calc_stats
from backtest.runner import C12_KWARGS, run_kfold
from backtest.stats import print_comparison, print_ranked

print("=" * 70)
print("EXP30: KELTNER MAX HOLD TIME OPTIMIZATION")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

BASE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
}

# ═══════════════════════════════════════════════════════════════
# Part 1: Max Hold Scan
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: MAX HOLD SCAN (no spread)")
print("=" * 70)

hold_values = [16, 20, 24, 28, 32, 36, 40, 48, 60, 80]

results = []
for hold in hold_values:
    label = f"Hold={hold} ({hold*0.25:.0f}h)"
    kwargs = {**BASE, "keltner_max_hold_m15": hold}
    stats = run_variant(data, label, **kwargs)
    stats['hold_m15'] = hold
    stats['hold_hours'] = hold * 0.25
    results.append(stats)

# Baseline is hold=60 (default)
baseline = next((r for r in results if r['hold_m15'] == 60), results[-1])

print("\n  RESULTS RANKED BY SHARPE:")
print(f"  {'Hold':<12} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'MaxDD':>10} {'WR%':>6} {'$/t':>7}")
print(f"  {'-'*70}")
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    d = r['sharpe'] - baseline['sharpe']
    ppt = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
    marker = " <-- CURRENT" if r['hold_m15'] == 60 else ""
    print(f"  {r['hold_m15']:>3} ({r['hold_hours']:.0f}h)   {r['n']:>6} {r['sharpe']:>8.2f} {d:>+7.2f} "
          f"${r['total_pnl']:>9,.0f} ${r['max_dd']:>9,.0f} {r['win_rate']:>5.1f}% ${ppt:>6.2f}{marker}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Exit reason breakdown for baseline vs best
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: EXIT REASON ANALYSIS")
print("=" * 70)

best = max(results, key=lambda x: x['sharpe'])
for label, r in [("BASELINE (60)", baseline), (f"BEST ({best['hold_m15']})", best)]:
    trades = r.get('_trades', [])
    if not trades:
        continue
    reasons = {}
    for t in trades:
        key = t.exit_reason.split(':')[0] if ':' in t.exit_reason else t.exit_reason
        if key not in reasons:
            reasons[key] = {'n': 0, 'pnl': 0}
        reasons[key]['n'] += 1
        reasons[key]['pnl'] += t.pnl
    
    print(f"\n  {label}:")
    print(f"  {'Reason':<20} {'N':>6} {'PnL':>10} {'$/t':>7} {'%trades':>8}")
    for reason, d in sorted(reasons.items(), key=lambda x: x[1]['pnl']):
        ppt = d['pnl'] / d['n'] if d['n'] > 0 else 0
        pct = d['n'] / len(trades) * 100
        print(f"  {reason:<20} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {pct:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: K-Fold validation for top 3
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: K-FOLD VALIDATION (6 FOLDS)")
print("=" * 70)

top3 = sorted(results, key=lambda x: x['sharpe'], reverse=True)[:3]
# Always include baseline if not in top 3
if all(r['hold_m15'] != 60 for r in top3):
    top3.append(baseline)

kfold_results = {}
for r in top3:
    hold = r['hold_m15']
    label = f"Hold={hold}"
    kwargs = {**BASE, "keltner_max_hold_m15": hold}
    folds = run_kfold(data, kwargs, n_folds=6, label_prefix=f"H{hold}_")
    fold_sharpes = [f['sharpe'] for f in folds]
    avg = np.mean(fold_sharpes)
    std = np.std(fold_sharpes)
    kfold_results[hold] = {'folds': fold_sharpes, 'avg': avg, 'std': std}
    print(f"\n  Hold={hold}: Avg={avg:.2f} Std={std:.2f}")
    for f in folds:
        print(f"    {f['fold']}: Sh={f['sharpe']:.2f}  N={f['n']:,}  PnL=${f['total_pnl']:,.0f}")

# Compare folds
base_folds = kfold_results.get(60, {}).get('folds', [])
print(f"\n  K-Fold Summary:")
print(f"  {'Hold':<10} {'Avg':>6} {'Std':>6} {'Wins vs 60':>12}")
for hold in sorted(kfold_results.keys()):
    res = kfold_results[hold]
    wins = sum(1 for a, b in zip(res['folds'], base_folds) if a > b) if base_folds else 0
    marker = " <-- CURRENT" if hold == 60 else ""
    print(f"  {hold:<10} {res['avg']:>6.2f} {res['std']:>6.2f} {wins:>5}/6{marker}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP30 SUMMARY")
print("=" * 70)
print(f"  Baseline (Hold=60): Sharpe={baseline['sharpe']:.2f}  PnL=${baseline['total_pnl']:,.0f}  MaxDD=${baseline['max_dd']:,.0f}")
best = max(results, key=lambda x: x['sharpe'])
d = best['sharpe'] - baseline['sharpe']
print(f"  Best (Hold={best['hold_m15']}): Sharpe={best['sharpe']:.2f} ({d:+.2f})  PnL=${best['total_pnl']:,.0f}  MaxDD=${best['max_dd']:,.0f}")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
