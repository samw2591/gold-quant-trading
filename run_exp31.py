#!/usr/bin/env python3
"""
EXP31: EMA100 斜率过滤 — 阴跌缓解
====================================
当 EMA100 下行时禁止 BUY，减少慢跌行情（S4 类）中的逆势做多亏损。

测试不同的斜率观测窗口和组合方案。
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
print("EXP31: EMA100 SLOPE FILTER (BLOCK BUY WHEN DECLINING)")
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
# Part 1: EMA Slope Lookback Scan
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: EMA100 SLOPE LOOKBACK SCAN (no spread)")
print("=" * 70)

# block_buy_ema_slope = N means: block BUY if EMA100[now] < EMA100[now-N]
lookbacks = [0, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]

results = []
for lb in lookbacks:
    label = f"EMA_slope={lb}" if lb > 0 else "Baseline (no filter)"
    kwargs = {**BASE, "block_buy_ema_slope": lb}
    stats = run_variant(data, label, **kwargs)
    stats['lookback'] = lb
    results.append(stats)

baseline = results[0]  # lookback=0

print("\n  RESULTS RANKED BY SHARPE:")
print(f"  {'Lookback':<12} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'MaxDD':>10} {'WR%':>6} {'Skipped':>8}")
print(f"  {'-'*75}")
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    d = r['sharpe'] - baseline['sharpe']
    marker = " <-- BASELINE" if r['lookback'] == 0 else ""
    print(f"  {r['lookback']:>3}         {r['n']:>6} {r['sharpe']:>8.2f} {d:>+7.2f} "
          f"${r['total_pnl']:>9,.0f} ${r['max_dd']:>9,.0f} {r['win_rate']:>5.1f}% {r['skipped_ema_slope']:>7}{marker}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Direction breakdown — BUY vs SELL impact
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: DIRECTION BREAKDOWN")
print("=" * 70)

for label_prefix, r in [("Baseline", baseline)] + [(f"Slope={r['lookback']}", r) for r in results if r['lookback'] in [4, 6, 10]]:
    trades = r.get('_trades', [])
    if not trades:
        continue
    buys = [t for t in trades if t.direction == 'BUY']
    sells = [t for t in trades if t.direction == 'SELL']
    buy_pnl = sum(t.pnl for t in buys)
    sell_pnl = sum(t.pnl for t in sells)
    buy_wr = sum(1 for t in buys if t.pnl > 0) / len(buys) * 100 if buys else 0
    sell_wr = sum(1 for t in sells if t.pnl > 0) / len(sells) * 100 if sells else 0
    print(f"\n  {label_prefix}:")
    print(f"    BUY:  N={len(buys):>5}  PnL=${buy_pnl:>8,.0f}  WR={buy_wr:.1f}%  $/t=${buy_pnl/len(buys) if buys else 0:.2f}")
    print(f"    SELL: N={len(sells):>5}  PnL=${sell_pnl:>8,.0f}  WR={sell_wr:.1f}%  $/t=${sell_pnl/len(sells) if sells else 0:.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 3: Stress test on S4 (阴跌) period
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: S4 SLOW DECLINE STRESS TEST (2022-09 to 2022-11)")
print("=" * 70)

s4_data = data.slice("2022-09-01", "2022-12-01")
print(f"  S4 period: {len(s4_data.m15_df)} M15 bars, {len(s4_data.h1_df)} H1 bars")

s4_results = []
for lb in [0, 4, 6, 8, 10]:
    label = f"S4_slope={lb}" if lb > 0 else "S4_Baseline"
    kwargs = {**BASE, "block_buy_ema_slope": lb}
    stats = run_variant(s4_data, label, **kwargs)
    stats['lookback'] = lb
    s4_results.append(stats)
    d = stats['sharpe'] - s4_results[0]['sharpe']
    print(f"    slope={lb}: N={stats['n']:>4}  Sh={stats['sharpe']:.2f} ({d:+.2f})  "
          f"PnL=${stats['total_pnl']:.0f}  MaxDD=${stats['max_dd']:.0f}")


# ═══════════════════════════════════════════════════════════════
# Part 4: K-Fold validation for best lookbacks
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: K-FOLD VALIDATION (6 FOLDS)")
print("=" * 70)

top_lookbacks = [0]  # always include baseline
# Add top 3 non-zero by Sharpe
non_zero = [r for r in results if r['lookback'] > 0]
non_zero.sort(key=lambda x: x['sharpe'], reverse=True)
for r in non_zero[:3]:
    if r['lookback'] not in top_lookbacks:
        top_lookbacks.append(r['lookback'])

kfold_results = {}
for lb in top_lookbacks:
    label = f"Slope={lb}" if lb > 0 else "Baseline"
    kwargs = {**BASE, "block_buy_ema_slope": lb}
    folds = run_kfold(data, kwargs, n_folds=6, label_prefix=f"S{lb}_")
    fold_sharpes = [f['sharpe'] for f in folds]
    avg = np.mean(fold_sharpes)
    std = np.std(fold_sharpes)
    kfold_results[lb] = {'folds': fold_sharpes, 'avg': avg, 'std': std}
    print(f"\n  Slope={lb}: Avg={avg:.2f} Std={std:.2f}")
    for f in folds:
        print(f"    {f['fold']}: Sh={f['sharpe']:.2f}  N={f['n']:,}  PnL=${f['total_pnl']:,.0f}")

base_folds = kfold_results.get(0, {}).get('folds', [])
print(f"\n  K-Fold Summary:")
print(f"  {'Slope':<10} {'Avg':>6} {'Std':>6} {'Wins vs BL':>12}")
for lb in sorted(kfold_results.keys()):
    res = kfold_results[lb]
    wins = sum(1 for a, b in zip(res['folds'], base_folds) if a > b) if base_folds and lb > 0 else '-'
    marker = " <-- BASELINE" if lb == 0 else ""
    w_str = f"{wins}/6" if isinstance(wins, int) else "  -"
    print(f"  {lb:<10} {res['avg']:>6.2f} {res['std']:>6.2f} {w_str:>9}{marker}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP31 SUMMARY")
print("=" * 70)
print(f"  Baseline (no filter): Sharpe={baseline['sharpe']:.2f}  PnL=${baseline['total_pnl']:,.0f}  MaxDD=${baseline['max_dd']:,.0f}")
best = max(non_zero, key=lambda x: x['sharpe']) if non_zero else baseline
d = best['sharpe'] - baseline['sharpe']
print(f"  Best (slope={best['lookback']}): Sharpe={best['sharpe']:.2f} ({d:+.2f})  PnL=${best['total_pnl']:,.0f}  MaxDD=${best['max_dd']:,.0f}")
if best['lookback'] in kfold_results:
    kf = kfold_results[best['lookback']]
    bkf = kfold_results.get(0, {})
    wins = sum(1 for a, b in zip(kf['folds'], bkf.get('folds', [])) if a > b) if bkf.get('folds') else 0
    print(f"  K-Fold: Avg={kf['avg']:.2f} Wins={wins}/6")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
