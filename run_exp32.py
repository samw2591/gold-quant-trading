#!/usr/bin/env python3
"""
EXP32: Mega Grid 最优参数 T0.5/D0.15 验证
============================================
Mega Grid 搜索冠军: Trail Act=0.5, Dist=0.15 (Sharpe 3.36 无成本)
验证它与当前实盘 T0.8/D0.25 的差异，含 K-Fold、Regime 压力、方向分析。

无点差，聚焦策略逻辑。
"""
import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant, calc_stats
from backtest.runner import C12_KWARGS, V3_REGIME, run_kfold
from backtest.stats import print_comparison, print_ranked

print("=" * 70)
print("EXP32: MEGA GRID T0.5/D0.15 VALIDATION")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

# Current production config
CURRENT = {
    **C12_KWARGS,
    "intraday_adaptive": True,
}

# Mega Grid champion — override trailing params only
MEGA = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

# Also test intermediate values
VARIANTS = [
    ("A: Current T0.8/D0.25", CURRENT),
    ("B: Mega T0.5/D0.15", MEGA),
    ("C: T0.6/D0.20", {
        **C12_KWARGS, "intraday_adaptive": True,
        "trailing_activate_atr": 0.6,
        "trailing_distance_atr": 0.20,
        "regime_config": {
            'low': {'trail_act': 0.8, 'trail_dist': 0.30},
            'normal': {'trail_act': 0.6, 'trail_dist': 0.20},
            'high': {'trail_act': 0.5, 'trail_dist': 0.15},
        },
    }),
    ("D: T0.7/D0.20", {
        **C12_KWARGS, "intraday_adaptive": True,
        "trailing_activate_atr": 0.7,
        "trailing_distance_atr": 0.20,
        "regime_config": {
            'low': {'trail_act': 0.9, 'trail_dist': 0.30},
            'normal': {'trail_act': 0.7, 'trail_dist': 0.20},
            'high': {'trail_act': 0.5, 'trail_dist': 0.15},
        },
    }),
    ("E: T0.5/D0.20", {
        **C12_KWARGS, "intraday_adaptive": True,
        "trailing_activate_atr": 0.5,
        "trailing_distance_atr": 0.20,
        "regime_config": {
            'low': {'trail_act': 0.7, 'trail_dist': 0.30},
            'normal': {'trail_act': 0.5, 'trail_dist': 0.20},
            'high': {'trail_act': 0.4, 'trail_dist': 0.15},
        },
    }),
    ("F: No V3 T0.5/D0.15", {
        **C12_KWARGS, "intraday_adaptive": True,
        "trailing_activate_atr": 0.5,
        "trailing_distance_atr": 0.15,
        "regime_config": None,
    }),
]

# ═══════════════════════════════════════════════════════════════
# Part 1: Full backtest comparison
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: FULL BACKTEST COMPARISON (no spread)")
print("=" * 70)

results = []
for label, kwargs in VARIANTS:
    stats = run_variant(data, label, **kwargs)
    results.append(stats)

baseline = results[0]  # Current config

print("\n  RESULTS:")
print(f"  {'Variant':<30} {'N':>6} {'Sharpe':>8} {'Delta':>7} {'PnL':>10} {'MaxDD':>10} {'WR%':>6} {'$/t':>7}")
print(f"  {'-'*85}")
for r in results:
    d = r['sharpe'] - baseline['sharpe']
    ppt = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
    print(f"  {r['label']:<30} {r['n']:>6} {r['sharpe']:>8.2f} {d:>+7.2f} "
          f"${r['total_pnl']:>9,.0f} ${r['max_dd']:>9,.0f} {r['win_rate']:>5.1f}% ${ppt:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Exit reason analysis — Current vs Mega
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: EXIT REASON COMPARISON")
print("=" * 70)

for r in [results[0], results[1]]:  # Current vs Mega
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
    
    print(f"\n  {r['label']}:")
    print(f"  {'Reason':<20} {'N':>6} {'PnL':>10} {'$/t':>7} {'%trades':>8}")
    for reason, d in sorted(reasons.items(), key=lambda x: x[1]['pnl']):
        ppt = d['pnl'] / d['n'] if d['n'] > 0 else 0
        pct = d['n'] / len(trades) * 100
        print(f"  {reason:<20} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {pct:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# Part 3: Stress test on specific periods
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: REGIME STRESS TEST")
print("=" * 70)

scenarios = [
    ("S1 War Escalation", "2020-01-01", "2020-03-01"),
    ("S2 Ceasefire Drop", "2026-03-15", "2026-03-31"),
    ("S3 Liquidity Crisis", "2020-03-09", "2020-03-23"),
    ("S4 Slow Decline", "2022-09-01", "2022-12-01"),
    ("S6 Tariff Whipsaw", "2025-04-01", "2025-04-10"),
    ("High Vol 2020", "2020-01-01", "2020-12-31"),
    ("Low Vol 2018", "2018-01-01", "2018-12-31"),
]

for s_name, s_start, s_end in scenarios:
    s_data = data.slice(s_start, s_end)
    if len(s_data.m15_df) < 100:
        print(f"\n  {s_name}: insufficient data")
        continue
    
    current_s = run_variant(s_data, f"Current_{s_name}", verbose=False, **CURRENT)
    mega_s = run_variant(s_data, f"Mega_{s_name}", verbose=False, **MEGA)
    d = mega_s['sharpe'] - current_s['sharpe']
    winner = "MEGA" if mega_s['sharpe'] > current_s['sharpe'] else "CURRENT"
    print(f"  {s_name:<25} Current: Sh={current_s['sharpe']:>6.2f} PnL=${current_s['total_pnl']:>7,.0f}  |  "
          f"Mega: Sh={mega_s['sharpe']:>6.2f} PnL=${mega_s['total_pnl']:>7,.0f}  [{winner} {d:+.2f}]")


# ═══════════════════════════════════════════════════════════════
# Part 4: K-Fold validation
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: K-FOLD VALIDATION (6 FOLDS)")
print("=" * 70)

kfold_configs = [
    ("Current", CURRENT),
    ("Mega T0.5/D0.15", MEGA),
    ("T0.6/D0.20", VARIANTS[2][1]),
]

kfold_results = {}
for name, kwargs in kfold_configs:
    folds = run_kfold(data, kwargs, n_folds=6, label_prefix=f"{name[:3]}_")
    fold_sharpes = [f['sharpe'] for f in folds]
    avg = np.mean(fold_sharpes)
    std = np.std(fold_sharpes)
    kfold_results[name] = {'folds': fold_sharpes, 'avg': avg, 'std': std}
    print(f"\n  {name}: Avg={avg:.2f} Std={std:.2f}")
    for f in folds:
        print(f"    {f['fold']}: Sh={f['sharpe']:.2f}  N={f['n']:,}  PnL=${f['total_pnl']:,.0f}")

base_folds = kfold_results["Current"]['folds']
print(f"\n  K-Fold Summary:")
print(f"  {'Config':<25} {'Avg':>6} {'Std':>6} {'Wins vs Current':>16}")
for name, res in kfold_results.items():
    wins = sum(1 for a, b in zip(res['folds'], base_folds) if a > b) if name != "Current" else "-"
    w_str = f"{wins}/6" if isinstance(wins, int) else "   -"
    marker = " <-- CURRENT" if name == "Current" else ""
    print(f"  {name:<25} {res['avg']:>6.2f} {res['std']:>6.2f} {w_str:>10}{marker}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Yearly breakdown
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: YEARLY COMPARISON (Current vs Mega)")
print("=" * 70)

years = range(2015, 2027)
print(f"  {'Year':<6} {'Current Sh':>11} {'Current PnL':>12} {'Mega Sh':>9} {'Mega PnL':>10} {'Winner':>8}")
print(f"  {'-'*60}")
for year in years:
    start = f"{year}-01-01"
    end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
    yr_data = data.slice(start, end)
    if len(yr_data.m15_df) < 500:
        continue
    c = run_variant(yr_data, f"Cur_{year}", verbose=False, **CURRENT)
    m = run_variant(yr_data, f"Meg_{year}", verbose=False, **MEGA)
    w = "MEGA" if m['sharpe'] > c['sharpe'] else "CURRENT"
    print(f"  {year:<6} {c['sharpe']:>11.2f} ${c['total_pnl']:>10,.0f} {m['sharpe']:>9.2f} ${m['total_pnl']:>9,.0f} {w:>8}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

elapsed = time.time() - t0
print("\n" + "=" * 70)
print("EXP32 SUMMARY")
print("=" * 70)
c = results[0]
m = results[1]
d = m['sharpe'] - c['sharpe']
print(f"  Current (T0.8/D0.25): Sharpe={c['sharpe']:.2f}  PnL=${c['total_pnl']:,.0f}  MaxDD=${c['max_dd']:,.0f}  N={c['n']:,}")
print(f"  Mega (T0.5/D0.15):    Sharpe={m['sharpe']:.2f} ({d:+.2f})  PnL=${m['total_pnl']:,.0f}  MaxDD=${m['max_dd']:,.0f}  N={m['n']:,}")
kc = kfold_results.get("Current", {})
km = kfold_results.get("Mega T0.5/D0.15", {})
if kc and km:
    wins = sum(1 for a, b in zip(km['folds'], kc['folds']) if a > b)
    print(f"  K-Fold: Mega Avg={km['avg']:.2f} vs Current Avg={kc['avg']:.2f}, Mega wins {wins}/6 folds")
print(f"\n  Runtime: {elapsed/60:.1f} minutes")
print(f"  Finished: {datetime.now()}")
