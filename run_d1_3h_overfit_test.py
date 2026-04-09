#!/usr/bin/env python3
"""
D1+3h Overfit Detection Suite
================================
核心问题: Sharpe 8.39 是否可信？D1+3h 比 C12 多了 4 个参数，过拟合风险是否上升？

Test 1: K-Fold 6 折交叉验证 — 每折 Sharpe 是否全正？跨折稳定性？
Test 2: 参数敏感性 — 4 个关键参数 ±1 步扰动，是否存在悬崖式下降？
Test 3: PBO — 回测过拟合概率（CSCV 方法）
Test 4: PSR/DSR — 概率夏普比率 + 通缩夏普比率（多重测试校正）
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold
from backtest.stats import (
    calc_stats,
    probabilistic_sharpe,
    deflated_sharpe,
    compute_pbo,
)

OUTPUT_FILE = "d1_3h_overfit_output.txt"


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
print("D1+3h OVERFIT DETECTION SUITE")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

# ── Configs ──

D1_3H = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high':   {'trail_act': 0.4, 'trail_dist': 0.10},
    },
    "keltner_max_hold_m15": 12,
    "time_decay_tp": True,
    "time_decay_start_hour": 1.0,
    "time_decay_atr_start": 0.30,
    "time_decay_atr_step": 0.10,
}

C12_BASELINE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.8,
    "trailing_distance_atr": 0.25,
    "regime_config": {
        'low':    {'trail_act': 1.0, 'trail_dist': 0.35},
        'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
        'high':   {'trail_act': 0.6, 'trail_dist': 0.20},
    },
}


# ══════════════════════════════════════════════════════════════
# TEST 1: K-Fold 6 折交叉验证
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("TEST 1: K-FOLD 6-FOLD CROSS VALIDATION")
print("=" * 80)

print("\n--- D1+3h ---")
d1_folds = run_kfold(data, D1_3H, n_folds=6, label_prefix="D1_")

print("\n--- C12 Baseline ---")
c12_folds = run_kfold(data, C12_BASELINE, n_folds=6, label_prefix="C12_")

print(f"\n{'Fold':<12} {'D1_Sharpe':>10} {'D1_PnL':>10} {'D1_N':>6} | {'C12_Sharpe':>11} {'C12_PnL':>10} {'C12_N':>6} | {'Delta':>7}")
print("-" * 85)
d1_sharpes = []
c12_sharpes = []
for d1, c12 in zip(d1_folds, c12_folds):
    d1_sharpes.append(d1['sharpe'])
    c12_sharpes.append(c12['sharpe'])
    delta = d1['sharpe'] - c12['sharpe']
    print(f"{d1.get('fold','?'):<12} {d1['sharpe']:>10.2f} ${d1['total_pnl']:>9,.0f} {d1['n']:>6} | "
          f"{c12['sharpe']:>11.2f} ${c12['total_pnl']:>9,.0f} {c12['n']:>6} | {delta:>+7.2f}")

d1_avg = np.mean(d1_sharpes)
d1_std = np.std(d1_sharpes)
c12_avg = np.mean(c12_sharpes)
c12_std = np.std(c12_sharpes)
d1_min = min(d1_sharpes)
c12_min = min(c12_sharpes)
all_positive_d1 = all(s > 0 for s in d1_sharpes)
all_positive_c12 = all(s > 0 for s in c12_sharpes)

print(f"\nD1+3h:  Avg={d1_avg:.2f}, Std={d1_std:.2f}, Min={d1_min:.2f}, All Positive={all_positive_d1}")
print(f"C12:    Avg={c12_avg:.2f}, Std={c12_std:.2f}, Min={c12_min:.2f}, All Positive={all_positive_c12}")
print(f"D1 wins {sum(1 for d, c in zip(d1_sharpes, c12_sharpes) if d > c)}/6 folds")

if all_positive_d1:
    print("\n>>> K-Fold PASSED: D1+3h positive in all 6 folds")
else:
    neg_folds = [d1_folds[i].get('fold','?') for i, s in enumerate(d1_sharpes) if s <= 0]
    print(f"\n>>> K-Fold WARNING: D1+3h negative in folds: {neg_folds}")


# ══════════════════════════════════════════════════════════════
# TEST 2: 参数敏感性
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("TEST 2: PARAMETER SENSITIVITY (±1 step perturbation)")
print("=" * 80)

perturbations = [
    # (label, param_name, value)
    ("Base: D1+3h", {}, None),

    ("trail_act=0.4", {"trailing_activate_atr": 0.4}, None),
    ("trail_act=0.6", {"trailing_activate_atr": 0.6}, None),

    ("trail_dist=0.10", {"trailing_distance_atr": 0.10}, None),
    ("trail_dist=0.20", {"trailing_distance_atr": 0.20}, None),

    ("decay_atr_start=0.20", {"time_decay_atr_start": 0.20}, None),
    ("decay_atr_start=0.40", {"time_decay_atr_start": 0.40}, None),

    ("decay_step=0.05", {"time_decay_atr_step": 0.05}, None),
    ("decay_step=0.15", {"time_decay_atr_step": 0.15}, None),

    ("max_hold=8 (2h)", {"keltner_max_hold_m15": 8}, None),
    ("max_hold=16 (4h)", {"keltner_max_hold_m15": 16}, None),

    ("decay_start=0.5h", {"time_decay_start_hour": 0.5}, None),
    ("decay_start=1.5h", {"time_decay_start_hour": 1.5}, None),

    ("SL=4.0", {"sl_atr_mult": 4.0}, None),
    ("SL=5.0", {"sl_atr_mult": 5.0}, None),

    ("no_time_decay", {"time_decay_tp": False}, None),
    ("no_regime", {"regime_config": None}, None),
]

sensitivity_results = []
daily_pnls_by_variant = {}

for label, overrides, _ in perturbations:
    kwargs = {**D1_3H, **overrides}
    stats = run_variant(data, label, **kwargs)

    # Collect daily PnL for PBO
    daily = defaultdict(float)
    for t in stats['_trades']:
        daily[t.entry_time.strftime('%Y-%m-%d')] += t.pnl
    sorted_days = sorted(daily.keys())
    daily_pnls_by_variant[label] = [daily[d] for d in sorted_days]

    sensitivity_results.append(stats)
    gc.collect()


print(f"\n\n{'Variant':<28} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}")
print("-" * 68)
base_sh = sensitivity_results[0]['sharpe']
for r in sensitivity_results:
    delta = r['sharpe'] - base_sh
    marker = ""
    if abs(delta) > 0.5:
        marker = " ***" if delta < -0.5 else " +++"
    print(f"{r['label']:<28} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f}{marker}")

sharpes = [r['sharpe'] for r in sensitivity_results]
sh_range = max(sharpes) - min(sharpes)
sh_min = min(sharpes)
sh_max = max(sharpes)

print(f"\nSharpe range: {sh_min:.2f} — {sh_max:.2f} (span={sh_range:.2f})")
print(f"All perturbations Sharpe > 0: {all(s > 0 for s in sharpes)}")
print(f"All perturbations Sharpe > baseline*0.8: {all(s > base_sh*0.8 for s in sharpes[1:])}")

if sh_range < 1.0:
    print("\n>>> Sensitivity PASSED: Sharpe range < 1.0, smooth parameter landscape")
elif sh_range < 2.0:
    print("\n>>> Sensitivity MODERATE: Sharpe range 1.0-2.0, some sensitivity")
else:
    print(f"\n>>> Sensitivity WARNING: Sharpe range {sh_range:.2f} > 2.0, possible cliff")

# Identify most sensitive parameter
param_groups = {
    'trail_act': [r for r in sensitivity_results if 'trail_act' in r['label']],
    'trail_dist': [r for r in sensitivity_results if 'trail_dist' in r['label']],
    'decay_atr_start': [r for r in sensitivity_results if 'decay_atr_start' in r['label']],
    'decay_step': [r for r in sensitivity_results if 'decay_step' in r['label']],
    'max_hold': [r for r in sensitivity_results if 'max_hold' in r['label']],
    'decay_start': [r for r in sensitivity_results if 'decay_start=' in r['label']],
    'SL': [r for r in sensitivity_results if r['label'].startswith('SL=')],
    'no_time_decay': [r for r in sensitivity_results if 'no_time_decay' in r['label']],
    'no_regime': [r for r in sensitivity_results if 'no_regime' in r['label']],
}

print(f"\nPER-PARAMETER IMPACT:")
for pname, variants in param_groups.items():
    if not variants:
        continue
    shs = [v['sharpe'] for v in variants]
    avg_delta = np.mean([s - base_sh for s in shs])
    print(f"  {pname:<20} avg Δ={avg_delta:+.2f}  range=[{min(shs):.2f}, {max(shs):.2f}]")


# ══════════════════════════════════════════════════════════════
# TEST 3: PBO (Probability of Backtest Overfitting)
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("TEST 3: PROBABILITY OF BACKTEST OVERFITTING (PBO)")
print("=" * 80)

pbo_result = compute_pbo(daily_pnls_by_variant, n_partitions=8)

print(f"\n  PBO = {pbo_result['pbo']:.4f}")
print(f"  Combinations tested: {pbo_result['n_combinations']}")
print(f"  Overfit risk: {pbo_result['overfit_risk']}")

if pbo_result['is_best_oos_ranks']:
    ranks = pbo_result['is_best_oos_ranks']
    n_variants = len(daily_pnls_by_variant)
    print(f"  IS-best OOS rank distribution: median={np.median(ranks):.1f}, "
          f"mean={np.mean(ranks):.1f} (of {n_variants} variants)")
    below_median = sum(1 for r in ranks if r > n_variants / 2)
    print(f"  IS-best ranked below OOS median: {below_median}/{len(ranks)} = {100*below_median/len(ranks):.1f}%")

if pbo_result['pbo'] < 0.10:
    print("\n>>> PBO PASSED: PBO < 0.10, low overfit risk")
elif pbo_result['pbo'] < 0.50:
    print(f"\n>>> PBO MODERATE: PBO = {pbo_result['pbo']:.2f}, some overfit concern")
else:
    print(f"\n>>> PBO FAILED: PBO = {pbo_result['pbo']:.2f} >= 0.50, HIGH overfit risk!")


# ══════════════════════════════════════════════════════════════
# TEST 4: PSR / DSR
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("TEST 4: PROBABILISTIC SHARPE RATIO (PSR) & DEFLATED SHARPE RATIO (DSR)")
print("=" * 80)

# Get daily returns for D1+3h full run
base_trades = sensitivity_results[0]['_trades']
daily_pnl = defaultdict(float)
for t in base_trades:
    daily_pnl[t.entry_time.strftime('%Y-%m-%d')] += t.pnl
sorted_days = sorted(daily_pnl.keys())
daily_returns = [daily_pnl[d] for d in sorted_days]

n_trials = len(perturbations)
all_variant_sharpes = [r['sharpe'] for r in sensitivity_results]
all_sharpes_var = float(np.var(all_variant_sharpes))

psr = probabilistic_sharpe(daily_returns, sharpe_benchmark=0.0)
dsr = deflated_sharpe(daily_returns, n_trials=n_trials, all_sharpes_var=all_sharpes_var)

print(f"\nPSR (vs benchmark=0):")
print(f"  Observed Sharpe: {psr['sharpe_obs']:.2f}")
print(f"  PSR probability: {psr['psr']:.4f}")
print(f"  p-value: {psr['p_value']:.6f}")
print(f"  Skewness: {psr['skew']:.3f}")
print(f"  Excess kurtosis: {psr['kurtosis']:.3f}")
print(f"  N (trading days): {psr['n']}")

if psr['p_value'] < 0.01:
    print("  >>> PSR PASSED: p < 0.01, Sharpe is highly significant")
elif psr['p_value'] < 0.05:
    print("  >>> PSR PASSED: p < 0.05, Sharpe is significant")
else:
    print(f"  >>> PSR FAILED: p = {psr['p_value']:.4f} >= 0.05, Sharpe NOT significant")

print(f"\nDSR (corrected for {n_trials} trials):")
print(f"  Expected max SR under null (SR*): {dsr['sr_star']:.2f}")
print(f"  DSR probability: {dsr['dsr']:.4f}")
print(f"  Passed (DSR > 0.95): {dsr['passed']}")
print(f"  Sharpes variance across variants: {all_sharpes_var:.4f}")

if dsr['passed']:
    print("  >>> DSR PASSED: Sharpe survives multiple-testing correction")
else:
    print("  >>> DSR FAILED: Sharpe does NOT survive multiple-testing correction")


# ══════════════════════════════════════════════════════════════
# OVERALL VERDICT
# ══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("OVERALL VERDICT")
print("=" * 80)

tests = {
    'K-Fold': all_positive_d1,
    'Sensitivity': sh_range < 2.0,
    'PBO': pbo_result['pbo'] < 0.50,
    'PSR': psr['p_value'] < 0.05,
    'DSR': dsr['passed'],
}

passed = sum(1 for v in tests.values() if v)
total = len(tests)

for name, result in tests.items():
    status = "PASS" if result else "FAIL"
    print(f"  {name:<15} {status}")

print(f"\n  Result: {passed}/{total} tests passed")

if passed == total:
    print("\n>>> D1+3h configuration shows LOW overfit risk. Sharpe is likely genuine.")
elif passed >= 3:
    print("\n>>> D1+3h shows MODERATE overfit risk. Most tests pass but review failed ones.")
else:
    print("\n>>> D1+3h shows HIGH overfit risk! Consider reverting to C12 or simplifying.")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
