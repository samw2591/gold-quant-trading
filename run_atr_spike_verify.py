#!/usr/bin/env python3
"""
ATR Spike Protection — Full Verification
==========================================
EXP52 post-hoc found: ATR spike >50% trades lose $10.88/trade avg.
Tightening trailing distance 0.7x when spiked → +0.48~0.57 Sharpe (post-hoc).

This script verifies with REAL engine implementation (not post-hoc):
  Phase 1: Current baseline vs Current+Spike on full dataset
  Phase 2: Mega baseline vs Mega+Spike on full dataset
  Phase 3: Parameter sensitivity (threshold × multiplier grid)
  Phase 4: K-Fold cross-validation (6 folds)
  Phase 5: PSR / DSR statistical tests
  Phase 6: Exit reason breakdown (how does spike change exit distribution)
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import (
    DataBundle, run_variant, run_kfold, calc_stats,
    print_comparison, print_ranked,
    probabilistic_sharpe, deflated_sharpe, compute_pbo,
)
from backtest.runner import C12_KWARGS
from backtest.stats import aggregate_daily_pnl

OUTPUT_FILE = "atr_spike_verify_output.txt"


class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

print("=" * 80)
print("ATR SPIKE PROTECTION — FULL VERIFICATION")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

# ═══════════════════════════════════════════════════════════════
# Phase 1 & 2: Full dataset comparison
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("PHASE 1 & 2: FULL DATASET — Current / Mega ± ATR Spike Protection")
print("=" * 80)

variants_full = []

for base_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    # Baseline (no spike protection)
    r = run_variant(data, f"{base_label}", **base_kwargs)
    variants_full.append(r)

    # With ATR spike protection (default: threshold=1.5, trail_mult=0.7)
    r = run_variant(data, f"{base_label}+Spike(1.5/0.7)", **base_kwargs,
                    atr_spike_protection=True,
                    atr_spike_threshold=1.5,
                    atr_spike_trail_mult=0.7)
    print(f"    ATR spike tightens: {r['atr_spike_tightens']}")
    variants_full.append(r)

print_comparison(variants_full, "Full Dataset: Baseline vs ATR Spike Protection")

# Exit reason breakdown
for i in range(0, len(variants_full), 2):
    base = variants_full[i]
    spike = variants_full[i + 1]
    base_label = base['label']
    print(f"\n  Exit Reason Breakdown: {base_label}")
    print(f"  {'Reason':<20} {'Base_N':>7} {'Base_PnL':>10} {'Spike_N':>7} {'Spike_PnL':>10} {'dPnL':>8}")
    print(f"  {'-'*65}")

    reasons_base = defaultdict(lambda: {'n': 0, 'pnl': 0.0})
    reasons_spike = defaultdict(lambda: {'n': 0, 'pnl': 0.0})

    for t in base['_trades']:
        r_key = t.exit_reason.split(':')[0]
        reasons_base[r_key]['n'] += 1
        reasons_base[r_key]['pnl'] += t.pnl
    for t in spike['_trades']:
        r_key = t.exit_reason.split(':')[0]
        reasons_spike[r_key]['n'] += 1
        reasons_spike[r_key]['pnl'] += t.pnl

    all_reasons = sorted(set(list(reasons_base.keys()) + list(reasons_spike.keys())))
    for r_key in all_reasons:
        bn = reasons_base[r_key]['n']
        bp = reasons_base[r_key]['pnl']
        sn = reasons_spike[r_key]['n']
        sp = reasons_spike[r_key]['pnl']
        print(f"  {r_key:<20} {bn:>7} ${bp:>9,.0f} {sn:>7} ${sp:>9,.0f} ${sp-bp:>+7,.0f}")

# ═══════════════════════════════════════════════════════════════
# Phase 3: Parameter sensitivity grid
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("PHASE 3: PARAMETER SENSITIVITY")
print("=" * 80)

for base_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    results_grid = []
    # Baseline
    r_base = run_variant(data, f"{base_label} (no spike)", verbose=False, **base_kwargs)
    results_grid.append(r_base)

    for threshold in [1.3, 1.5, 1.8, 2.0]:
        for trail_mult in [0.5, 0.6, 0.7, 0.8]:
            label = f"T{threshold}/M{trail_mult}"
            r = run_variant(data, label, verbose=False, **base_kwargs,
                            atr_spike_protection=True,
                            atr_spike_threshold=threshold,
                            atr_spike_trail_mult=trail_mult)
            results_grid.append(r)

    base_sh = r_base['sharpe']
    print(f"\n  {base_label} Parameter Grid (base Sharpe={base_sh:.2f}):")
    print(f"  {'Threshold':>10} {'Trail_x':>8} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} {'WR%':>6} {'Spikes':>7}")
    print(f"  {'-'*65}")

    for r in results_grid:
        if r == r_base:
            print(f"  {'(base)':>10} {'---':>8} {r['n']:>6} {r['sharpe']:>8.2f} {'---':>6} ${r['total_pnl']:>9,.0f} "
                  f"{r['win_rate']:>5.1f}% {'---':>7}")
        else:
            parts = r['label'].split('/')
            th_str = parts[0][1:]
            tm_str = parts[1][1:]
            ds = r['sharpe'] - base_sh
            print(f"  {th_str:>10} {tm_str:>8} {r['n']:>6} {r['sharpe']:>8.2f} {ds:>+5.2f} ${r['total_pnl']:>9,.0f} "
                  f"{r['win_rate']:>5.1f}% {r.get('atr_spike_tightens', 0):>7}")

# ═══════════════════════════════════════════════════════════════
# Phase 4: K-Fold cross-validation
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("PHASE 4: K-FOLD CROSS-VALIDATION (6 folds)")
print("=" * 80)

for base_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    print(f"\n  {base_label}:")

    # Baseline K-Fold
    base_folds = run_kfold(data, base_kwargs, n_folds=6, label_prefix=f"{base_label}_")

    # Spike K-Fold
    spike_kwargs = {**base_kwargs,
                    "atr_spike_protection": True,
                    "atr_spike_threshold": 1.5,
                    "atr_spike_trail_mult": 0.7}
    spike_folds = run_kfold(data, spike_kwargs, n_folds=6, label_prefix=f"{base_label}+Spike_")

    print(f"\n  {'Fold':<12} {'Base_Sh':>8} {'Spike_Sh':>9} {'Delta':>7} {'Base_PnL':>10} {'Spike_PnL':>10}")
    print(f"  {'-'*60}")

    base_sharpes = []
    spike_sharpes = []
    wins = 0
    for bf, sf in zip(base_folds, spike_folds):
        ds = sf['sharpe'] - bf['sharpe']
        base_sharpes.append(bf['sharpe'])
        spike_sharpes.append(sf['sharpe'])
        if ds > 0:
            wins += 1
        marker = "  ✓" if ds > 0 else ""
        print(f"  {bf['fold']:<12} {bf['sharpe']:>8.2f} {sf['sharpe']:>9.2f} {ds:>+7.2f} "
              f"${bf['total_pnl']:>9,.0f} ${sf['total_pnl']:>9,.0f}{marker}")

    if base_sharpes and spike_sharpes:
        avg_base = np.mean(base_sharpes)
        avg_spike = np.mean(spike_sharpes)
        std_base = np.std(base_sharpes)
        std_spike = np.std(spike_sharpes)
        print(f"  {'Avg':<12} {avg_base:>8.2f} {avg_spike:>9.2f} {avg_spike-avg_base:>+7.2f}")
        print(f"  {'Std':<12} {std_base:>8.2f} {std_spike:>9.2f}")
        print(f"  Spike wins: {wins}/{len(base_folds)} folds")

# ═══════════════════════════════════════════════════════════════
# Phase 5: PSR / DSR statistical tests
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("PHASE 5: STATISTICAL SIGNIFICANCE (PSR / DSR / PBO)")
print("=" * 80)

for base_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    print(f"\n  {base_label}:")

    base_full = run_variant(data, f"{base_label}", verbose=False, **base_kwargs)
    spike_full = run_variant(data, f"{base_label}+Spike", verbose=False, **base_kwargs,
                             atr_spike_protection=True,
                             atr_spike_threshold=1.5,
                             atr_spike_trail_mult=0.7)

    base_daily = aggregate_daily_pnl(base_full['_trades'])
    spike_daily = aggregate_daily_pnl(spike_full['_trades'])

    # PSR: Is the Sharpe significantly > 0?
    psr_base = probabilistic_sharpe(base_daily, sharpe_benchmark=0)
    psr_spike = probabilistic_sharpe(spike_daily, sharpe_benchmark=0)

    print(f"\n  Probabilistic Sharpe Ratio (PSR, benchmark=0):")
    print(f"    Base:  Sharpe={psr_base['sharpe_obs']:.2f}  PSR={psr_base['psr']:.4f}  p={psr_base['p_value']:.6f}")
    print(f"    Spike: Sharpe={psr_spike['sharpe_obs']:.2f}  PSR={psr_spike['psr']:.4f}  p={psr_spike['p_value']:.6f}")

    # PSR: Is spike significantly better than base?
    psr_vs = probabilistic_sharpe(spike_daily, sharpe_benchmark=psr_base['sharpe_obs'])
    print(f"\n  PSR (spike vs base benchmark={psr_base['sharpe_obs']:.2f}):")
    print(f"    PSR={psr_vs['psr']:.4f}  p={psr_vs['p_value']:.6f}")
    if psr_vs['p_value'] < 0.05:
        print(f"    ✓ Spike is significantly better than base at 5% level")
    else:
        print(f"    ✗ Spike is NOT significantly better than base at 5% level")

    # DSR: Adjusted for parameter searching (we tested ~16 configs in Phase 3)
    n_trials = 16
    dsr_spike = deflated_sharpe(spike_daily, n_trials=n_trials)
    print(f"\n  Deflated Sharpe Ratio (DSR, {n_trials} trials):")
    print(f"    Sharpe={dsr_spike['sharpe_obs']:.2f}  SR*={dsr_spike['sr_star']:.2f}  "
          f"DSR={dsr_spike['dsr']:.4f}  Passed={'✓' if dsr_spike['passed'] else '✗'}")

    # PBO: Probability of Backtest Overfitting
    daily_by_variant = {
        f"{base_label}_base": base_daily,
        f"{base_label}_spike": spike_daily,
    }
    pbo = compute_pbo(daily_by_variant, n_partitions=8)
    print(f"\n  Probability of Backtest Overfitting (PBO):")
    print(f"    PBO={pbo['pbo']:.2f}  Risk={pbo['overfit_risk']}  "
          f"Combinations={pbo['n_combinations']}")

# ═══════════════════════════════════════════════════════════════
# Phase 6: Year-by-year consistency
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("PHASE 6: YEAR-BY-YEAR CONSISTENCY")
print("=" * 80)

for base_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    base_full = run_variant(data, f"{base_label}", verbose=False, **base_kwargs)
    spike_full = run_variant(data, f"{base_label}+Spike", verbose=False, **base_kwargs,
                             atr_spike_protection=True,
                             atr_spike_threshold=1.5,
                             atr_spike_trail_mult=0.7)

    print(f"\n  {base_label} Year-by-Year:")
    print(f"  {'Year':<6} {'Base_N':>7} {'Spike_N':>8} {'Base_PnL':>10} {'Spike_PnL':>10} {'dPnL':>8} {'Base_WR':>8} {'Spike_WR':>9}")
    print(f"  {'-'*72}")

    for year in range(2015, 2027):
        start = f"{year}-01-01"
        end = f"{year+1}-01-01" if year < 2026 else "2026-04-01"
        bt = [t for t in base_full['_trades'] if start <= t.entry_time.strftime('%Y-%m-%d') < end]
        st = [t for t in spike_full['_trades'] if start <= t.entry_time.strftime('%Y-%m-%d') < end]
        if len(bt) < 10:
            continue
        bp = sum(t.pnl for t in bt)
        sp = sum(t.pnl for t in st)
        bw = 100 * sum(1 for t in bt if t.pnl > 0) / len(bt)
        sw = 100 * sum(1 for t in st if t.pnl > 0) / len(st) if st else 0
        marker = "  ✓" if sp > bp else ""
        print(f"  {year:<6} {len(bt):>7} {len(st):>8} ${bp:>9,.0f} ${sp:>9,.0f} ${sp-bp:>+7,.0f} "
              f"{bw:>7.1f}% {sw:>8.1f}%{marker}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

total_elapsed = time.time() - t_total
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"  Total runtime: {total_elapsed/60:.1f} minutes")
print(f"  Output saved to: {OUTPUT_FILE}")
print(f"  Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
