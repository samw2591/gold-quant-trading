#!/usr/bin/env python3
"""
Trailing Stop Evolution — Maximize the Core Alpha
====================================================
核心认知: "追踪止盈是核心 alpha，5542 笔 trailing +$41,088 (97.7%WR)"
探索更先进的追踪止盈机制:

1. 阶梯式追踪 (breakeven → 0.3ATR → 0.15ATR)
2. ATR 自适应追踪距离 (低波动紧追/高波动松追，但反过来测)
3. 利润百分比追踪 (回撤 30%/40%/50% 浮盈即出场)
4. 加速追踪 (持仓越久追踪越紧)
5. Ratchet 追踪 (每 0.5ATR 利润向上锁一阶)
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "trailing_evolution_output.txt"


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
print("TRAILING STOP EVOLUTION — MAXIMIZE CORE ALPHA")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

MEGA_D1_3H_BASE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
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

# The engine already has regime_config with different trail params per ATR regime.
# We test different base trail parameters and regime spread combinations.

variants = [
    # Current live
    {"label": "B: Current (0.5/0.15)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15},

    # Even tighter trail
    {"label": "T1: Tight (0.4/0.10)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.4, "trailing_distance_atr": 0.10},
    {"label": "T2: Very tight (0.3/0.08)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.3, "trailing_distance_atr": 0.08},

    # Wider trail (capture more trend)
    {"label": "T3: Wide (0.6/0.20)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.6, "trailing_distance_atr": 0.20},
    {"label": "T4: Very wide (0.8/0.25)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25},

    # Asymmetric: easy activate, wide trail
    {"label": "T5: EasyAct+WideDist (0.3/0.20)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.3, "trailing_distance_atr": 0.20},

    # Asymmetric: hard activate, tight trail
    {"label": "T6: HardAct+TightDist (0.7/0.10)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.7, "trailing_distance_atr": 0.10},

    # Extreme regime spread: very different low vs high
    {"label": "T7: ExtremeRegime", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
     "regime_config": {
         'low':    {'trail_act': 1.0, 'trail_dist': 0.35},
         'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
         'high':   {'trail_act': 0.25, 'trail_dist': 0.05},
     }},

    # No regime adaptation
    {"label": "T8: NoRegime (flat 0.5/0.15)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
     "regime_config": None},

    # No time-decay TP (isolate trail effect)
    {"label": "T9: NoDecay+Trail (0.5/0.15)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
     "time_decay_tp": False},

    # Tighter but no time-decay
    {"label": "T10: NoDecay+Tight (0.4/0.10)", **MEGA_D1_3H_BASE,
     "trailing_activate_atr": 0.4, "trailing_distance_atr": 0.10,
     "time_decay_tp": False},
]


results = []
for v in variants:
    label = v.pop("label")
    stats = run_variant(data, label, **v)

    kc = [t for t in stats['_trades'] if t.strategy == 'keltner']
    kc_pnl = sum(t.pnl for t in kc)
    kc_wins = sum(1 for t in kc if t.pnl > 0)
    trail_exits = sum(1 for t in kc if 'Trailing' in getattr(t, 'exit_reason', ''))
    decay_exits = sum(1 for t in kc if 'TimeDecay' in getattr(t, 'exit_reason', '') or 'time_decay' in getattr(t, 'exit_reason', ''))
    timeout_exits = sum(1 for t in kc if getattr(t, 'exit_reason', '').startswith('Timeout'))
    sl_exits = sum(1 for t in kc if getattr(t, 'exit_reason', '') == 'SL')

    stats['kc_n'] = len(kc)
    stats['kc_pnl'] = kc_pnl
    stats['kc_wr'] = 100.0 * kc_wins / len(kc) if kc else 0
    stats['kc_ppt'] = kc_pnl / len(kc) if kc else 0
    stats['trail'] = trail_exits
    stats['decay'] = decay_exits
    stats['timeout'] = timeout_exits
    stats['sl'] = sl_exits

    results.append(stats)
    v["label"] = label
    gc.collect()


print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
hdr = (f"{'Variant':<35} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} "
       f"{'KC_$/t':>7} {'Trail':>5} {'Decay':>5} {'SL':>4} {'TM':>4}")
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:<35} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} "
          f"${r['kc_ppt']:>6.2f} {r['trail']:>5} {r['decay']:>5} {r['sl']:>4} {r['timeout']:>4}")


# Year-by-year best vs baseline
base_sh = results[0]['sharpe']
non_base = [r for r in results[1:]]
if non_base:
    best = max(non_base, key=lambda r: r['sharpe'])
    print(f"\n\nYEAR-BY-YEAR: {best['label']} vs Baseline")
    print("-" * 70)

    def yearly(trades):
        yrs = defaultdict(list)
        for t in trades:
            yrs[t.entry_time.year].append(t)
        out = {}
        for yr in sorted(yrs.keys()):
            ts = yrs[yr]
            pnl = sum(t.pnl for t in ts)
            eq = [0.0]
            for t in ts:
                eq.append(eq[-1] + t.pnl)
            s = calc_stats(ts, eq)
            out[yr] = {'n': len(ts), 'pnl': pnl, 'sharpe': s['sharpe']}
        return out

    b_yr = yearly(results[0]['_trades'])
    best_yr = yearly(best['_trades'])

    print(f"{'Year':<6} {'B_N':>5} {'B_Sh':>7} {'B_PnL':>10} | {'Best_N':>5} {'Best_Sh':>7} {'Best_PnL':>10} {'ΔSh':>7}")
    for yr in sorted(set(list(b_yr.keys()) + list(best_yr.keys()))):
        b = b_yr.get(yr, {'n': 0, 'sharpe': 0, 'pnl': 0})
        d = best_yr.get(yr, {'n': 0, 'sharpe': 0, 'pnl': 0})
        print(f"{yr:<6} {b['n']:>5} {b['sharpe']:>7.2f} ${b['pnl']:>9,.0f} | "
              f"{d['n']:>5} {d['sharpe']:>7.2f} ${d['pnl']:>9,.0f} {d['sharpe']-b['sharpe']:>+7.2f}")


print(f"\n\nCONCLUSION")
print("=" * 80)
print(f"Baseline: Sharpe={base_sh:.2f}")
for r in sorted(results[1:], key=lambda x: x['sharpe'], reverse=True):
    delta = r['sharpe'] - base_sh
    marker = " <<<" if delta > 0.05 else ""
    print(f"  {r['label']:<35} Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f}) KC_$/t=${r['kc_ppt']:.2f}{marker}")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
