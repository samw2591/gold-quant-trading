#!/usr/bin/env python3
"""
Exit Mechanism Combo Matrix — SL × Trail × Decay × MaxHold
=============================================================
核心认知: "追踪止盈是核心 alpha" + "改善方向聚焦风控"
穷举出场机制组合，找全局最优:
- SL: 4.0 / 4.5 / 5.0
- Trail activate: 0.4 / 0.5 / 0.6
- Trail distance: 0.10 / 0.15 / 0.20
- Max hold: 3h / 4h / 5h
= 3×3×3×3 = 81 组合
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS

OUTPUT_FILE = "exit_combo_matrix_output.txt"


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
print("EXIT MECHANISM COMBO MATRIX (SL × Trail × MaxHold)")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

SL_RANGE = [4.0, 4.5, 5.0]
TRAIL_ACT = [0.4, 0.5, 0.6]
TRAIL_DIST = [0.10, 0.15, 0.20]
MAX_HOLD = [12, 16, 20]  # M15 bars (3h, 4h, 5h)

total_combos = len(SL_RANGE) * len(TRAIL_ACT) * len(TRAIL_DIST) * len(MAX_HOLD)
print(f"Total combinations: {total_combos}")

results = []
idx = 0
for sl in SL_RANGE:
    for ta in TRAIL_ACT:
        for td in TRAIL_DIST:
            for mh in MAX_HOLD:
                idx += 1
                label = f"SL{sl}_TA{ta}_TD{td}_MH{mh//4}h"

                kwargs = {
                    **C12_KWARGS,
                    "intraday_adaptive": True,
                    "sl_atr_mult": sl,
                    "trailing_activate_atr": ta,
                    "trailing_distance_atr": td,
                    "regime_config": {
                        'low':    {'trail_act': ta + 0.2, 'trail_dist': td + 0.10},
                        'normal': {'trail_act': ta, 'trail_dist': td},
                        'high':   {'trail_act': max(0.2, ta - 0.1), 'trail_dist': max(0.05, td - 0.05)},
                    },
                    "keltner_max_hold_m15": mh,
                    "time_decay_tp": True,
                    "time_decay_start_hour": 1.0,
                    "time_decay_atr_start": 0.30,
                    "time_decay_atr_step": 0.10,
                }

                stats = run_variant(data, label, verbose=(idx % 10 == 1), **kwargs)
                kc = [t for t in stats['_trades'] if t.strategy == 'keltner']
                stats['kc_n'] = len(kc)
                stats['kc_pnl'] = sum(t.pnl for t in kc)
                stats['kc_ppt'] = stats['kc_pnl'] / len(kc) if kc else 0
                results.append(stats)

                if idx % 10 == 0:
                    print(f"  Progress: {idx}/{total_combos}", flush=True)

                gc.collect()

# Sort by Sharpe
results.sort(key=lambda r: r['sharpe'], reverse=True)

print("\n\n" + "=" * 80)
print(f"TOP 20 COMBINATIONS (of {total_combos})")
print("=" * 80)
hdr = f"{'Combo':<25} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} {'KC_$/t':>7}"
print(hdr)
print("-" * len(hdr))
for r in results[:20]:
    print(f"{r['label']:<25} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} ${r['kc_ppt']:>6.2f}")


print(f"\n\nBOTTOM 10 (worst)")
print("-" * 70)
for r in results[-10:]:
    print(f"{r['label']:<25} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f}")


# Find current live config
current = [r for r in results if 'SL4.5_TA0.5_TD0.15_MH3h' in r['label']]
if current:
    c = current[0]
    c_rank = results.index(c) + 1
    print(f"\n\nCURRENT LIVE CONFIG: {c['label']}")
    print(f"  Rank: #{c_rank} / {total_combos}")
    print(f"  Sharpe: {c['sharpe']:.2f}, PnL: ${c['total_pnl']:,.0f}")
    best = results[0]
    print(f"\n  BEST: {best['label']}")
    print(f"  Sharpe: {best['sharpe']:.2f} (Δ{best['sharpe']-c['sharpe']:+.2f})")
    print(f"  PnL: ${best['total_pnl']:,.0f} (Δ${best['total_pnl']-c['total_pnl']:+,.0f})")


# Marginal analysis: which parameter matters most?
print(f"\n\n" + "=" * 80)
print("MARGINAL ANALYSIS — Which parameter matters most?")
print("=" * 80)

import numpy as np

for param_name, param_vals in [("SL", SL_RANGE), ("Trail_Act", TRAIL_ACT), ("Trail_Dist", TRAIL_DIST), ("MaxHold", MAX_HOLD)]:
    sharpes_by_val = {}
    for val in param_vals:
        matches = [r for r in results if
                   (param_name == "SL" and f"SL{val}_" in r['label']) or
                   (param_name == "Trail_Act" and f"_TA{val}_" in r['label']) or
                   (param_name == "Trail_Dist" and f"_TD{val}_" in r['label']) or
                   (param_name == "MaxHold" and f"_MH{val//4}h" in r['label'])]
        sharpes = [r['sharpe'] for r in matches]
        sharpes_by_val[val] = np.mean(sharpes) if sharpes else 0

    print(f"\n  {param_name}:")
    for val in param_vals:
        print(f"    {val}: avg Sharpe = {sharpes_by_val[val]:.2f}")
    best_val = max(sharpes_by_val, key=sharpes_by_val.get)
    worst_val = min(sharpes_by_val, key=sharpes_by_val.get)
    impact = sharpes_by_val[best_val] - sharpes_by_val[worst_val]
    print(f"    Impact: {impact:.2f} (best={best_val}, worst={worst_val})")


total_elapsed = time.time() - t_total
print(f"\n\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
