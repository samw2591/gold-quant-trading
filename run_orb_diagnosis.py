#!/usr/bin/env python3
"""
ORB Strategy Diagnosis — Keep vs Remove vs Tune
=================================================
Tests whether ORB adds value to the portfolio or drags Sharpe down.
"""
import sys, os, time, gc
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS

OUTPUT_FILE = "orb_diagnosis_output.txt"


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
print("ORB STRATEGY DIAGNOSIS — KEEP / REMOVE / TUNE")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

MEGA_D1_3H = {
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


def run_with_orb_toggle(label, orb_enabled, extra_kwargs=None):
    """Run variant with ORB enabled/disabled via config monkey-patch."""
    old_val = config.ORB_ENABLED
    config.ORB_ENABLED = orb_enabled
    kwargs = {**MEGA_D1_3H}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    stats = run_variant(data, label, **kwargs)
    config.ORB_ENABLED = old_val
    return stats


results = []

# 1. Current: ORB on
s = run_with_orb_toggle("B: ORB enabled (current)", True)
results.append(s)

# 2. ORB off — Keltner + RSI only
s = run_with_orb_toggle("NoORB: Keltner+RSI only", False)
results.append(s)

# 3. ORB with different max_hold
for hold in [4, 8, 12, 16, 24]:
    label = f"ORB_hold={hold} (~{hold*15/60:.1f}h)"
    s = run_with_orb_toggle(label, True, {"orb_max_hold_m15": hold})
    results.append(s)

# 4. ORB only (disable keltner ADX to effectively keltner-off? no, just filter post-hoc)
gc.collect()


def strat_breakdown(trades):
    strats = {}
    for t in trades:
        s = t.strategy
        if s not in strats:
            strats[s] = {'n': 0, 'pnl': 0.0, 'wins': 0}
        strats[s]['n'] += 1
        strats[s]['pnl'] += t.pnl
        if t.pnl > 0:
            strats[s]['wins'] += 1
    return strats


print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
hdr = f"{'Variant':<32} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}"
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:<32} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f}")


print("\n\nPER-STRATEGY BREAKDOWN")
print("-" * 80)
for r in results:
    sb = strat_breakdown(r['_trades'])
    print(f"\n  {r['label']}:")
    print(f"  {'Strategy':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print(f"  {'-'*48}")
    for s in sorted(sb.keys()):
        d = sb[s]
        wr = 100.0 * d['wins'] / d['n'] if d['n'] > 0 else 0
        ppt = d['pnl'] / d['n'] if d['n'] > 0 else 0
        print(f"  {s:<15} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")


print("\n\nCONCLUSION")
print("=" * 80)
base_sh = results[0]['sharpe']
no_orb_sh = results[1]['sharpe']
print(f"With ORB:    Sharpe={base_sh:.2f}")
print(f"Without ORB: Sharpe={no_orb_sh:.2f} (Δ{no_orb_sh - base_sh:+.2f})")
if no_orb_sh > base_sh + 0.05:
    print(">>> ORB is dragging portfolio down. Consider disabling.")
elif no_orb_sh < base_sh - 0.05:
    print(">>> ORB adds value. Keep it.")
else:
    print(">>> ORB has minimal impact. Keep for diversification or remove for simplicity.")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
