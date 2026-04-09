#!/usr/bin/env python3
"""
Time-Decay Take-Profit Backtest
================================
Tests 8 variants of time-decay TP on 11-year H1 Keltner data.
Uses Mega Trail as baseline, compares with/without decay + parameter combos.

Output: Sharpe, PnL, Trades, WinRate, MaxDD, PF, exit-reason breakdown.
"""
import sys, os, time, gc
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "time_decay_tp_output.txt"


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
print("TIME-DECAY TAKE-PROFIT BACKTEST")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

MEGA_BASE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high':   {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

# keltner_max_hold_m15: 5h = 20 M15 bars
MAX_HOLD_5H = 20
MAX_HOLD_3H = 12

variants = [
    # Section 1: Baselines
    {
        "label": "B1: Mega (no hold limit)",
        **MEGA_BASE,
    },
    {
        "label": "B2: Mega + max_hold=5h",
        **MEGA_BASE,
        "keltner_max_hold_m15": MAX_HOLD_5H,
    },

    # Section 2: Decay parameter grid (no hold limit)
    {
        "label": "D1: start=1h atr=0.30 step=0.10/h",
        **MEGA_BASE,
        "time_decay_tp": True,
        "time_decay_start_hour": 1.0,
        "time_decay_atr_start": 0.30,
        "time_decay_atr_step": 0.10,
    },
    {
        "label": "D2: start=1h atr=0.20 step=0.07/h",
        **MEGA_BASE,
        "time_decay_tp": True,
        "time_decay_start_hour": 1.0,
        "time_decay_atr_start": 0.20,
        "time_decay_atr_step": 0.07,
    },
    {
        "label": "D3: start=0.5h atr=0.30 step=0.15/h",
        **MEGA_BASE,
        "time_decay_tp": True,
        "time_decay_start_hour": 0.5,
        "time_decay_atr_start": 0.30,
        "time_decay_atr_step": 0.15,
    },
    {
        "label": "D4: start=1h atr=0.40 step=0.10/h",
        **MEGA_BASE,
        "time_decay_tp": True,
        "time_decay_start_hour": 1.0,
        "time_decay_atr_start": 0.40,
        "time_decay_atr_step": 0.10,
    },

    # Section 3: Decay + short hold combos
    {
        "label": "D1+5h: decay + max_hold=5h",
        **MEGA_BASE,
        "keltner_max_hold_m15": MAX_HOLD_5H,
        "time_decay_tp": True,
        "time_decay_start_hour": 1.0,
        "time_decay_atr_start": 0.30,
        "time_decay_atr_step": 0.10,
    },
    {
        "label": "D1+3h: decay + max_hold=3h",
        **MEGA_BASE,
        "keltner_max_hold_m15": MAX_HOLD_3H,
        "time_decay_tp": True,
        "time_decay_start_hour": 1.0,
        "time_decay_atr_start": 0.30,
        "time_decay_atr_step": 0.10,
    },
]


def exit_breakdown(trades):
    """Count exits by reason category for keltner trades."""
    counts = Counter()
    for t in trades:
        if t.strategy != 'keltner':
            continue
        r = t.exit_reason
        if r == "Trailing":
            counts['Trail'] += 1
        elif r == "TimeDecayTP":
            counts['Decay'] += 1
        elif r.startswith("Timeout"):
            counts['Tmout'] += 1
        elif r == "SL":
            counts['SL'] += 1
        elif r == "TP":
            counts['TP'] += 1
        else:
            counts['Other'] += 1
    return counts


def profit_factor(trades):
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    return gross_win / gross_loss if gross_loss > 0 else float('inf')


# Run all variants
results = []
for v in variants:
    label = v.pop("label")
    stats = run_variant(data, label, **v)
    stats['pf'] = profit_factor(stats['_trades'])
    stats['exits'] = exit_breakdown(stats['_trades'])

    kc_trades = [t for t in stats['_trades'] if t.strategy == 'keltner']
    kc_pnl = sum(t.pnl for t in kc_trades)
    kc_wins = sum(1 for t in kc_trades if t.pnl > 0)
    stats['kc_n'] = len(kc_trades)
    stats['kc_pnl'] = kc_pnl
    stats['kc_wr'] = 100.0 * kc_wins / len(kc_trades) if kc_trades else 0

    results.append(stats)
    v["label"] = label  # restore
    gc.collect()


# ── Summary Table ──
print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

hdr = (f"{'Variant':<35} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} "
       f"{'MaxDD':>8} {'PF':>6}  {'Trail':>5} {'Decay':>5} {'Tmout':>5} {'SL':>5} {'TP':>5}")
print(hdr)
print("-" * len(hdr))

for i, r in enumerate(results):
    e = r['exits']
    print(f"{r['label']:<35} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} {r['pf']:>6.2f}  "
          f"{e.get('Trail',0):>5} {e.get('Decay',0):>5} {e.get('Tmout',0):>5} "
          f"{e.get('SL',0):>5} {e.get('TP',0):>5}")

# ── Keltner-only breakdown ──
print("\n\nKELTNER-ONLY BREAKDOWN")
print("-" * 70)
hdr2 = f"{'Variant':<35} {'KC_N':>6} {'KC_PnL':>10} {'KC_WR%':>7} {'$/t':>7}"
print(hdr2)
print("-" * len(hdr2))
for r in results:
    ppt = r['kc_pnl'] / r['kc_n'] if r['kc_n'] > 0 else 0
    print(f"{r['label']:<35} {r['kc_n']:>6} ${r['kc_pnl']:>9,.0f} {r['kc_wr']:>6.1f}% ${ppt:>6.2f}")

# ── Year-by-year for best decay variant ──
baseline_sharpe = results[0]['sharpe']
best_decay = max(results[2:6], key=lambda r: r['sharpe'])
print(f"\n\nYEAR-BY-YEAR: {best_decay['label']} vs B1 (Mega baseline)")
print("-" * 70)

from collections import defaultdict


def yearly_stats(trades):
    yearly = defaultdict(list)
    for t in trades:
        yr = t.entry_time.year
        yearly[yr].append(t)
    result = {}
    for yr in sorted(yearly.keys()):
        ts = yearly[yr]
        pnl = sum(t.pnl for t in ts)
        wins = sum(1 for t in ts if t.pnl > 0)
        eq = [0.0]
        for t in ts:
            eq.append(eq[-1] + t.pnl)
        s = calc_stats(ts, eq)
        result[yr] = {
            'n': len(ts), 'pnl': pnl,
            'wr': 100.0 * wins / len(ts) if ts else 0,
            'sharpe': s['sharpe'],
        }
    return result


base_yearly = yearly_stats(results[0]['_trades'])
best_yearly = yearly_stats(best_decay['_trades'])

print(f"{'Year':<6} {'B1_N':>5} {'B1_Sh':>7} {'B1_PnL':>10} | "
      f"{'Best_N':>5} {'Best_Sh':>7} {'Best_PnL':>10} {'ΔSh':>7}")
print("-" * 72)
for yr in sorted(set(list(base_yearly.keys()) + list(best_yearly.keys()))):
    b = base_yearly.get(yr, {'n': 0, 'sharpe': 0, 'pnl': 0})
    d = best_yearly.get(yr, {'n': 0, 'sharpe': 0, 'pnl': 0})
    delta_sh = d['sharpe'] - b['sharpe']
    print(f"{yr:<6} {b['n']:>5} {b['sharpe']:>7.2f} ${b['pnl']:>9,.0f} | "
          f"{d['n']:>5} {d['sharpe']:>7.2f} ${d['pnl']:>9,.0f} {delta_sh:>+7.2f}")


# ── Conclusion ──
print("\n\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

improvements = [(r['label'], r['sharpe'] - baseline_sharpe, r['sharpe'], r['total_pnl'])
                for r in results[2:]]
improvements.sort(key=lambda x: x[1], reverse=True)

print(f"\nBaseline (Mega): Sharpe={baseline_sharpe:.2f} PnL=${results[0]['total_pnl']:,.0f}")
print(f"\nDecay variants ranked by Sharpe improvement:")
for label, delta, sh, pnl in improvements:
    marker = " <<<" if delta > 0.05 else ""
    print(f"  {label:<35} Sharpe={sh:.2f} (Δ{delta:+.2f}) PnL=${pnl:,.0f}{marker}")

if improvements[0][1] > 0:
    print(f"\n  >>> BEST: {improvements[0][0]} with Sharpe +{improvements[0][1]:.2f}")
    print(f"      Consider deploying to live after review.")
else:
    print(f"\n  >>> No decay variant improves Sharpe. Time-decay TP may not help.")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
