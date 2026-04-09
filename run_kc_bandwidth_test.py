#!/usr/bin/env python3
"""
KC Bandwidth Expanding Filter Test
====================================
Tests whether requiring KC bandwidth to be expanding at entry improves Keltner.
Uses various lookback periods (3, 5, 8, 12 bars).
"""
import sys, os, time, gc
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS

OUTPUT_FILE = "kc_bandwidth_output.txt"


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
print("KC BANDWIDTH EXPANDING FILTER TEST")
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

variants = [
    {"label": "B: D1+3h (no BW filter)", **MEGA_D1_3H},
    {"label": "BW3: expanding over 3 bars", **MEGA_D1_3H, "kc_bw_filter_bars": 3},
    {"label": "BW5: expanding over 5 bars", **MEGA_D1_3H, "kc_bw_filter_bars": 5},
    {"label": "BW8: expanding over 8 bars", **MEGA_D1_3H, "kc_bw_filter_bars": 8},
    {"label": "BW12: expanding over 12 bars", **MEGA_D1_3H, "kc_bw_filter_bars": 12},
]

results = []
for v in variants:
    label = v.pop("label")
    stats = run_variant(data, label, **v)

    kc = [t for t in stats['_trades'] if t.strategy == 'keltner']
    kc_pnl = sum(t.pnl for t in kc)
    kc_wins = sum(1 for t in kc if t.pnl > 0)
    stats['kc_n'] = len(kc)
    stats['kc_pnl'] = kc_pnl
    stats['kc_wr'] = 100.0 * kc_wins / len(kc) if kc else 0
    stats['kc_ppt'] = kc_pnl / len(kc) if kc else 0

    results.append(stats)
    v["label"] = label
    gc.collect()


print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
hdr = (f"{'Variant':<32} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} "
       f"{'KC_N':>6} {'KC_PnL':>9} {'KC_$/t':>7} {'KC_WR':>6} {'BW_skip':>7}")
print(hdr)
print("-" * len(hdr))

for r in results:
    print(f"{r['label']:<32} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} "
          f"{r['kc_n']:>6} ${r['kc_pnl']:>8,.0f} ${r['kc_ppt']:>6.2f} {r['kc_wr']:>5.1f}% "
          f"{r.get('skipped_kc_bw', 0):>7}")


# Year-by-year for best BW filter
from collections import defaultdict

baseline_sh = results[0]['sharpe']
bw_results = results[1:]
if bw_results:
    best_bw = max(bw_results, key=lambda r: r['sharpe'])
    print(f"\n\nYEAR-BY-YEAR: {best_bw['label']} vs Baseline")
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

    from backtest.stats import calc_stats
    b_yr = yearly(results[0]['_trades'])
    best_yr = yearly(best_bw['_trades'])

    print(f"{'Year':<6} {'B_N':>5} {'B_Sh':>7} {'B_PnL':>10} | {'BW_N':>5} {'BW_Sh':>7} {'BW_PnL':>10} {'ΔSh':>7}")
    print("-" * 70)
    for yr in sorted(set(list(b_yr.keys()) + list(best_yr.keys()))):
        b = b_yr.get(yr, {'n': 0, 'sharpe': 0, 'pnl': 0})
        d = best_yr.get(yr, {'n': 0, 'sharpe': 0, 'pnl': 0})
        print(f"{yr:<6} {b['n']:>5} {b['sharpe']:>7.2f} ${b['pnl']:>9,.0f} | "
              f"{d['n']:>5} {d['sharpe']:>7.2f} ${d['pnl']:>9,.0f} {d['sharpe']-b['sharpe']:>+7.2f}")

print(f"\n\nCONCLUSION")
print("=" * 80)
print(f"Baseline: Sharpe={baseline_sh:.2f}")
for r in results[1:]:
    delta = r['sharpe'] - baseline_sh
    marker = " <<<" if delta > 0.05 else ""
    print(f"  {r['label']:<32} Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f}) KC_$/t=${r['kc_ppt']:.2f}{marker}")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
