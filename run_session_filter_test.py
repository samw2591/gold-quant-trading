#!/usr/bin/env python3
"""
Session Filter Test — Which trading hours produce best Keltner signals?
========================================================================
Post-hoc analysis + filter backtests.
Asian(0-7), London(7-14), NY(14-21), Overlap(14-17), Late(21-24)
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "session_filter_output.txt"


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
print("SESSION FILTER TEST — KELTNER ENTRY HOUR ANALYSIS")
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

ASIAN = list(range(0, 7))
LONDON = list(range(7, 14))
NY = list(range(14, 21))
OVERLAP = list(range(14, 18))
LONDON_NY = list(range(7, 21))
NO_ASIAN = list(range(7, 24))
PRIME = list(range(8, 20))

variants = [
    {"label": "B: No session filter", **MEGA_D1_3H},
    {"label": "S1: London only (7-14)", **MEGA_D1_3H, "h1_allowed_sessions": LONDON},
    {"label": "S2: NY only (14-21)", **MEGA_D1_3H, "h1_allowed_sessions": NY},
    {"label": "S3: London+NY (7-21)", **MEGA_D1_3H, "h1_allowed_sessions": LONDON_NY},
    {"label": "S4: No Asian (7-24)", **MEGA_D1_3H, "h1_allowed_sessions": NO_ASIAN},
    {"label": "S5: Prime hours (8-20)", **MEGA_D1_3H, "h1_allowed_sessions": PRIME},
    {"label": "S6: Overlap only (14-18)", **MEGA_D1_3H, "h1_allowed_sessions": OVERLAP},
]


# ── Part 1: Run baseline for post-hoc analysis ──
print("\n--- Part 1: Post-hoc hour-by-hour analysis ---")
base_stats = run_variant(data, "Baseline", **MEGA_D1_3H)
trades = base_stats['_trades']

kc_trades = [t for t in trades if t.strategy == 'keltner']

hour_stats = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0})
for t in kc_trades:
    h = t.entry_time.hour
    hour_stats[h]['n'] += 1
    hour_stats[h]['pnl'] += t.pnl
    if t.pnl > 0:
        hour_stats[h]['wins'] += 1

print(f"\nKELTNER TRADES BY ENTRY HOUR (UTC)")
print(f"{'Hour':>4} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 38)
for h in range(24):
    d = hour_stats.get(h, {'n': 0, 'pnl': 0, 'wins': 0})
    if d['n'] > 0:
        print(f"{h:>4} {d['n']:>6} ${d['pnl']:>9,.0f} ${d['pnl']/d['n']:>6.2f} {100*d['wins']/d['n']:>5.1f}%")

# Session grouping
sessions = {
    'Asian (0-7)': [t for t in kc_trades if 0 <= t.entry_time.hour < 7],
    'London (7-14)': [t for t in kc_trades if 7 <= t.entry_time.hour < 14],
    'NY (14-21)': [t for t in kc_trades if 14 <= t.entry_time.hour < 21],
    'Late (21-24)': [t for t in kc_trades if 21 <= t.entry_time.hour < 24],
}

print(f"\nKELTNER BY SESSION")
print(f"{'Session':<18} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 50)
for s, ts in sessions.items():
    if ts:
        pnl = sum(t.pnl for t in ts)
        wins = sum(1 for t in ts if t.pnl > 0)
        print(f"{s:<18} {len(ts):>6} ${pnl:>9,.0f} ${pnl/len(ts):>6.2f} {100*wins/len(ts):>5.1f}%")

# M15 RSI by session
rsi_trades = [t for t in trades if t.strategy == 'm15_rsi']
rsi_sessions = {
    'Asian (0-7)': [t for t in rsi_trades if 0 <= t.entry_time.hour < 7],
    'London (7-14)': [t for t in rsi_trades if 7 <= t.entry_time.hour < 14],
    'NY (14-21)': [t for t in rsi_trades if 14 <= t.entry_time.hour < 21],
    'Late (21-24)': [t for t in rsi_trades if 21 <= t.entry_time.hour < 24],
}

print(f"\nM15 RSI BY SESSION")
print(f"{'Session':<18} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 50)
for s, ts in rsi_sessions.items():
    if ts:
        pnl = sum(t.pnl for t in ts)
        wins = sum(1 for t in ts if t.pnl > 0)
        print(f"{s:<18} {len(ts):>6} ${pnl:>9,.0f} ${pnl/len(ts):>6.2f} {100*wins/len(ts):>5.1f}%")


# ── Part 2: Filter backtests ──
print(f"\n\n--- Part 2: Session filter backtests ---")
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
       f"{'KC_N':>6} {'KC_$/t':>7} {'Sess_skip':>9}")
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:<32} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} "
          f"{r['kc_n']:>6} ${r['kc_ppt']:>6.2f} {r.get('skipped_session',0):>9}")

print(f"\nCONCLUSION")
print("=" * 80)
base_sh = results[0]['sharpe']
for r in results[1:]:
    delta = r['sharpe'] - base_sh
    marker = " <<<" if delta > 0.05 else ""
    print(f"  {r['label']:<32} Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f}) KC_$/t=${r['kc_ppt']:.2f}{marker}")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
