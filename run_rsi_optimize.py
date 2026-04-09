#!/usr/bin/env python3
"""
M15 RSI Strategy Parameter Optimization
=========================================
Tests RSI max_hold, trailing stop for RSI, and time-decay TP for RSI.
Based on D1+3h Mega Trail baseline.
"""
import sys, os, time, gc
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "rsi_optimize_output.txt"


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
print("M15 RSI STRATEGY PARAMETER OPTIMIZATION")
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
    "keltner_max_hold_m15": 12,  # 3h
    "time_decay_tp": True,
    "time_decay_start_hour": 1.0,
    "time_decay_atr_start": 0.30,
    "time_decay_atr_step": 0.10,
}

variants = [
    # Baseline
    {"label": "B: D1+3h (current live)", **MEGA_D1_3H},

    # RSI max_hold sweep
    {"label": "R1: rsi_hold=8 (~2h)", **MEGA_D1_3H, "rsi_max_hold_m15": 8},
    {"label": "R2: rsi_hold=12 (~3h)", **MEGA_D1_3H, "rsi_max_hold_m15": 12},
    {"label": "R3: rsi_hold=16 (~4h)", **MEGA_D1_3H, "rsi_max_hold_m15": 16},
    {"label": "R4: rsi_hold=24 (~6h)", **MEGA_D1_3H, "rsi_max_hold_m15": 24},
    {"label": "R5: rsi_hold=4 (~1h)", **MEGA_D1_3H, "rsi_max_hold_m15": 4},

    # RSI sell disabled
    {"label": "R6: rsi_sell=off", **MEGA_D1_3H, "rsi_sell_enabled": False},

    # RSI ADX filter
    {"label": "R7: rsi_adx<30", **MEGA_D1_3H, "rsi_adx_filter": 30},
    {"label": "R8: rsi_adx<35", **MEGA_D1_3H, "rsi_adx_filter": 35},

    # RSI threshold tuning
    {"label": "R9: buy<10 sell>90", **MEGA_D1_3H, "rsi_buy_threshold": 10, "rsi_sell_threshold": 90},
    {"label": "R10: buy<3 sell>97", **MEGA_D1_3H, "rsi_buy_threshold": 3, "rsi_sell_threshold": 97},
]


def exit_breakdown(trades, strategy_filter=None):
    counts = Counter()
    for t in trades:
        if strategy_filter and t.strategy != strategy_filter:
            continue
        r = t.exit_reason
        if "Trailing" in r:
            counts['Trail'] += 1
        elif "TimeDecay" in r:
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


def rsi_stats(trades):
    rsi_t = [t for t in trades if t.strategy == 'm15_rsi']
    if not rsi_t:
        return {'n': 0, 'pnl': 0, 'wr': 0, 'ppt': 0}
    pnl = sum(t.pnl for t in rsi_t)
    wins = sum(1 for t in rsi_t if t.pnl > 0)
    return {'n': len(rsi_t), 'pnl': pnl, 'wr': 100.0 * wins / len(rsi_t), 'ppt': pnl / len(rsi_t)}


results = []
for v in variants:
    label = v.pop("label")
    stats = run_variant(data, label, **v)
    stats['rsi'] = rsi_stats(stats['_trades'])
    stats['rsi_exits'] = exit_breakdown(stats['_trades'], 'm15_rsi')
    results.append(stats)
    v["label"] = label
    gc.collect()


print("\n\n" + "=" * 80)
print("SUMMARY — FULL PORTFOLIO")
print("=" * 80)
hdr = f"{'Variant':<30} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8}"
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:<30} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f}")

print("\n\nRSI-ONLY BREAKDOWN")
print("-" * 80)
hdr2 = f"{'Variant':<30} {'RSI_N':>6} {'RSI_PnL':>10} {'RSI_WR%':>8} {'$/t':>7}  {'Trail':>5} {'Decay':>5} {'Tmout':>5} {'SL':>5} {'Other':>5}"
print(hdr2)
print("-" * len(hdr2))
for r in results:
    rs = r['rsi']
    e = r['rsi_exits']
    print(f"{r['label']:<30} {rs['n']:>6} ${rs['pnl']:>9,.0f} {rs['wr']:>7.1f}% ${rs['ppt']:>6.2f}  "
          f"{e.get('Trail',0):>5} {e.get('Decay',0):>5} {e.get('Tmout',0):>5} {e.get('SL',0):>5} {e.get('Other',0):>5}")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
