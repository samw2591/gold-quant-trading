#!/usr/bin/env python3
"""
TP ATR Multiplier Sweep (with Mega Trail + Time Decay)
========================================================
核心认知: "SL=4.5, TP=8.0 是旧配置下的最优, 现在有 trail+decay 后 TP 可能需要调整"
TP 太大 → 永远不触达，全靠 trail/decay 出场 → TP 形同虚设
TP 太小 → 提前锁利，但可能截断大趋势

测试 TP 从 4.0 到 12.0 ATR，以及完全关闭 TP (TP=0, 全靠 trail+decay+timeout)
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "tp_atr_sweep_output.txt"


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
print("TP ATR MULTIPLIER SWEEP (Mega Trail + Time Decay)")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

BASE = {
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

variants = []
for tp in [0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]:
    if tp == 0:
        label = "TP=off (trail+decay only)"
    else:
        label = f"TP={tp:.1f} ATR"
    variants.append({"label": label, **BASE, "tp_atr_mult": tp})

results = []
for v in variants:
    label = v.pop("label")
    stats = run_variant(data, label, **v)

    kc = [t for t in stats['_trades'] if t.strategy == 'keltner']
    kc_pnl = sum(t.pnl for t in kc)
    tp_exits = sum(1 for t in kc if getattr(t, 'exit_reason', '') == 'TP')
    trail_exits = sum(1 for t in kc if 'Trailing' in getattr(t, 'exit_reason', ''))
    decay_exits = sum(1 for t in kc if 'TimeDecay' in getattr(t, 'exit_reason', '') or 'time_decay' in getattr(t, 'exit_reason', ''))
    sl_exits = sum(1 for t in kc if getattr(t, 'exit_reason', '') == 'SL')

    stats['kc_n'] = len(kc)
    stats['kc_pnl'] = kc_pnl
    stats['kc_ppt'] = kc_pnl / len(kc) if kc else 0
    stats['tp_exits'] = tp_exits
    stats['trail_exits'] = trail_exits
    stats['decay_exits'] = decay_exits
    stats['sl_exits'] = sl_exits

    results.append(stats)
    v["label"] = label
    gc.collect()


print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
hdr = (f"{'TP':>28} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} "
       f"{'KC_$/t':>7} {'TP':>4} {'Trail':>5} {'Decay':>5} {'SL':>4}")
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:>28} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} "
          f"${r['kc_ppt']:>6.2f} {r['tp_exits']:>4} {r['trail_exits']:>5} {r['decay_exits']:>5} {r['sl_exits']:>4}")


print(f"\n\nCONCLUSION")
print("=" * 80)
best = max(results, key=lambda r: r['sharpe'])
current = [r for r in results if 'TP=8.0' in r['label']][0]
print(f"Best: {best['label']} with Sharpe={best['sharpe']:.2f}")
print(f"Current: {current['label']} with Sharpe={current['sharpe']:.2f}")
if best['label'] != current['label']:
    delta = best['sharpe'] - current['sharpe']
    print(f"  Improvement: Δ{delta:+.2f} Sharpe")

tp_off = [r for r in results if 'TP=off' in r['label']][0]
print(f"\nTP=off (trail+decay only): Sharpe={tp_off['sharpe']:.2f} (Δ{tp_off['sharpe']-current['sharpe']:+.2f})")
if tp_off['sharpe'] > current['sharpe']:
    print("  >>> TP is redundant when trail+decay exits are active!")
else:
    print("  >>> TP still adds value as a safety net.")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
