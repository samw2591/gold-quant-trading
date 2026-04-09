#!/usr/bin/env python3
"""
Stop Loss Fine-tuning with Mega Trail + Time Decay
=====================================================
核心认知: "SL 4.5 ATR 已实装，但配合新的 Mega Trail + Time Decay 后最优 SL 可能变了"
测试 SL 从 3.0 到 6.0 ATR，在当前 D1+3h 配置下的表现。
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "sl_optimization_output.txt"


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
print("STOP LOSS FINE-TUNING (Mega Trail + Time Decay context)")
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
for sl in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
    label = f"SL={sl:.1f} ATR"
    v = {**BASE, "sl_atr_mult": sl, "label": label}
    variants.append(v)

# Also test asymmetric: different SL for BUY vs SELL (post-hoc analysis)
results = []
for v in variants:
    label = v.pop("label")
    stats = run_variant(data, label, **v)

    kc = [t for t in stats['_trades'] if t.strategy == 'keltner']
    sl_exits = sum(1 for t in kc if getattr(t, 'exit_reason', '') == 'SL')
    trail_exits = sum(1 for t in kc if 'Trailing' in getattr(t, 'exit_reason', ''))
    kc_pnl = sum(t.pnl for t in kc)
    stats['kc_n'] = len(kc)
    stats['kc_pnl'] = kc_pnl
    stats['kc_ppt'] = kc_pnl / len(kc) if kc else 0
    stats['sl_exits'] = sl_exits
    stats['trail_exits'] = trail_exits
    stats['sl_rate'] = 100.0 * sl_exits / len(kc) if kc else 0

    # BUY vs SELL breakdown
    buy_kc = [t for t in kc if t.direction == 'BUY']
    sell_kc = [t for t in kc if t.direction == 'SELL']
    stats['buy_pnl'] = sum(t.pnl for t in buy_kc)
    stats['sell_pnl'] = sum(t.pnl for t in sell_kc)
    stats['buy_n'] = len(buy_kc)
    stats['sell_n'] = len(sell_kc)

    results.append(stats)
    v["label"] = label
    gc.collect()


print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
hdr = (f"{'SL':>8} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} "
       f"{'KC_$/t':>7} {'SL_exits':>8} {'SL%':>5} {'Trail':>5}")
print(hdr)
print("-" * len(hdr))
for r in results:
    print(f"{r['label']:>8} {r['n']:>5} {r['sharpe']:>7.2f} ${r['total_pnl']:>9,.0f} "
          f"{r['win_rate']:>5.1f}% ${r['max_dd']:>7,.0f} "
          f"${r['kc_ppt']:>6.2f} {r['sl_exits']:>8} {r['sl_rate']:>4.1f}% {r['trail_exits']:>5}")

print(f"\n\nBUY vs SELL BREAKDOWN")
print("-" * 70)
hdr2 = f"{'SL':>8} {'BUY_N':>6} {'BUY_PnL':>10} {'BUY_$/t':>8} | {'SELL_N':>6} {'SELL_PnL':>10} {'SELL_$/t':>8}"
print(hdr2)
print("-" * len(hdr2))
for r in results:
    buy_ppt = r['buy_pnl'] / r['buy_n'] if r['buy_n'] > 0 else 0
    sell_ppt = r['sell_pnl'] / r['sell_n'] if r['sell_n'] > 0 else 0
    print(f"{r['label']:>8} {r['buy_n']:>6} ${r['buy_pnl']:>9,.0f} ${buy_ppt:>7.2f} | "
          f"{r['sell_n']:>6} ${r['sell_pnl']:>9,.0f} ${sell_ppt:>7.2f}")


print(f"\n\nCONCLUSION")
print("=" * 80)
best = max(results, key=lambda r: r['sharpe'])
print(f"Best SL: {best['label']} with Sharpe={best['sharpe']:.2f}")
current = [r for r in results if 'SL=4.5' in r['label']][0]
print(f"Current: {current['label']} with Sharpe={current['sharpe']:.2f}")
if best['label'] != current['label']:
    delta = best['sharpe'] - current['sharpe']
    print(f"  Improvement: Δ{delta:+.2f} Sharpe by switching to {best['label']}")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
