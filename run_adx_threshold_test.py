#!/usr/bin/env python3
"""
ADX Threshold Grid Test — 震荡期假突破过滤
============================================
对比 ADX 阈值 18(当前)/22/24/26/28 对 Keltner 策略的影响。
同时测试 Mega Trail 参数 + ADX 提高的组合效果。

目标: 找到减少震荡期假突破止损的最优 ADX 阈值。
"""
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "adx_threshold_output.txt"


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

print("=" * 70)
print("ADX THRESHOLD GRID TEST")
print(f"Started: {datetime.now()}")
print("=" * 70)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

CURRENT_LIVE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
}

MEGA_BASE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

results = {}

# ── Baseline: current live (ADX=18) ──
print("\n" + "=" * 60)
print("SECTION 1: ADX Threshold with Current Trail (live params)")
print("=" * 60)

for adx in [18, 22, 24, 26, 28]:
    label = f"Current_ADX{adx}"
    kwargs = {**CURRENT_LIVE, "keltner_adx_threshold": adx}
    stats = run_variant(data, label, **kwargs)
    results[label] = stats

# ── Mega Trail + ADX variants ──
print("\n" + "=" * 60)
print("SECTION 2: ADX Threshold with Mega Trail (P7/P8 params)")
print("=" * 60)

for adx in [18, 22, 24, 26, 28]:
    label = f"Mega_ADX{adx}"
    kwargs = {**MEGA_BASE, "keltner_adx_threshold": adx}
    stats = run_variant(data, label, **kwargs)
    results[label] = stats

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

header = f"{'Variant':<22} {'Sharpe':>7} {'PnL':>10} {'Trades':>7} {'WinR%':>7} {'MaxDD':>8} {'PF':>6} {'AvgW':>7} {'AvgL':>7}"
print(header)
print("-" * len(header))

for label, s in sorted(results.items()):
    sharpe = s.get('sharpe', 0)
    pnl = s.get('total_pnl', 0)
    trades = s.get('total_trades', 0)
    winr = s.get('win_rate', 0) * 100
    maxdd = s.get('max_drawdown', 0)
    pf = s.get('profit_factor', 0)
    avg_w = s.get('avg_win', 0)
    avg_l = s.get('avg_loss', 0)
    print(f"{label:<22} {sharpe:>7.2f} {pnl:>10.0f} {trades:>7} {winr:>6.1f}% {maxdd:>8.0f} {pf:>6.2f} {avg_w:>7.1f} {avg_l:>7.1f}")

# ── ADX impact analysis ──
print("\n" + "=" * 70)
print("ADX IMPACT ANALYSIS (trade count reduction)")
print("=" * 70)

base_current = results.get("Current_ADX18", {}).get("total_trades", 1)
base_mega = results.get("Mega_ADX18", {}).get("total_trades", 1)

for adx in [22, 24, 26, 28]:
    c = results.get(f"Current_ADX{adx}", {})
    m = results.get(f"Mega_ADX{adx}", {})
    c_trades = c.get("total_trades", 0)
    m_trades = m.get("total_trades", 0)
    c_pct = (1 - c_trades / base_current) * 100 if base_current else 0
    m_pct = (1 - m_trades / base_mega) * 100 if base_mega else 0
    c_sharpe = c.get("sharpe", 0)
    m_sharpe = m.get("sharpe", 0)
    print(f"ADX {adx}: Current -{c_pct:.0f}% trades (Sharpe={c_sharpe:.2f}) | "
          f"Mega -{m_pct:.0f}% trades (Sharpe={m_sharpe:.2f})")

elapsed = time.time() - t_total
print(f"\nTotal time: {elapsed:.0f}s")
print(f"Finished: {datetime.now()}")

tee.close()
sys.stdout = tee.stdout
print(f"\nResults saved to {OUTPUT_FILE}")
