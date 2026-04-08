#!/usr/bin/env python3
"""
EXP-MOM: IntradayTrendMeter 短窗口动量旁路回测
================================================
问题: IntradayTrendMeter 用全天数据计算 trend_score, 对日内末段突发趋势
     (如 $50 暴跌) 反应太慢, score=0.33 被判 choppy 而错失入场机会.

方案: 在 _update_intraday_score 中加入短窗口检测, 当最近 N 根 K 线
     出现强动量信号时, 绕过 choppy 门控升级到 neutral 或 trending.

测试方案:
  A) Baseline: 当前配置 (intraday_adaptive=True, choppy=0.35)
  B) KC连续突破: 最近3根中>=2根突破KC → 升级到 neutral
  C) ATR暴涨: 最近1根ATR > 前5根均值*1.5 → 升级到 neutral
  D) 价格动量: 最近3根累计涨跌幅 > 1.5*ATR → 升级到 neutral
  E) 组合旁路: B+C+D 任一触发即升级
  F) 无门控: intraday_adaptive=False (上限参考)
  G) 组合旁路+升级到trending: 旁路直接升级到trending(允许M15也入场)

每个方案跑 Current 和 Mega 两套配置, 对比 Sharpe/PnL/MaxDD/交易笔数.
"""
import sys, os, time, copy
from datetime import datetime
from typing import Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.engine import BacktestEngine

OUTPUT_FILE = "exp_momentum_bypass_output.txt"


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
print("EXP-MOM: INTRADAY TREND METER MOMENTUM BYPASS")
print(f"Started: {datetime.now()}")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# Save original method
# ═══════════════════════════════════════════════════════════════
_original_update = BacktestEngine._update_intraday_score

# ═══════════════════════════════════════════════════════════════
# Bypass strategies
# ═══════════════════════════════════════════════════════════════

def _make_bypass_update(bypass_mode="none", upgrade_to="neutral",
                        kc_window=3, kc_min_breaks=2,
                        atr_window=5, atr_spike_mult=1.5,
                        mom_window=3, mom_atr_mult=1.5):
    """
    Factory: returns a patched _update_intraday_score that adds momentum bypass.
    After computing the normal score/regime, if regime == 'choppy' and bypass
    condition is met, upgrade regime.
    """
    def _patched_update(self, h1_window, bar_time):
        _original_update(self, h1_window, bar_time)

        if self._current_regime != 'choppy':
            return

        if h1_window is None or len(h1_window) < max(kc_window, atr_window, mom_window) + 1:
            return

        triggered = False
        tail = h1_window.iloc[-kc_window:]

        if bypass_mode in ("kc", "combo"):
            kc_upper = tail.get('KC_upper')
            kc_lower = tail.get('KC_lower')
            if kc_upper is not None and kc_lower is not None:
                breaks = ((tail['Close'] > kc_upper) | (tail['Close'] < kc_lower)).sum()
                if breaks >= kc_min_breaks:
                    triggered = True

        if bypass_mode in ("atr", "combo") and not triggered:
            atr_col = h1_window['ATR']
            if len(atr_col) >= atr_window + 1:
                recent_atr = float(atr_col.iloc[-1])
                prev_mean = float(atr_col.iloc[-(atr_window+1):-1].mean())
                if prev_mean > 0 and recent_atr > prev_mean * atr_spike_mult:
                    triggered = True

        if bypass_mode in ("mom", "combo") and not triggered:
            closes = h1_window['Close'].iloc[-(mom_window+1):]
            if len(closes) >= mom_window + 1:
                move = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))
                atr_now = float(h1_window.iloc[-1].get('ATR', 0))
                if atr_now > 0 and move > mom_atr_mult * atr_now:
                    triggered = True

        if triggered:
            self._current_regime = upgrade_to
            if not hasattr(self, 'bypass_count'):
                self.bypass_count = 0
            self.bypass_count += 1

    return _patched_update


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════
t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}
NO_GATE = {**C12_KWARGS, "intraday_adaptive": False}

print(f"\n  Data loaded: H1={len(data.h1_df):,} bars, M15={len(data.m15_df):,} bars")
print(f"  Range: {data.h1_df.index[0]} to {data.h1_df.index[-1]}")


# ═══════════════════════════════════════════════════════════════
# Run variants
# ═══════════════════════════════════════════════════════════════

VARIANTS = [
    ("A: Baseline (choppy=0.35)", "none", "neutral", CURRENT),
    ("B: KC 3bar>=2 bypass", "kc", "neutral", CURRENT),
    ("C: ATR spike bypass", "atr", "neutral", CURRENT),
    ("D: Price mom bypass", "mom", "neutral", CURRENT),
    ("E: Combo bypass (any)", "combo", "neutral", CURRENT),
    ("F: No gate (upper bound)", "none", "neutral", NO_GATE),
    ("G: Combo→trending", "combo", "trending", CURRENT),
]

PARAM_VARIANTS = [
    ("E1: Combo kc2/atr1.3/mom1.2", "combo", "neutral", CURRENT,
     {"kc_window": 3, "kc_min_breaks": 2, "atr_spike_mult": 1.3, "mom_atr_mult": 1.2}),
    ("E2: Combo kc2/atr1.5/mom1.5", "combo", "neutral", CURRENT,
     {"kc_window": 3, "kc_min_breaks": 2, "atr_spike_mult": 1.5, "mom_atr_mult": 1.5}),
    ("E3: Combo kc2/atr2.0/mom2.0", "combo", "neutral", CURRENT,
     {"kc_window": 3, "kc_min_breaks": 2, "atr_spike_mult": 2.0, "mom_atr_mult": 2.0}),
    ("E4: KC window=2 min=2", "kc", "neutral", CURRENT,
     {"kc_window": 2, "kc_min_breaks": 2}),
    ("E5: Mom window=2 mult=1.0", "mom", "neutral", CURRENT,
     {"mom_window": 2, "mom_atr_mult": 1.0}),
]


def run_one(label, bypass_mode, upgrade_to, base_kwargs, extra_bypass_params=None):
    bp = extra_bypass_params or {}
    if bypass_mode != "none" or base_kwargs.get("intraday_adaptive", False) is False:
        if base_kwargs.get("intraday_adaptive", False) and bypass_mode != "none":
            BacktestEngine._update_intraday_score = _make_bypass_update(
                bypass_mode=bypass_mode, upgrade_to=upgrade_to, **bp
            )
        elif not base_kwargs.get("intraday_adaptive", False):
            BacktestEngine._update_intraday_score = _original_update
        else:
            BacktestEngine._update_intraday_score = _original_update
    else:
        BacktestEngine._update_intraday_score = _original_update

    stats = run_variant(data, label, **base_kwargs)
    bypass_total = getattr(stats.get('_engine', None), 'bypass_count', 0) if '_engine' in stats else '?'

    BacktestEngine._update_intraday_score = _original_update
    return stats


print("\n" + "=" * 70)
print("PART 1: INDIVIDUAL BYPASS STRATEGIES (Current config)")
print("=" * 70)

results_current = []
for label, bm, ut, kw in VARIANTS:
    print(f"\n--- {label} ---")
    stats = run_one(label, bm, ut, kw)
    results_current.append((label, stats))

print("\n" + "=" * 70)
print("PART 2: PARAMETER SENSITIVITY (Current config)")
print("=" * 70)

results_params = []
for label, bm, ut, kw, bp in PARAM_VARIANTS:
    print(f"\n--- {label} ---")
    stats = run_one(label, bm, ut, kw, bp)
    results_params.append((label, stats))

print("\n" + "=" * 70)
print("PART 3: BEST BYPASS ON MEGA CONFIG")
print("=" * 70)

MEGA_VARIANTS = [
    ("M-A: Mega Baseline", "none", "neutral", MEGA),
    ("M-E: Mega + Combo bypass", "combo", "neutral", MEGA),
    ("M-G: Mega + Combo→trending", "combo", "trending", MEGA),
    ("M-F: Mega No gate", "none", "neutral", {**C12_KWARGS, "intraday_adaptive": False,
        "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
        "regime_config": {
            'low': {'trail_act': 0.7, 'trail_dist': 0.25},
            'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
            'high': {'trail_act': 0.4, 'trail_dist': 0.10},
        },
    }),
]

results_mega = []
for label, bm, ut, kw in MEGA_VARIANTS:
    print(f"\n--- {label} ---")
    stats = run_one(label, bm, ut, kw)
    results_mega.append((label, stats))


# ═══════════════════════════════════════════════════════════════
# PART 4: K-Fold validation on best bypass
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: 6-FOLD CROSS-VALIDATION (Combo bypass vs Baseline)")
print("=" * 70)

from backtest.runner import kfold_backtest

print("\n--- Baseline K-Fold ---")
BacktestEngine._update_intraday_score = _original_update
baseline_folds = kfold_backtest(data, n_folds=6, engine_kwargs=CURRENT)

print("\n--- Combo Bypass K-Fold ---")
BacktestEngine._update_intraday_score = _make_bypass_update(
    bypass_mode="combo", upgrade_to="neutral"
)
combo_folds = kfold_backtest(data, n_folds=6, engine_kwargs=CURRENT)
BacktestEngine._update_intraday_score = _original_update


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

header = f"{'Variant':<40} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6} {'Skip_C':>7}"
print(f"\n--- Part 1: Individual Bypass (Current) ---")
print(header)
for label, s in results_current:
    sc = s.get('skipped_choppy', '?')
    print(f"{label:<40} {s['n']:>5} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['win_rate']*100:>5.1f}% {sc:>7}")

print(f"\n--- Part 2: Parameter Sensitivity ---")
print(header)
for label, s in results_params:
    sc = s.get('skipped_choppy', '?')
    print(f"{label:<40} {s['n']:>5} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['win_rate']*100:>5.1f}% {sc:>7}")

print(f"\n--- Part 3: Mega Config ---")
print(header)
for label, s in results_mega:
    sc = s.get('skipped_choppy', '?')
    print(f"{label:<40} {s['n']:>5} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['win_rate']*100:>5.1f}% {sc:>7}")

print(f"\n--- Part 4: K-Fold Validation ---")
if baseline_folds and combo_folds:
    b_sharpes = [f['sharpe'] for f in baseline_folds]
    c_sharpes = [f['sharpe'] for f in combo_folds]
    print(f"  Baseline:      Avg={np.mean(b_sharpes):.2f}  Std={np.std(b_sharpes):.2f}  Folds={['%.2f'%s for s in b_sharpes]}")
    print(f"  Combo Bypass:  Avg={np.mean(c_sharpes):.2f}  Std={np.std(c_sharpes):.2f}  Folds={['%.2f'%s for s in c_sharpes]}")
    delta = np.mean(c_sharpes) - np.mean(b_sharpes)
    print(f"  Delta Sharpe:  {delta:+.3f}")
    pos_b = sum(1 for s in b_sharpes if s > 0)
    pos_c = sum(1 for s in c_sharpes if s > 0)
    print(f"  Positive folds: Baseline {pos_b}/{len(b_sharpes)}, Combo {pos_c}/{len(c_sharpes)}")
else:
    print("  K-Fold failed or no results")

elapsed = time.time() - t_total
print(f"\n  Total time: {elapsed:.0f}s")

tee.close()
