#!/usr/bin/env python3
"""
EXP29: Adaptive Trend Gating — 亚盘冷启动优化
==============================================

问题: 每天 UTC 00:00 日切后, 前 4-6 根 H1 bar 不够, trend_score 系统性偏低,
      亚盘如果出现趋势行情会被 choppy 门控拦截 (如 2026-04-06 关税冲击日).

两个优化方向:
  A. 降低 choppy 阈值 (0.35 → 0.20/0.25/0.30) — 放宽门控, 让亚盘更容易入场
  B. 引入昨日尾部 bar 作为评分补充 — 解决 bar 不够时评分不准的冷启动问题

Part 1: 方向 A — choppy 阈值扫描
Part 2: 方向 B — 昨日 bar 补充 (需 monkey-patch 引擎)
Part 3: A+B 组合最优
Part 4: K-Fold 验证 top 方案
Part 5: 亚盘 vs 非亚盘分时段分析

用法: python -u run_exp29.py 2>&1 | tee logs/exp29.log
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant, calc_stats
from backtest.engine import BacktestEngine
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, prepare_indicators_custom,
    add_atr_percentile, run_kfold, H1_CSV_PATH
)

SPREAD = 0.50
LABEL_FMT = "{}"

print("=" * 70)
print("EXP29: ADAPTIVE TREND GATING — COLD START FIX")
print(f"Started: {datetime.now()}")
print("=" * 70)


# ── Load data ──────────────────────────────────────────────────

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Range: {data.m15_df.index[0]} to {data.m15_df.index[-1]}")
print(f"  Load time: {time.time()-t0:.1f}s")


# ── Common kwargs ──────────────────────────────────────────────

BASE_KWARGS = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
}


# ═══════════════════════════════════════════════════════════════
# Part 1: 方向 A — choppy 阈值扫描
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: CHOPPY THRESHOLD SCAN")
print("=" * 70)

choppy_values = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
kc_only_values = [0.50, 0.55, 0.60]

part1_results = []
for chop in choppy_values:
    for kco in kc_only_values:
        if chop >= kco:
            continue
        label = f"chop={chop:.2f}/kco={kco:.2f}"
        kwargs = {**BASE_KWARGS, "choppy_threshold": chop, "kc_only_threshold": kco}
        stats = run_variant(data, label, verbose=False, **kwargs)
        stats['choppy'] = chop
        stats['kc_only'] = kco
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        part1_results.append(stats)
        print(f"  {label}: N={stats['n']:,}  Sh={stats['sharpe']:.2f}  "
              f"PnL=${stats['total_pnl']:,.0f}  MaxDD=${stats['max_dd']:,.0f}  "
              f"$/t=${ppt:.2f}")

# Baseline for comparison
baseline_stats = run_variant(data, "BASELINE (0.35/0.60)", verbose=False, **BASE_KWARGS)
baseline_stats['choppy'] = 0.35
baseline_stats['kc_only'] = 0.60
print(f"\n  BASELINE (0.35/0.60): N={baseline_stats['n']:,}  Sh={baseline_stats['sharpe']:.2f}  "
      f"PnL=${baseline_stats['total_pnl']:,.0f}  MaxDD=${baseline_stats['max_dd']:,.0f}")

# Rank
part1_results.sort(key=lambda x: x['sharpe'], reverse=True)
print("\n  TOP 5 by Sharpe:")
for i, r in enumerate(part1_results[:5]):
    delta = r['sharpe'] - baseline_stats['sharpe']
    print(f"    #{i+1} chop={r['choppy']:.2f}/kco={r['kc_only']:.2f}: "
          f"Sh={r['sharpe']:.2f} ({delta:+.2f})  "
          f"N={r['n']:,}  PnL=${r['total_pnl']:,.0f}  MaxDD=${r['max_dd']:,.0f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: 方向 B — 引入昨日 bar 补充
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: YESTERDAY BAR WARMUP (COLD START FIX)")
print("=" * 70)

# Monkey-patch _update_intraday_score to include yesterday's trailing bars
_original_update = BacktestEngine._update_intraday_score
_original_calc = BacktestEngine._calc_realtime_score

def _make_warmup_update(warmup_bars: int):
    """Create a patched _update_intraday_score that uses N bars from yesterday as warmup."""
    def _patched_update(self, h1_window, bar_time):
        if h1_window is None or len(h1_window) < 2:
            return
        bar_date = pd.Timestamp(bar_time).date()
        h1_len = len(h1_window)
        if bar_date == self._cached_date and h1_len == self._cached_h1_count:
            return

        indices_today = self._h1_date_map.get(bar_date)
        if indices_today:
            valid_today = [i for i in indices_today if i < h1_len]

            # If today has few bars, add yesterday's trailing bars
            if len(valid_today) < 6 and warmup_bars > 0:
                prev_date = bar_date - pd.Timedelta(days=1)
                # Try up to 3 days back for weekends
                for offset in range(1, 4):
                    prev_date = bar_date - pd.Timedelta(days=offset)
                    prev_indices = self._h1_date_map.get(prev_date)
                    if prev_indices:
                        break
                else:
                    prev_indices = None

                if prev_indices:
                    valid_prev = [i for i in prev_indices if i < h1_len]
                    tail = valid_prev[-warmup_bars:]
                    combined = tail + valid_today
                    if len(combined) >= 2:
                        combined_bars = self.h1_df.iloc[combined]
                        self._current_score = self._calc_realtime_score(combined_bars)
                        if self._current_score >= self._kc_only_threshold:
                            self._current_regime = 'trending'
                        elif self._current_score >= self._choppy_threshold:
                            self._current_regime = 'neutral'
                        else:
                            self._current_regime = 'choppy'
                        self._cached_date = bar_date
                        self._cached_h1_count = h1_len
                        return

            # Normal path: enough today bars or no prev data
            if len(valid_today) >= 2:
                today_bars = self.h1_df.iloc[valid_today]
                self._current_score = self._calc_realtime_score(today_bars)
                if self._current_score >= self._kc_only_threshold:
                    self._current_regime = 'trending'
                elif self._current_score >= self._choppy_threshold:
                    self._current_regime = 'neutral'
                else:
                    self._current_regime = 'choppy'
        elif bar_date != self._cached_date:
            self._current_score = 0.5
            self._current_regime = 'neutral'

        self._cached_date = bar_date
        self._cached_h1_count = h1_len
    return _patched_update

warmup_values = [0, 2, 4, 6, 8, 12]
part2_results = []

for wb in warmup_values:
    label = f"warmup={wb}"
    if wb == 0:
        BacktestEngine._update_intraday_score = _original_update
    else:
        BacktestEngine._update_intraday_score = _make_warmup_update(wb)

    kwargs = {**BASE_KWARGS, "choppy_threshold": 0.35, "kc_only_threshold": 0.60}
    stats = run_variant(data, label, verbose=False, **kwargs)
    stats['warmup_bars'] = wb
    part2_results.append(stats)
    delta = stats['sharpe'] - baseline_stats['sharpe']
    ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
    print(f"  {label}: N={stats['n']:,}  Sh={stats['sharpe']:.2f} ({delta:+.2f})  "
          f"PnL=${stats['total_pnl']:,.0f}  MaxDD=${stats['max_dd']:,.0f}  "
          f"$/t=${ppt:.2f}")

# Restore
BacktestEngine._update_intraday_score = _original_update


# ═══════════════════════════════════════════════════════════════
# Part 3: A+B 组合最优
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: COMBINED A+B — BEST THRESHOLD × BEST WARMUP")
print("=" * 70)

# Pick top 3 from Part 1 + top 3 from Part 2
top_chop = [(r['choppy'], r['kc_only']) for r in part1_results[:3]]
top_warmup = [r['warmup_bars'] for r in sorted(part2_results, key=lambda x: x['sharpe'], reverse=True)[:3] if r['warmup_bars'] > 0]

if not top_warmup:
    top_warmup = [4, 6]

part3_results = []
for chop, kco in top_chop:
    for wb in top_warmup:
        label = f"chop={chop:.2f}/kco={kco:.2f}/warmup={wb}"
        BacktestEngine._update_intraday_score = _make_warmup_update(wb)
        kwargs = {**BASE_KWARGS, "choppy_threshold": chop, "kc_only_threshold": kco}
        stats = run_variant(data, label, verbose=False, **kwargs)
        stats['choppy'] = chop
        stats['kc_only'] = kco
        stats['warmup_bars'] = wb
        part3_results.append(stats)
        delta = stats['sharpe'] - baseline_stats['sharpe']
        print(f"  {label}: N={stats['n']:,}  Sh={stats['sharpe']:.2f} ({delta:+.2f})  "
              f"PnL=${stats['total_pnl']:,.0f}  MaxDD=${stats['max_dd']:,.0f}")

BacktestEngine._update_intraday_score = _original_update

# Rank all Part 3
part3_results.sort(key=lambda x: x['sharpe'], reverse=True)
print("\n  TOP 5 COMBINED:")
for i, r in enumerate(part3_results[:5]):
    delta = r['sharpe'] - baseline_stats['sharpe']
    print(f"    #{i+1} chop={r['choppy']:.2f}/kco={r['kc_only']:.2f}/warmup={r['warmup_bars']}: "
          f"Sh={r['sharpe']:.2f} ({delta:+.2f})  N={r['n']:,}  "
          f"PnL=${r['total_pnl']:,.0f}  MaxDD=${r['max_dd']:,.0f}")


# ═══════════════════════════════════════════════════════════════
# Part 4: K-Fold 验证 top 方案
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: K-FOLD VALIDATION (6 FOLDS)")
print("=" * 70)

# Collect top 3 unique configs from all parts
all_candidates = []
seen = set()

# Add baseline
all_candidates.append(("BASELINE", {**BASE_KWARGS, "choppy_threshold": 0.35, "kc_only_threshold": 0.60}, 0))
seen.add((0.35, 0.60, 0))

# Add best from Part 1
for r in part1_results[:2]:
    key = (r['choppy'], r['kc_only'], 0)
    if key not in seen:
        all_candidates.append((
            f"A: chop={r['choppy']:.2f}/kco={r['kc_only']:.2f}",
            {**BASE_KWARGS, "choppy_threshold": r['choppy'], "kc_only_threshold": r['kc_only']},
            0
        ))
        seen.add(key)

# Add best from Part 2
for r in sorted(part2_results, key=lambda x: x['sharpe'], reverse=True)[:2]:
    if r['warmup_bars'] > 0:
        key = (0.35, 0.60, r['warmup_bars'])
        if key not in seen:
            all_candidates.append((
                f"B: warmup={r['warmup_bars']}",
                {**BASE_KWARGS, "choppy_threshold": 0.35, "kc_only_threshold": 0.60},
                r['warmup_bars']
            ))
            seen.add(key)

# Add best from Part 3
for r in part3_results[:2]:
    key = (r['choppy'], r['kc_only'], r['warmup_bars'])
    if key not in seen:
        all_candidates.append((
            f"A+B: chop={r['choppy']:.2f}/kco={r['kc_only']:.2f}/warmup={r['warmup_bars']}",
            {**BASE_KWARGS, "choppy_threshold": r['choppy'], "kc_only_threshold": r['kc_only']},
            r['warmup_bars']
        ))
        seen.add(key)

n_folds = 6
total_days = (data.m15_df.index[-1] - data.m15_df.index[0]).days
fold_days = total_days // n_folds

kfold_results = {}
for name, kwargs, wb in all_candidates:
    if wb > 0:
        BacktestEngine._update_intraday_score = _make_warmup_update(wb)
    else:
        BacktestEngine._update_intraday_score = _original_update

    fold_sharpes = []
    for fold in range(n_folds):
        start = data.m15_df.index[0] + pd.Timedelta(days=fold * fold_days)
        end = start + pd.Timedelta(days=fold_days)
        fold_data = data.slice(str(start), str(end))
        if fold_data.m15_df is None or len(fold_data.m15_df) < 100:
            continue
        fold_stats = run_variant(fold_data, f"{name}_F{fold+1}", verbose=False, **kwargs)
        fold_sharpes.append(fold_stats['sharpe'])

    avg_sh = np.mean(fold_sharpes) if fold_sharpes else 0
    std_sh = np.std(fold_sharpes) if fold_sharpes else 0
    kfold_results[name] = {
        'fold_sharpes': fold_sharpes,
        'avg': avg_sh,
        'std': std_sh,
        'warmup': wb,
    }

    # Compare vs baseline
    base_folds = kfold_results.get("BASELINE", {}).get('fold_sharpes', [])
    if base_folds and fold_sharpes:
        wins = sum(1 for a, b in zip(fold_sharpes, base_folds) if a > b)
    else:
        wins = 0

    print(f"  {name}:")
    print(f"    Folds: {['%.2f' % s for s in fold_sharpes]}")
    print(f"    Avg={avg_sh:.2f} (Std={std_sh:.2f})  Wins={wins}/{len(fold_sharpes)} vs BASELINE")

BacktestEngine._update_intraday_score = _original_update


# ═══════════════════════════════════════════════════════════════
# Part 5: 亚盘 vs 非亚盘分时段分析
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: ASIA SESSION ANALYSIS")
print("=" * 70)
print("  Running baseline to extract per-trade timestamps...")

# Run baseline with full output to get trade details
baseline_full = run_variant(data, "BASELINE_FULL", verbose=False, **BASE_KWARGS)

# We need the engine's trade log for timing analysis
# Run a quick engine directly to get trade records
engine = BacktestEngine(
    m15_df=data.m15_df,
    h1_df=data.h1_df,
    **BASE_KWARGS
)
engine.run()
trades = engine.trades

# Classify trades by UTC hour of entry
asia_trades = []    # UTC 22:00-07:00 (Beijing 06:00-15:00)
london_trades = []  # UTC 07:00-13:00
ny_trades = []      # UTC 13:00-22:00

for t in trades:
    h = t.entry_time.hour
    if h >= 22 or h < 7:
        asia_trades.append(t)
    elif 7 <= h < 13:
        london_trades.append(t)
    else:
        ny_trades.append(t)

def _session_stats(trade_list, name):
    if not trade_list:
        print(f"  {name}: 0 trades")
        return
    pnls = [t.pnl for t in trade_list]
    n = len(pnls)
    total = sum(pnls)
    avg = total / n
    wins = sum(1 for p in pnls if p > 0)
    print(f"  {name}: N={n:,}  PnL=${total:,.0f}  $/t=${avg:.2f}  WR={wins/n*100:.1f}%")

_session_stats(asia_trades, "Asia   (UTC 22-07)")
_session_stats(london_trades, "London (UTC 07-13)")
_session_stats(ny_trades, "NY     (UTC 13-22)")

# Sub-analysis: early Asia (cold start zone UTC 00-04) vs late Asia
early_asia = [t for t in asia_trades if 0 <= t.entry_time.hour < 4]
late_asia = [t for t in asia_trades if t.entry_time.hour >= 4 or t.entry_time.hour >= 22]
_session_stats(early_asia, "  Early Asia (UTC 00-04, cold start)")
_session_stats(late_asia, "  Late Asia  (UTC 22-00,04-07)")

# Count choppy skips by session (approximate from trade count difference)
print(f"\n  Total trades: {len(trades):,}")
print(f"  Asia: {len(asia_trades)} ({len(asia_trades)/len(trades)*100:.1f}%)")
print(f"  London: {len(london_trades)} ({len(london_trades)/len(trades)*100:.1f}%)")
print(f"  NY: {len(ny_trades)} ({len(ny_trades)/len(trades)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXP29 FINAL SUMMARY")
print("=" * 70)

print(f"\nBASELINE: choppy=0.35/kc_only=0.60, warmup=0")
print(f"  Sharpe={baseline_stats['sharpe']:.2f}  PnL=${baseline_stats['total_pnl']:,.0f}  "
      f"MaxDD=${baseline_stats['max_dd']:,.0f}  N={baseline_stats['n']:,}")

print(f"\nPART 1 BEST (threshold only):")
if part1_results:
    best1 = part1_results[0]
    d = best1['sharpe'] - baseline_stats['sharpe']
    print(f"  chop={best1['choppy']:.2f}/kco={best1['kc_only']:.2f}: "
          f"Sh={best1['sharpe']:.2f} ({d:+.2f})  PnL=${best1['total_pnl']:,.0f}")

print(f"\nPART 2 BEST (warmup only):")
best2 = max(part2_results, key=lambda x: x['sharpe'])
d = best2['sharpe'] - baseline_stats['sharpe']
print(f"  warmup={best2['warmup_bars']}: "
      f"Sh={best2['sharpe']:.2f} ({d:+.2f})  PnL=${best2['total_pnl']:,.0f}")

print(f"\nPART 3 BEST (combined):")
if part3_results:
    best3 = part3_results[0]
    d = best3['sharpe'] - baseline_stats['sharpe']
    print(f"  chop={best3['choppy']:.2f}/kco={best3['kc_only']:.2f}/warmup={best3['warmup_bars']}: "
          f"Sh={best3['sharpe']:.2f} ({d:+.2f})  PnL=${best3['total_pnl']:,.0f}")

print(f"\nPART 4 K-FOLD:")
for name, res in kfold_results.items():
    base_folds = kfold_results.get("BASELINE", {}).get('fold_sharpes', [])
    folds = res['fold_sharpes']
    wins = sum(1 for a, b in zip(folds, base_folds) if a > b) if base_folds else 0
    print(f"  {name}: Avg={res['avg']:.2f} (Std={res['std']:.2f}) Wins={wins}/{len(folds)}")

# Recommendation
print(f"\nRECOMMENDATION:")
# Find best K-Fold candidate (non-baseline)
best_kf = None
best_kf_name = None
base_kf = kfold_results.get("BASELINE", {})
for name, res in kfold_results.items():
    if name == "BASELINE":
        continue
    if best_kf is None or res['avg'] > best_kf['avg']:
        best_kf = res
        best_kf_name = name

if best_kf and base_kf:
    base_folds = base_kf.get('fold_sharpes', [])
    best_folds = best_kf.get('fold_sharpes', [])
    wins = sum(1 for a, b in zip(best_folds, base_folds) if a > b) if base_folds else 0
    if wins >= 4:
        print(f"  IMPLEMENT: {best_kf_name} (K-Fold {wins}/6 wins, Avg Sharpe {best_kf['avg']:.2f})")
    elif wins >= 3:
        print(f"  CONSIDER: {best_kf_name} (K-Fold {wins}/6 wins — marginal)")
    else:
        print(f"  NO CHANGE: Best candidate {best_kf_name} only wins {wins}/6 folds — keep current 0.35/0.60")
else:
    print("  NO CHANGE: No valid candidates found")

elapsed = time.time() - t0
print(f"\nTotal runtime: {elapsed/60:.1f} minutes")
print(f"Finished: {datetime.now()}")

# Save results
def _serializable(d):
    """Filter out non-serializable items from stats dict."""
    skip_types = (pd.DataFrame, pd.Series, list)
    return {k: v for k, v in d.items() if k not in ('_trades', '_equity_curve') and not isinstance(v, skip_types)}

results = {
    'baseline': _serializable(baseline_stats),
    'part1_top5': [_serializable(r) for r in part1_results[:5]],
    'part2': [_serializable(r) for r in part2_results],
    'part3_top5': [_serializable(r) for r in part3_results[:5]],
    'kfold': {name: {k: v for k, v in res.items()} for name, res in kfold_results.items()},
}
out_path = os.path.join(os.path.dirname(__file__), 'data', 'exp29_results.json')
try:
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
except Exception as e:
    print(f"\nWarning: Could not save results: {e}")
