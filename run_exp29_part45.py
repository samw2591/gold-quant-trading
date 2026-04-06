#!/usr/bin/env python3
"""
EXP29 Part 4+5 Only — 跳过已完成的 Part 1-3
(Part 1-3 结果：所有变体与 baseline 完全一致，无差异)
"""
import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant, calc_stats
from backtest.engine import BacktestEngine
from backtest.runner import C12_KWARGS, H1_CSV_PATH

SPREAD = 0.50

print("=" * 70)
print("EXP29 Part 4+5: K-Fold + Session Analysis")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Range: {data.m15_df.index[0]} to {data.m15_df.index[-1]}")
print(f"  Load time: {time.time()-t0:.1f}s")

BASE_KWARGS = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
}

# ═══════════════════════════════════════════════════════════════
# Part 4: K-Fold Validation (6 folds) — only BASELINE
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: K-FOLD VALIDATION (6 FOLDS)")
print("=" * 70)

n_folds = 6
total_days = (data.m15_df.index[-1] - data.m15_df.index[0]).days
fold_days = total_days // n_folds

fold_sharpes = []
for fold in range(n_folds):
    start = data.m15_df.index[0] + pd.Timedelta(days=fold * fold_days)
    end = start + pd.Timedelta(days=fold_days)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    fold_data = data.slice(start_str, end_str)
    if fold_data.m15_df is None or len(fold_data.m15_df) < 100:
        print(f"  Fold {fold+1}: skipped (too few bars)")
        continue
    kwargs = {**BASE_KWARGS, "choppy_threshold": 0.35, "kc_only_threshold": 0.60}
    fold_stats = run_variant(fold_data, f"BASELINE_F{fold+1}", verbose=False, **kwargs)
    fold_sharpes.append(fold_stats['sharpe'])
    print(f"  Fold {fold+1} ({start_str} to {end_str}): "
          f"N={fold_stats['n']:,}  Sh={fold_stats['sharpe']:.2f}  "
          f"PnL=${fold_stats['total_pnl']:,.0f}  MaxDD=${fold_stats['max_dd']:,.0f}")

avg_sh = np.mean(fold_sharpes) if fold_sharpes else 0
std_sh = np.std(fold_sharpes) if fold_sharpes else 0
print(f"\n  K-Fold Summary: Avg={avg_sh:.2f}  Std={std_sh:.2f}")
print(f"  Folds: {['%.2f' % s for s in fold_sharpes]}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Session Analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: SESSION ANALYSIS")
print("=" * 70)

engine = BacktestEngine(
    m15_df=data.m15_df,
    h1_df=data.h1_df,
    **BASE_KWARGS
)
engine.run()
trades = engine.trades

asia_trades = []
london_trades = []
ny_trades = []

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

early_asia = [t for t in asia_trades if 0 <= t.entry_time.hour < 4]
late_asia = [t for t in asia_trades if t.entry_time.hour >= 4 or t.entry_time.hour >= 22]
_session_stats(early_asia, "  Early Asia (UTC 00-04, cold start)")
_session_stats(late_asia, "  Late Asia  (UTC 22-00,04-07)")

print(f"\n  Total trades: {len(trades):,}")
print(f"  Skipped (choppy): {engine.skipped_choppy}")
print(f"  Skipped (neutral M15): {engine.skipped_neutral_m15}")
if trades:
    print(f"  Asia: {len(asia_trades)} ({len(asia_trades)/len(trades)*100:.1f}%)")
    print(f"  London: {len(london_trades)} ({len(london_trades)/len(trades)*100:.1f}%)")
    print(f"  NY: {len(ny_trades)} ({len(ny_trades)/len(trades)*100:.1f}%)")

# Hourly PnL breakdown
print("\n  Hourly PnL breakdown:")
hourly = {}
for t in trades:
    h = t.entry_time.hour
    if h not in hourly:
        hourly[h] = {'n': 0, 'pnl': 0}
    hourly[h]['n'] += 1
    hourly[h]['pnl'] += t.pnl

for h in sorted(hourly.keys()):
    d = hourly[h]
    avg = d['pnl'] / d['n'] if d['n'] > 0 else 0
    bar = "+" * int(max(0, avg * 2)) + "-" * int(max(0, -avg * 2))
    print(f"    UTC {h:02d}: N={d['n']:>5}  PnL=${d['pnl']:>8.0f}  $/t=${avg:>6.2f}  {bar}")


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXP29 COMPLETE SUMMARY")
print("=" * 70)

print(f"""
FINDINGS:
  1. Part 1 (Choppy Threshold): ALL choppy values (0.15-0.40) produce
     IDENTICAL results when kco=0.60. The choppy threshold has NO effect
     on trade selection in the backtest engine.
     → Only kco (kc_only_threshold) matters for filtering.

  2. Part 2 (Yesterday Warmup): ALL warmup values (0-12 bars) produce
     IDENTICAL results. The monkey-patched warmup bars have no impact.
     → The scoring change doesn't alter which trades pass the gate.

  3. Part 3 (Combined): Same — no combination changes anything.

  4. Part 4 K-Fold: Avg Sharpe = {avg_sh:.2f} (Std = {std_sh:.2f})
     Folds: {['%.2f' % s for s in fold_sharpes]}

  5. The intraday_adaptive gating in the backtest engine may not be
     functioning as intended, OR the gate thresholds never trigger in
     the historical data because the default score (0.5 for <2 bars)
     always passes the NEUTRAL check.

DIAGNOSIS:
  The default score of 0.5 when today_bars < 2 means:
    - score 0.5 >= CHOPPY_THRESHOLD (0.35) → regime = NEUTRAL
    - NEUTRAL allows H1 entries → Keltner can trade
    - Changing choppy_threshold below 0.5 has NO effect
    - Warmup bars only help when today_bars >= 2 (score calculated)
    - But calculated scores in practice rarely fall below 0.35

  The real fix should be in the DEFAULT SCORE, not the thresholds.
""")

elapsed = time.time() - t0
print(f"Total runtime: {elapsed/60:.1f} minutes")
print(f"Finished: {datetime.now()}")
