#!/usr/bin/env python3
"""
Regime Validation Backtest
===========================
验证 MacroRegimeDetector 的分类是否对交易表现有真实的预测价值。

三部分实验:
  Part 1: 基线回测（当前最优参数，不加 regime 权重）
  Part 2: 按 regime 分组交叉分析（基线交易在各 regime 下的表现差异）
  Part 3: Regime 加权回测（应用 get_strategy_weights 的 lot_multiplier / direction filter）

如果 Part 2 显示 regime 分组间表现差异显著，且 Part 3 的 Sharpe 优于 Part 1，
则说明 regime 分类器有价值，可以上线。

用法:
  python run_regime_validation.py > regime_validation_output.txt 2>&1
"""
import sys
import os
import time
import json
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant, calc_stats
from backtest.runner import C12_KWARGS, run_kfold
from backtest.stats import aggregate_daily_pnl, print_comparison
from macro.data_provider import load_macro_for_backtest
from macro.regime_detector import MacroRegimeDetector, add_regime_column

print("=" * 80)
print("  REGIME VALIDATION BACKTEST")
print(f"  Started: {datetime.now()}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Data load time: {time.time()-t0:.1f}s")

# Load macro history and classify regimes
print("\nLoading macro history...")
macro_df = load_macro_for_backtest("data/macro_history.csv")
macro_df = add_regime_column(macro_df)
print(f"  Macro: {len(macro_df)} days, {macro_df.index[0].date()} -> {macro_df.index[-1].date()}")

# Regime distribution
regime_counts = macro_df['macro_regime'].value_counts()
print(f"\n  Regime Distribution (full period):")
for regime, count in regime_counts.items():
    pct = count / len(macro_df) * 100
    print(f"    {regime:<35} {count:>5} days ({pct:.1f}%)")

# Current best params (Mega Trail + Time Decay + Intraday Adaptive)
MEGA_REGIME = {
    'low': {'trail_act': 0.7, 'trail_dist': 0.25},
    'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
    'high': {'trail_act': 0.4, 'trail_dist': 0.10},
}

CURRENT_BEST = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": MEGA_REGIME,
    "keltner_max_hold_m15": 12,
    "time_decay_tp": True,
    "time_decay_start_hour": 1.0,
    "time_decay_atr_start": 0.30,
    "time_decay_atr_step": 0.10,
    "spread_cost": 0.30,
}


# ═══════════════════════════════════════════════════════════════
# Part 1: Baseline (no macro regime)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PART 1: BASELINE (no macro regime weighting)")
print("=" * 80)

baseline = run_variant(data, "A: Baseline (current best)", **CURRENT_BEST)
baseline_trades = baseline['_trades']

print(f"\n  Baseline: {baseline['n']} trades, Sharpe={baseline['sharpe']:.2f}, "
      f"PnL=${baseline['total_pnl']:.0f}, MaxDD=${baseline['max_dd']:.0f}")


# ═══════════════════════════════════════════════════════════════
# Part 2: Cross-analysis — baseline trades grouped by macro regime
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PART 2: CROSS-ANALYSIS — Baseline trades by macro regime")
print("=" * 80)

detector = MacroRegimeDetector()

regime_trades = defaultdict(list)
regime_not_found = 0

for trade in baseline_trades:
    trade_date = pd.Timestamp(trade.entry_time).normalize()
    if trade_date.tz is not None:
        trade_date = trade_date.tz_localize(None)

    if trade_date in macro_df.index:
        regime = macro_df.loc[trade_date, 'macro_regime']
        regime_trades[regime].append(trade)
    else:
        nearest = macro_df.index[macro_df.index.get_indexer([trade_date], method='ffill')]
        if len(nearest) > 0 and pd.notna(nearest[0]):
            regime = macro_df.loc[nearest[0], 'macro_regime']
            regime_trades[regime].append(trade)
        else:
            regime_not_found += 1

if regime_not_found > 0:
    print(f"\n  Warning: {regime_not_found} trades could not be matched to a macro date")

print(f"\n  {'Regime':<35} {'N':>5} {'Wins':>5} {'WR%':>6} {'PnL':>10} "
      f"{'AvgWin':>8} {'AvgLoss':>8} {'RR':>5} {'Sharpe':>8}")
print(f"  {'-'*35} {'-'*5} {'-'*5} {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*5} {'-'*8}")

regime_summary = {}
for regime in sorted(regime_trades.keys()):
    trades = regime_trades[regime]
    n = len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr = len(wins) / n * 100 if n > 0 else 0
    total_pnl = sum(t.pnl for t in trades)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    daily_pnl = aggregate_daily_pnl(trades)
    sharpe = 0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)

    print(f"  {regime:<35} {n:>5} {len(wins):>5} {wr:>5.1f}% ${total_pnl:>9.0f} "
          f"${avg_win:>7.1f} ${avg_loss:>7.1f} {rr:>4.2f} {sharpe:>7.2f}")

    regime_summary[regime] = {
        'n': n, 'win_rate': wr, 'total_pnl': total_pnl,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'sharpe': sharpe,
    }

# Per-strategy breakdown by regime
print(f"\n  --- Per-Strategy Breakdown by Regime ---")
for strategy in ['keltner', 'm15_rsi', 'orb']:
    print(f"\n  Strategy: {strategy}")
    print(f"  {'Regime':<35} {'N':>5} {'WR%':>6} {'PnL':>10} {'AvgPnL':>8}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*10} {'-'*8}")
    for regime in sorted(regime_trades.keys()):
        strat_trades = [t for t in regime_trades[regime] if t.strategy == strategy]
        n = len(strat_trades)
        if n == 0:
            continue
        wins = len([t for t in strat_trades if t.pnl > 0])
        wr = wins / n * 100
        pnl = sum(t.pnl for t in strat_trades)
        avg = pnl / n
        print(f"  {regime:<35} {n:>5} {wr:>5.1f}% ${pnl:>9.0f} ${avg:>7.1f}")


# ═══════════════════════════════════════════════════════════════
# Part 3: Regime-weighted backtest
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PART 3: REGIME-WEIGHTED BACKTEST")
print("=" * 80)

# B: Full macro regime integration (uses engine's built-in macro_regime_enabled)
regime_weighted = run_variant(
    data, "B: Macro Regime Weighted",
    **CURRENT_BEST,
    macro_df=macro_df,
    macro_regime_enabled=True,
)

print_comparison(
    [baseline, regime_weighted],
    title="BASELINE vs MACRO REGIME WEIGHTED"
)

# K-Fold validation
print("\n" + "=" * 80)
print("  PART 4: K-FOLD VALIDATION (6 folds)")
print("=" * 80)

print("\n  --- Baseline K-Fold ---")
kf_baseline = run_kfold(data, CURRENT_BEST, label_prefix="Base_")

print("\n  --- Regime-Weighted K-Fold ---")
kf_regime = run_kfold(
    data,
    {**CURRENT_BEST, "macro_df": macro_df, "macro_regime_enabled": True},
    label_prefix="Regime_"
)

print(f"\n  {'Fold':<10} {'Base Sharpe':>12} {'Regime Sharpe':>14} {'Delta':>8} {'Winner':>10}")
print(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*8} {'-'*10}")

base_wins = 0
regime_wins = 0
for bf, rf in zip(kf_baseline, kf_regime):
    b_sh = bf['sharpe']
    r_sh = rf['sharpe']
    delta = r_sh - b_sh
    winner = "Regime" if delta > 0 else "Base"
    if delta > 0:
        regime_wins += 1
    else:
        base_wins += 1
    print(f"  {bf.get('fold', '?'):<10} {b_sh:>12.2f} {r_sh:>14.2f} {delta:>+7.2f} {winner:>10}")

avg_base = np.mean([f['sharpe'] for f in kf_baseline]) if kf_baseline else 0
avg_regime = np.mean([f['sharpe'] for f in kf_regime]) if kf_regime else 0
print(f"\n  Average:   {avg_base:>12.2f} {avg_regime:>14.2f} {avg_regime-avg_base:>+7.2f}")
print(f"  Wins:      {base_wins:>12} {regime_wins:>14}")


# ═══════════════════════════════════════════════════════════════
# Part 5: Optimal regime weights (data-driven)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  PART 5: DATA-DRIVEN REGIME WEIGHT RECOMMENDATIONS")
print("=" * 80)

weights = detector.get_strategy_weights
print(f"\n  Current regime weights from MacroRegimeDetector.get_strategy_weights():")
for regime_name in sorted(regime_summary.keys()):
    from macro.regime_detector import MacroRegime
    try:
        r_enum = MacroRegime(regime_name)
        w = weights(r_enum)
        print(f"\n  {regime_name}:")
        print(f"    lot_multiplier={w['lot_multiplier']}, buy={w['buy_enabled']}, "
              f"sell={w['sell_enabled']}, allow={w['allow_trading']}")
        if regime_name in regime_summary:
            s = regime_summary[regime_name]
            print(f"    Actual: {s['n']} trades, WR={s['win_rate']:.1f}%, "
                  f"PnL=${s['total_pnl']:.0f}, Sharpe={s['sharpe']:.2f}")

            if s['sharpe'] < 0 and s['n'] >= 10:
                print(f"    >>> NEGATIVE SHARPE with {s['n']} trades — consider lot_multiplier=0.5 or lower")
            elif s['sharpe'] > 2 and s['n'] >= 10:
                print(f"    >>> STRONG positive Sharpe — consider lot_multiplier=1.2 or higher")
            elif s['n'] < 10:
                print(f"    >>> Too few trades ({s['n']}) — insufficient evidence")
    except ValueError:
        pass


# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  FINAL SUMMARY")
print("=" * 80)

print(f"""
  Baseline:       Sharpe={baseline['sharpe']:.2f}, PnL=${baseline['total_pnl']:.0f}, N={baseline['n']}
  Regime-Weighted: Sharpe={regime_weighted['sharpe']:.2f}, PnL=${regime_weighted['total_pnl']:.0f}, N={regime_weighted['n']}
  Delta Sharpe:   {regime_weighted['sharpe'] - baseline['sharpe']:+.2f}
  K-Fold Wins:    Base={base_wins}, Regime={regime_wins}

  Regime discrimination (Cross-analysis):""")

sharpes = {k: v['sharpe'] for k, v in regime_summary.items() if v['n'] >= 10}
if sharpes:
    best = max(sharpes, key=sharpes.get)
    worst = min(sharpes, key=sharpes.get)
    spread = sharpes[best] - sharpes[worst]
    print(f"    Best regime:  {best} (Sharpe={sharpes[best]:.2f})")
    print(f"    Worst regime: {worst} (Sharpe={sharpes[worst]:.2f})")
    print(f"    Spread:       {spread:.2f}")

    if spread > 2.0:
        print(f"\n  VERDICT: STRONG regime discrimination (spread={spread:.2f} > 2.0)")
        print(f"  >>> Regime-adaptive parameters are likely valuable")
    elif spread > 1.0:
        print(f"\n  VERDICT: MODERATE regime discrimination (spread={spread:.2f})")
        print(f"  >>> Worth exploring, but test with K-Fold before committing")
    else:
        print(f"\n  VERDICT: WEAK regime discrimination (spread={spread:.2f} < 1.0)")
        print(f"  >>> Regime classifier may not add value for short-term trading")

print(f"\n  Elapsed: {time.time()-t0:.0f}s")
print("=" * 80)
