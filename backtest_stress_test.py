"""
Strategy Stress Test - Challenge the "too perfect" results
============================================================
5 challenges to the C12+Adaptive strategy:

  Challenge 1: Spread/Slippage cost (回测无交易成本)
  Challenge 2: Baseline isolation (Baseline 是否被污染)
  Challenge 3: Signal clustering (信号是否聚集/重复)
  Challenge 4: Worst drawdown deep-dive (最差时段放大)
  Challenge 5: Parameter cliff test (参数悬崖检测)

Usage: python backtest_stress_test.py
"""
import json
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import config
from backtest import DataBundle, run_variant
from backtest.runner import (
    C12_KWARGS, TRUE_BASELINE_KWARGS,
    load_m15, load_h1_aligned, add_atr_percentile,
    prepare_indicators_custom, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREAD = 0.50
RESULTS = {}

ADAPTIVE_KWARGS = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
}


def main():
    t_start = time.time()
    print("=" * 100)
    print("  STRATEGY STRESS TEST - Challenging C12+Adaptive")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15_default = prepare_indicators(m15_raw)
    h1_default = prepare_indicators(h1_raw)
    h1_default = add_atr_percentile(h1_default)

    print(f"  M15: {len(m15_default)} bars, H1: {len(h1_default)} bars\n")

    bundle = DataBundle(m15_default, h1_default)

    # ══════════════════════════════════════════════════════════════
    # Get C12+Adaptive trades
    # ══════════════════════════════════════════════════════════════
    print("  Running C12+Adaptive to get trade list...", flush=True)
    d_stats = run_variant(bundle, "C12+Adaptive", verbose=False, **ADAPTIVE_KWARGS)
    d_trades = d_stats['_trades']
    print(f"  {len(d_trades)} trades, Sharpe={d_stats['sharpe']:.2f}, PnL=${d_stats['total_pnl']:.0f}\n")

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 1: Transaction costs
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  CHALLENGE 1: Transaction Costs (Spread + Slippage)")
    print("=" * 100)

    spread_scenarios = [
        ("No cost", 0),
        ("Spread $0.30 (tight)", 0.30),
        ("Spread $0.50 (normal)", 0.50),
        ("Spread $0.80 (wide)", 0.80),
        ("Spread $1.00 (adverse)", 1.00),
        ("Spread $1.50 (extreme)", 1.50),
    ]

    for label, spread_points in spread_scenarios:
        adjusted_pnls = []
        for t in d_trades:
            cost = spread_points * t.lots * config.POINT_VALUE_PER_LOT
            adjusted_pnls.append(t.pnl - cost)

        total_pnl = sum(adjusted_pnls)
        total_cost = sum(spread_points * t.lots * config.POINT_VALUE_PER_LOT for t in d_trades)
        winning = sum(1 for p in adjusted_pnls if p > 0)
        wr = winning / len(adjusted_pnls) * 100 if adjusted_pnls else 0

        equity = [config.CAPITAL]
        for pnl in adjusted_pnls:
            equity.append(equity[-1] + pnl)
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        max_dd = abs((eq_arr - peak).min())

        print(f"  {label:<30}: PnL=${total_pnl:>9.0f}, Cost=${total_cost:>7.0f}, "
              f"WR={wr:>5.1f}%, MaxDD=${max_dd:>7.0f}, Net/Trade=${total_pnl/len(d_trades):>.2f}")

    avg_lots = np.mean([t.lots for t in d_trades])
    avg_cost_50 = 0.50 * avg_lots * config.POINT_VALUE_PER_LOT
    total_cost_50 = avg_cost_50 * len(d_trades)
    print(f"\n  Avg lot size: {avg_lots:.3f}")
    print(f"  Avg cost per trade ($0.50 spread): ${avg_cost_50:.2f}")
    print(f"  Total cost ($0.50 spread, {len(d_trades)} trades): ${total_cost_50:.0f}")
    gross_pnl = d_stats['total_pnl']
    if gross_pnl > 0:
        print(f"  Cost as % of gross PnL: {total_cost_50/gross_pnl*100:.1f}%")

    RESULTS['transaction_costs'] = {
        'avg_lots': float(avg_lots),
        'n_trades': len(d_trades),
        'gross_pnl': float(gross_pnl),
        'cost_at_050': float(total_cost_50),
        'net_pnl_at_050': float(gross_pnl - total_cost_50),
        'cost_pct': float(total_cost_50 / gross_pnl * 100) if gross_pnl > 0 else 0,
    }

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 2: True Baseline
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  CHALLENGE 2: True Baseline Isolation")
    print("  Compare C12+Adaptive vs original params (Trail 1.5/0.5, SL 2.5, TP 3.0, ADX 24)")
    print("=" * 100)

    print("\n  Running true baseline...", flush=True)
    baseline_stats = run_variant(bundle, "TRUE Baseline", verbose=False,
                                 spread_cost=SPREAD, **TRUE_BASELINE_KWARGS)

    print(f"  TRUE Baseline: {baseline_stats['n']} trades, Sharpe={baseline_stats['sharpe']:.2f}, "
          f"PnL=${baseline_stats['total_pnl']:.0f}, MaxDD=${baseline_stats['max_dd']:.0f}")
    print(f"  C12+Adaptive:  {d_stats['n']} trades, Sharpe={d_stats['sharpe']:.2f}, "
          f"PnL=${d_stats['total_pnl']:.0f}, MaxDD=${d_stats['max_dd']:.0f}")
    real_improvement = d_stats['sharpe'] - baseline_stats['sharpe']
    print(f"\n  REAL improvement from TRUE baseline: +{real_improvement:.2f} Sharpe")

    RESULTS['true_baseline'] = {
        'baseline_sharpe': float(baseline_stats['sharpe']),
        'baseline_pnl': float(baseline_stats['total_pnl']),
        'baseline_dd': float(baseline_stats['max_dd']),
        'combo_sharpe': float(d_stats['sharpe']),
        'real_improvement': float(real_improvement),
    }

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 3: Signal Clustering
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  CHALLENGE 3: Signal Clustering Analysis")
    print("=" * 100)

    entry_times = [(t.entry_time, t.strategy, t.direction) for t in d_trades]
    sorted_entries = sorted(entry_times, key=lambda x: x[0])

    time_gaps = []
    for i in range(1, len(sorted_entries)):
        gap = (sorted_entries[i][0] - sorted_entries[i-1][0]).total_seconds() / 3600
        time_gaps.append(gap)

    same_hour_entries = sum(1 for g in time_gaps if g < 1)
    same_4h_entries = sum(1 for g in time_gaps if g < 4)

    print(f"\n  Total trades: {len(d_trades)}")
    print(f"  H1 entries: {d_stats.get('h1_entries', 0)}, M15 entries: {d_stats.get('m15_entries', 0)}")
    if time_gaps:
        print(f"\n  Time gap between consecutive entries:")
        print(f"    < 1 hour:  {same_hour_entries} ({same_hour_entries/len(time_gaps)*100:.1f}%)")
        print(f"    < 4 hours: {same_4h_entries} ({same_4h_entries/len(time_gaps)*100:.1f}%)")
        print(f"    Median gap: {np.median(time_gaps):.1f} hours")
        print(f"    Mean gap:   {np.mean(time_gaps):.1f} hours")

    strategy_counts = Counter(t.strategy for t in d_trades)
    print(f"\n  Trades by strategy:")
    for s, c in strategy_counts.most_common():
        pnl = sum(t.pnl for t in d_trades if t.strategy == s)
        wr = sum(1 for t in d_trades if t.strategy == s and t.pnl > 0) / c * 100
        print(f"    {s:<20}: {c:>5} trades, PnL=${pnl:>9.0f}, WR={wr:>5.1f}%")

    entry_dates = Counter(t.entry_time.date() for t in d_trades)
    max_day = entry_dates.most_common(1)[0]
    avg_per_day = np.mean(list(entry_dates.values()))
    days_with_5plus = sum(1 for c in entry_dates.values() if c >= 5)
    print(f"\n  Trades per day:")
    print(f"    Average: {avg_per_day:.1f}")
    print(f"    Max: {max_day[1]} (on {max_day[0]})")
    print(f"    Days with 5+ trades: {days_with_5plus} ({days_with_5plus/len(entry_dates)*100:.1f}%)")

    RESULTS['signal_clustering'] = {
        'same_hour_pct': float(same_hour_entries / len(time_gaps) * 100) if time_gaps else 0,
        'median_gap_hours': float(np.median(time_gaps)) if time_gaps else 0,
        'avg_trades_per_day': float(avg_per_day),
        'max_trades_day': int(max_day[1]),
    }

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 4: Worst Drawdown Analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  CHALLENGE 4: Worst Drawdown Deep-Dive")
    print("=" * 100)

    equity = [config.CAPITAL]
    for t in d_trades:
        equity.append(equity[-1] + t.pnl)
    eq_arr = np.array(equity)
    peak = np.maximum.accumulate(eq_arr)
    dd_arr = eq_arr - peak

    worst_dd_idx = np.argmin(dd_arr)
    peak_before_dd = np.argmax(eq_arr[:worst_dd_idx+1])

    dd_trades = d_trades[peak_before_dd:worst_dd_idx]
    dd_start = d_trades[peak_before_dd].entry_time if peak_before_dd < len(d_trades) else "N/A"
    dd_end = d_trades[worst_dd_idx-1].exit_time if worst_dd_idx > 0 and worst_dd_idx-1 < len(d_trades) else "N/A"

    print(f"\n  Worst drawdown: ${abs(dd_arr.min()):.0f}")
    print(f"  DD period: {dd_start} -> {dd_end}")
    print(f"  Trades in DD: {len(dd_trades)}")
    if dd_trades:
        dd_losing = sum(1 for t in dd_trades if t.pnl < 0)
        dd_total_loss = sum(t.pnl for t in dd_trades if t.pnl < 0)
        print(f"  Losing trades: {dd_losing}/{len(dd_trades)} ({dd_losing/len(dd_trades)*100:.0f}%)")
        print(f"  Total loss in DD: ${dd_total_loss:.0f}")

    losing_streaks = []
    streak = 0
    for t in d_trades:
        if t.pnl < 0:
            streak += 1
        else:
            if streak > 0:
                losing_streaks.append(streak)
            streak = 0
    if streak > 0:
        losing_streaks.append(streak)

    print(f"\n  Consecutive losing streaks:")
    print(f"    Max streak: {max(losing_streaks) if losing_streaks else 0}")
    print(f"    5+ streaks: {sum(1 for s in losing_streaks if s >= 5)}")
    print(f"    10+ streaks: {sum(1 for s in losing_streaks if s >= 10)}")

    monthly_pnl = {}
    for t in d_trades:
        key = t.entry_time.strftime('%Y-%m')
        monthly_pnl[key] = monthly_pnl.get(key, 0) + t.pnl
    negative_months = {k: v for k, v in monthly_pnl.items() if v < 0}
    print(f"\n  Monthly breakdown:")
    print(f"    Total months: {len(monthly_pnl)}")
    print(f"    Negative months: {len(negative_months)} ({len(negative_months)/len(monthly_pnl)*100:.1f}%)")
    if negative_months:
        print(f"    Top 5 worst months:")
        for m, p in sorted(negative_months.items(), key=lambda x: x[1])[:5]:
            print(f"      {m}: ${p:.0f}")

    RESULTS['drawdown'] = {
        'worst_dd': float(abs(dd_arr.min())),
        'max_losing_streak': max(losing_streaks) if losing_streaks else 0,
        'negative_months': len(negative_months),
        'total_months': len(monthly_pnl),
        'worst_month_pnl': float(min(monthly_pnl.values())) if monthly_pnl else 0,
    }

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 5: Parameter Cliff Test
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  CHALLENGE 5: Parameter Cliff Test")
    print("  Does small perturbation around ADX/SL/TP cause sharp drops?")
    print("=" * 100)

    cliff_variants = [
        ("ADX=15", {"keltner_adx_threshold": 15}),
        ("ADX=16", {"keltner_adx_threshold": 16}),
        ("ADX=17", {"keltner_adx_threshold": 17}),
        ("ADX=18 [OURS]", {"keltner_adx_threshold": 18}),
        ("ADX=19", {"keltner_adx_threshold": 19}),
        ("ADX=20", {"keltner_adx_threshold": 20}),
        ("ADX=22", {"keltner_adx_threshold": 22}),
        ("ADX=24", {"keltner_adx_threshold": 24}),
        ("SL=2.5", {"sl_atr_mult": 2.5}),
        ("SL=3.0", {"sl_atr_mult": 3.0}),
        ("SL=3.5 [OURS]", {"sl_atr_mult": 3.5}),
        ("SL=4.0", {"sl_atr_mult": 4.0}),
        ("SL=4.5", {"sl_atr_mult": 4.5}),
        ("TP=3.0", {"tp_atr_mult": 3.0}),
        ("TP=4.0", {"tp_atr_mult": 4.0}),
        ("TP=5.0 [OURS]", {"tp_atr_mult": 5.0}),
        ("TP=6.0", {"tp_atr_mult": 6.0}),
        ("TP=7.0", {"tp_atr_mult": 7.0}),
    ]

    cliff_results = []
    for i, (label, overrides) in enumerate(cliff_variants, 1):
        print(f"  [{i}/{len(cliff_variants)}] {label}", end="", flush=True)
        t0 = time.time()
        kw = {**ADAPTIVE_KWARGS, **overrides}
        stats = run_variant(bundle, label, verbose=False, **kw)
        elapsed = time.time() - t0
        print(f"  Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"N={stats['n']}, {elapsed:.0f}s")
        cliff_results.append({
            'label': label, 'sharpe': float(stats['sharpe']),
            'pnl': float(stats['total_pnl']), 'dd': float(stats['max_dd']),
            'n': int(stats['n']),
        })

    print(f"\n  --- ADX Sensitivity ---")
    adx_series = [r for r in cliff_results if r['label'].startswith('ADX')]
    for r in adx_series:
        marker = " <<" if "[OURS]" in r['label'] else ""
        print(f"    {r['label']:<16}: Sharpe={r['sharpe']:.2f}, PnL=${r['pnl']:>9.0f}, N={r['n']:>6}{marker}")

    print(f"\n  --- SL Sensitivity ---")
    sl_series = [r for r in cliff_results if r['label'].startswith('SL')]
    for r in sl_series:
        marker = " <<" if "[OURS]" in r['label'] else ""
        print(f"    {r['label']:<16}: Sharpe={r['sharpe']:.2f}, PnL=${r['pnl']:>9.0f}, N={r['n']:>6}{marker}")

    print(f"\n  --- TP Sensitivity ---")
    tp_series = [r for r in cliff_results if r['label'].startswith('TP')]
    for r in tp_series:
        marker = " <<" if "[OURS]" in r['label'] else ""
        print(f"    {r['label']:<16}: Sharpe={r['sharpe']:.2f}, PnL=${r['pnl']:>9.0f}, N={r['n']:>6}{marker}")

    def sharpe_range(series):
        vals = [r['sharpe'] for r in series]
        return max(vals) - min(vals) if vals else 0

    adx_range = sharpe_range(adx_series)
    sl_range = sharpe_range(sl_series)
    tp_range = sharpe_range(tp_series)
    print(f"\n  ADX Sharpe range: {adx_range:.2f} ({'CLIFF' if adx_range > 1.0 else 'smooth' if adx_range < 0.5 else 'moderate'})")
    print(f"  SL  Sharpe range: {sl_range:.2f} ({'CLIFF' if sl_range > 1.0 else 'smooth' if sl_range < 0.5 else 'moderate'})")
    print(f"  TP  Sharpe range: {tp_range:.2f} ({'CLIFF' if tp_range > 1.0 else 'smooth' if tp_range < 0.5 else 'moderate'})")

    RESULTS['parameter_cliff'] = cliff_results

    # ══════════════════════════════════════════════════════════════
    # FINAL ASSESSMENT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  FINAL STRESS TEST ASSESSMENT")
    print("=" * 100)

    issues = []
    cost_pct = RESULTS['transaction_costs']['cost_pct']
    if cost_pct > 30:
        issues.append(f"HIGH: Transaction costs eat {cost_pct:.0f}% of PnL")
    elif cost_pct > 15:
        issues.append(f"MEDIUM: Transaction costs eat {cost_pct:.0f}% of PnL")

    if real_improvement < 0.5:
        issues.append(f"HIGH: Real improvement from true baseline only +{real_improvement:.2f}")

    cluster_pct = RESULTS['signal_clustering']['same_hour_pct']
    if cluster_pct > 30:
        issues.append(f"MEDIUM: {cluster_pct:.0f}% signals within 1 hour")

    for dim, rng in [("ADX", adx_range), ("SL", sl_range), ("TP", tp_range)]:
        if rng > 1.0:
            issues.append(f"HIGH: {dim} parameter cliff (range={rng:.2f})")

    neg_ratio = RESULTS['drawdown']['negative_months'] / max(RESULTS['drawdown']['total_months'], 1)
    if neg_ratio > 0.3:
        issues.append(f"MEDIUM: {RESULTS['drawdown']['negative_months']}/{RESULTS['drawdown']['total_months']} negative months")

    if issues:
        print(f"\n  Issues found ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  No critical issues found.")

    print(f"\n  Summary:")
    print(f"    Sharpe:                {d_stats['sharpe']:.2f}")
    print(f"    PnL (with spread):     ${d_stats['total_pnl']:.0f}")
    print(f"    True baseline Sharpe:  {baseline_stats['sharpe']:.2f}")
    print(f"    Real Sharpe gain:      +{real_improvement:.2f}")
    print(f"    Negative months:       {RESULTS['drawdown']['negative_months']}/{RESULTS['drawdown']['total_months']}")

    out = Path("data/stress_test_results.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  TOTAL: {elapsed/60:.0f} min ({elapsed:.0f}s)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
