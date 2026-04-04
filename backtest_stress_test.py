"""
Strategy Stress Test - Challenge the "too perfect" results
============================================================
5 challenges to the Combo (Adaptive Trail + KC1.25 + KC_EMA30):

  Challenge 1: Spread/Slippage cost (回测无交易成本)
  Challenge 2: Baseline isolation (Baseline 是否被污染)
  Challenge 3: Signal clustering (信号是否聚集/重复)
  Challenge 4: Worst drawdown deep-dive (最差时段放大)
  Challenge 5: Parameter cliff test (参数悬崖检测)
"""
import json
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import config
from strategies.signals import get_orb_strategy, prepare_indicators
import strategies.signals as signals_mod
from backtest import Position, TradeRecord
from backtest_m15 import (
    load_m15, load_h1_aligned, build_h1_lookup,
    MultiTimeframeEngine, calc_stats,
    M15_CSV_PATH, H1_CSV_PATH,
)
from backtest_round2 import Round2Engine
from backtest_advanced import (
    C12_KWARGS, RegimeEngine, prepare_indicators_custom,
)
from backtest_combo_verify import V3_REGIME, add_atr_percentile

RESULTS = {}


def run_fixed(m15_df, h1_df, label, **kwargs):
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = Round2Engine(m15_df, h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    return stats, trades, engine


def run_regime(m15_df, h1_df, label, regime_config, **kwargs):
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = RegimeEngine(m15_df, h1_df, regime_config=regime_config, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    return stats, trades, engine


def main():
    t_start = time.time()
    print("=" * 100)
    print("  STRATEGY STRESS TEST - Challenging the Combo")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15_default = prepare_indicators(m15_raw)
    h1_default = prepare_indicators(h1_raw)
    h1_default = add_atr_percentile(h1_default)

    m15_custom = prepare_indicators_custom(m15_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = prepare_indicators_custom(h1_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = add_atr_percentile(h1_custom)

    print(f"  M15: {len(m15_default)} bars, H1: {len(h1_default)} bars\n")

    # ══════════════════════════════════════════════════════════════
    # Get Combo D trades
    # ══════════════════════════════════════════════════════════════
    print("  Running Combo D to get trade list...", flush=True)
    d_stats, d_trades, d_engine = run_regime(m15_custom, h1_custom, "D: Combo", V3_REGIME, **C12_KWARGS)
    print(f"  {len(d_trades)} trades, Sharpe={d_stats['sharpe']:.2f}, PnL=${d_stats['total_pnl']:.0f}\n")

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 1: Transaction costs
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  CHALLENGE 1: Transaction Costs (Spread + Slippage)")
    print("  Current backtest: ZERO transaction costs!")
    print("=" * 100)

    spread_scenarios = [
        ("No cost (current)", 0),
        ("Spread $0.30 (tight)", 0.30),
        ("Spread $0.50 (normal)", 0.50),
        ("Spread $0.80 (wide)", 0.80),
        ("Spread $1.00 (adverse)", 1.00),
        ("Spread $1.50 (extreme)", 1.50),
    ]

    print(f"\n  XAUUSD typical spread: $0.30-0.80 per lot")
    print(f"  Each trade uses ~0.01-0.03 lots")
    print(f"  Cost per trade = spread * lots * POINT_VALUE_PER_LOT")

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
    print(f"  Cost as % of gross PnL: {total_cost_50/d_stats['total_pnl']*100:.1f}%")

    RESULTS['transaction_costs'] = {
        'avg_lots': float(avg_lots),
        'n_trades': len(d_trades),
        'gross_pnl': float(d_stats['total_pnl']),
        'cost_at_050': float(total_cost_50),
        'net_pnl_at_050': float(d_stats['total_pnl'] - total_cost_50),
        'cost_pct': float(total_cost_50 / d_stats['total_pnl'] * 100),
    }

    # ══════════════════════════════════════════════════════════════
    # CHALLENGE 2: True Baseline (original parameters before ANY optimization)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  CHALLENGE 2: True Baseline Isolation")
    print("  Problem: config.py already has Trail 0.8/0.25, so 'Baseline' = Trail-only")
    print("  Fix: Force original parameters (Trail 1.5/0.5, SL 2.5, TP 3.0, ADX 24)")
    print("=" * 100)

    true_baseline_kwargs = {
        "trailing_activate_atr": 1.5,
        "trailing_distance_atr": 0.5,
        "sl_atr_mult": 2.5,
        "tp_atr_mult": 3.0,
        "keltner_adx_threshold": 24,
    }

    orig_trail_act = config.TRAILING_ACTIVATE_ATR
    orig_trail_dist = config.TRAILING_DISTANCE_ATR
    config.TRAILING_ACTIVATE_ATR = 1.5
    config.TRAILING_DISTANCE_ATR = 0.5

    print("\n  Running true baseline (original params)...", flush=True)
    baseline_stats, baseline_trades, _ = run_fixed(
        m15_default, h1_default, "TRUE Baseline (Trail1.5/0.5, SL2.5, TP3, ADX24)",
        **true_baseline_kwargs)

    config.TRAILING_ACTIVATE_ATR = orig_trail_act
    config.TRAILING_DISTANCE_ATR = orig_trail_dist

    print(f"  TRUE Baseline: {len(baseline_trades)} trades, Sharpe={baseline_stats['sharpe']:.2f}, "
          f"PnL=${baseline_stats['total_pnl']:.0f}, MaxDD=${baseline_stats['max_dd']:.0f}")
    print(f"  Combo D:       {len(d_trades)} trades, Sharpe={d_stats['sharpe']:.2f}, "
          f"PnL=${d_stats['total_pnl']:.0f}, MaxDD=${d_stats['max_dd']:.0f}")
    real_improvement = d_stats['sharpe'] - baseline_stats['sharpe']
    print(f"\n  REAL improvement from TRUE baseline: +{real_improvement:.2f} Sharpe")
    print(f"  Previous claim (vs polluted baseline): +0.92 Sharpe")
    print(f"  Difference: {real_improvement - 0.92:+.2f}")

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
    print("  Are the extra signals from narrower KC just duplicates?")
    print("=" * 100)

    entry_times = [(t.entry_time, t.strategy, t.direction) for t in d_trades]
    h1_entries = [(t.entry_time, t.strategy) for t in d_trades if t.strategy in ('keltner', 'orb', 'gap_fill')]
    m15_entries = [(t.entry_time, t.strategy) for t in d_trades if t.strategy == 'm15_rsi']

    time_gaps = []
    sorted_entries = sorted(entry_times, key=lambda x: x[0])
    for i in range(1, len(sorted_entries)):
        gap = (sorted_entries[i][0] - sorted_entries[i-1][0]).total_seconds() / 3600
        time_gaps.append(gap)

    same_hour_entries = sum(1 for g in time_gaps if g < 1)
    same_4h_entries = sum(1 for g in time_gaps if g < 4)

    print(f"\n  Total trades: {len(d_trades)}")
    print(f"  H1 entries: {d_engine.h1_entry_count}, M15 entries: {d_engine.m15_entry_count}")
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
        print(f"    {s:<15}: {c:>5} trades, PnL=${pnl:>9.0f}, WR={wr:>5.1f}%")

    entry_dates = Counter(t.entry_time.date() for t in d_trades)
    max_day = entry_dates.most_common(1)[0]
    avg_per_day = np.mean(list(entry_dates.values()))
    days_with_5plus = sum(1 for c in entry_dates.values() if c >= 5)
    print(f"\n  Trades per day:")
    print(f"    Average: {avg_per_day:.1f}")
    print(f"    Max: {max_day[1]} (on {max_day[0]})")
    print(f"    Days with 5+ trades: {days_with_5plus} ({days_with_5plus/len(entry_dates)*100:.1f}%)")

    RESULTS['signal_clustering'] = {
        'same_hour_pct': float(same_hour_entries / len(time_gaps) * 100),
        'median_gap_hours': float(np.median(time_gaps)),
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
    dd_end = d_trades[worst_dd_idx-1].exit_time if worst_dd_idx-1 < len(d_trades) else "N/A"

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
        worst_month = min(negative_months, key=negative_months.get)
        print(f"    Worst month: {worst_month} (${negative_months[worst_month]:.0f})")
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
    print("  Does small perturbation around KC_mult=1.25 / KC_EMA=30 cause sharp drops?")
    print("=" * 100)

    cliff_variants = [
        ("KC 1.00 / EMA 30", 1.00, 30),
        ("KC 1.10 / EMA 30", 1.10, 30),
        ("KC 1.20 / EMA 30", 1.20, 30),
        ("KC 1.25 / EMA 30 [OURS]", 1.25, 30),
        ("KC 1.30 / EMA 30", 1.30, 30),
        ("KC 1.40 / EMA 30", 1.40, 30),
        ("KC 1.50 / EMA 30", 1.50, 30),
        ("KC 1.25 / EMA 20", 1.25, 20),
        ("KC 1.25 / EMA 25", 1.25, 25),
        ("KC 1.25 / EMA 30 [OURS]", 1.25, 30),
        ("KC 1.25 / EMA 35", 1.25, 35),
        ("KC 1.25 / EMA 40", 1.25, 40),
    ]

    cliff_results = []
    for i, (label, kc_mult, kc_ema) in enumerate(cliff_variants, 1):
        print(f"\n  [{i}/{len(cliff_variants)}] {label}", flush=True)
        t0 = time.time()
        m15_v = prepare_indicators_custom(m15_raw, kc_ema=kc_ema, kc_mult=kc_mult)
        h1_v = prepare_indicators_custom(h1_raw, kc_ema=kc_ema, kc_mult=kc_mult)
        h1_v = add_atr_percentile(h1_v)

        stats = run_regime(m15_v, h1_v, label, V3_REGIME, **C12_KWARGS)[0]
        elapsed = time.time() - t0
        print(f"    Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"MaxDD=${stats['max_dd']:.0f}, N={stats['n']}, {elapsed:.0f}s")
        cliff_results.append({
            'label': label, 'kc_mult': kc_mult, 'kc_ema': kc_ema,
            'sharpe': float(stats['sharpe']), 'pnl': float(stats['total_pnl']),
            'dd': float(stats['max_dd']), 'n': int(stats['n']),
        })

    print(f"\n  --- KC Multiplier Sensitivity (EMA=30) ---")
    kc_series = [r for r in cliff_results if r['kc_ema'] == 30]
    for r in sorted(kc_series, key=lambda x: x['kc_mult']):
        marker = " <<" if r['kc_mult'] == 1.25 else ""
        print(f"    KC {r['kc_mult']:.2f}: Sharpe={r['sharpe']:.2f}, PnL=${r['pnl']:>9.0f}, "
              f"MaxDD=${r['dd']:>7.0f}, N={r['n']:>6}{marker}")

    print(f"\n  --- KC EMA Sensitivity (mult=1.25) ---")
    ema_series = [r for r in cliff_results if r['kc_mult'] == 1.25]
    for r in sorted(ema_series, key=lambda x: x['kc_ema']):
        marker = " <<" if r['kc_ema'] == 30 else ""
        print(f"    EMA {r['kc_ema']:>2}: Sharpe={r['sharpe']:.2f}, PnL=${r['pnl']:>9.0f}, "
              f"MaxDD=${r['dd']:>7.0f}, N={r['n']:>6}{marker}")

    sharpes_kc = [r['sharpe'] for r in kc_series]
    sharpes_ema = [r['sharpe'] for r in ema_series]
    kc_range = max(sharpes_kc) - min(sharpes_kc) if sharpes_kc else 0
    ema_range = max(sharpes_ema) - min(sharpes_ema) if sharpes_ema else 0
    print(f"\n  KC mult Sharpe range: {kc_range:.2f} ({'CLIFF' if kc_range > 1.0 else 'smooth' if kc_range < 0.5 else 'moderate'})")
    print(f"  KC EMA Sharpe range:  {ema_range:.2f} ({'CLIFF' if ema_range > 1.0 else 'smooth' if ema_range < 0.5 else 'moderate'})")

    RESULTS['parameter_cliff'] = cliff_results

    # ══════════════════════════════════════════════════════════════
    # FINAL ASSESSMENT
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("  FINAL STRESS TEST ASSESSMENT")
    print("=" * 100)

    issues = []
    if RESULTS['transaction_costs']['cost_pct'] > 30:
        issues.append(f"HIGH: Transaction costs eat {RESULTS['transaction_costs']['cost_pct']:.0f}% of PnL")
    elif RESULTS['transaction_costs']['cost_pct'] > 15:
        issues.append(f"MEDIUM: Transaction costs eat {RESULTS['transaction_costs']['cost_pct']:.0f}% of PnL")

    if real_improvement < 0.5:
        issues.append(f"HIGH: Real improvement from true baseline only +{real_improvement:.2f}")

    if RESULTS['signal_clustering']['same_hour_pct'] > 30:
        issues.append(f"MEDIUM: {RESULTS['signal_clustering']['same_hour_pct']:.0f}% signals within 1 hour")

    if kc_range > 1.0:
        issues.append(f"HIGH: KC mult parameter cliff (range={kc_range:.2f})")
    if ema_range > 1.0:
        issues.append(f"HIGH: KC EMA parameter cliff (range={ema_range:.2f})")

    if RESULTS['drawdown']['negative_months'] / RESULTS['drawdown']['total_months'] > 0.3:
        issues.append(f"MEDIUM: {RESULTS['drawdown']['negative_months']}/{RESULTS['drawdown']['total_months']} negative months")

    if issues:
        print(f"\n  Issues found ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  No critical issues found.")

    print(f"\n  Summary:")
    print(f"    Gross Sharpe:          {d_stats['sharpe']:.2f}")
    net_pnl = d_stats['total_pnl'] - RESULTS['transaction_costs']['cost_at_050']
    print(f"    Net PnL (0.50 spread): ${net_pnl:.0f} (was ${d_stats['total_pnl']:.0f})")
    print(f"    True baseline Sharpe:  {baseline_stats['sharpe']:.2f}")
    print(f"    Real Sharpe gain:      +{real_improvement:.2f}")
    print(f"    KC parameter cliff:    {kc_range:.2f}")
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
