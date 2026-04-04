"""
Hold Time & Exit Reason Analysis
==================================
Dissects C12+Adaptive trades by:
  - Hold time buckets (how long positions are held)
  - Exit reason (SL / TP / trailing / timeout)
  - Hold time x Strategy
  - Optimal exit timing: would earlier/later exits improve results?

Usage: python backtest_hold_time_analysis.py
"""
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREAD = 0.50

ADAPTIVE_KWARGS = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
}


def bucket_stats(trades):
    if not trades:
        return 0, 0, 0, 0, 0
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    avg_bars = np.mean([t.bars_held for t in trades])
    return len(trades), sum(pnls), wins / len(trades) * 100, sum(pnls) / len(trades), avg_bars


def main():
    print("=" * 90)
    print("  HOLD TIME & EXIT REASON ANALYSIS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    print("\n  Running C12+Adaptive backtest...", flush=True)
    stats = run_variant(bundle, "C12+Adaptive", verbose=False, **ADAPTIVE_KWARGS)
    trades = stats['_trades']
    print(f"  {len(trades)} trades, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}")

    # Part 1: Hold time distribution
    print("\n" + "=" * 90)
    print("  Part 1: Hold Time Distribution (M15 bars)")
    print("=" * 90)

    hold_buckets = [
        ("1-4 bars (15-60min)", 1, 4),
        ("5-8 bars (1-2h)", 5, 8),
        ("9-16 bars (2-4h)", 9, 16),
        ("17-32 bars (4-8h)", 17, 32),
        ("33-64 bars (8-16h)", 33, 64),
        ("65-96 bars (16-24h)", 65, 96),
        ("97+ bars (24h+)", 97, 9999),
    ]

    print(f"\n  {'Bucket':<28} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9} {'AvgBars':>8}")
    print("  " + "-" * 72)

    for label, lo, hi in hold_buckets:
        bucket_trades = [t for t in trades if lo <= t.bars_held <= hi]
        n, pnl, wr, avg_pnl, avg_bars = bucket_stats(bucket_trades)
        if n > 0:
            print(f"  {label:<28} {n:>6} ${pnl:>9.0f} {wr:>6.1f}% ${avg_pnl:>8.2f} {avg_bars:>7.1f}")

    bars_held = [t.bars_held for t in trades]
    print(f"\n  Overall: median={np.median(bars_held):.0f} bars, "
          f"mean={np.mean(bars_held):.0f} bars, "
          f"max={max(bars_held)} bars ({max(bars_held)*15/60:.0f}h)")

    # Part 2: Exit reason breakdown
    print("\n" + "=" * 90)
    print("  Part 2: Exit Reason Breakdown")
    print("=" * 90)

    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t.exit_reason].append(t)

    print(f"\n  {'Exit Reason':<25} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9} {'AvgBars':>8}")
    print("  " + "-" * 68)

    for reason in sorted(by_exit.keys()):
        rt = by_exit[reason]
        n, pnl, wr, avg_pnl, avg_bars = bucket_stats(rt)
        print(f"  {reason:<25} {n:>6} ${pnl:>9.0f} {wr:>6.1f}% ${avg_pnl:>8.2f} {avg_bars:>7.1f}")

    # Part 3: Strategy x Hold time
    print("\n" + "=" * 90)
    print("  Part 3: Strategy x Hold Time")
    print("=" * 90)

    strategies = sorted(set(t.strategy for t in trades))
    for strat in strategies:
        strat_trades = [t for t in trades if t.strategy == strat]
        if not strat_trades:
            continue
        bars = [t.bars_held for t in strat_trades]
        pnls = [t.pnl for t in strat_trades]
        print(f"\n  {strat}:")
        print(f"    N={len(strat_trades)}, PnL=${sum(pnls):.0f}")
        print(f"    Hold: median={np.median(bars):.0f}, mean={np.mean(bars):.0f}, "
              f"min={min(bars)}, max={max(bars)} bars")

        for label, lo, hi in hold_buckets:
            bt = [t for t in strat_trades if lo <= t.bars_held <= hi]
            if bt:
                n, pnl, wr, avg_pnl, _ = bucket_stats(bt)
                print(f"      {label:<28} N={n:>4}, PnL=${pnl:>7.0f}, WR={wr:.1f}%")

    # Part 4: Strategy x Exit reason
    print("\n" + "=" * 90)
    print("  Part 4: Strategy x Exit Reason")
    print("=" * 90)

    for strat in strategies:
        strat_trades = [t for t in trades if t.strategy == strat]
        if not strat_trades:
            continue
        print(f"\n  {strat}:")
        strat_exits = defaultdict(list)
        for t in strat_trades:
            strat_exits[t.exit_reason].append(t)
        for reason in sorted(strat_exits.keys()):
            rt = strat_exits[reason]
            n, pnl, wr, avg_pnl, avg_bars = bucket_stats(rt)
            print(f"    {reason:<25} N={n:>4}, PnL=${pnl:>7.0f}, WR={wr:.1f}%, "
                  f"avg=${avg_pnl:.2f}, bars={avg_bars:.0f}")

    # Part 5: Winner/Loser hold time comparison
    print("\n" + "=" * 90)
    print("  Part 5: Winners vs Losers Hold Time")
    print("=" * 90)

    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]

    w_bars = [t.bars_held for t in winners]
    l_bars = [t.bars_held for t in losers]

    print(f"\n  {'':>20} {'Winners':>12} {'Losers':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    print(f"  {'Count':<20} {len(winners):>12} {len(losers):>12}")
    print(f"  {'Avg hold (bars)':<20} {np.mean(w_bars):>12.1f} {np.mean(l_bars):>12.1f}")
    print(f"  {'Med hold (bars)':<20} {np.median(w_bars):>12.0f} {np.median(l_bars):>12.0f}")
    print(f"  {'Avg PnL':<20} ${np.mean([t.pnl for t in winners]):>11.2f} "
          f"${np.mean([t.pnl for t in losers]):>11.2f}")
    print(f"  {'Max hold':<20} {max(w_bars):>12} {max(l_bars):>12}")

    # Part 6: Time-of-day entry performance
    print("\n" + "=" * 90)
    print("  Part 6: Hold Time vs Profitability Correlation")
    print("=" * 90)

    bars_arr = np.array([t.bars_held for t in trades])
    pnl_arr = np.array([t.pnl for t in trades])
    corr = np.corrcoef(bars_arr, pnl_arr)[0, 1]
    print(f"\n  Correlation(hold_time, pnl): {corr:.3f}")

    short_trades = [t for t in trades if t.bars_held <= np.median(bars_held)]
    long_trades = [t for t in trades if t.bars_held > np.median(bars_held)]
    s_n, s_pnl, s_wr, s_avg, _ = bucket_stats(short_trades)
    l_n, l_pnl, l_wr, l_avg, _ = bucket_stats(long_trades)
    print(f"  Short holds (<=median): N={s_n}, PnL=${s_pnl:.0f}, WR={s_wr:.1f}%, avg=${s_avg:.2f}")
    print(f"  Long holds  (>median):  N={l_n}, PnL=${l_pnl:.0f}, WR={l_wr:.1f}%, avg=${l_avg:.2f}")

    verdict = "longer" if l_avg > s_avg else "shorter"
    print(f"\n  VERDICT: {verdict} holds tend to be more profitable per trade")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
