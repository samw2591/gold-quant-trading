"""
Yearly / Monthly / Weekday Strategy Breakdown
===============================================
Runs C12+Adaptive once, then dissects trades by:
  - Year x Strategy
  - Month (1-12) x Strategy
  - Weekday (Mon-Fri) x Strategy

Usage: python backtest_yearly_strategy_breakdown.py
"""
import time
from collections import defaultdict

import pandas as pd

from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from backtest.engine import BacktestEngine
from strategies.signals import prepare_indicators

SPREAD = 0.50


def group_stats(trades):
    if not trades:
        return 0, 0, 0, 0
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return len(trades), sum(pnls), wins / len(trades) * 100, sum(pnls) / len(trades)


def main():
    print("=" * 90)
    print("  YEARLY / MONTHLY / WEEKDAY STRATEGY BREAKDOWN")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)

    kw = {
        **C12_KWARGS, "spread_cost": SPREAD,
        "intraday_adaptive": True, "choppy_threshold": 0.35, "kc_only_threshold": 0.60,
    }
    print("\n  Running C12+Adaptive backtest...")
    engine = BacktestEngine(m15_df, h1_df, label="C12+Adaptive", **kw)
    engine.run()
    trades = engine.trades
    print(f"  Total trades: {len(trades)}")

    strategies = sorted(set(t.strategy for t in trades))

    # Year x Strategy
    print("\n" + "=" * 90)
    print("  Part 1: Year x Strategy")
    print("=" * 90)

    by_year = defaultdict(lambda: defaultdict(list))
    for t in trades:
        year = pd.Timestamp(t.entry_time).year
        by_year[year][t.strategy].append(t)
        by_year[year]['_all'].append(t)

    header = f"  {'Year':<6}"
    for s in strategies:
        header += f" {s[:8]:>8}N {s[:8]:>9}$ {s[:8]:>6}WR"
    header += f" {'Total':>8}N {'Total':>9}$ {'Total':>6}WR"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for year in sorted(by_year.keys()):
        line = f"  {year:<6}"
        for s in strategies:
            n, pnl, wr, avg = group_stats(by_year[year].get(s, []))
            line += f" {n:>9} ${pnl:>8.0f} {wr:>5.1f}%"
        n, pnl, wr, avg = group_stats(by_year[year]['_all'])
        line += f" {n:>9} ${pnl:>8.0f} {wr:>5.1f}%"
        print(line)

    # Month x Strategy
    print("\n" + "=" * 90)
    print("  Part 2: Month x Strategy")
    print("=" * 90)

    by_month = defaultdict(lambda: defaultdict(list))
    for t in trades:
        month = pd.Timestamp(t.entry_time).month
        by_month[month][t.strategy].append(t)
        by_month[month]['_all'].append(t)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"\n  {'Month':<6} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 42)
    for m in range(1, 13):
        n, pnl, wr, avg = group_stats(by_month[m]['_all'])
        print(f"  {month_names[m-1]:<6} {n:>6} ${pnl:>9.0f} {wr:>6.1f}% ${avg:>8.2f}")

    # Weekday x Strategy
    print("\n" + "=" * 90)
    print("  Part 3: Weekday x Strategy")
    print("=" * 90)

    by_dow = defaultdict(lambda: defaultdict(list))
    for t in trades:
        dow = pd.Timestamp(t.entry_time).dayofweek
        by_dow[dow][t.strategy].append(t)
        by_dow[dow]['_all'].append(t)

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    print(f"\n  {'Day':<6} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 42)
    for d in range(7):
        if d not in by_dow:
            continue
        n, pnl, wr, avg = group_stats(by_dow[d]['_all'])
        print(f"  {day_names[d]:<6} {n:>6} ${pnl:>9.0f} {wr:>6.1f}% ${avg:>8.2f}")

    # Hour distribution
    print("\n" + "=" * 90)
    print("  Part 4: Hour of Day (Entry Time)")
    print("=" * 90)

    by_hour = defaultdict(list)
    for t in trades:
        hour = pd.Timestamp(t.entry_time).hour
        by_hour[hour].append(t)

    print(f"\n  {'Hour':>6} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 42)
    for h in range(24):
        if h not in by_hour:
            continue
        n, pnl, wr, avg = group_stats(by_hour[h])
        if n > 0:
            print(f"  {h:>4}:00 {n:>6} ${pnl:>9.0f} {wr:>6.1f}% ${avg:>8.2f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
