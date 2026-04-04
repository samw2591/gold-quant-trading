"""
BUY vs SELL Direction Bias Analysis
======================================
Gold has a long-term upward bias. Does this create a structural
advantage for BUY trades and penalty for SELL trades?

Analysis:
  1. BUY vs SELL overall performance
  2. BUY vs SELL by strategy
  3. BUY vs SELL by year (has the bias changed?)
  4. BUY vs SELL by session (Asian/London/NY)
  5. What if we disable SELL entirely?
  6. Consecutive direction patterns

Usage: python backtest_direction_bias.py
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
from backtest.engine import BacktestEngine
from strategies.signals import prepare_indicators

SPREAD = 0.50

ADAPTIVE_KWARGS = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
}


def dir_stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "avg": 0, "avg_bars": 0}
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    avg_bars = np.mean([t.bars_held for t in trades])
    return {
        "n": len(trades),
        "pnl": sum(pnls),
        "wr": wins / len(trades) * 100,
        "avg": sum(pnls) / len(trades),
        "avg_bars": avg_bars,
    }


def session_of(ts):
    """Classify entry time into trading session."""
    hour = pd.Timestamp(ts).hour
    if 0 <= hour < 8:
        return "Asian"
    elif 8 <= hour < 13:
        return "London"
    elif 13 <= hour < 21:
        return "New York"
    else:
        return "Late NY"


def main():
    print("=" * 90)
    print("  BUY vs SELL DIRECTION BIAS ANALYSIS")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    print("\n  Running C12+Adaptive full backtest...", flush=True)
    engine = BacktestEngine(m15_df, h1_df, label="C12+Adaptive", **ADAPTIVE_KWARGS)
    engine.run()
    all_trades = engine.trades
    print(f"  {len(all_trades)} trades total")

    buys = [t for t in all_trades if t.direction == 'BUY']
    sells = [t for t in all_trades if t.direction == 'SELL']

    # Part 1: Overall BUY vs SELL
    print("\n" + "=" * 90)
    print("  Part 1: Overall BUY vs SELL")
    print("=" * 90)

    b = dir_stats(buys)
    s = dir_stats(sells)

    print(f"\n  {'':>20} {'BUY':>12} {'SELL':>12} {'Ratio':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Count':<20} {b['n']:>12} {s['n']:>12} {b['n']/max(s['n'],1):>10.2f}")
    print(f"  {'Total PnL':<20} ${b['pnl']:>11.0f} ${s['pnl']:>11.0f}")
    print(f"  {'Win Rate':<20} {b['wr']:>11.1f}% {s['wr']:>11.1f}%")
    print(f"  {'Avg $/trade':<20} ${b['avg']:>11.2f} ${s['avg']:>11.2f}")
    print(f"  {'Avg hold (bars)':<20} {b['avg_bars']:>12.1f} {s['avg_bars']:>12.1f}")

    buy_pct_of_pnl = b['pnl'] / (b['pnl'] + s['pnl']) * 100 if (b['pnl'] + s['pnl']) != 0 else 0
    print(f"\n  BUY contributes {buy_pct_of_pnl:.1f}% of total PnL")

    # Part 2: BUY vs SELL by Strategy
    print("\n" + "=" * 90)
    print("  Part 2: BUY vs SELL by Strategy")
    print("=" * 90)

    strategies = sorted(set(t.strategy for t in all_trades))

    print(f"\n  {'Strategy':<20} {'Dir':>5} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 60)

    for strat in strategies:
        for direction in ['BUY', 'SELL']:
            dt = [t for t in all_trades if t.strategy == strat and t.direction == direction]
            ds = dir_stats(dt)
            if ds['n'] > 0:
                print(f"  {strat:<20} {direction:>5} {ds['n']:>6} ${ds['pnl']:>9.0f} "
                      f"{ds['wr']:>6.1f}% ${ds['avg']:>8.2f}")
        print()

    # Part 3: BUY vs SELL by Year
    print("=" * 90)
    print("  Part 3: BUY vs SELL by Year")
    print("=" * 90)

    years = sorted(set(pd.Timestamp(t.entry_time).year for t in all_trades))

    print(f"\n  {'Year':<6} {'BUY_N':>6} {'BUY_PnL':>10} {'BUY_WR':>7} "
          f"{'SELL_N':>7} {'SELL_PnL':>10} {'SELL_WR':>8} {'BUY_edge':>10}")
    print("  " + "-" * 68)

    yearly_buy_edge = []
    for year in years:
        yb = [t for t in buys if pd.Timestamp(t.entry_time).year == year]
        ys = [t for t in sells if pd.Timestamp(t.entry_time).year == year]
        bs = dir_stats(yb)
        ss = dir_stats(ys)
        edge = bs['avg'] - ss['avg']
        yearly_buy_edge.append(edge)
        print(f"  {year:<6} {bs['n']:>6} ${bs['pnl']:>9.0f} {bs['wr']:>6.1f}% "
              f"{ss['n']:>7} ${ss['pnl']:>9.0f} {ss['wr']:>7.1f}% ${edge:>9.2f}")

    buy_wins_years = sum(1 for e in yearly_buy_edge if e > 0)
    print(f"\n  BUY has higher avg $/trade in {buy_wins_years}/{len(years)} years")

    # Part 4: BUY vs SELL by Session
    print("\n" + "=" * 90)
    print("  Part 4: BUY vs SELL by Trading Session")
    print("=" * 90)

    sessions = ["Asian", "London", "New York", "Late NY"]
    print(f"\n  {'Session':<12} {'Dir':>5} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 52)

    for sess in sessions:
        for direction in ['BUY', 'SELL']:
            dt = [t for t in all_trades
                  if t.direction == direction and session_of(t.entry_time) == sess]
            ds = dir_stats(dt)
            if ds['n'] > 0:
                print(f"  {sess:<12} {direction:>5} {ds['n']:>6} ${ds['pnl']:>9.0f} "
                      f"{ds['wr']:>6.1f}% ${ds['avg']:>8.2f}")
        print()

    # Part 5: What if SELL disabled?
    print("=" * 90)
    print("  Part 5: Hypothetical — Disable SELL?")
    print("=" * 90)

    buy_only_pnl = sum(t.pnl for t in buys)
    sell_only_pnl = sum(t.pnl for t in sells)
    total_pnl = buy_only_pnl + sell_only_pnl

    from backtest.stats import aggregate_daily_pnl

    buy_daily = aggregate_daily_pnl(buys)
    sell_daily = aggregate_daily_pnl(sells)
    all_daily = aggregate_daily_pnl(all_trades)

    def daily_sharpe(dpnl):
        if len(dpnl) > 1 and np.std(dpnl) > 0:
            return np.mean(dpnl) / np.std(dpnl) * np.sqrt(252)
        return 0

    print(f"\n  {'Scenario':<25} {'N':>6} {'PnL':>10} {'Sharpe':>8}")
    print("  " + "-" * 52)
    print(f"  {'BUY + SELL (current)':<25} {len(all_trades):>6} ${total_pnl:>9.0f} "
          f"{daily_sharpe(all_daily):>8.2f}")
    print(f"  {'BUY only':<25} {len(buys):>6} ${buy_only_pnl:>9.0f} "
          f"{daily_sharpe(buy_daily):>8.2f}")
    print(f"  {'SELL only':<25} {len(sells):>6} ${sell_only_pnl:>9.0f} "
          f"{daily_sharpe(sell_daily):>8.2f}")

    if daily_sharpe(buy_daily) > daily_sharpe(all_daily):
        print(f"\n  NOTE: Disabling SELL would IMPROVE Sharpe!")
    else:
        print(f"\n  NOTE: Keeping SELL improves overall Sharpe (diversification benefit)")

    # Part 6: Consecutive same-direction patterns
    print("\n" + "=" * 90)
    print("  Part 6: Consecutive Direction Patterns")
    print("=" * 90)

    same_dir_streaks = []
    streak = 1
    for i in range(1, len(all_trades)):
        if all_trades[i].direction == all_trades[i-1].direction:
            streak += 1
        else:
            same_dir_streaks.append((all_trades[i-1].direction, streak))
            streak = 1
    same_dir_streaks.append((all_trades[-1].direction, streak))

    buy_streaks = [s for d, s in same_dir_streaks if d == 'BUY']
    sell_streaks = [s for d, s in same_dir_streaks if d == 'SELL']

    print(f"\n  BUY streaks:  max={max(buy_streaks)}, avg={np.mean(buy_streaks):.1f}, "
          f"count={len(buy_streaks)}")
    print(f"  SELL streaks: max={max(sell_streaks)}, avg={np.mean(sell_streaks):.1f}, "
          f"count={len(sell_streaks)}")

    # After long BUY streak, what happens?
    for threshold in [3, 5, 7]:
        after_buy = []
        after_sell = []
        streak = 1
        for i in range(1, len(all_trades)):
            if all_trades[i-1].direction == all_trades[i].direction:
                streak += 1
            else:
                streak = 1
            if streak >= threshold and i + 1 < len(all_trades):
                next_trade = all_trades[i + 1]
                if all_trades[i].direction == 'BUY':
                    after_buy.append(next_trade.pnl)
                else:
                    after_sell.append(next_trade.pnl)
        if after_buy:
            print(f"  After {threshold}+ BUY streak: next avg=${np.mean(after_buy):.2f} (N={len(after_buy)})")
        if after_sell:
            print(f"  After {threshold}+ SELL streak: next avg=${np.mean(after_sell):.2f} (N={len(after_sell)})")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
