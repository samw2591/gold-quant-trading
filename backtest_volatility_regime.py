"""
Volatility Regime Breakdown
==============================
Splits the entire backtest period by ATR percentile into 4 regimes:
  - Very Low  (ATR pct < 25%)
  - Low       (25-50%)
  - High      (50-75%)
  - Very High (75-100%)

Runs C12+Adaptive on each regime window and full period, then
compares strategy-level performance across regimes.

Also tests: should we disable certain strategies in certain regimes?

Usage: python backtest_volatility_regime.py
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


def trade_stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "avg": 0, "sharpe": 0}
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    daily = defaultdict(float)
    for t in trades:
        daily[pd.Timestamp(t.exit_time).date()] += t.pnl
    daily_vals = list(daily.values())
    if len(daily_vals) > 1 and np.std(daily_vals) > 0:
        sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)
    else:
        sharpe = 0
    return {
        "n": len(trades),
        "pnl": sum(pnls),
        "wr": wins / len(trades) * 100,
        "avg": sum(pnls) / len(trades),
        "sharpe": sharpe,
    }


def main():
    print("=" * 90)
    print("  VOLATILITY REGIME BREAKDOWN")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    # Run full backtest to get all trades with timestamps
    print("\n  Running C12+Adaptive full backtest...", flush=True)
    engine = BacktestEngine(m15_df, h1_df, label="C12+Adaptive", **ADAPTIVE_KWARGS)
    engine.run()
    all_trades = engine.trades
    print(f"  {len(all_trades)} trades total")

    # Build daily ATR regime lookup from H1 data
    h1_daily_atr = h1_df[['ATR', 'atr_percentile']].copy()
    h1_daily_atr['date'] = h1_daily_atr.index.date
    daily_atr_pct = h1_daily_atr.groupby('date')['atr_percentile'].mean()

    regimes = [
        ("Very Low (ATR<25%)", 0.00, 0.25),
        ("Low (25-50%)", 0.25, 0.50),
        ("High (50-75%)", 0.50, 0.75),
        ("Very High (>75%)", 0.75, 1.01),
    ]

    # Part 1: Overall performance by regime
    print("\n" + "=" * 90)
    print("  Part 1: Performance by Volatility Regime")
    print("=" * 90)

    regime_trades = {}
    for label, lo, hi in regimes:
        regime_dates = set(daily_atr_pct[(daily_atr_pct >= lo) & (daily_atr_pct < hi)].index)
        rt = [t for t in all_trades
              if pd.Timestamp(t.entry_time).date() in regime_dates]
        regime_trades[label] = rt
        s = trade_stats(rt)
        n_days = len(regime_dates)
        print(f"\n  {label}:")
        print(f"    Days: {n_days}, Trades: {s['n']}, Trades/day: {s['n']/max(n_days,1):.1f}")
        print(f"    PnL: ${s['pnl']:.0f}, Sharpe: {s['sharpe']:.2f}")
        print(f"    WR: {s['wr']:.1f}%, avg: ${s['avg']:.2f}")

    # Part 2: Strategy x Regime
    print("\n" + "=" * 90)
    print("  Part 2: Strategy x Volatility Regime")
    print("=" * 90)

    strategies = sorted(set(t.strategy for t in all_trades))

    print(f"\n  {'Strategy':<20} {'Regime':<22} {'N':>5} {'PnL':>9} {'WR%':>6} {'$/trade':>9}")
    print("  " + "-" * 74)

    for strat in strategies:
        for label, _, _ in regimes:
            st = [t for t in regime_trades[label] if t.strategy == strat]
            if st:
                s = trade_stats(st)
                print(f"  {strat:<20} {label:<22} {s['n']:>5} ${s['pnl']:>8.0f} "
                      f"{s['wr']:>5.1f}% ${s['avg']:>8.2f}")
        print()

    # Part 3: Direction x Regime
    print("=" * 90)
    print("  Part 3: Direction (BUY/SELL) x Volatility Regime")
    print("=" * 90)

    for direction in ['BUY', 'SELL']:
        print(f"\n  {direction}:")
        for label, _, _ in regimes:
            dt = [t for t in regime_trades[label] if t.direction == direction]
            s = trade_stats(dt)
            if s['n'] > 0:
                print(f"    {label:<22} N={s['n']:>4}, PnL=${s['pnl']:>8.0f}, "
                      f"WR={s['wr']:.1f}%, avg=${s['avg']:.2f}")

    # Part 4: What if we disable strategies in bad regimes?
    print("\n" + "=" * 90)
    print("  Part 4: Hypothetical Regime Filters")
    print("=" * 90)

    scenarios = [
        ("Full (no filter)", lambda t, d: True),
        ("Skip very-low vol", lambda t, d: d.get(pd.Timestamp(t.entry_time).date(), 0.5) >= 0.25),
        ("Skip very-high vol", lambda t, d: d.get(pd.Timestamp(t.entry_time).date(), 0.5) < 0.75),
        ("Only mid-vol (25-75%)", lambda t, d: 0.25 <= d.get(pd.Timestamp(t.entry_time).date(), 0.5) < 0.75),
        ("Skip RSI in low vol", lambda t, d: not (t.strategy == 'm15_rsi' and d.get(pd.Timestamp(t.entry_time).date(), 0.5) < 0.25)),
        ("Skip RSI in high vol", lambda t, d: not (t.strategy == 'm15_rsi' and d.get(pd.Timestamp(t.entry_time).date(), 0.5) >= 0.75)),
        ("Skip ORB in low vol", lambda t, d: not (t.strategy == 'orb' and d.get(pd.Timestamp(t.entry_time).date(), 0.5) < 0.25)),
    ]

    atr_dict = daily_atr_pct.to_dict()

    print(f"\n  {'Scenario':<30} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9} {'Sharpe':>8}")
    print("  " + "-" * 74)

    for label, filter_fn in scenarios:
        filtered = [t for t in all_trades if filter_fn(t, atr_dict)]
        s = trade_stats(filtered)
        print(f"  {label:<30} {s['n']:>6} ${s['pnl']:>9.0f} {s['wr']:>6.1f}% "
              f"${s['avg']:>8.2f} {s['sharpe']:>8.2f}")

    # Part 5: ATR level at entry vs trade outcome
    print("\n" + "=" * 90)
    print("  Part 5: ATR at Entry vs Trade Outcome")
    print("=" * 90)

    trade_atrs = []
    trade_pnls = []
    for t in all_trades:
        d = pd.Timestamp(t.entry_time).date()
        atr_pct = atr_dict.get(d, None)
        if atr_pct is not None:
            trade_atrs.append(atr_pct)
            trade_pnls.append(t.pnl)

    if trade_atrs:
        corr = np.corrcoef(trade_atrs, trade_pnls)[0, 1]
        print(f"\n  Correlation(ATR_pct, PnL): {corr:.3f}")

        for q in [0.25, 0.50, 0.75]:
            low = [p for a, p in zip(trade_atrs, trade_pnls) if a < q]
            high = [p for a, p in zip(trade_atrs, trade_pnls) if a >= q]
            print(f"  ATR < {q:.0%}: avg PnL=${np.mean(low):.2f} (N={len(low)})")
            print(f"  ATR >= {q:.0%}: avg PnL=${np.mean(high):.2f} (N={len(high)})")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
