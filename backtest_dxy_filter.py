"""
DXY Direction Filter Analysis
===============================
Post-hoc analysis: does DXY direction predict C12+Adaptive trade profitability?

Steps:
  1. Run C12+Adaptive once to get all trades
  2. Match each trade to DXY status (bullish/bearish vs SMA20)
  3. Split trades by DXY alignment and compute stats

Usage: python backtest_dxy_filter.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import DataBundle, run_variant, aggregate_daily_pnl
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from backtest.engine import BacktestEngine
from strategies.signals import prepare_indicators

SPREAD = 0.50
MACRO_CSV = Path("data/macro_history.csv")


def trade_stats(trades):
    if not trades:
        return {"n": 0, "pnl": 0, "wr": 0, "avg": 0}
    pnls = [t.pnl for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n": len(trades),
        "pnl": sum(pnls),
        "wr": wins / len(trades) * 100,
        "avg": sum(pnls) / len(trades),
    }


def main():
    print("=" * 90)
    print("  DXY DIRECTION FILTER ANALYSIS")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)

    macro_df = pd.read_csv(str(MACRO_CSV), index_col='date', parse_dates=True)
    macro_df = macro_df.ffill()
    print(f"  Macro: {len(macro_df)} days")

    print("\n  Running C12+Adaptive backtest...")
    kw = {
        **C12_KWARGS, "spread_cost": SPREAD,
        "intraday_adaptive": True, "choppy_threshold": 0.35, "kc_only_threshold": 0.60,
    }
    engine = BacktestEngine(m15_df, h1_df, label="C12+Adaptive", **kw)
    engine.run()
    all_trades = engine.trades
    print(f"  Total trades: {len(all_trades)}")

    dxy_bull = []
    dxy_bear = []
    dxy_unknown = []
    buy_on_bear = []
    sell_on_bull = []
    aligned = []
    misaligned = []

    for t in all_trades:
        td = pd.Timestamp(t.entry_time).normalize()
        if td.tz is not None:
            td = td.tz_localize(None)

        if td not in macro_df.index:
            dxy_unknown.append(t)
            continue

        row = macro_df.loc[td]
        dxy_val = row.get('dxy', None)
        dxy_sma = row.get('dxy_sma20', None)

        if pd.isna(dxy_val) or pd.isna(dxy_sma):
            dxy_unknown.append(t)
            continue

        is_dxy_bull = dxy_val > dxy_sma

        if is_dxy_bull:
            dxy_bull.append(t)
            if t.direction == 'SELL':
                sell_on_bull.append(t)
                aligned.append(t)
            else:
                misaligned.append(t)
        else:
            dxy_bear.append(t)
            if t.direction == 'BUY':
                buy_on_bear.append(t)
                aligned.append(t)
            else:
                misaligned.append(t)

    groups = [
        ("All trades", all_trades),
        ("DXY bullish days (gold bearish)", dxy_bull),
        ("DXY bearish days (gold bullish)", dxy_bear),
        ("BUY on DXY-bearish (aligned)", buy_on_bear),
        ("SELL on DXY-bullish (aligned)", sell_on_bull),
        ("All DXY-aligned trades", aligned),
        ("All DXY-misaligned trades", misaligned),
        ("No DXY data", dxy_unknown),
    ]

    print("\n" + "=" * 90)
    print("  DXY Filter Results")
    print("=" * 90)
    print(f"\n  {'Group':<38} {'N':>6} {'PnL':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 74)

    for label, trades in groups:
        s = trade_stats(trades)
        print(f"  {label:<38} {s['n']:>6} ${s['pnl']:>9.0f} {s['wr']:>6.1f}% ${s['avg']:>8.2f}")

    print("\n" + "=" * 90)
    print("  By Strategy x DXY")
    print("=" * 90)

    for strat in ['keltner_breakout', 'orb_breakout', 'm15_rsi']:
        strat_trades = [t for t in all_trades if t.strategy == strat]
        if not strat_trades:
            continue
        strat_aligned = [t for t in aligned if t.strategy == strat]
        strat_mis = [t for t in misaligned if t.strategy == strat]
        s_all = trade_stats(strat_trades)
        s_aln = trade_stats(strat_aligned)
        s_mis = trade_stats(strat_mis)
        print(f"\n  {strat}:")
        print(f"    All:       N={s_all['n']:>5}, PnL=${s_all['pnl']:>8.0f}, WR={s_all['wr']:.1f}%, avg=${s_all['avg']:.2f}")
        print(f"    Aligned:   N={s_aln['n']:>5}, PnL=${s_aln['pnl']:>8.0f}, WR={s_aln['wr']:.1f}%, avg=${s_aln['avg']:.2f}")
        print(f"    Misalign:  N={s_mis['n']:>5}, PnL=${s_mis['pnl']:>8.0f}, WR={s_mis['wr']:.1f}%, avg=${s_mis['avg']:.2f}")

    a_s = trade_stats(aligned)
    m_s = trade_stats(misaligned)
    verdict = "YES" if a_s['avg'] > m_s['avg'] and a_s['pnl'] > m_s['pnl'] else "NO"
    print(f"\n  VERDICT: DXY alignment improves trades? {verdict}")
    print(f"    Aligned avg: ${a_s['avg']:.2f}/trade ({a_s['n']} trades)")
    print(f"    Misaligned avg: ${m_s['avg']:.2f}/trade ({m_s['n']} trades)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
