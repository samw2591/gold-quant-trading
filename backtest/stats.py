"""
Backtest Statistics & Reporting
================================
Unified statistics calculation and report formatting.
Replaces: backtest.print_report, backtest_m15.calc_stats/print_comparison
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from backtest.engine import TradeRecord


def aggregate_daily_pnl(trades: List[TradeRecord]) -> List[float]:
    """Aggregate PnL by exit date."""
    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl
    return list(daily.values())


def calc_stats(trades: List[TradeRecord], equity_curve: List[float]) -> Dict:
    """Calculate comprehensive backtest statistics.

    Returns a dict with keys: n, total_pnl, sharpe, win_rate, max_dd, max_dd_pct,
    rr, avg_win, avg_loss, plus per-strategy and per-year breakdowns.
    """
    if not trades:
        return {
            'n': 0, 'total_pnl': 0, 'sharpe': 0, 'win_rate': 0,
            'max_dd': 0, 'max_dd_pct': 0, 'rr': 0,
            'avg_win': 0, 'avg_loss': 0, 'year_pnl': {},
            'rsi_n': 0, 'rsi_pnl': 0, 'rsi_wr': 0,
            'rsi_buy_n': 0, 'rsi_buy_pnl': 0,
            'rsi_sell_n': 0, 'rsi_sell_pnl': 0,
            'keltner_n': 0, 'keltner_pnl': 0, 'keltner_wr': 0,
            'orb_n': 0, 'orb_pnl': 0, 'orb_wr': 0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    daily_pnl = aggregate_daily_pnl(trades)
    sharpe = 0.0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)

    eq = np.array(equity_curve) if equity_curve else np.array([config.CAPITAL])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    # Per-strategy breakdown
    def _strat_stats(strat_name):
        st = [t for t in trades if t.strategy == strat_name]
        n = len(st)
        pnl = sum(t.pnl for t in st)
        wr = len([t for t in st if t.pnl > 0]) / n * 100 if n else 0
        return n, pnl, wr

    rsi_n, rsi_pnl, rsi_wr = _strat_stats('m15_rsi')
    keltner_n, keltner_pnl, keltner_wr = _strat_stats('keltner')
    orb_n, orb_pnl, orb_wr = _strat_stats('orb')

    rsi_buy = [t for t in trades if t.strategy == 'm15_rsi' and t.direction == 'BUY']
    rsi_sell = [t for t in trades if t.strategy == 'm15_rsi' and t.direction == 'SELL']

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] = year_pnl.get(y, 0) + t.pnl

    return {
        'n': len(pnls), 'total_pnl': total_pnl, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'sharpe': sharpe, 'max_dd': max_dd, 'max_dd_pct': max_dd_pct,
        'rsi_n': rsi_n, 'rsi_pnl': rsi_pnl, 'rsi_wr': rsi_wr,
        'rsi_buy_n': len(rsi_buy), 'rsi_buy_pnl': sum(t.pnl for t in rsi_buy),
        'rsi_sell_n': len(rsi_sell), 'rsi_sell_pnl': sum(t.pnl for t in rsi_sell),
        'keltner_n': keltner_n, 'keltner_pnl': keltner_pnl, 'keltner_wr': keltner_wr,
        'orb_n': orb_n, 'orb_pnl': orb_pnl, 'orb_wr': orb_wr,
        'year_pnl': year_pnl,
    }


# ═══════════════════════════════════════════════════════════════
# Report printing
# ═══════════════════════════════════════════════════════════════

def print_comparison(variants: List[Dict], title: str = "Backtest Comparison"):
    """Print a ranked comparison table of multiple variants."""
    print("\n")
    print("=" * 120)
    print(f"  {title}")
    print("=" * 120)

    header = (f"  {'Variant':<40} {'Trades':>6} {'Sharpe':>8} {'PnL':>10} "
              f"{'MaxDD':>10} {'DD%':>6} {'WinR%':>7} {'RR':>6}")
    print(header)
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    baseline = variants[0] if variants else {}
    base_sharpe = baseline.get('sharpe', 0)

    for v in variants:
        if v.get('n', 0) == 0:
            print(f"  {v.get('label', '?'):<40}   (no trades)")
            continue
        sharpe_diff = v['sharpe'] - base_sharpe
        marker = f"({sharpe_diff:+.2f})" if v != baseline else ""
        print(f"  {v.get('label', '?'):<40} {v['n']:>6} {v['sharpe']:>8.2f} ${v['total_pnl']:>9.2f} "
              f"${v['max_dd']:>9.2f} {v['max_dd_pct']:>5.1f}% {v['win_rate']:>6.1f}% "
              f"{v['rr']:>5.2f} {marker}")

    # Strategy breakdown
    print(f"\n  --- Strategy Breakdown ---")
    print(f"  {'Variant':<40} {'K_N':>5} {'K_PnL':>10} {'ORB_N':>5} {'ORB_PnL':>10} "
          f"{'RSI_N':>5} {'RSI_PnL':>10} {'RSI_WR':>6} {'Filt':>5}")
    print(f"  {'-'*40} {'-'*5} {'-'*10} {'-'*5} {'-'*10} "
          f"{'-'*5} {'-'*10} {'-'*6} {'-'*5}")
    for v in variants:
        if v.get('n', 0) == 0:
            continue
        print(f"  {v.get('label', '?'):<40} {v.get('keltner_n', 0):>5} ${v.get('keltner_pnl', 0):>9.2f} "
              f"{v.get('orb_n', 0):>5} ${v.get('orb_pnl', 0):>9.2f} "
              f"{v.get('rsi_n', 0):>5} ${v.get('rsi_pnl', 0):>9.2f} "
              f"{v.get('rsi_wr', 0):>5.1f}% {v.get('rsi_filtered', 0):>5}")

    # Year-by-year
    all_years = set()
    for v in variants:
        all_years.update(v.get('year_pnl', {}).keys())
    years = sorted(all_years)

    if years:
        print(f"\n  --- Year-by-Year PnL ($) ---")
        print(f"  {'Year':<6}", end='')
        for v in variants:
            print(f"  {v.get('label', '?')[:18]:>20}", end='')
        print()
        for y in years:
            print(f"  {y:<6}", end='')
            for v in variants:
                pnl = v.get('year_pnl', {}).get(y, 0)
                print(f"  ${pnl:>19.2f}", end='')
            print()

    print("\n" + "=" * 120)


def print_ranked(results: List[Dict], title: str = "Ranked Results"):
    """Print results sorted by Sharpe, with delta from baseline."""
    results_sorted = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)
    baseline_list = [r for r in results if 'Baseline' in r.get('label', '') or r == results[0]]
    base_sharpe = baseline_list[0].get('sharpe', 0) if baseline_list else 0

    print(f"\n  {'='*130}")
    print(f"  {title}")
    print(f"  {'='*130}")

    print(f"\n  {'Rank':<5} {'Variant':<35} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} "
          f"{'MaxDD':>10} {'DD%':>6} {'WR%':>6} {'RR':>5}")
    print(f"  {'-'*5} {'-'*35} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*5}")

    for rank, v in enumerate(results_sorted, 1):
        if v.get('n', 0) == 0:
            print(f"  {rank:<5} {v.get('label', '?'):<35}   (no trades)")
            continue
        ds = v['sharpe'] - base_sharpe
        print(f"  {rank:<5} {v.get('label', '?'):<35} {v['n']:>6} {v['sharpe']:>8.2f} {ds:>+5.2f} "
              f"${v['total_pnl']:>9.0f} ${v['max_dd']:>9.0f} {v['max_dd_pct']:>5.1f}% "
              f"{v['win_rate']:>5.1f}% {v['rr']:>4.2f}")

    print(f"\n  {'='*130}")
