"""
Monte Carlo Confidence Intervals
==================================
Shuffles trade order 2000 times to build PnL / MaxDD / Sharpe distributions.
Answers: "How much of our result is due to trade ordering luck?"

Usage: python backtest_monte_carlo.py
"""
import time
from datetime import datetime

import numpy as np

import config
from backtest import DataBundle, run_variant
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREAD = 0.50
N_SIMULATIONS = 2000

ADAPTIVE_KWARGS = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
}


def main():
    print("=" * 90)
    print(f"  MONTE CARLO CONFIDENCE INTERVALS ({N_SIMULATIONS} simulations)")
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
    trade_pnls = np.array([t.pnl for t in trades])
    print(f"  {len(trades)} trades, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}")

    print(f"\n  Running {N_SIMULATIONS} Monte Carlo simulations...", flush=True)
    t1 = time.time()

    mc_sharpes = []
    mc_max_dds = []
    mc_final_pnls = []
    mc_max_equity = []
    mc_calmar = []

    for i in range(N_SIMULATIONS):
        shuffled = np.random.permutation(trade_pnls)
        equity = np.concatenate([[config.CAPITAL], config.CAPITAL + np.cumsum(shuffled)])
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = abs(dd.min())

        final_pnl = equity[-1] - config.CAPITAL

        chunks = [shuffled[j:j+6] for j in range(0, len(shuffled), 6)]
        daily_pnl = np.array([chunk.sum() for chunk in chunks])
        if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
            sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
        else:
            sharpe = 0

        mc_sharpes.append(sharpe)
        mc_max_dds.append(max_dd)
        mc_final_pnls.append(final_pnl)
        mc_max_equity.append(equity.max() - config.CAPITAL)
        mc_calmar.append(final_pnl / max_dd if max_dd > 0 else 0)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{N_SIMULATIONS}...", flush=True)

    elapsed_mc = time.time() - t1
    print(f"  Done in {elapsed_mc:.1f}s")

    mc_sharpes = np.array(mc_sharpes)
    mc_max_dds = np.array(mc_max_dds)
    mc_final_pnls = np.array(mc_final_pnls)
    mc_max_equity = np.array(mc_max_equity)
    mc_calmar = np.array(mc_calmar)

    # Results
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    print("\n" + "=" * 90)
    print("  Percentile Distributions")
    print("=" * 90)

    header = f"  {'Metric':<20}"
    for p in pcts:
        header += f" {'P'+str(p):>8}"
    header += f" {'Mean':>10}"
    print(header)
    print("  " + "-" * (20 + 9 * len(pcts) + 10))

    for name, arr, fmt in [
        ("PnL ($)", mc_final_pnls, ".0f"),
        ("MaxDD ($)", mc_max_dds, ".0f"),
        ("Sharpe", mc_sharpes, ".2f"),
        ("Max Equity ($)", mc_max_equity, ".0f"),
        ("Calmar", mc_calmar, ".2f"),
    ]:
        row = f"  {name:<20}"
        for p in pcts:
            val = np.percentile(arr, p)
            row += f" {val:>8{fmt}}"
        row += f" {np.mean(arr):>10{fmt}}"
        print(row)

    print(f"\n  Original (actual order):")
    print(f"    PnL:    ${stats['total_pnl']:.0f}")
    print(f"    Sharpe: {stats['sharpe']:.2f}")
    print(f"    MaxDD:  ${stats['max_dd']:.0f}")

    orig_pnl_pct = np.mean(mc_final_pnls <= stats['total_pnl']) * 100
    orig_sharpe_pct = np.mean(mc_sharpes <= stats['sharpe']) * 100
    orig_dd_pct = np.mean(mc_max_dds >= stats['max_dd']) * 100

    print(f"\n  Original vs Monte Carlo:")
    print(f"    PnL percentile:    {orig_pnl_pct:.1f}% (higher = PnL is robust)")
    print(f"    Sharpe percentile: {orig_sharpe_pct:.1f}% (higher = Sharpe is robust)")
    print(f"    MaxDD percentile:  {orig_dd_pct:.1f}% (higher = DD is worse than avg)")

    # Risk metrics
    print("\n" + "=" * 90)
    print("  Risk Metrics")
    print("=" * 90)

    prob_positive = np.mean(mc_final_pnls > 0) * 100
    prob_dd_500 = np.mean(mc_max_dds < 500) * 100
    prob_dd_1000 = np.mean(mc_max_dds < 1000) * 100
    prob_dd_1500 = np.mean(mc_max_dds < 1500) * 100
    prob_sharpe_pos = np.mean(mc_sharpes > 0) * 100
    prob_sharpe_1 = np.mean(mc_sharpes > 1.0) * 100

    print(f"\n  P(PnL > 0):        {prob_positive:.1f}%")
    print(f"  P(Sharpe > 0):     {prob_sharpe_pos:.1f}%")
    print(f"  P(Sharpe > 1.0):   {prob_sharpe_1:.1f}%")
    print(f"  P(MaxDD < $500):   {prob_dd_500:.1f}%")
    print(f"  P(MaxDD < $1000):  {prob_dd_1000:.1f}%")
    print(f"  P(MaxDD < $1500):  {prob_dd_1500:.1f}%")
    print(f"  Worst case PnL:    ${np.min(mc_final_pnls):.0f}")
    print(f"  Worst case MaxDD:  ${np.max(mc_max_dds):.0f}")
    print(f"  95% VaR (DD):      ${np.percentile(mc_max_dds, 95):.0f}")
    print(f"  99% VaR (DD):      ${np.percentile(mc_max_dds, 99):.0f}")

    # Per-strategy contribution
    print("\n" + "=" * 90)
    print("  Strategy Contribution Analysis")
    print("=" * 90)

    strategies = set(t.strategy for t in trades)
    for strat in sorted(strategies):
        strat_pnls = [t.pnl for t in trades if t.strategy == strat]
        n = len(strat_pnls)
        total = sum(strat_pnls)
        wins = sum(1 for p in strat_pnls if p > 0)
        avg = total / n if n > 0 else 0
        print(f"  {strat:<20}: {n:>5} trades, PnL=${total:>9.0f}, "
              f"WR={wins/n*100:.1f}%, avg=${avg:.2f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
