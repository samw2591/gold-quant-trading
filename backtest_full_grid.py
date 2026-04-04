"""
Full Parameter Grid Search + Statistical Validation
=====================================================
81 combinations (3x3x3x3) with DSR and PBO validation.

Grid:
  trailing_activate_atr: [0.6, 0.8, 1.0]
  trailing_distance_atr: [0.20, 0.25, 0.30]
  keltner_adx_threshold: [16, 18, 20]
  choppy_threshold: [0.30, 0.35, 0.40]

All with: C12 SL/TP (3.5/5.0), spread=$0.50, adaptive, kc_only=0.60

Usage: python backtest_full_grid.py
"""
import time
import itertools

import numpy as np

from backtest import DataBundle, run_variant, aggregate_daily_pnl
from backtest.stats import deflated_sharpe, compute_pbo
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREAD = 0.50

TRAIL_ACT = [0.6, 0.8, 1.0]
TRAIL_DIST = [0.20, 0.25, 0.30]
ADX_THRESH = [16, 18, 20]
CHOPPY = [0.30, 0.35, 0.40]


def main():
    total_combos = len(TRAIL_ACT) * len(TRAIL_DIST) * len(ADX_THRESH) * len(CHOPPY)
    print("=" * 90)
    print(f"  FULL PARAMETER GRID SEARCH ({total_combos} combinations)")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    results = []
    daily_pnls = {}

    combos = list(itertools.product(TRAIL_ACT, TRAIL_DIST, ADX_THRESH, CHOPPY))

    for i, (ta, td, adx, ch) in enumerate(combos):
        label = f"T{ta}/D{td}/A{adx}/C{ch}"
        kw = {
            "trailing_activate_atr": ta,
            "trailing_distance_atr": td,
            "sl_atr_mult": C12_KWARGS["sl_atr_mult"],
            "tp_atr_mult": C12_KWARGS["tp_atr_mult"],
            "keltner_adx_threshold": adx,
            "spread_cost": SPREAD,
            "intraday_adaptive": True,
            "choppy_threshold": ch,
            "kc_only_threshold": 0.60,
        }

        if (i + 1) % 10 == 0 or i == 0:
            elapsed_so_far = time.time() - t0
            eta = elapsed_so_far / max(i, 1) * (total_combos - i)
            print(f"  [{i+1}/{total_combos}] ETA: {eta/60:.0f}min ...", flush=True)

        stats = run_variant(bundle, label, verbose=False, **kw)
        results.append(stats)

        trades = stats.get('_trades', [])
        daily = aggregate_daily_pnl(trades)
        daily_pnls[label] = daily

    # Sort by Sharpe
    results.sort(key=lambda s: s['sharpe'], reverse=True)

    print("\n" + "=" * 90)
    print("  TOP 20 by Sharpe")
    print("=" * 90)
    print(f"\n  {'Rank':>4} {'Label':<22} {'N':>6} {'Sharpe':>8} "
          f"{'PnL':>10} {'MaxDD':>10} {'WR%':>7}")
    print("  " + "-" * 72)

    for rank, s in enumerate(results[:20], 1):
        print(f"  {rank:>4} {s['label']:<22} {s['n']:>6} {s['sharpe']:>8.2f} "
              f"${s['total_pnl']:>9.0f} ${s['max_dd']:>9.0f} {s['win_rate']:>6.1f}%")

    # Statistical validation on best
    print("\n" + "=" * 90)
    print("  Statistical Validation")
    print("=" * 90)

    best = results[0]
    best_daily = daily_pnls.get(best['label'], [])
    all_sharpes = [s['sharpe'] for s in results]
    sharpe_var = float(np.var(all_sharpes))

    dsr = deflated_sharpe(best_daily, n_trials=total_combos, all_sharpes_var=sharpe_var)
    sr_star = f"{dsr['sr_star']:.2f}" if np.isfinite(dsr.get('sr_star', float('nan'))) else "N/A"
    print(f"\n  Best variant: {best['label']}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  DSR: {dsr['dsr']:.4f} (SR*={sr_star}, n_trials={total_combos})")
    print(f"  DSR passed (>0.95): {'YES' if dsr['passed'] else 'NO'}")

    # PBO
    print(f"\n  Computing PBO ({total_combos} variants, 8 partitions)...")
    pbo = compute_pbo(daily_pnls, n_partitions=8)
    print(f"  PBO = {pbo['pbo']:.2f} ({pbo['n_combinations']} combinations)")
    print(f"  Overfit Risk: {pbo['overfit_risk']}")

    if 'is_best_oos_ranks' in pbo and pbo['is_best_oos_ranks']:
        ranks = pbo['is_best_oos_ranks']
        print(f"  IS-best OOS rank: mean={np.mean(ranks):.1f}, "
              f"median={np.median(ranks):.1f}, worst={max(ranks)}")

    # Current C12 position in grid
    c12_label = f"T0.8/D0.25/A18/C0.35"
    c12_rank = next((i+1 for i, s in enumerate(results) if s['label'] == c12_label), None)
    if c12_rank:
        print(f"\n  Current C12 (T0.8/D0.25/A18/C0.35) rank: #{c12_rank}/{total_combos}")

    # Sharpe distribution
    print(f"\n  Sharpe distribution across {total_combos} combos:")
    print(f"    Mean: {np.mean(all_sharpes):.2f}")
    print(f"    Std:  {np.std(all_sharpes):.2f}")
    print(f"    Min:  {min(all_sharpes):.2f}")
    print(f"    Max:  {max(all_sharpes):.2f}")
    print(f"    >1.0: {sum(1 for s in all_sharpes if s > 1.0)}/{total_combos}")
    print(f"    >0.0: {sum(1 for s in all_sharpes if s > 0)}/{total_combos}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
