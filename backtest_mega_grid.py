"""
Mega Grid Search — 5D Parameter Space + Statistical Validation
================================================================
1440 combinations across 5 dimensions, with multiprocessing.

Grid (6×5×4×4×3 = 1440):
  trailing_activate_atr: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  trailing_distance_atr: [0.15, 0.20, 0.25, 0.30, 0.35]
  keltner_adx_threshold: [15, 17, 19, 21]
  sl_atr_mult:           [2.5, 3.0, 3.5, 4.0]
  choppy_threshold:      [0.30, 0.35, 0.40]

Fixed: TP=5.0, spread=$0.50, adaptive, kc_only=0.60

After grid:
  - DSR validation on best variant
  - PBO (8 partitions) on top-50 variants
  - Stability heatmaps (printed as tables)
  - Current C12 ranking

Server usage:
  Uses multiprocessing.Pool with N_WORKERS (default 8).
  Each worker loads data independently to avoid pickle issues.

Estimated time: ~7 hours on 8 cores (single core ~50h).

Usage: python backtest_mega_grid.py [--workers N]
"""
import sys
import time
import json
import itertools
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Grid definition
# ═══════════════════════════════════════════════════════════════

TRAIL_ACT  = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TRAIL_DIST = [0.15, 0.20, 0.25, 0.30, 0.35]
ADX_THRESH = [15, 17, 19, 21]
SL_MULT    = [2.5, 3.0, 3.5, 4.0]
CHOPPY     = [0.30, 0.35, 0.40]

TP_MULT    = 5.0
SPREAD     = 0.50
KC_ONLY    = 0.60

N_WORKERS  = 8

ALL_COMBOS = list(itertools.product(TRAIL_ACT, TRAIL_DIST, ADX_THRESH, SL_MULT, CHOPPY))
TOTAL      = len(ALL_COMBOS)


def combo_label(ta, td, adx, sl, ch):
    return f"T{ta}/D{td}/A{adx}/SL{sl}/C{ch}"


# ═══════════════════════════════════════════════════════════════
# Worker: each process loads data independently
# ═══════════════════════════════════════════════════════════════

_worker_bundle = None


def _worker_init():
    """Each worker loads and prepares data once."""
    global _worker_bundle
    import warnings
    warnings.filterwarnings("ignore")

    from backtest.runner import load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH, DataBundle
    from strategies.signals import prepare_indicators
    import pandas as pd

    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    _worker_bundle = DataBundle(m15_df, h1_df)


def _run_single(combo: Tuple) -> Dict:
    """Run one backtest combo, return lightweight result dict."""
    global _worker_bundle
    ta, td, adx, sl, ch = combo
    label = combo_label(ta, td, adx, sl, ch)

    from backtest.runner import run_variant
    from backtest.stats import aggregate_daily_pnl

    kw = {
        "trailing_activate_atr": ta,
        "trailing_distance_atr": td,
        "keltner_adx_threshold": adx,
        "sl_atr_mult": sl,
        "tp_atr_mult": TP_MULT,
        "spread_cost": SPREAD,
        "intraday_adaptive": True,
        "choppy_threshold": ch,
        "kc_only_threshold": KC_ONLY,
    }

    stats = run_variant(_worker_bundle, label, verbose=False, **kw)
    trades = stats.get('_trades', [])
    daily = aggregate_daily_pnl(trades)

    return {
        "label": label,
        "combo": combo,
        "n": int(stats["n"]),
        "sharpe": float(stats["sharpe"]),
        "total_pnl": float(stats["total_pnl"]),
        "max_dd": float(stats["max_dd"]),
        "max_dd_pct": float(stats["max_dd_pct"]),
        "win_rate": float(stats["win_rate"]),
        "rr": float(stats["rr"]),
        "keltner_n": int(stats.get("keltner_n", 0)),
        "keltner_pnl": float(stats.get("keltner_pnl", 0)),
        "orb_n": int(stats.get("orb_n", 0)),
        "orb_pnl": float(stats.get("orb_pnl", 0)),
        "rsi_n": int(stats.get("rsi_n", 0)),
        "rsi_pnl": float(stats.get("rsi_pnl", 0)),
        "daily_pnl": daily,
    }


# ═══════════════════════════════════════════════════════════════
# Stability analysis helpers
# ═══════════════════════════════════════════════════════════════

def _analyze_dimension(results: List[Dict], dim_idx: int, dim_name: str, dim_values: list):
    """Analyze Sharpe sensitivity along one dimension, averaging over others."""
    print(f"\n  --- {dim_name} Sensitivity ---")
    print(f"  {'Value':>8} {'AvgSharpe':>10} {'StdSharpe':>10} {'AvgPnL':>10} {'Count':>6}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    for val in dim_values:
        subset = [r for r in results if r['combo'][dim_idx] == val]
        if not subset:
            continue
        sharpes = [r['sharpe'] for r in subset]
        pnls = [r['total_pnl'] for r in subset]
        print(f"  {val:>8} {np.mean(sharpes):>10.3f} {np.std(sharpes):>10.3f} "
              f"${np.mean(pnls):>9.0f} {len(subset):>6}")

    all_sharpes_by_val = {}
    for val in dim_values:
        subset = [r['sharpe'] for r in results if r['combo'][dim_idx] == val]
        if subset:
            all_sharpes_by_val[val] = np.mean(subset)

    if all_sharpes_by_val:
        best_val = max(all_sharpes_by_val, key=all_sharpes_by_val.get)
        worst_val = min(all_sharpes_by_val, key=all_sharpes_by_val.get)
        rng = all_sharpes_by_val[best_val] - all_sharpes_by_val[worst_val]
        label = "CLIFF" if rng > 0.5 else "moderate" if rng > 0.2 else "stable"
        print(f"  Range: {rng:.3f} ({label}), Best={best_val}, Worst={worst_val}")


def _print_2d_heatmap(results: List[Dict], dim1_idx, dim1_name, dim1_vals,
                       dim2_idx, dim2_name, dim2_vals):
    """Print 2D Sharpe heatmap (averaged over other dims)."""
    print(f"\n  --- 2D Heatmap: {dim1_name} x {dim2_name} (avg Sharpe) ---")
    header = f"  {dim1_name:>10}"
    for v2 in dim2_vals:
        header += f" {v2:>8}"
    print(header)
    print(f"  {'-'*10}" + f" {'-'*8}" * len(dim2_vals))

    for v1 in dim1_vals:
        row = f"  {v1:>10}"
        for v2 in dim2_vals:
            subset = [r['sharpe'] for r in results
                      if r['combo'][dim1_idx] == v1 and r['combo'][dim2_idx] == v2]
            if subset:
                row += f" {np.mean(subset):>8.3f}"
            else:
                row += f" {'N/A':>8}"
        print(row)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    workers = N_WORKERS
    if len(sys.argv) > 2 and sys.argv[1] == '--workers':
        workers = int(sys.argv[2])

    print("=" * 100)
    print(f"  MEGA GRID SEARCH — {TOTAL} combinations, {workers} workers")
    print(f"  Grid: TrailAct({len(TRAIL_ACT)}) x TrailDist({len(TRAIL_DIST)}) "
          f"x ADX({len(ADX_THRESH)}) x SL({len(SL_MULT)}) x Choppy({len(CHOPPY)})")
    print(f"  Fixed: TP={TP_MULT}, Spread=${SPREAD}, Adaptive, KC_only={KC_ONLY}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    t0 = time.time()

    # Phase 1: Parallel grid search
    print(f"\n  Phase 1: Running {TOTAL} backtests across {workers} processes...")
    print(f"  Estimated time: ~{TOTAL * 2.5 / workers / 60:.0f} hours")
    print(f"  Each '.' = {max(TOTAL // 50, 1)} combos completed\n  ", end="", flush=True)

    results = []
    dot_interval = max(TOTAL // 50, 1)

    with mp.Pool(processes=workers, initializer=_worker_init) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_single, ALL_COMBOS)):
            results.append(result)
            if (i + 1) % dot_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (TOTAL - i - 1) / rate
                pct = (i + 1) / TOTAL * 100
                print(".", end="", flush=True)
                if (i + 1) % (dot_interval * 10) == 0:
                    print(f" [{pct:.0f}% ETA:{eta/60:.0f}min]", end="", flush=True)

    elapsed_grid = time.time() - t0
    print(f"\n\n  Grid complete: {elapsed_grid/3600:.1f}h ({elapsed_grid:.0f}s)")

    # Sort by Sharpe
    results.sort(key=lambda r: r['sharpe'], reverse=True)

    # ════════════════════════════════════════════════════════════
    # Phase 2: Results analysis
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  Phase 2: Results Analysis")
    print("=" * 100)

    # Top 30
    print(f"\n  TOP 30 / {TOTAL} by Sharpe:")
    print(f"  {'Rank':>4} {'Label':<28} {'N':>6} {'Sharpe':>8} "
          f"{'PnL':>10} {'MaxDD':>10} {'WR%':>7} {'RR':>5}")
    print(f"  {'-'*4} {'-'*28} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*5}")

    for rank, r in enumerate(results[:30], 1):
        print(f"  {rank:>4} {r['label']:<28} {r['n']:>6} {r['sharpe']:>8.2f} "
              f"${r['total_pnl']:>9.0f} ${r['max_dd']:>9.0f} {r['win_rate']:>6.1f}% "
              f"{r['rr']:>4.2f}")

    # Bottom 10
    print(f"\n  BOTTOM 10:")
    for rank, r in enumerate(results[-10:], TOTAL - 9):
        print(f"  {rank:>4} {r['label']:<28} {r['n']:>6} {r['sharpe']:>8.2f} "
              f"${r['total_pnl']:>9.0f}")

    # Current C12 position
    c12_label = combo_label(0.8, 0.25, 18, 3.5, 0.35)
    c12_rank = next((i+1 for i, r in enumerate(results) if r['label'] == c12_label), None)
    c12_result = next((r for r in results if r['label'] == c12_label), None)
    if c12_rank:
        print(f"\n  Current C12 ({c12_label}):")
        print(f"    Rank: #{c12_rank}/{TOTAL} (top {c12_rank/TOTAL*100:.1f}%)")
        print(f"    Sharpe: {c12_result['sharpe']:.2f}, PnL: ${c12_result['total_pnl']:.0f}")

    # Sharpe distribution
    all_sharpes = [r['sharpe'] for r in results]
    print(f"\n  Sharpe distribution ({TOTAL} combos):")
    print(f"    Mean:  {np.mean(all_sharpes):.3f}")
    print(f"    Std:   {np.std(all_sharpes):.3f}")
    print(f"    Min:   {min(all_sharpes):.3f}")
    print(f"    P25:   {np.percentile(all_sharpes, 25):.3f}")
    print(f"    P50:   {np.percentile(all_sharpes, 50):.3f}")
    print(f"    P75:   {np.percentile(all_sharpes, 75):.3f}")
    print(f"    Max:   {max(all_sharpes):.3f}")
    print(f"    >1.0:  {sum(1 for s in all_sharpes if s > 1.0)}/{TOTAL}")
    print(f"    >0.5:  {sum(1 for s in all_sharpes if s > 0.5)}/{TOTAL}")
    print(f"    >0.0:  {sum(1 for s in all_sharpes if s > 0)}/{TOTAL}")

    # ════════════════════════════════════════════════════════════
    # Phase 3: Dimensional sensitivity
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  Phase 3: Parameter Sensitivity")
    print("=" * 100)

    dims = [
        (0, "TrailAct", TRAIL_ACT),
        (1, "TrailDist", TRAIL_DIST),
        (2, "ADX", ADX_THRESH),
        (3, "SL", SL_MULT),
        (4, "Choppy", CHOPPY),
    ]

    for idx, name, vals in dims:
        _analyze_dimension(results, idx, name, vals)

    # Key 2D heatmaps
    _print_2d_heatmap(results, 0, "TrailAct", TRAIL_ACT, 1, "TrailDist", TRAIL_DIST)
    _print_2d_heatmap(results, 2, "ADX", ADX_THRESH, 3, "SL", SL_MULT)
    _print_2d_heatmap(results, 0, "TrailAct", TRAIL_ACT, 3, "SL", SL_MULT)

    # ════════════════════════════════════════════════════════════
    # Phase 4: Statistical validation
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  Phase 4: Statistical Validation")
    print("=" * 100)

    from backtest.stats import deflated_sharpe, compute_pbo

    best = results[0]
    best_daily = best.get('daily_pnl', [])
    sharpe_var = float(np.var(all_sharpes))

    dsr = deflated_sharpe(best_daily, n_trials=TOTAL, all_sharpes_var=sharpe_var)
    sr_star = f"{dsr['sr_star']:.2f}" if np.isfinite(dsr.get('sr_star', float('nan'))) else "N/A"
    print(f"\n  Best variant: {best['label']}")
    print(f"  Sharpe: {best['sharpe']:.2f}, PnL: ${best['total_pnl']:.0f}")
    print(f"  DSR: {dsr['dsr']:.4f} (SR*={sr_star}, n_trials={TOTAL})")
    print(f"  DSR passed (>0.95): {'YES' if dsr['passed'] else 'NO'}")

    # PBO on top-50
    top_n_pbo = min(50, TOTAL)
    print(f"\n  Computing PBO on top-{top_n_pbo} variants (8 partitions)...")
    top_daily = {r['label']: r['daily_pnl'] for r in results[:top_n_pbo]}
    pbo = compute_pbo(top_daily, n_partitions=8)
    print(f"  PBO = {pbo['pbo']:.2f} ({pbo['n_combinations']} combinations)")
    print(f"  Overfit Risk: {pbo['overfit_risk']}")

    if pbo.get('is_best_oos_ranks'):
        ranks = pbo['is_best_oos_ranks']
        print(f"  IS-best OOS rank: mean={np.mean(ranks):.1f}, "
              f"median={np.median(ranks):.1f}, worst={max(ranks)}")

    # PBO on full grid (sample 200)
    if TOTAL > 50:
        print(f"\n  Computing PBO on full {TOTAL} variants...")
        full_daily = {r['label']: r['daily_pnl'] for r in results}
        pbo_full = compute_pbo(full_daily, n_partitions=8)
        print(f"  PBO (full) = {pbo_full['pbo']:.2f} ({pbo_full['n_combinations']} combinations)")
        print(f"  Overfit Risk (full): {pbo_full['overfit_risk']}")

    # ════════════════════════════════════════════════════════════
    # Phase 5: Robust parameter region
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  Phase 5: Robust Parameter Regions")
    print("=" * 100)

    sharpe_threshold = np.percentile(all_sharpes, 75)
    robust = [r for r in results if r['sharpe'] >= sharpe_threshold]
    print(f"\n  Robust region: Sharpe >= {sharpe_threshold:.2f} ({len(robust)} combos, top 25%)")

    for idx, name, vals in dims:
        counts = {}
        for val in vals:
            c = sum(1 for r in robust if r['combo'][idx] == val)
            total_v = sum(1 for r in results if r['combo'][idx] == val)
            counts[val] = (c, total_v)
        print(f"\n  {name}:")
        for val in vals:
            c, t = counts[val]
            pct = c / t * 100 if t > 0 else 0
            bar = "#" * int(pct / 5)
            print(f"    {val:>6}: {c:>4}/{t:<4} ({pct:>5.1f}%) {bar}")

    # Find the most common values in robust region
    print(f"\n  Most common parameter values in robust region:")
    for idx, name, vals in dims:
        from collections import Counter
        val_counts = Counter(r['combo'][idx] for r in robust)
        best_val = val_counts.most_common(1)[0]
        print(f"    {name}: {best_val[0]} ({best_val[1]}/{len(robust)} = "
              f"{best_val[1]/len(robust)*100:.0f}%)")

    # ════════════════════════════════════════════════════════════
    # Phase 6: Save results
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  Phase 6: Saving Results")
    print("=" * 100)

    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'daily_pnl'}
        sr['combo'] = list(sr['combo'])
        save_results.append(sr)

    out_path = Path("data/mega_grid_results.json")
    out_path.write_text(json.dumps({
        "meta": {
            "total_combos": TOTAL,
            "workers": workers,
            "elapsed_hours": round((time.time() - t0) / 3600, 2),
            "grid": {
                "trailing_activate_atr": TRAIL_ACT,
                "trailing_distance_atr": TRAIL_DIST,
                "keltner_adx_threshold": ADX_THRESH,
                "sl_atr_mult": SL_MULT,
                "choppy_threshold": CHOPPY,
                "tp_atr_mult": TP_MULT,
                "spread_cost": SPREAD,
            },
            "timestamp": datetime.now().isoformat(),
        },
        "sharpe_distribution": {
            "mean": round(float(np.mean(all_sharpes)), 4),
            "std": round(float(np.std(all_sharpes)), 4),
            "min": round(float(min(all_sharpes)), 4),
            "max": round(float(max(all_sharpes)), 4),
            "p25": round(float(np.percentile(all_sharpes, 25)), 4),
            "p50": round(float(np.percentile(all_sharpes, 50)), 4),
            "p75": round(float(np.percentile(all_sharpes, 75)), 4),
        },
        "c12_rank": c12_rank,
        "dsr": {
            "best_label": best['label'],
            "best_sharpe": round(best['sharpe'], 4),
            "dsr_value": round(dsr['dsr'], 4),
            "sr_star": round(dsr['sr_star'], 4) if np.isfinite(dsr.get('sr_star', float('nan'))) else None,
            "passed": dsr['passed'],
        },
        "pbo_top50": {
            "pbo": round(pbo['pbo'], 4),
            "overfit_risk": pbo['overfit_risk'],
        },
        "results": save_results[:100],
    }, indent=2, default=str), encoding='utf-8')
    print(f"  Saved to {out_path} (top 100 combos)")

    # Full CSV for analysis
    csv_path = Path("data/mega_grid_all.csv")
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trail_act', 'trail_dist', 'adx', 'sl', 'choppy',
                         'n', 'sharpe', 'total_pnl', 'max_dd', 'win_rate', 'rr',
                         'keltner_n', 'keltner_pnl', 'orb_n', 'orb_pnl',
                         'rsi_n', 'rsi_pnl'])
        for r in results:
            ta, td, adx, sl, ch = r['combo']
            writer.writerow([ta, td, adx, sl, ch,
                             r['n'], round(r['sharpe'], 4), round(r['total_pnl'], 2),
                             round(r['max_dd'], 2), round(r['win_rate'], 2),
                             round(r['rr'], 3),
                             r['keltner_n'], round(r['keltner_pnl'], 2),
                             r['orb_n'], round(r['orb_pnl'], 2),
                             r['rsi_n'], round(r['rsi_pnl'], 2)])
    print(f"  Full CSV: {csv_path} ({TOTAL} rows)")

    # Final summary
    elapsed_total = time.time() - t0
    print(f"\n{'='*100}")
    print(f"  MEGA GRID SEARCH COMPLETE")
    print(f"  {TOTAL} combinations in {elapsed_total/3600:.1f} hours ({elapsed_total:.0f}s)")
    print(f"  Best: {results[0]['label']} (Sharpe={results[0]['sharpe']:.2f})")
    if c12_rank:
        print(f"  C12:  #{c12_rank}/{TOTAL} (Sharpe={c12_result['sharpe']:.2f})")
    print(f"  DSR passed: {'YES' if dsr['passed'] else 'NO'} (DSR={dsr['dsr']:.4f})")
    print(f"  PBO (top-50): {pbo['pbo']:.2f} ({pbo['overfit_risk']})")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
