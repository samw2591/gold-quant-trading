"""
Keltner Parameter Sensitivity Scan
====================================
One-dimensional sweeps of ADX threshold, SL multiplier, TP multiplier
on C12+Adaptive (spread=$0.50). Then top-3 combinations.

Usage: python backtest_kc_sensitivity.py
"""
import time
from backtest import DataBundle, run_variant, print_comparison
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREAD = 0.50
ADAPTIVE_BASE = {
    **C12_KWARGS,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
}


def main():
    print("=" * 90)
    print("  KELTNER PARAMETER SENSITIVITY")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    all_results = []

    # Sweep 1: ADX threshold
    print("\n  --- ADX Threshold Sweep (SL=3.5, TP=5.0) ---")
    adx_results = []
    for adx in [15, 16, 17, 18, 19, 20, 21, 22]:
        label = f"ADX={adx}"
        kw = {**ADAPTIVE_BASE, "keltner_adx_threshold": adx}
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        adx_results.append(stats)
        all_results.append((adx, 3.5, 5.0, stats))
        print(f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, N={stats['n']}")

    # Sweep 2: SL multiplier
    print("\n  --- SL ATR Multiplier Sweep (ADX=18, TP=5.0) ---")
    sl_results = []
    for sl in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        label = f"SL={sl}"
        kw = {**ADAPTIVE_BASE, "sl_atr_mult": sl}
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        sl_results.append(stats)
        all_results.append((18, sl, 5.0, stats))
        print(f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, N={stats['n']}")

    # Sweep 3: TP multiplier
    print("\n  --- TP ATR Multiplier Sweep (ADX=18, SL=3.5) ---")
    tp_results = []
    for tp in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        label = f"TP={tp}"
        kw = {**ADAPTIVE_BASE, "tp_atr_mult": tp}
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        tp_results.append(stats)
        all_results.append((18, 3.5, tp, stats))
        print(f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, N={stats['n']}")

    # Top-3 combinations from sweeps
    print("\n  --- Top Combinations ---")
    sorted_by_sharpe = sorted(all_results, key=lambda x: x[3]['sharpe'], reverse=True)
    seen = set()
    combo_results = []
    for adx, sl, tp, _ in sorted_by_sharpe:
        key = (adx, sl, tp)
        if key in seen or key == (18, 3.5, 5.0):
            continue
        seen.add(key)
        label = f"Combo ADX={adx}/SL={sl}/TP={tp}"
        kw = {**ADAPTIVE_BASE, "keltner_adx_threshold": adx, "sl_atr_mult": sl, "tp_atr_mult": tp}
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        combo_results.append(stats)
        print(f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}")
        if len(combo_results) >= 3:
            break

    # Summary
    print("\n" + "=" * 90)
    print("  Sensitivity Summary")
    print("=" * 90)

    print("\n  ADX Sweep:")
    print(f"  {'ADX':>5} {'Sharpe':>8} {'PnL':>10} {'N':>6}")
    for s in adx_results:
        print(f"  {s['label']:>5} {s['sharpe']:>8.2f} ${s['total_pnl']:>9.0f} {s['n']:>6}")

    print("\n  SL Sweep:")
    print(f"  {'SL':>5} {'Sharpe':>8} {'PnL':>10} {'N':>6}")
    for s in sl_results:
        print(f"  {s['label']:>5} {s['sharpe']:>8.2f} ${s['total_pnl']:>9.0f} {s['n']:>6}")

    print("\n  TP Sweep:")
    print(f"  {'TP':>5} {'Sharpe':>8} {'PnL':>10} {'N':>6}")
    for s in tp_results:
        print(f"  {s['label']:>5} {s['sharpe']:>8.2f} ${s['total_pnl']:>9.0f} {s['n']:>6}")

    if combo_results:
        print("\n  Top Combos:")
        print_comparison(combo_results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
