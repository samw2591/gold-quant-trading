"""
Cooldown Period Fine-Grained Scan
==================================
Scans cooldown durations from 15 minutes to 60 minutes (plus current 3h baseline).

Tests on C12+Adaptive (0.35/0.60) with spread=$0.50.
The engine already supports float cooldown_hours (e.g. 0.25 = 15 min).

Usage:
    python backtest_cooldown_scan.py
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

COOLDOWN_SCAN = [
    (0.25,  "15min"),
    (1/3,   "20min"),
    (0.50,  "30min"),
    (35/60, "35min"),
    (2/3,   "40min"),
    (0.75,  "45min"),
    (50/60, "50min"),
    (55/60, "55min"),
    (1.0,   "60min"),
    (3.0,   "3h (current)"),
]


def main():
    print("=" * 90)
    print("  COOLDOWN PERIOD SCAN")
    print("  15min -> 60min fine-grained + 3h baseline")
    print("=" * 90)

    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    results = []

    print(f"\n  {len(COOLDOWN_SCAN)} cooldown values to test\n")

    for cd_hours, cd_label in COOLDOWN_SCAN:
        label = f"CD={cd_label}"
        kw = {**ADAPTIVE_BASE, "cooldown_hours": cd_hours}
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        results.append(stats)
        print(f"N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, MaxDD=${stats['max_dd']:.0f}")

    # Summary table
    print("\n" + "=" * 90)
    print("  Cooldown Scan Results")
    print("=" * 90)

    print(f"\n  {'Cooldown':<18} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>10} {'WR%':>7} {'$/trade':>9}")
    print("  " + "-" * 72)

    best_sharpe = -999
    best_label = ""
    for i, (cd_hours, cd_label) in enumerate(COOLDOWN_SCAN):
        s = results[i]
        avg_trade = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
        marker = ""
        if s['sharpe'] > best_sharpe:
            best_sharpe = s['sharpe']
            best_label = cd_label
        if cd_label == "3h (current)":
            marker = " <-- current"
        print(f"  {cd_label:<18} {s['n']:>6} {s['sharpe']:>8.2f} "
              f"${s['total_pnl']:>9.0f} ${s['max_dd']:>9.0f} "
              f"{s['win_rate']:>6.1f}% ${avg_trade:>8.2f}{marker}")

    print(f"\n  Best: {best_label} (Sharpe={best_sharpe:.2f})")

    # Full comparison
    print("\n" + "-" * 90)
    print_comparison(results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
