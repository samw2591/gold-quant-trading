"""
Spread x Cooldown x Adaptive Cross Experiment
===============================================
2 spreads x 5 cooldowns x 2 adaptive = 20 runs.

Usage: python backtest_spread_cooldown_cross.py
"""
import time
from backtest import DataBundle, run_variant, print_comparison
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREADS = [0.30, 0.50]
COOLDOWNS = [
    (0.25, "15m"),
    (0.50, "30m"),
    (0.75, "45m"),
    (1.0,  "60m"),
    (3.0,  "3h"),
]
ADAPTIVE_OPTIONS = [False, True]


def main():
    print("=" * 90)
    print("  SPREAD x COOLDOWN x ADAPTIVE CROSS EXPERIMENT")
    print(f"  {len(SPREADS)} spreads x {len(COOLDOWNS)} cooldowns x 2 adaptive = "
          f"{len(SPREADS)*len(COOLDOWNS)*2} runs")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    results = []

    for spread in SPREADS:
        for cd_hours, cd_label in COOLDOWNS:
            for adaptive in ADAPTIVE_OPTIONS:
                adapt_str = "adapt" if adaptive else "raw"
                label = f"sp${spread}/cd={cd_label}/{adapt_str}"

                kw = {**C12_KWARGS, "spread_cost": spread, "cooldown_hours": cd_hours}
                if adaptive:
                    kw.update({
                        "intraday_adaptive": True,
                        "choppy_threshold": 0.35,
                        "kc_only_threshold": 0.60,
                    })

                print(f"  {label}...", end=" ", flush=True)
                stats = run_variant(bundle, label, verbose=False, **kw)
                results.append(stats)
                print(f"N={stats['n']}, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}")

    # Pivot table
    print("\n" + "=" * 90)
    print("  Pivot: Sharpe by Spread x Cooldown (Adaptive vs Raw)")
    print("=" * 90)

    print(f"\n  {'Spread':<10} {'Cooldown':<10} {'Raw':>8} {'Adaptive':>10} {'Delta':>8}")
    print("  " + "-" * 50)

    for spread in SPREADS:
        for cd_hours, cd_label in COOLDOWNS:
            raw_s = next((r for r in results
                          if f"sp${spread}" in r['label'] and f"cd={cd_label}" in r['label']
                          and "raw" in r['label']), None)
            adapt_s = next((r for r in results
                            if f"sp${spread}" in r['label'] and f"cd={cd_label}" in r['label']
                            and "adapt" in r['label']), None)
            if raw_s and adapt_s:
                delta = adapt_s['sharpe'] - raw_s['sharpe']
                print(f"  ${spread:<9} {cd_label:<10} {raw_s['sharpe']:>8.2f} "
                      f"{adapt_s['sharpe']:>10.2f} {delta:>+8.2f}")

    print("\n" + "-" * 90)
    print_comparison(results)

    best = max(results, key=lambda r: r['sharpe'])
    print(f"\n  BEST: {best['label']} (Sharpe={best['sharpe']:.2f}, PnL=${best['total_pnl']:.0f})")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
