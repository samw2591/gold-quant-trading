"""
Spread Model Comparison
========================
Compare 3 spread models on C12 and Combo configs:
  1. fixed   — flat $0.50 per trade
  2. atr_scaled — base $0.30, scales with ATR percentile
  3. session_aware — varies by trading session (Asia/London/NY)

Usage:
    python backtest_spread_model.py
"""
import time
from backtest import DataBundle, run_variant, calc_stats, print_comparison
from backtest.runner import (
    C12_KWARGS, load_m15, load_h1_aligned, add_atr_percentile,
    prepare_indicators_custom, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators


def main():
    print("=" * 90)
    print("  SPREAD MODEL COMPARISON")
    print("=" * 90)

    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)

    bundle = DataBundle(m15_df, h1_df)

    spread_configs = [
        ("Fixed $0.50", {"spread_cost": 0.50, "spread_model": "fixed"}),
        ("Fixed $0.30", {"spread_cost": 0.30, "spread_model": "fixed"}),
        ("ATR-Scaled (base $0.30)", {"spread_model": "atr_scaled", "spread_base": 0.30, "spread_max": 3.0}),
        ("Session-Aware (base $0.30)", {"spread_model": "session_aware", "spread_base": 0.30, "spread_max": 3.0}),
        ("Session-Aware (base $0.50)", {"spread_model": "session_aware", "spread_base": 0.50, "spread_max": 3.0}),
    ]

    # ── C12 Config ──
    print("\n" + "=" * 90)
    print("  Part 1: C12 Configuration")
    print("=" * 90)

    c12_results = []
    for label, spread_kw in spread_configs:
        kw = {**C12_KWARGS, **spread_kw}
        print(f"\n  Running C12 + {label}...", flush=True)
        stats = run_variant(bundle, f"C12 {label}", verbose=False, **kw)
        c12_results.append(stats)
        print(f"    N={stats['n_trades']}, Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, MaxDD=${stats['max_dd']:.0f}")

    print("\n" + "-" * 90)
    print("  C12 Spread Model Comparison")
    print("-" * 90)
    print_comparison(c12_results)

    # ── C12 + Intraday Adaptive ──
    print("\n" + "=" * 90)
    print("  Part 2: C12 + Intraday Adaptive (0.35/0.60)")
    print("=" * 90)

    adaptive_results = []
    for label, spread_kw in spread_configs:
        kw = {
            **C12_KWARGS,
            "intraday_adaptive": True,
            "choppy_threshold": 0.35,
            "kc_only_threshold": 0.60,
            **spread_kw,
        }
        print(f"\n  Running C12+Adaptive + {label}...", flush=True)
        stats = run_variant(bundle, f"Adaptive {label}", verbose=False, **kw)
        adaptive_results.append(stats)
        print(f"    N={stats['n_trades']}, Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, MaxDD=${stats['max_dd']:.0f}")

    print("\n" + "-" * 90)
    print("  Adaptive Spread Model Comparison")
    print("-" * 90)
    print_comparison(adaptive_results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
