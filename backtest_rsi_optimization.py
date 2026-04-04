"""
M15 RSI Optimization
=====================
Tests whether M15 RSI can be rescued with tighter thresholds and filters.
If no variant achieves RSI PnL > 0 and WR > 55%, conclusion: disable RSI.

Variants tested on C12+Adaptive (0.35/0.60) with spread=$0.50:

  A. RSI threshold tightening (buy<10/sell>90, buy<5/sell>95)
  B. Direction filter (buy-only, no short RSI)
  C. ADX filter (block RSI when H1 ADX > 25/30/35)
  D. ATR volatility filter (block extreme volatility)
  E. Best combinations from A-D

Usage:
    python backtest_rsi_optimization.py
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
    print("  M15 RSI OPTIMIZATION")
    print("  Can we rescue M15 RSI with tighter filters?")
    print("=" * 90)

    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    variants = {}

    # Baseline: current RSI (buy<15, sell>85, no filters)
    variants["RSI default (15/85)"] = {}

    # A. Threshold tightening
    variants["RSI 10/90"] = {"rsi_buy_threshold": 10, "rsi_sell_threshold": 90}
    variants["RSI 5/95"] = {"rsi_buy_threshold": 5, "rsi_sell_threshold": 95}
    variants["RSI 8/92"] = {"rsi_buy_threshold": 8, "rsi_sell_threshold": 92}

    # B. Direction filter (buy-only)
    variants["RSI buy-only (15/85)"] = {"rsi_sell_enabled": False}
    variants["RSI buy-only (10/90)"] = {"rsi_buy_threshold": 10, "rsi_sell_threshold": 90,
                                        "rsi_sell_enabled": False}
    variants["RSI buy-only (5/95)"] = {"rsi_buy_threshold": 5, "rsi_sell_threshold": 95,
                                       "rsi_sell_enabled": False}

    # C. ADX filter
    variants["RSI + ADX<25"] = {"rsi_adx_filter": 25}
    variants["RSI + ADX<30"] = {"rsi_adx_filter": 30}
    variants["RSI + ADX<35"] = {"rsi_adx_filter": 35}

    # D. ATR volatility filter
    variants["RSI + ATR<0.70"] = {"rsi_atr_pct_filter": 0.70}
    variants["RSI + ATR>0.20"] = {"rsi_atr_pct_min_filter": 0.20}
    variants["RSI + ATR 0.20-0.70"] = {"rsi_atr_pct_filter": 0.70,
                                        "rsi_atr_pct_min_filter": 0.20}

    # E. Combinations
    variants["RSI 10/90 buy-only + ADX<30"] = {
        "rsi_buy_threshold": 10, "rsi_sell_threshold": 90,
        "rsi_sell_enabled": False, "rsi_adx_filter": 30,
    }
    variants["RSI 5/95 buy-only + ADX<30"] = {
        "rsi_buy_threshold": 5, "rsi_sell_threshold": 95,
        "rsi_sell_enabled": False, "rsi_adx_filter": 30,
    }
    variants["RSI 10/90 + ADX<30 + ATR<0.70"] = {
        "rsi_buy_threshold": 10, "rsi_sell_threshold": 90,
        "rsi_adx_filter": 30, "rsi_atr_pct_filter": 0.70,
    }
    variants["RSI 5/95 buy-only + ADX<30 + ATR band"] = {
        "rsi_buy_threshold": 5, "rsi_sell_threshold": 95,
        "rsi_sell_enabled": False, "rsi_adx_filter": 30,
        "rsi_atr_pct_filter": 0.70, "rsi_atr_pct_min_filter": 0.20,
    }

    # Also test: no RSI at all (Keltner+ORB only) as reference
    # We can't directly disable RSI via a param, but kc_only_threshold=0.65
    # effectively does it (from threshold scan results)
    variants["NO RSI (kc_only=0.65)"] = {"kc_only_threshold": 0.65}

    # Run all variants
    results = []
    print(f"\n  {len(variants)} variants to test\n")

    for label, extra_kw in variants.items():
        kw = {**ADAPTIVE_BASE, **extra_kw}
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        results.append(stats)
        rsi_n = stats.get('rsi_n', 0)
        rsi_pnl = stats.get('rsi_pnl', 0)
        rsi_wr = stats.get('rsi_wr', 0)
        print(f"N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
              f"RSI: {rsi_n} trades, PnL=${rsi_pnl:.0f}, WR={rsi_wr:.1f}%")

    # Summary table
    print("\n" + "=" * 90)
    print("  RSI-Focused Summary")
    print("=" * 90)
    print(f"\n  {'Variant':<38} {'Total':>6} {'Sharpe':>7} {'RSI_N':>6} "
          f"{'RSI_PnL':>9} {'RSI_WR':>7} {'K_PnL':>9} {'Verdict':>8}")
    print("  " + "-" * 95)

    for s in results:
        rsi_n = s.get('rsi_n', 0)
        rsi_pnl = s.get('rsi_pnl', 0)
        rsi_wr = s.get('rsi_wr', 0)
        k_pnl = s.get('keltner_pnl', 0)
        verdict = "PASS" if rsi_pnl > 0 and rsi_wr > 55 else "FAIL"
        if rsi_n == 0:
            verdict = "N/A"
        print(f"  {s['label']:<38} {s['n']:>6} {s['sharpe']:>7.2f} {rsi_n:>6} "
              f"${rsi_pnl:>8.0f} {rsi_wr:>6.1f}% ${k_pnl:>8.0f} {verdict:>8}")

    # Full comparison
    print("\n" + "-" * 90)
    print_comparison(results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Final verdict
    passed = [s for s in results if s.get('rsi_pnl', 0) > 0
              and s.get('rsi_wr', 0) > 55 and s.get('rsi_n', 0) > 0]
    if passed:
        best = max(passed, key=lambda s: s.get('rsi_pnl', 0))
        print(f"\n  VERDICT: RSI CAN BE RESCUED")
        print(f"  Best variant: {best['label']}")
        print(f"  RSI PnL=${best.get('rsi_pnl', 0):.0f}, "
              f"WR={best.get('rsi_wr', 0):.1f}%, N={best.get('rsi_n', 0)}")
    else:
        print(f"\n  VERDICT: RSI CANNOT BE RESCUED — recommend disabling M15 RSI")

    print("  Done!")


if __name__ == "__main__":
    main()
