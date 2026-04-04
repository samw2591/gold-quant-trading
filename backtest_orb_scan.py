"""
ORB Max Hold Scan
==================
Scans ORB max hold time: 4/6/8/12/16/20/24 M15 bars (1h to 6h).
On C12+Adaptive (spread=$0.50).

Usage: python backtest_orb_scan.py
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

ORB_VALUES = [4, 6, 8, 12, 16, 20, 24, 0]


def main():
    print("=" * 90)
    print("  ORB MAX HOLD SCAN")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    results = []
    for hold in ORB_VALUES:
        label = f"ORB hold={hold} bars ({hold*15}min)" if hold > 0 else "ORB no limit (default)"
        kw = {**ADAPTIVE_BASE}
        if hold > 0:
            kw["orb_max_hold_m15"] = hold
        print(f"  {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        results.append(stats)
        orb_n = stats.get('orb_n', 0)
        orb_pnl = stats.get('orb_pnl', 0)
        orb_wr = stats.get('orb_wr', 0)
        print(f"N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
              f"ORB: {orb_n} trades, PnL=${orb_pnl:.0f}, WR={orb_wr:.1f}%")

    print("\n" + "=" * 90)
    print("  ORB Hold Summary")
    print("=" * 90)
    print(f"\n  {'Hold':<28} {'Total':>6} {'Sharpe':>8} {'ORB_N':>6} "
          f"{'ORB_PnL':>9} {'ORB_WR':>7} {'K_PnL':>9}")
    print("  " + "-" * 78)
    for s in results:
        orb_n = s.get('orb_n', 0)
        orb_pnl = s.get('orb_pnl', 0)
        orb_wr = s.get('orb_wr', 0)
        k_pnl = s.get('keltner_pnl', 0)
        print(f"  {s['label']:<28} {s['n']:>6} {s['sharpe']:>8.2f} {orb_n:>6} "
              f"${orb_pnl:>8.0f} {orb_wr:>6.1f}% ${k_pnl:>8.0f}")

    print("\n" + "-" * 90)
    print_comparison(results)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
