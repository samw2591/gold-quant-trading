"""
Walk-Forward Validation
========================
Rolling window analysis: 2-year train, 6-month test, 6-month step.
Tests C12, Baseline, and C12+Adaptive on each window.

Usage: python backtest_walkforward.py
"""
import time
import pandas as pd

from backtest import DataBundle, run_variant
from backtest.runner import (
    C12_KWARGS, TRUE_BASELINE_KWARGS,
    load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH,
)
from strategies.signals import prepare_indicators

SPREAD = 0.50

CONFIGS = {
    "C12": {**C12_KWARGS, "spread_cost": SPREAD},
    "Baseline": {**TRUE_BASELINE_KWARGS, "spread_cost": SPREAD},
    "C12+Adaptive": {
        **C12_KWARGS, "spread_cost": SPREAD,
        "intraday_adaptive": True, "choppy_threshold": 0.35, "kc_only_threshold": 0.60,
    },
}


def main():
    print("=" * 90)
    print("  WALK-FORWARD VALIDATION")
    print("  Train: 2yr | Test: 6mo | Step: 6mo")
    print("=" * 90)
    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)

    data_end = m15_df.index[-1]

    starts = pd.date_range('2015-01-01', '2024-07-01', freq='6MS', tz='UTC')

    windows = []
    for s in starts:
        train_start = s
        train_end = s + pd.DateOffset(years=2)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=6)
        if test_end > data_end:
            break
        windows.append((train_start, train_end, test_start, test_end))

    print(f"  {len(windows)} walk-forward windows\n")

    wf_results = []

    for i, (tr_s, tr_e, te_s, te_e) in enumerate(windows):
        label = f"W{i+1}: train {tr_s.strftime('%Y-%m')}..{tr_e.strftime('%Y-%m')} | test {te_s.strftime('%Y-%m')}..{te_e.strftime('%Y-%m')}"
        print(f"  {label}")

        m15_train = m15_df[(m15_df.index >= tr_s) & (m15_df.index < tr_e)]
        h1_train = h1_df[(h1_df.index >= tr_s - pd.Timedelta(hours=200)) & (h1_df.index < tr_e)]
        m15_test = m15_df[(m15_df.index >= te_s) & (m15_df.index < te_e)]
        h1_test = h1_df[(h1_df.index >= te_s - pd.Timedelta(hours=200)) & (h1_df.index < te_e)]

        if len(m15_train) < 100 or len(m15_test) < 100:
            print(f"    Skipping (insufficient data)")
            continue

        is_results = {}
        for name, kw in CONFIGS.items():
            try:
                b = DataBundle(m15_train, h1_train)
                stats = run_variant(b, f"IS-{name}", verbose=False, **kw)
                is_results[name] = stats['sharpe']
            except Exception:
                is_results[name] = -999

        is_best = max(is_results, key=is_results.get)

        oos_results = {}
        for name, kw in CONFIGS.items():
            try:
                b = DataBundle(m15_test, h1_test)
                stats = run_variant(b, f"OOS-{name}", verbose=False, **kw)
                oos_results[name] = stats['sharpe']
            except Exception:
                oos_results[name] = -999

        oos_best = max(oos_results, key=oos_results.get)
        held = is_best == oos_best

        print(f"    IS-best: {is_best} ({is_results[is_best]:.2f}) | "
              f"OOS: {oos_results.get(is_best, -999):.2f} | "
              f"OOS-best: {oos_best} ({oos_results[oos_best]:.2f}) | "
              f"{'HELD' if held else 'SHIFTED'}")

        wf_results.append({
            "window": f"W{i+1}",
            "train": f"{tr_s.strftime('%Y-%m')}..{tr_e.strftime('%Y-%m')}",
            "test": f"{te_s.strftime('%Y-%m')}..{te_e.strftime('%Y-%m')}",
            "is_best": is_best,
            "is_sharpe": is_results[is_best],
            "oos_sharpe": oos_results.get(is_best, -999),
            "oos_best": oos_best,
            "oos_best_sharpe": oos_results[oos_best],
            "held": held,
            "adaptive_oos": oos_results.get("C12+Adaptive", -999),
        })

    print("\n" + "=" * 90)
    print("  Walk-Forward Summary")
    print("=" * 90)

    print(f"\n  {'Window':<6} {'Train':<16} {'Test':<16} {'IS-Best':<14} "
          f"{'IS-SR':>6} {'OOS-SR':>7} {'OOS-Best':<14} {'Status':>8}")
    print("  " + "-" * 90)

    held_count = 0
    adaptive_oos_sharpes = []
    for r in wf_results:
        status = "HELD" if r['held'] else "SHIFT"
        if r['held']:
            held_count += 1
        adaptive_oos_sharpes.append(r['adaptive_oos'])
        print(f"  {r['window']:<6} {r['train']:<16} {r['test']:<16} "
              f"{r['is_best']:<14} {r['is_sharpe']:>6.2f} {r['oos_sharpe']:>7.2f} "
              f"{r['oos_best']:<14} {status:>8}")

    n_windows = len(wf_results)
    if n_windows > 0:
        consistency = held_count / n_windows * 100
        avg_adaptive_oos = sum(adaptive_oos_sharpes) / len(adaptive_oos_sharpes)
        neg_windows = sum(1 for s in adaptive_oos_sharpes if s < 0)

        print(f"\n  Consistency: {held_count}/{n_windows} ({consistency:.0f}%)")
        print(f"  C12+Adaptive avg OOS Sharpe: {avg_adaptive_oos:.2f}")
        print(f"  Negative OOS windows for Adaptive: {neg_windows}/{n_windows}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
