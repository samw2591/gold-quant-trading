"""
Statistical Validation Suite
==============================
Runs PSR / DSR / PBO on C12 and its parameter variants to detect overfitting.

  1. PSR — Is the observed Sharpe statistically significant?
  2. DSR — After testing N parameter combos, is the best still significant?
  3. PBO — Combinatorial cross-validation: how likely is our in-sample winner
           to fail out-of-sample?

Usage:
    python backtest_statistical_validation.py
"""
import time
import numpy as np

from backtest import (
    DataBundle, run_variant, calc_stats,
    probabilistic_sharpe, deflated_sharpe, compute_pbo,
    aggregate_daily_pnl,
)
from backtest.runner import (
    C12_KWARGS, TRUE_BASELINE_KWARGS,
    load_m15, load_h1_aligned, add_atr_percentile,
    H1_CSV_PATH,
)
from strategies.signals import prepare_indicators


SPREAD = 0.50

VARIANTS = {
    "C12 (main)": {**C12_KWARGS, "spread_cost": SPREAD},
    "Baseline": {**TRUE_BASELINE_KWARGS, "spread_cost": SPREAD},
    "C12 ADX=15": {**C12_KWARGS, "keltner_adx_threshold": 15, "spread_cost": SPREAD},
    "C12 ADX=21": {**C12_KWARGS, "keltner_adx_threshold": 21, "spread_cost": SPREAD},
    "C12 ADX=25": {**C12_KWARGS, "keltner_adx_threshold": 25, "spread_cost": SPREAD},
    "C12 SL=2.5": {**C12_KWARGS, "sl_atr_mult": 2.5, "spread_cost": SPREAD},
    "C12 SL=4.5": {**C12_KWARGS, "sl_atr_mult": 4.5, "spread_cost": SPREAD},
    "C12 TP=3.0": {**C12_KWARGS, "tp_atr_mult": 3.0, "spread_cost": SPREAD},
    "C12 TP=7.0": {**C12_KWARGS, "tp_atr_mult": 7.0, "spread_cost": SPREAD},
    "C12 Trail=0.6/0.20": {**C12_KWARGS, "trailing_activate_atr": 0.6,
                            "trailing_distance_atr": 0.20, "spread_cost": SPREAD},
    "C12 Trail=1.0/0.35": {**C12_KWARGS, "trailing_activate_atr": 1.0,
                            "trailing_distance_atr": 0.35, "spread_cost": SPREAD},
    "C12+Adaptive": {**C12_KWARGS, "intraday_adaptive": True,
                     "choppy_threshold": 0.35, "kc_only_threshold": 0.60,
                     "spread_cost": SPREAD},
}


def main():
    print("=" * 90)
    print("  STATISTICAL VALIDATION SUITE")
    print(f"  {len(VARIANTS)} parameter variants | spread=${SPREAD}")
    print("=" * 90)

    t0 = time.time()

    m15_raw = load_m15()
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    all_stats = {}
    all_daily_pnls = {}
    all_sharpes = []

    print("\n  Running all variants...")
    for label, kw in VARIANTS.items():
        print(f"    {label}...", end=" ", flush=True)
        stats = run_variant(bundle, label, verbose=False, **kw)
        all_stats[label] = stats
        trades = stats.get('_trades', [])
        daily = aggregate_daily_pnl(trades)
        all_daily_pnls[label] = daily
        all_sharpes.append(stats['sharpe'])
        print(f"N={stats['n']}, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}")

    # ── Part 1: PSR ──
    print("\n" + "=" * 90)
    print("  Part 1: Probabilistic Sharpe Ratio (PSR)")
    print("  H0: True Sharpe <= 0  |  Reject if PSR > 0.95")
    print("=" * 90)

    print(f"\n  {'Variant':<25} {'SR_obs':>8} {'PSR':>8} {'p-val':>8} {'Result':>10}")
    print("  " + "-" * 65)

    for label in VARIANTS:
        daily = all_daily_pnls[label]
        psr = probabilistic_sharpe(daily, sharpe_benchmark=0)
        sig = "PASS" if psr['psr'] > 0.95 else "FAIL"
        print(f"  {label:<25} {psr['sharpe_obs']:>8.2f} {psr['psr']:>8.4f} "
              f"{psr['p_value']:>8.4f} {sig:>10}")

    # ── Part 2: DSR ──
    print("\n" + "=" * 90)
    print("  Part 2: Deflated Sharpe Ratio (DSR)")
    print(f"  Adjusting for {len(VARIANTS)} trials tested")
    print("=" * 90)

    sharpe_var = float(np.var(all_sharpes))
    print(f"\n  Variance of all tested Sharpes: {sharpe_var:.4f}")

    print(f"\n  {'Variant':<25} {'SR_obs':>8} {'SR*':>8} {'DSR':>8} {'Result':>10}")
    print("  " + "-" * 65)

    for label in VARIANTS:
        daily = all_daily_pnls[label]
        dsr = deflated_sharpe(daily, n_trials=len(VARIANTS), all_sharpes_var=sharpe_var)
        sig = "PASS" if dsr['passed'] else "FAIL"
        sr_star = f"{dsr['sr_star']:.2f}" if np.isfinite(dsr['sr_star']) else "N/A"
        print(f"  {label:<25} {dsr['sharpe_obs']:>8.2f} {sr_star:>8} "
              f"{dsr['dsr']:>8.4f} {sig:>10}")

    # ── Part 3: PBO ──
    print("\n" + "=" * 90)
    print("  Part 3: Probability of Backtest Overfitting (PBO)")
    print("  CSCV with 8 partitions")
    print("=" * 90)

    pbo = compute_pbo(all_daily_pnls, n_partitions=8)
    print(f"\n  PBO = {pbo['pbo']:.2f}  ({pbo['n_combinations']} combinations)")
    print(f"  Overfit Risk: {pbo['overfit_risk']}")

    if 'is_best_oos_ranks' in pbo and pbo['is_best_oos_ranks']:
        ranks = pbo['is_best_oos_ranks']
        print(f"  IS-best OOS rank: mean={np.mean(ranks):.1f}, "
              f"median={np.median(ranks):.1f}, worst={max(ranks)}")

    if 'logit_distribution' in pbo and pbo['logit_distribution']:
        logits = [x for x in pbo['logit_distribution'] if np.isfinite(x)]
        if logits:
            print(f"  Logit λ: mean={np.mean(logits):.2f}, "
                  f"median={np.median(logits):.2f}")
            pct_positive = sum(1 for x in logits if x > 0) / len(logits)
            print(f"  Logit > 0 (overfit): {pct_positive:.1%}")

    # ── Summary ──
    print("\n" + "=" * 90)
    print("  INTERPRETATION GUIDE")
    print("=" * 90)
    print("  PSR > 0.95 → SR is statistically significant (not just luck)")
    print("  DSR > 0.95 → SR survives multiple-testing adjustment")
    print("  PBO < 0.20 → LOW overfit risk")
    print("  PBO 0.20-0.40 → MEDIUM risk, proceed with caution")
    print("  PBO > 0.40 → HIGH risk, likely overfitted")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("  Done!")


if __name__ == "__main__":
    main()
