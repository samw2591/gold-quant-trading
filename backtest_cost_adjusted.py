"""
Cost-Adjusted Backtest
======================
Re-evaluate all configs with realistic transaction costs.
Then test frequency-reduction strategies to find cost-viable configs.

Phase 1: Spread sensitivity (0 / 0.30 / 0.50 / 0.80) x 5 configs
Phase 2: Frequency reduction strategies with $0.50 spread
Phase 3: Best config K-Fold validation with costs
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.signals import prepare_indicators

from backtest import DataBundle, run_variant
from backtest.runner import (
    C12_KWARGS,
    V3_REGIME,
    TRUE_BASELINE_KWARGS,
    add_atr_percentile,
    prepare_indicators_custom,
    load_m15,
    load_h1_aligned,
    H1_CSV_PATH,
)

RESULTS = {}


def run_engine(m15_df, h1_df, label, regime_config=None, spread=0.0, **kwargs):
    stats = run_variant(
        DataBundle(m15_df, h1_df),
        label,
        verbose=False,
        regime_config=regime_config,
        spread_cost=spread,
        **kwargs,
    )
    return stats, stats['_trades']


def run_on_window(m15_df, h1_df, start, end, label, regime_config=None, spread=0.0, **kwargs):
    m15 = m15_df[(m15_df.index >= start) & (m15_df.index < end)]
    h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
    if len(m15) < 1000 or len(h1) < 200:
        return None
    stats, _ = run_engine(m15, h1, label, regime_config=regime_config, spread=spread, **kwargs)
    return stats


def run_cooldown(
    m15_df, h1_df, label, gap_hours, regime_config=None, spread=0.0, **kwargs
):
    stats = run_variant(
        DataBundle(m15_df, h1_df),
        label,
        verbose=False,
        regime_config=regime_config,
        spread_cost=spread,
        min_entry_gap_hours=gap_hours,
        **kwargs,
    )
    return stats, stats['_trades']


def print_table(results, title=""):
    if title:
        print(f"\n  --- {title} ---")
    print(f"\n  {'Rank':<5} {'Config':<45} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'DD%':>6} {'WR%':>6} {'$/trade':>8}")
    print(f"  {'-'*5} {'-'*45} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
    for rank, r in enumerate(results, 1):
        ppt = r['pnl'] / r['n'] if r['n'] > 0 else 0
        print(f"  {rank:<5} {r['label']:<45} {r['n']:>6} {r['sharpe']:>8.2f} "
              f"${r['pnl']:>9.0f} ${r['dd']:>7.0f} {r['dd_pct']:>5.1f}% "
              f"{r['wr']:>5.1f}% ${ppt:>7.2f}")


def main():
    t_start = time.time()
    print("=" * 100)
    print("  COST-ADJUSTED BACKTEST")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15_default = prepare_indicators(m15_raw)
    h1_default = prepare_indicators(h1_raw)
    h1_default = add_atr_percentile(h1_default)

    m15_custom = prepare_indicators_custom(m15_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = prepare_indicators_custom(h1_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = add_atr_percentile(h1_custom)

    print(f"  M15: {len(m15_default)} bars, H1: {len(h1_default)} bars\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Spread sensitivity
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  PHASE 1: Spread Sensitivity (5 configs x 4 spreads)")
    print("=" * 100)

    configs = [
        ("True Baseline (orig params)", m15_default, h1_default, None, TRUE_BASELINE_KWARGS),
        ("C12 (current live)", m15_default, h1_default, None, C12_KWARGS),
        ("C12 + Adaptive Trail", m15_default, h1_default, V3_REGIME, C12_KWARGS),
        ("C12 + KC1.25+EMA30", m15_custom, h1_custom, None, C12_KWARGS),
        ("Combo (Trail+KC1.25+EMA30)", m15_custom, h1_custom, V3_REGIME, C12_KWARGS),
    ]

    spreads = [0, 0.30, 0.50, 0.80]
    phase1 = []

    for spread in spreads:
        print(f"\n  --- Spread = ${spread:.2f} ---")
        for cfg_name, m15, h1, regime, kwargs in configs:
            label = f"{cfg_name} [sp={spread}]"

            t0 = time.time()
            stats, trades = run_engine(
                m15, h1, label, regime_config=regime, spread=spread, **kwargs
            )
            elapsed = time.time() - t0
            ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
            print(f"    {cfg_name:<40} N={stats['n']:>6}, Sharpe={stats['sharpe']:>6.2f}, "
                  f"PnL=${stats['total_pnl']:>9.0f}, $/trade={ppt:>6.2f}, {elapsed:.0f}s")
            phase1.append({
                'config': cfg_name, 'spread': spread,
                'label': label, 'sharpe': float(stats['sharpe']),
                'pnl': float(stats['total_pnl']), 'wr': float(stats['win_rate']),
                'rr': float(stats['rr']), 'dd': float(stats['max_dd']),
                'dd_pct': float(stats['max_dd_pct']), 'n': int(stats['n']),
                'ppt': float(ppt),
            })

    print(f"\n  === Spread Sensitivity Summary ===")
    print(f"  {'Config':<40} {'sp=0':>8} {'sp=0.3':>8} {'sp=0.5':>8} {'sp=0.8':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cfg_name in [c[0] for c in configs]:
        rows = [r for r in phase1 if r['config'] == cfg_name]
        sharpes = {r['spread']: r['sharpe'] for r in rows}
        print(f"  {cfg_name:<40} {sharpes.get(0,0):>8.2f} {sharpes.get(0.3,0):>8.2f} "
              f"{sharpes.get(0.5,0):>8.2f} {sharpes.get(0.8,0):>8.2f}")

    RESULTS['phase1'] = phase1

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Frequency reduction with $0.50 spread
    # ══════════════════════════════════════════════════════════════
    SPREAD = 0.50
    print(f"\n{'='*100}")
    print(f"  PHASE 2: Frequency Reduction Strategies (spread=${SPREAD})")
    print(f"  Goal: fewer trades -> lower total cost -> higher net profit")
    print(f"{'='*100}")

    gap_hours_list = [0, 2, 4, 6, 8, 12, 24]
    phase2 = []

    for gap in gap_hours_list:
        label = f"Combo gap={gap}h [sp={SPREAD}]"
        print(f"\n  [{gap}h gap] ", end='', flush=True)
        t0 = time.time()
        stats, trades = run_cooldown(
            m15_custom, h1_custom, label, gap,
            regime_config=V3_REGIME, spread=SPREAD, **C12_KWARGS)
        elapsed = time.time() - t0
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"N={stats['n']:>6}, Sharpe={stats['sharpe']:>6.2f}, PnL=${stats['total_pnl']:>9.0f}, "
              f"$/trade={ppt:>6.2f}, {elapsed:.0f}s")
        phase2.append({
            'gap_hours': gap, 'label': label,
            'sharpe': float(stats['sharpe']), 'pnl': float(stats['total_pnl']),
            'wr': float(stats['win_rate']), 'rr': float(stats['rr']),
            'dd': float(stats['max_dd']), 'dd_pct': float(stats['max_dd_pct']),
            'n': int(stats['n']), 'ppt': float(ppt),
        })

    print(f"\n  {'Gap':>5} {'N':>7} {'Sharpe':>8} {'PnL':>10} {'$/trade':>8} {'MaxDD':>8} {'WR%':>6}")
    print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for r in phase2:
        marker = " **" if r['sharpe'] == max(x['sharpe'] for x in phase2) else ""
        print(f"  {r['gap_hours']:>4}h {r['n']:>7} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} "
              f"${r['ppt']:>7.2f} ${r['dd']:>7.0f} {r['wr']:>5.1f}%{marker}")

    RESULTS['phase2_combo'] = phase2

    # Also test gap on C12 baseline (no KC change)
    phase2b = []
    for gap in [0, 4, 8, 12, 24]:
        label = f"C12 gap={gap}h [sp={SPREAD}]"
        print(f"\n  C12 [{gap}h gap] ", end='', flush=True)
        t0 = time.time()
        stats, trades = run_cooldown(
            m15_default, h1_default, label, gap,
            spread=SPREAD, **C12_KWARGS)
        elapsed = time.time() - t0
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"N={stats['n']:>6}, Sharpe={stats['sharpe']:>6.2f}, PnL=${stats['total_pnl']:>9.0f}, "
              f"$/trade={ppt:>6.2f}, {elapsed:.0f}s")
        phase2b.append({
            'gap_hours': gap, 'label': label,
            'sharpe': float(stats['sharpe']), 'pnl': float(stats['total_pnl']),
            'wr': float(stats['win_rate']), 'rr': float(stats['rr']),
            'dd': float(stats['max_dd']), 'dd_pct': float(stats['max_dd_pct']),
            'n': int(stats['n']), 'ppt': float(ppt),
        })

    print(f"\n  C12 gap results (spread=${SPREAD}):")
    print(f"  {'Gap':>5} {'N':>7} {'Sharpe':>8} {'PnL':>10} {'$/trade':>8} {'MaxDD':>8}")
    print(f"  {'-'*5} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for r in phase2b:
        print(f"  {r['gap_hours']:>4}h {r['n']:>7} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} "
              f"${r['ppt']:>7.2f} ${r['dd']:>7.0f}")

    RESULTS['phase2_c12'] = phase2b

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: K-Fold validation for best cost-adjusted config
    # ══════════════════════════════════════════════════════════════
    all_phase2 = phase2 + phase2b
    best = max(all_phase2, key=lambda x: x['sharpe'])
    print(f"\n{'='*100}")
    print(f"  PHASE 3: K-Fold Validation (spread=${SPREAD})")
    print(f"  Best from Phase 2: {best['label']} (Sharpe={best['sharpe']:.2f})")
    print(f"{'='*100}")

    best_gap = best['gap_hours']
    is_combo = 'Combo' in best['label']

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    fold_results = []
    for fold_name, test_start, test_end in folds:
        ts = pd.Timestamp(test_start, tz='UTC')
        te = pd.Timestamp(test_end, tz='UTC')
        print(f"\n  {fold_name}: {test_start}~{test_end}", flush=True)

        # Best config
        m15_b = m15_custom if is_combo else m15_default
        h1_b = h1_custom if is_combo else h1_default
        regime_b = V3_REGIME if is_combo else None

        m15_w = m15_b[(m15_b.index >= ts) & (m15_b.index < te)]
        h1_w = h1_b[(h1_b.index >= ts) & (h1_b.index < te)]
        if len(m15_w) < 1000:
            continue

        stats_best, _ = run_cooldown(m15_w, h1_w, f"Best [{fold_name}]", best_gap,
                                     regime_config=regime_b, spread=SPREAD, **C12_KWARGS)

        # C12 no-gap for comparison
        m15_c = m15_default[(m15_default.index >= ts) & (m15_default.index < te)]
        h1_c = h1_default[(h1_default.index >= ts) & (h1_default.index < te)]
        stats_c12, _ = run_engine(m15_c, h1_c, f"C12 [{fold_name}]", spread=SPREAD, **C12_KWARGS)

        fold_results.append({
            'fold': fold_name,
            'best_sharpe': float(stats_best['sharpe']),
            'best_pnl': float(stats_best['total_pnl']),
            'best_n': int(stats_best['n']),
            'c12_sharpe': float(stats_c12['sharpe']),
            'c12_pnl': float(stats_c12['total_pnl']),
            'c12_n': int(stats_c12['n']),
        })

    print(f"\n  {'Fold':<8} {'Best Sh':>8} {'Best PnL':>10} {'Best N':>7} "
          f"{'C12 Sh':>8} {'C12 PnL':>10} {'C12 N':>7} {'Winner':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
    best_wins = 0
    for r in fold_results:
        w = "Best" if r['best_sharpe'] > r['c12_sharpe'] else "C12"
        if w == "Best":
            best_wins += 1
        print(f"  {r['fold']:<8} {r['best_sharpe']:>8.2f} ${r['best_pnl']:>9.0f} {r['best_n']:>7} "
              f"{r['c12_sharpe']:>8.2f} ${r['c12_pnl']:>9.0f} {r['c12_n']:>7} {w:>8}")

    avg_best = np.mean([r['best_sharpe'] for r in fold_results])
    avg_c12 = np.mean([r['c12_sharpe'] for r in fold_results])
    print(f"\n  Avg Sharpe: Best={avg_best:.2f}, C12={avg_c12:.2f}")
    print(f"  Best wins {best_wins}/{len(fold_results)} folds")
    all_pos = all(r['best_sharpe'] > 0 for r in fold_results)
    print(f"  All folds positive: {all_pos}")

    RESULTS['phase3'] = fold_results

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  FINAL SUMMARY (with spread=${SPREAD})")
    print(f"{'='*100}")

    p1_at_050 = [r for r in phase1 if r['spread'] == SPREAD]
    p1_at_050.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Phase 1 ranking (no frequency reduction):")
    for i, r in enumerate(p1_at_050, 1):
        print(f"    {i}. {r['config']:<40} Sharpe={r['sharpe']:>6.2f}, PnL=${r['pnl']:>9.0f}, N={r['n']}")

    print(f"\n  Phase 2 best: {best['label']}")
    print(f"    Sharpe={best['sharpe']:.2f}, PnL=${best['pnl']:.0f}, N={best['n']}, $/trade=${best['ppt']:.2f}")

    if fold_results:
        print(f"\n  Phase 3 K-Fold: Avg Sharpe={avg_best:.2f}, wins {best_wins}/{len(fold_results)}")

    out = Path("data/cost_adjusted_results.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  TOTAL: {elapsed/60:.0f} min ({elapsed/3600:.1f}h)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
