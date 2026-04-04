"""
Intraday Adaptive Backtest (Phase 5)
=====================================
Test the IntradayTrendMeter in backtest: compute trend_score in real-time
from today's H1 bars (no look-ahead), gate entries based on regime.

Compares:
  A: Baseline (all entries, spread=$0.50)
  B: Adaptive (choppy=skip all, neutral=H1 only, trending=all)
  C: Adaptive strict (choppy=skip, neutral=skip M15, trending=all, threshold sweep)

Also runs K-Fold and threshold sensitivity analysis.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.signals import get_orb_strategy, prepare_indicators
import strategies.signals as signals_mod

from backtest import BacktestEngine, DataBundle, calc_stats
from backtest.runner import (
    C12_KWARGS,
    V3_REGIME,
    add_atr_percentile,
    prepare_indicators_custom,
    load_m15,
    load_h1_aligned,
    M15_CSV_PATH,
    H1_CSV_PATH,
)

RESULTS = {}
SPREAD = 0.50


def _reset_signal_state():
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False


def run_adaptive(
    m15_df,
    h1_df,
    label,
    choppy_threshold=0.35,
    kc_only_threshold=0.60,
    regime_config=None,
    spread=SPREAD,
    **kwargs,
):
    _reset_signal_state()
    engine = BacktestEngine(
        m15_df,
        h1_df,
        intraday_adaptive=True,
        choppy_threshold=choppy_threshold,
        kc_only_threshold=kc_only_threshold,
        regime_config=regime_config,
        label=label,
        spread_cost=spread,
        **kwargs,
    )
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    stats['skipped_choppy'] = engine.skipped_choppy
    stats['skipped_neutral_m15'] = engine.skipped_neutral_m15
    return stats


def run_baseline(m15_df, h1_df, label, regime_config=None, spread=SPREAD, **kwargs):
    _reset_signal_state()
    if regime_config:
        engine = BacktestEngine(
            m15_df,
            h1_df,
            regime_config=regime_config,
            label=label,
            spread_cost=spread,
            **kwargs,
        )
    else:
        engine = BacktestEngine(
            m15_df,
            h1_df,
            label=label,
            spread_cost=spread,
            **kwargs,
        )
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    return stats


def stats_to_row(stats, label=None):
    return {
        'label': label or stats.get('label', ''),
        'n': int(stats['n']),
        'sharpe': float(stats['sharpe']),
        'pnl': float(stats['total_pnl']),
        'wr': float(stats['win_rate']),
        'rr': float(stats['rr']),
        'dd': float(stats['max_dd']),
        'dd_pct': float(stats['max_dd_pct']),
        'h1_n': int(stats.get('h1_entries', 0)),
        'm15_n': int(stats.get('m15_entries', 0)),
        'skip_choppy': int(stats.get('skipped_choppy', 0)),
        'skip_m15': int(stats.get('skipped_neutral_m15', 0)),
    }


def print_table(results, title=""):
    if title:
        print(f"\n  --- {title} ---")
    print(f"\n  {'Rank':<5} {'Config':<55} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'WR%':>6} {'$/trade':>8}")
    print(f"  {'-'*5} {'-'*55} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
    for rank, r in enumerate(results, 1):
        ppt = r['pnl'] / r['n'] if r['n'] > 0 else 0
        print(f"  {rank:<5} {r['label']:<55} {r['n']:>6} {r['sharpe']:>8.2f} "
              f"${r['pnl']:>9.0f} ${r['dd']:>7.0f} {r['wr']:>5.1f}% ${ppt:>7.2f}")


def main():
    t_start = time.time()
    print("=" * 100)
    print("  INTRADAY ADAPTIVE BACKTEST (Phase 5)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spread: ${SPREAD}")
    print("=" * 100)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15_custom = prepare_indicators_custom(m15_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = prepare_indicators_custom(h1_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = add_atr_percentile(h1_custom)

    m15_default = prepare_indicators(m15_raw)
    h1_default = prepare_indicators(h1_raw)
    h1_default = add_atr_percentile(h1_default)

    print(f"  M15: {len(m15_custom)} bars, H1: {len(h1_custom)} bars\n")

    combo_bundle = DataBundle(m15_custom, h1_custom)

    print("=" * 100)
    print("  TEST 1: Baseline vs Intraday Adaptive (Combo config)")
    print("=" * 100)

    print("\n  [A: Combo Baseline (all entries)]", flush=True)
    t0 = time.time()
    stats_a = run_baseline(
        m15_custom,
        h1_custom,
        "A: Combo Baseline",
        regime_config=V3_REGIME,
        spread=SPREAD,
        **C12_KWARGS,
    )
    ppt_a = stats_a['total_pnl'] / stats_a['n'] if stats_a['n'] > 0 else 0
    print(f"    N={stats_a['n']}, Sharpe={stats_a['sharpe']:.2f}, PnL=${stats_a['total_pnl']:.0f}, "
          f"$/trade={ppt_a:.2f}, {time.time()-t0:.0f}s")

    print("\n  [B: Adaptive (choppy<0.35 skip, neutral<0.60 H1-only)]", flush=True)
    t0 = time.time()
    stats_b = run_adaptive(
        m15_custom,
        h1_custom,
        "B: Adaptive (0.35/0.60)",
        choppy_threshold=0.35,
        kc_only_threshold=0.60,
        regime_config=V3_REGIME,
        spread=SPREAD,
        **C12_KWARGS,
    )
    ppt_b = stats_b['total_pnl'] / stats_b['n'] if stats_b['n'] > 0 else 0
    print(f"    N={stats_b['n']}, Sharpe={stats_b['sharpe']:.2f}, PnL=${stats_b['total_pnl']:.0f}, "
          f"$/trade={ppt_b:.2f}, skipped_choppy={stats_b.get('skipped_choppy',0)}, "
          f"skipped_m15={stats_b.get('skipped_neutral_m15',0)}, {time.time()-t0:.0f}s")

    test1 = [stats_to_row(stats_a), stats_to_row(stats_b)]

    print("\n  [C: C12 Baseline]", flush=True)
    stats_c = run_baseline(
        m15_default,
        h1_default,
        "C: C12 Baseline",
        spread=SPREAD,
        **C12_KWARGS,
    )
    print(f"    N={stats_c['n']}, Sharpe={stats_c['sharpe']:.2f}, PnL=${stats_c['total_pnl']:.0f}")

    print("\n  [D: C12 Adaptive (0.35/0.60)]", flush=True)
    stats_d = run_adaptive(
        m15_default,
        h1_default,
        "D: C12 Adaptive (0.35/0.60)",
        choppy_threshold=0.35,
        kc_only_threshold=0.60,
        spread=SPREAD,
        **C12_KWARGS,
    )
    print(f"    N={stats_d['n']}, Sharpe={stats_d['sharpe']:.2f}, PnL=${stats_d['total_pnl']:.0f}")

    test1.extend([stats_to_row(stats_c), stats_to_row(stats_d)])
    print_table(test1, "Test 1: Baseline vs Adaptive")
    RESULTS['test1'] = test1

    print(f"\n{'='*100}")
    print(f"  TEST 2: Threshold Sensitivity (Combo config)")
    print(f"{'='*100}")

    threshold_configs = [
        (0.25, 0.50, "Low/Low (0.25/0.50)"),
        (0.30, 0.55, "Med-Low (0.30/0.55)"),
        (0.35, 0.60, "Default (0.35/0.60)"),
        (0.40, 0.65, "Med-High (0.40/0.65)"),
        (0.45, 0.70, "High (0.45/0.70)"),
        (0.35, 0.50, "Wide neutral (0.35/0.50)"),
        (0.35, 0.70, "Narrow neutral (0.35/0.70)"),
        (0.30, 0.60, "Lower choppy (0.30/0.60)"),
        (0.40, 0.60, "Higher choppy (0.40/0.60)"),
    ]

    test2 = []
    for choppy_th, kc_th, name in threshold_configs:
        label = f"Adaptive {name}"
        print(f"\n  [{label}]", flush=True)
        t0 = time.time()
        stats = run_adaptive(
            m15_custom,
            h1_custom,
            label,
            choppy_threshold=choppy_th,
            kc_only_threshold=kc_th,
            regime_config=V3_REGIME,
            spread=SPREAD,
            **C12_KWARGS,
        )
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"    N={stats['n']}, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"$/trade={ppt:.2f}, {time.time()-t0:.0f}s")
        row = stats_to_row(stats, label)
        row['choppy_th'] = choppy_th
        row['kc_th'] = kc_th
        test2.append(row)

    test2.sort(key=lambda x: x['sharpe'], reverse=True)
    print_table(test2, "Test 2: Threshold Sensitivity (sorted by Sharpe)")

    print(f"\n  {'Config':<35} {'Choppy':>7} {'KC_only':>7} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'$/trade':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for r in test2:
        ppt = r['pnl'] / r['n'] if r['n'] > 0 else 0
        print(f"  {r['label']:<35} {r['choppy_th']:>7.2f} {r['kc_th']:>7.2f} "
              f"{r['n']:>6} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} ${ppt:>7.2f}")

    RESULTS['test2'] = test2
    best_config = test2[0]
    best_choppy = best_config['choppy_th']
    best_kc = best_config['kc_th']
    print(f"\n  Best config: choppy<{best_choppy}, kc_only<{best_kc} "
          f"(Sharpe={best_config['sharpe']:.2f})")

    print(f"\n{'='*100}")
    print(f"  TEST 3: K-Fold Validation (best: choppy<{best_choppy}, kc_only<{best_kc})")
    print(f"{'='*100}")

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
        print(f"\n  {fold_name}: {test_start} ~ {test_end}", flush=True)

        fold_data = combo_bundle.slice(test_start, test_end)
        if len(fold_data.m15_df) < 1000:
            continue

        stats_best = run_adaptive(
            fold_data.m15_df,
            fold_data.h1_df,
            f"Adaptive [{fold_name}]",
            choppy_threshold=best_choppy,
            kc_only_threshold=best_kc,
            regime_config=V3_REGIME,
            spread=SPREAD,
            **C12_KWARGS,
        )
        stats_base = run_baseline(
            fold_data.m15_df,
            fold_data.h1_df,
            f"Baseline [{fold_name}]",
            regime_config=V3_REGIME,
            spread=SPREAD,
            **C12_KWARGS,
        )

        fold_results.append({
            'fold': fold_name,
            'adapt_sharpe': float(stats_best['sharpe']),
            'adapt_pnl': float(stats_best['total_pnl']),
            'adapt_n': int(stats_best['n']),
            'base_sharpe': float(stats_base['sharpe']),
            'base_pnl': float(stats_base['total_pnl']),
            'base_n': int(stats_base['n']),
        })

    print(f"\n  {'Fold':<8} {'Adapt Sh':>9} {'Adapt PnL':>10} {'Adapt N':>8} "
          f"{'Base Sh':>8} {'Base PnL':>10} {'Base N':>7} {'Winner':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
    adapt_wins = 0
    for r in fold_results:
        w = "Adapt" if r['adapt_sharpe'] > r['base_sharpe'] else "Base"
        if w == "Adapt":
            adapt_wins += 1
        print(f"  {r['fold']:<8} {r['adapt_sharpe']:>9.2f} ${r['adapt_pnl']:>9.0f} {r['adapt_n']:>8} "
              f"{r['base_sharpe']:>8.2f} ${r['base_pnl']:>9.0f} {r['base_n']:>7} {w:>8}")

    if fold_results:
        avg_adapt = np.mean([r['adapt_sharpe'] for r in fold_results])
        avg_base = np.mean([r['base_sharpe'] for r in fold_results])
        print(f"\n  Avg Sharpe: Adaptive={avg_adapt:.2f}, Baseline={avg_base:.2f}")
        print(f"  Adaptive wins {adapt_wins}/{len(fold_results)} folds")

    RESULTS['test3_kfold'] = fold_results

    print(f"\n{'='*100}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*100}")

    print(f"\n  Test 1: Combo Baseline Sharpe={test1[0]['sharpe']:.2f} vs "
          f"Adaptive Sharpe={test1[1]['sharpe']:.2f}")
    print(f"          C12 Baseline Sharpe={test1[2]['sharpe']:.2f} vs "
          f"C12 Adaptive Sharpe={test1[3]['sharpe']:.2f}")

    print(f"\n  Test 2: Best thresholds: choppy<{best_choppy}, kc_only<{best_kc}")
    print(f"          Sharpe={best_config['sharpe']:.2f}, N={best_config['n']}, "
          f"PnL=${best_config['pnl']:.0f}")

    if fold_results:
        print(f"\n  Test 3: K-Fold: Adaptive wins {adapt_wins}/{len(fold_results)}")
        print(f"          Avg Sharpe: Adaptive={avg_adapt:.2f} vs Baseline={avg_base:.2f}")

    out = Path("data/intraday_adaptive_results.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  TOTAL: {elapsed/60:.0f} min ({elapsed/3600:.1f}h)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
