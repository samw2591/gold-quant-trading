"""
Fine-grained kc_only threshold scan (0.55 ~ 0.72)
===================================================
Confirms whether the 0.60→0.65 jump is a real regime boundary
or an artifact of coarse threshold spacing.

Uses Combo config (KC1.25+EMA30 + V3_REGIME) with spread=$0.50.
Also runs K-Fold on the best to check stability.
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
    C12_KWARGS, V3_REGIME,
    add_atr_percentile, prepare_indicators_custom,
    load_m15, load_h1_aligned, H1_CSV_PATH,
)

SPREAD = 0.50


def _reset():
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False


def run_adaptive(m15, h1, label, choppy_th, kc_th, **kw):
    _reset()
    engine = BacktestEngine(
        m15, h1,
        intraday_adaptive=True,
        choppy_threshold=choppy_th,
        kc_only_threshold=kc_th,
        regime_config=V3_REGIME,
        spread_cost=SPREAD,
        label=label,
        **kw,
    )
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['n'] = len(trades)
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    stats['skipped_choppy'] = engine.skipped_choppy
    stats['skipped_neutral_m15'] = engine.skipped_neutral_m15
    return stats


def run_baseline(m15, h1, label, **kw):
    _reset()
    engine = BacktestEngine(
        m15, h1,
        regime_config=V3_REGIME,
        spread_cost=SPREAD,
        label=label,
        **kw,
    )
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['n'] = len(trades)
    return stats


def main():
    t_start = time.time()
    print("=" * 100)
    print("  FINE-GRAINED KC_ONLY THRESHOLD SCAN")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spread: ${SPREAD}, Config: Combo (KC1.25+EMA30 + V3_REGIME)")
    print("=" * 100)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15 = prepare_indicators_custom(m15_raw, kc_ema=30, kc_mult=1.25)
    h1 = prepare_indicators_custom(h1_raw, kc_ema=30, kc_mult=1.25)
    h1 = add_atr_percentile(h1)
    print(f"  M15: {len(m15)} bars, H1: {len(h1)} bars\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Fine scan kc_only from 0.55 to 0.72 (step 0.01)
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  PHASE 1: kc_only threshold sweep (choppy fixed at 0.35)")
    print("=" * 100)

    thresholds = [round(0.55 + i * 0.01, 2) for i in range(18)]
    results = []

    for kc_th in thresholds:
        label = f"kc_only={kc_th:.2f}"
        print(f"\n  [{label}]", flush=True)
        t0 = time.time()
        stats = run_adaptive(m15, h1, label, choppy_th=0.35, kc_th=kc_th, **C12_KWARGS)
        elapsed = time.time() - t0
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"    N={stats['n']:>6}, Sharpe={stats['sharpe']:>6.2f}, PnL=${stats['total_pnl']:>9.0f}, "
              f"$/trade={ppt:>6.2f}, skip_m15={stats['skipped_neutral_m15']}, {elapsed:.0f}s")
        results.append({
            'kc_only': kc_th, 'n': stats['n'],
            'sharpe': float(stats['sharpe']),
            'pnl': float(stats['total_pnl']),
            'wr': float(stats['win_rate']),
            'dd': float(stats['max_dd']),
            'dd_pct': float(stats['max_dd_pct']),
            'ppt': float(ppt),
            'h1': stats['h1_entries'], 'm15': stats['m15_entries'],
            'skip_m15': stats['skipped_neutral_m15'],
        })

    print(f"\n  {'kc_only':>8} {'N':>7} {'Sharpe':>8} {'PnL':>10} {'MaxDD':>8} {'WR%':>6} "
          f"{'$/trade':>8} {'H1':>6} {'M15':>7} {'skip_m15':>9}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*6} "
          f"{'-'*8} {'-'*6} {'-'*7} {'-'*9}")
    for r in results:
        marker = " **" if r['sharpe'] == max(x['sharpe'] for x in results) else ""
        print(f"  {r['kc_only']:>8.2f} {r['n']:>7} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} "
              f"${r['dd']:>7.0f} {r['wr']:>5.1f}% ${r['ppt']:>7.2f} "
              f"{r['h1']:>6} {r['m15']:>7} {r['skip_m15']:>9}{marker}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Also scan choppy threshold with best kc_only
    # ══════════════════════════════════════════════════════════════
    best_kc = max(results, key=lambda x: x['sharpe'])['kc_only']
    print(f"\n{'='*100}")
    print(f"  PHASE 2: choppy threshold sweep (kc_only fixed at {best_kc:.2f})")
    print(f"{'='*100}")

    choppy_thresholds = [round(0.25 + i * 0.05, 2) for i in range(7)]
    phase2 = []
    for ch_th in choppy_thresholds:
        label = f"choppy={ch_th:.2f}/kc={best_kc:.2f}"
        print(f"\n  [{label}]", flush=True)
        t0 = time.time()
        stats = run_adaptive(m15, h1, label, choppy_th=ch_th, kc_th=best_kc, **C12_KWARGS)
        elapsed = time.time() - t0
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"    N={stats['n']:>6}, Sharpe={stats['sharpe']:>6.2f}, PnL=${stats['total_pnl']:>9.0f}, "
              f"$/trade={ppt:>6.2f}, {elapsed:.0f}s")
        phase2.append({
            'choppy': ch_th, 'kc_only': best_kc,
            'n': stats['n'], 'sharpe': float(stats['sharpe']),
            'pnl': float(stats['total_pnl']), 'ppt': float(ppt),
        })

    print(f"\n  {'choppy':>8} {'N':>7} {'Sharpe':>8} {'PnL':>10} {'$/trade':>8}")
    print(f"  {'-'*8} {'-'*7} {'-'*8} {'-'*10} {'-'*8}")
    for r in phase2:
        print(f"  {r['choppy']:>8.2f} {r['n']:>7} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} ${r['ppt']:>7.2f}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: K-Fold on best + neighbors
    # ══════════════════════════════════════════════════════════════
    test_kc_values = sorted(set([best_kc - 0.02, best_kc - 0.01, best_kc, best_kc + 0.01, best_kc + 0.02]))
    test_kc_values = [round(v, 2) for v in test_kc_values if 0.55 <= v <= 0.72]

    print(f"\n{'='*100}")
    print(f"  PHASE 3: K-Fold validation for kc_only = {test_kc_values}")
    print(f"{'='*100}")

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    bundle = DataBundle(m15, h1)
    kfold_results = {}

    for kc_val in test_kc_values:
        kfold_results[kc_val] = []
        for fold_name, start, end in folds:
            fd = bundle.slice(start, end)
            if len(fd.m15_df) < 1000:
                continue
            print(f"  kc={kc_val:.2f} {fold_name}...", end='', flush=True)
            stats = run_adaptive(fd.m15_df, fd.h1_df, f"kc={kc_val} {fold_name}",
                                 choppy_th=0.35, kc_th=kc_val, **C12_KWARGS)
            print(f" Sharpe={stats['sharpe']:.2f}, N={stats['n']}")
            kfold_results[kc_val].append({
                'fold': fold_name, 'sharpe': float(stats['sharpe']),
                'pnl': float(stats['total_pnl']), 'n': stats['n'],
            })

    print(f"\n  {'kc_only':>8}", end='')
    for f in folds:
        print(f"  {f[0]:>8}", end='')
    print(f"  {'Avg':>8} {'Std':>6} {'Min':>6} {'AllPos':>7}")

    print(f"  {'-'*8}", end='')
    for _ in folds:
        print(f"  {'-'*8}", end='')
    print(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*7}")

    for kc_val in test_kc_values:
        rows = kfold_results[kc_val]
        sharpes = [r['sharpe'] for r in rows]
        print(f"  {kc_val:>8.2f}", end='')
        for r in rows:
            print(f"  {r['sharpe']:>8.2f}", end='')
        avg = np.mean(sharpes)
        std = np.std(sharpes)
        mn = min(sharpes)
        ap = all(s > 0 for s in sharpes)
        print(f"  {avg:>8.2f} {std:>6.2f} {mn:>6.2f} {'Yes' if ap else 'No':>7}")

    # Save
    out = Path("data/threshold_fine_scan.json")
    save_data = {
        'phase1': results,
        'phase2': phase2,
        'phase3': {str(k): v for k, v in kfold_results.items()},
    }
    out.write_text(json.dumps(save_data, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  TOTAL: {elapsed/60:.0f} min ({elapsed/3600:.1f}h)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
