"""
EUR/USD Deep Backtest with 11-year Dukascopy data
==================================================
Uses 2015-2026 H1 data for comprehensive strategy evaluation.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from run_eurusd_research import (
    load_forex_csv, prepare_forex_indicators, ForexBacktestEngine,
    calc_forex_stats, print_stats_table, V3_REGIME
)


def merge_dukascopy_csvs(directory, pattern="*.csv"):
    """Merge multiple Dukascopy CSV chunks into one DataFrame."""
    data_dir = Path(directory)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    chunks = []
    for f in files:
        df = pd.read_csv(str(f))
        chunks.append(df)
        print(f"  {f.name}: {len(df)} bars")

    merged = pd.concat(chunks, ignore_index=True)
    merged.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    merged.sort_values('timestamp', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def run_single(h1_df, label, **kwargs):
    """Run one backtest and return stats."""
    print(f"\n  [{label}]")
    engine = ForexBacktestEngine(h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_forex_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['_trades'] = trades
    stats['_equity'] = engine.equity_curve

    strat_map = {}
    for t in trades:
        s = t.strategy
        if s not in strat_map:
            strat_map[s] = {'n': 0, 'pnl': 0, 'wins': 0}
        strat_map[s]['n'] += 1
        strat_map[s]['pnl'] += t.pnl
        if t.pnl > 0:
            strat_map[s]['wins'] += 1
    stats['_strats'] = strat_map

    print(f"    {stats['n']} trades, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
          f"MaxDD=${stats['max_dd']:.0f} ({stats['dd_pct']:.1f}%), WR={stats['win_rate']:.1f}%")
    for sn, sv in strat_map.items():
        wr = sv['wins'] / sv['n'] * 100 if sv['n'] > 0 else 0
        print(f"      {sn}: {sv['n']} trades, PnL=${sv['pnl']:.0f}, WR={wr:.1f}%")

    return stats


def run_kfold(h1_raw, label_prefix, n_folds=6, **kwargs):
    """K-Fold cross validation by 2-year windows."""
    years = sorted(h1_raw.index.year.unique())
    folds = []
    fold_years = 2
    y = min(years)
    fold_num = 1
    while y <= max(years) and fold_num <= n_folds:
        y_end = y + fold_years
        start = pd.Timestamp(f"{y}-01-01", tz='UTC')
        end = pd.Timestamp(f"{y_end}-01-01", tz='UTC')
        fold_df = h1_raw[(h1_raw.index >= start) & (h1_raw.index < end)]
        if len(fold_df) > 500:
            folds.append((f"Fold{fold_num}({y}-{y_end-1})", fold_df))
            fold_num += 1
        y = y_end

    if not folds:
        print("  No valid folds!")
        return []

    results = []
    for fname, fdf in folds:
        fdf_prep = prepare_forex_indicators(fdf)
        stats = run_single(fdf_prep, f"{label_prefix}_{fname}", **kwargs)
        stats['fold'] = fname
        results.append(stats)
    return results


if __name__ == '__main__':
    start_time = datetime.now()
    print(f"{'=' * 80}")
    print(f"EUR/USD DEEP BACKTEST (11-YEAR DATA)")
    print(f"Started: {start_time}")
    print(f"{'=' * 80}")

    # ─── Load H1 data ───
    print(f"\n{'=' * 80}")
    print("LOADING H1 DATA (Dukascopy 2015-2026)")
    print(f"{'=' * 80}")

    h1_merged = merge_dukascopy_csvs("data/download/eurusd_h1", "eurusd-h1-bid-*.csv")
    print(f"\n  Total merged: {len(h1_merged)} H1 bars")

    # Save merged CSV
    merged_path = Path("data/download/eurusd-h1-merged-2015-2026.csv")
    h1_merged.to_csv(str(merged_path), index=False)
    print(f"  Saved: {merged_path}")

    # Load into standard format
    h1_raw = load_forex_csv(merged_path)
    print(f"  H1: {len(h1_raw)} bars, {h1_raw.index[0]} -> {h1_raw.index[-1]}")

    h1_df = prepare_forex_indicators(h1_raw)

    # ─── Phase 1: Full-period strategy comparison ───
    print(f"\n{'=' * 80}")
    print("PHASE 1: FULL-PERIOD STRATEGY COMPARISON (2015-2026)")
    print(f"{'=' * 80}")

    results = []

    # 1a: Gold-equivalent params
    results.append(run_single(h1_df, "KC Gold-params(ADX18,SL4.5,TP8)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 1b: Best from initial test (KC mult=2.0)
    h1_wide = prepare_forex_indicators(h1_raw, kc_mult=2.0)
    results.append(run_single(h1_wide, "KC mult=2.0(ADX18)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 1c: KC mult=2.0 + ADX>=20
    results.append(run_single(h1_wide, "KC mult=2.0(ADX20)",
        kc_adx_threshold=20, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 1d: KC mult=2.0 + ADX>=25
    results.append(run_single(h1_wide, "KC mult=2.0(ADX25)",
        kc_adx_threshold=25, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 1e: Wider SL
    results.append(run_single(h1_wide, "KC mult=2.0 SL6 TP10",
        kc_adx_threshold=18, sl_atr_mult=6.0, tp_atr_mult=10.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 1f: Tighter trail
    results.append(run_single(h1_wide, "KC mult=2.0 Trail(0.6/0.15)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.6, trail_distance_atr=0.15,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 1g: No regime
    results.append(run_single(h1_wide, "KC mult=2.0 No-regime",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=None, spread_pips=1.8, max_hold_bars=15))

    # 1h: Longer hold
    results.append(run_single(h1_wide, "KC mult=2.0 MaxHold=20",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=20))

    # 1i: No spread (theoretical)
    results.append(run_single(h1_wide, "KC mult=2.0 No-spread",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=0.0, max_hold_bars=15))

    # 1j: Larger lot sizing (0.05 lot target to match $50 risk better)
    results.append(run_single(h1_wide, "KC mult=2.0 risk$30",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15,
        risk_per_trade=30))

    print_stats_table(results, "FULL PERIOD STRATEGY COMPARISON (2015-2026)")

    # ─── Phase 2: Year-by-year breakdown of best config ───
    print(f"\n{'=' * 80}")
    print("PHASE 2: YEAR-BY-YEAR BREAKDOWN")
    print(f"{'=' * 80}")

    # Find best config (excluding no-spread)
    real_results = [r for r in results if 'No-spread' not in r['label']]
    best = max(real_results, key=lambda r: r['sharpe'])
    print(f"\n  Best config: {best['label']} (Sharpe={best['sharpe']:.2f})")

    best_trades = best.get('_trades', [])
    if best_trades:
        yearly = {}
        for t in best_trades:
            y = t.exit_time.year if hasattr(t.exit_time, 'year') else pd.Timestamp(t.exit_time).year
            if y not in yearly:
                yearly[y] = {'n': 0, 'pnl': 0, 'wins': 0, 'losses': 0}
            yearly[y]['n'] += 1
            yearly[y]['pnl'] += t.pnl
            if t.pnl > 0:
                yearly[y]['wins'] += 1
            else:
                yearly[y]['losses'] += 1

        print(f"\n  {'Year':<8} {'N':>6} {'PnL':>10} {'WR%':>8} {'$/trade':>8} {'Profit':>8}")
        print(f"  {'-'*52}")
        profitable_years = 0
        for y in sorted(yearly.keys()):
            v = yearly[y]
            wr = v['wins'] / v['n'] * 100 if v['n'] > 0 else 0
            avg = v['pnl'] / v['n'] if v['n'] > 0 else 0
            marker = "+" if v['pnl'] > 0 else "-"
            if v['pnl'] > 0:
                profitable_years += 1
            print(f"  {y:<8} {v['n']:>6} ${v['pnl']:>9.0f} {wr:>7.1f}% ${avg:>7.2f} {marker}")
        total_years = len(yearly)
        print(f"\n  Profitable years: {profitable_years}/{total_years} ({profitable_years/total_years*100:.0f}%)")

    # ─── Phase 3: K-Fold Cross-Validation ───
    print(f"\n{'=' * 80}")
    print("PHASE 3: K-FOLD CROSS-VALIDATION (6 x 2-year folds)")
    print(f"{'=' * 80}")

    # Determine best non-no-spread config's params
    # Re-run K-Fold for top 3 configs
    print(f"\n  Config A: KC Gold-params (mult=1.2, ADX18)")
    kfold_a = run_kfold(h1_raw, "KF-Gold",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15)
    print_stats_table(kfold_a, "K-Fold: KC Gold-params")

    print(f"\n  Config B: KC mult=2.0 (ADX18)")
    h1_raw_wide = prepare_forex_indicators(h1_raw, kc_mult=2.0)
    # Need raw for kfold, which calls prepare_forex_indicators itself
    # We need a different approach: pass kc_mult through
    kfold_results_b = []
    years = sorted(h1_raw.index.year.unique())
    fold_num = 1
    y = min(years)
    while y <= max(years) and fold_num <= 6:
        y_end = y + 2
        start = pd.Timestamp(f"{y}-01-01", tz='UTC')
        end = pd.Timestamp(f"{y_end}-01-01", tz='UTC')
        fold_df = h1_raw[(h1_raw.index >= start) & (h1_raw.index < end)]
        if len(fold_df) > 500:
            fname = f"Fold{fold_num}({y}-{y_end-1})"
            fdf_prep = prepare_forex_indicators(fold_df, kc_mult=2.0)
            stats = run_single(fdf_prep, f"KF-Wide_{fname}",
                kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
                trail_activate_atr=0.8, trail_distance_atr=0.25,
                regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15)
            stats['fold'] = fname
            kfold_results_b.append(stats)
            fold_num += 1
        y = y_end

    print_stats_table(kfold_results_b, "K-Fold: KC mult=2.0")

    # K-Fold summary
    print(f"\n{'=' * 80}")
    print("K-FOLD SUMMARY")
    print(f"{'=' * 80}")

    for name, kf in [("Gold-params(1.2)", kfold_a), ("KC mult=2.0", kfold_results_b)]:
        if not kf:
            continue
        sharpes = [r['sharpe'] for r in kf]
        pnls = [r['total_pnl'] for r in kf]
        positive = sum(1 for s in sharpes if s > 0)
        print(f"\n  {name}:")
        print(f"    Folds: {len(kf)}")
        print(f"    Sharpe: mean={np.mean(sharpes):.2f}, min={min(sharpes):.2f}, max={max(sharpes):.2f}")
        print(f"    PnL: mean=${np.mean(pnls):.0f}, total=${sum(pnls):.0f}")
        print(f"    Positive Sharpe folds: {positive}/{len(kf)}")

    # ─── Phase 4: ATR Study (full data) ───
    print(f"\n{'=' * 80}")
    print("PHASE 4: ATR STUDY WITH FULL DATA (2015-2026)")
    print(f"{'=' * 80}")

    atr = h1_df['ATR'].dropna() * 10000
    daily = h1_df.resample('D').agg({'High': 'max', 'Low': 'min'}).dropna()
    daily['range_pips'] = (daily['High'] - daily['Low']) * 10000
    daily = daily[daily['range_pips'] > 0]

    print(f"\n  H1 ATR (11yr): mean={atr.mean():.1f} pips, median={atr.median():.1f}")
    print(f"  Daily range (11yr): mean={daily['range_pips'].mean():.0f} pips, median={daily['range_pips'].median():.0f}")

    adx = h1_df['ADX'].dropna()
    print(f"  ADX (11yr): mean={adx.mean():.1f}, ADX>18={((adx>18).mean()*100):.1f}%, ADX>25={((adx>25).mean()*100):.1f}%")

    # ATR by year
    print(f"\n  ATR by Year:")
    for yr in sorted(h1_df.index.year.unique()):
        yr_df = h1_df[h1_df.index.year == yr]
        yr_atr = yr_df['ATR'].dropna() * 10000
        if len(yr_atr) > 0:
            print(f"    {yr}: mean={yr_atr.mean():.1f} pips, median={yr_atr.median():.1f}")

    # ─── Summary ───
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print(f"DEEP BACKTEST COMPLETE")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 80}")

    # Save results
    summary = {
        'timestamp': str(datetime.now()),
        'data_range': f"{h1_raw.index[0]} -> {h1_raw.index[-1]}",
        'total_bars': len(h1_raw),
        'results': [{k: v for k, v in r.items() if not k.startswith('_')} for r in results],
        'kfold_gold': [{k: v for k, v in r.items() if not k.startswith('_')} for r in kfold_a],
        'kfold_wide': [{k: v for k, v in r.items() if not k.startswith('_')} for r in kfold_results_b],
    }
    out_path = Path("data/eurusd_deep_results.json")
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
