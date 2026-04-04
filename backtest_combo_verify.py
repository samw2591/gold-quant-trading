"""
Combo Verification Backtest
============================
验证三个升级方案的叠加效果是否存在交互损耗。

配置矩阵：
  A: C12 基准（当前实盘）
  B: C12 + 自适应 Trail (V3)
  C: C12 + KC 1.25 + KC_EMA30
  D: C12 + 自适应 Trail + KC 1.25 + KC_EMA30（三合一）

同时对冠军方案做 K-Fold 交叉验证（6 折）确认泛化能力。
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from strategies.signals import get_orb_strategy, calc_rsi, calc_adx
import strategies.signals as signals_mod
from backtest_m15 import (
    load_m15, load_h1_aligned, build_h1_lookup,
    calc_stats, M15_CSV_PATH, H1_CSV_PATH,
)
from backtest_round2 import Round2Engine
from backtest_advanced import (
    C12_KWARGS, RegimeEngine, prepare_indicators_custom,
)

V3_REGIME = {
    'low': {'trail_act': 1.0, 'trail_dist': 0.35},
    'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
    'high': {'trail_act': 0.6, 'trail_dist': 0.20},
}


def add_atr_percentile(h1_df):
    if 'atr_percentile' not in h1_df.columns:
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        h1_df['atr_percentile'] = h1_df['atr_percentile'].fillna(0.5)
    return h1_df


def run_fixed(m15_df, h1_df, label, **kwargs):
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = Round2Engine(m15_df, h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    return stats


def run_regime(m15_df, h1_df, label, regime_config, **kwargs):
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = RegimeEngine(m15_df, h1_df, regime_config=regime_config, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    return stats


def prep_custom(m15_raw, h1_raw, kc_ema=20, kc_mult=1.5):
    m15 = prepare_indicators_custom(m15_raw, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=100)
    h1 = prepare_indicators_custom(h1_raw, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=100)
    h1 = add_atr_percentile(h1)
    return m15, h1


def run_on_window(m15_df, h1_df, start, end, label, regime_config=None, **kwargs):
    m15 = m15_df[(m15_df.index >= start) & (m15_df.index < end)]
    h1 = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
    if len(m15) < 1000 or len(h1) < 200:
        return None
    if regime_config:
        return run_regime(m15, h1, label, regime_config, **kwargs)
    else:
        return run_fixed(m15, h1, label, **kwargs)


def print_results(results, title=""):
    if title:
        print(f"\n  --- {title} ---")
    base_sh = results[0]['sharpe'] if results else 0
    print(f"\n  {'Rank':<5} {'Config':<45} {'N':>6} {'Sharpe':>8} {'dSh':>6} "
          f"{'PnL':>10} {'MaxDD':>8} {'DD%':>6} {'WR%':>6} {'RR':>5}")
    print(f"  {'-'*5} {'-'*45} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*5}")
    for rank, r in enumerate(results, 1):
        ds = r['sharpe'] - base_sh
        print(f"  {rank:<5} {r['label']:<45} {r['n']:>6} {r['sharpe']:>8.2f} {ds:>+5.2f} "
              f"${r['pnl']:>9.0f} ${r['dd']:>7.0f} {r['dd_pct']:>5.1f}% "
              f"{r['wr']:>5.1f}% {r['rr']:>4.2f}")


def main():
    t_start = time.time()
    print("=" * 100)
    print("  COMBO VERIFICATION BACKTEST")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    from strategies.signals import prepare_indicators
    m15_default = prepare_indicators(m15_raw)
    h1_default = prepare_indicators(h1_raw)
    h1_default = add_atr_percentile(h1_default)

    print("  Preparing KC1.25+EMA30 indicators...", flush=True)
    m15_custom, h1_custom = prep_custom(m15_raw, h1_raw, kc_ema=30, kc_mult=1.25)

    print(f"  M15: {len(m15_default)} bars, H1: {len(h1_default)} bars\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Full-period comparison (4 configs)
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  PHASE 1: Full Period Comparison (2015-2026)")
    print("=" * 100)

    configs = [
        ("A: C12 Baseline", m15_default, h1_default, None),
        ("B: C12 + Adaptive Trail", m15_default, h1_default, V3_REGIME),
        ("C: C12 + KC1.25 + KC_EMA30", m15_custom, h1_custom, None),
        ("D: Combo (Trail+KC1.25+EMA30)", m15_custom, h1_custom, V3_REGIME),
    ]

    results = []
    for i, (label, m15, h1, regime) in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] {label}", flush=True)
        t0 = time.time()
        if regime:
            stats = run_regime(m15, h1, label, regime, **C12_KWARGS)
        else:
            stats = run_fixed(m15, h1, label, **C12_KWARGS)
        elapsed = time.time() - t0
        print(f"    {stats['n']} trades (H1={stats['h1_entries']}, M15={stats['m15_entries']}), "
              f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"MaxDD=${stats['max_dd']:.0f}, {elapsed:.0f}s")
        results.append({
            'label': label,
            'sharpe': float(stats['sharpe']),
            'pnl': float(stats['total_pnl']),
            'wr': float(stats['win_rate']),
            'rr': float(stats['rr']),
            'dd': float(stats['max_dd']),
            'dd_pct': float(stats['max_dd_pct']),
            'n': int(stats['n']),
        })

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print_results(results, "Full Period Ranking")

    a_sh = next(r['sharpe'] for r in results if r['label'].startswith('A:'))
    d_result = next(r for r in results if r['label'].startswith('D:'))
    b_result = next(r for r in results if r['label'].startswith('B:'))
    c_result = next(r for r in results if r['label'].startswith('C:'))

    expected_additive = (b_result['sharpe'] - a_sh) + (c_result['sharpe'] - a_sh)
    actual_combo = d_result['sharpe'] - a_sh
    synergy = actual_combo - expected_additive

    print(f"\n  --- Synergy Analysis ---")
    print(f"  B delta (Adaptive Trail):    +{b_result['sharpe'] - a_sh:.2f}")
    print(f"  C delta (KC1.25+EMA30):     +{c_result['sharpe'] - a_sh:.2f}")
    print(f"  Expected additive:           +{expected_additive:.2f}")
    print(f"  D actual combo:              +{actual_combo:.2f}")
    print(f"  Synergy (D - B - C + A):     {synergy:+.2f}")
    if synergy >= 0:
        print(f"  [OK] positive/neutral synergy, safe to implement")
    else:
        print(f"  [WARN] negative synergy ({synergy:.2f}), combo < sum of parts")
        if actual_combo > 0:
            print(f"    but combo still beats baseline (+{actual_combo:.2f}), still viable")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: K-Fold validation for the winner
    # ══════════════════════════════════════════════════════════════
    winner = results[0]
    print(f"\n{'='*100}")
    print(f"  PHASE 2: K-Fold Validation for Winner ({winner['label']})")
    print(f"{'='*100}")

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    fold_configs = {
        "A: C12 Baseline": (m15_default, h1_default, None),
        "D: Combo": (m15_custom, h1_custom, V3_REGIME),
    }

    fold_results = []
    for fold_name, test_start, test_end in folds:
        ts = pd.Timestamp(test_start, tz='UTC')
        te = pd.Timestamp(test_end, tz='UTC')
        print(f"\n  {fold_name}: {test_start} ~ {test_end}", flush=True)

        for cfg_name, (m15, h1, regime) in fold_configs.items():
            stats = run_on_window(m15, h1, ts, te,
                                  f"{cfg_name} [{fold_name}]",
                                  regime_config=regime, **C12_KWARGS)
            if stats:
                fold_results.append({
                    'fold': fold_name, 'test_start': test_start, 'test_end': test_end,
                    'config': cfg_name,
                    'sharpe': float(stats['sharpe']),
                    'pnl': float(stats['total_pnl']),
                    'wr': float(stats['win_rate']),
                    'rr': float(stats['rr']),
                    'dd': float(stats['max_dd']),
                    'n': int(stats['n']),
                })

    print(f"\n  {'Fold':<8} {'Config':<20} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'MaxDD':>8} {'N':>6}")
    print(f"  {'-'*8} {'-'*20} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*6}")
    for r in fold_results:
        print(f"  {r['fold']:<8} {r['config']:<20} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} "
              f"{r['wr']:>5.1f}% ${r['dd']:>7.0f} {r['n']:>6}")

    for cfg_name in fold_configs:
        rows = [r for r in fold_results if r['config'] == cfg_name]
        if not rows:
            continue
        sharpes = [r['sharpe'] for r in rows]
        print(f"\n  {cfg_name}: Avg Sharpe={np.mean(sharpes):.2f}, Std={np.std(sharpes):.2f}, "
              f"Min={min(sharpes):.2f}, All positive={all(s > 0 for s in sharpes)}")

    a_folds = [r for r in fold_results if r['config'].startswith('A:')]
    d_folds = [r for r in fold_results if r['config'].startswith('D:')]
    if a_folds and d_folds:
        wins = sum(1 for a, d in zip(a_folds, d_folds) if d['sharpe'] > a['sharpe'])
        print(f"\n  Combo beats Baseline in {wins}/{len(a_folds)} folds")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Year-by-year breakdown
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  PHASE 3: Year-by-Year Breakdown (A vs D)")
    print(f"{'='*100}")

    years = range(2015, 2027)
    yearly_results = []
    for year in years:
        ts = pd.Timestamp(f'{year}-01-01', tz='UTC')
        te = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
        for cfg_name, (m15, h1, regime) in fold_configs.items():
            stats = run_on_window(m15, h1, ts, te, f"{cfg_name} {year}",
                                  regime_config=regime, **C12_KWARGS)
            if stats:
                yearly_results.append({
                    'year': year, 'config': cfg_name,
                    'sharpe': float(stats['sharpe']),
                    'pnl': float(stats['total_pnl']),
                    'n': int(stats['n']),
                })

    print(f"\n  {'Year':<6} {'A Sharpe':>10} {'A PnL':>10} {'D Sharpe':>10} {'D PnL':>10} {'Winner':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    d_wins = 0
    for year in years:
        a_row = next((r for r in yearly_results if r['year'] == year and r['config'].startswith('A:')), None)
        d_row = next((r for r in yearly_results if r['year'] == year and r['config'].startswith('D:')), None)
        if a_row and d_row:
            w = 'D' if d_row['sharpe'] > a_row['sharpe'] else 'A'
            if w == 'D':
                d_wins += 1
            print(f"  {year:<6} {a_row['sharpe']:>10.2f} ${a_row['pnl']:>9.0f} "
                  f"{d_row['sharpe']:>10.2f} ${d_row['pnl']:>9.0f} {w:>8}")
    total_years = len([y for y in years if any(r['year'] == y for r in yearly_results)])
    print(f"\n  Combo wins {d_wins}/{total_years} years")

    # ══════════════════════════════════════════════════════════════
    # Final verdict
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  FINAL VERDICT")
    print(f"{'='*100}")
    print(f"  A (C12 Baseline): Sharpe={a_sh:.2f}")
    print(f"  D (Combo):        Sharpe={d_result['sharpe']:.2f} ({d_result['sharpe']-a_sh:+.2f})")
    print(f"  Synergy:          {synergy:+.2f} ({'positive' if synergy >= 0 else 'negative'})")

    a_fold_avg = np.mean([r['sharpe'] for r in fold_results if r['config'].startswith('A:')])
    d_fold_avg = np.mean([r['sharpe'] for r in fold_results if r['config'].startswith('D:')])
    print(f"  K-Fold Avg:     A={a_fold_avg:.2f}, D={d_fold_avg:.2f}")

    if d_result['sharpe'] > a_sh and synergy >= -0.1 and d_fold_avg > a_fold_avg:
        print(f"\n  [VERDICT] IMPLEMENT combo - all checks passed")
    elif d_result['sharpe'] > a_sh and synergy < -0.1:
        print(f"\n  [VERDICT] PARTIAL - negative synergy, implement best single upgrade only")
    else:
        print(f"\n  [VERDICT] HOLD - combo does not beat baseline")

    # Save results
    out = Path("data/combo_verify_results.json")
    save_data = {
        'full_period': [{k: v for k, v in r.items()} for r in results],
        'kfold': fold_results,
        'yearly': yearly_results,
        'synergy': float(synergy),
        'verdict': 'IMPLEMENT' if d_result['sharpe'] > a_sh and synergy >= -0.1 else 'HOLD',
    }
    out.write_text(json.dumps(save_data, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  TOTAL: {elapsed/60:.0f} min ({elapsed:.0f}s)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
