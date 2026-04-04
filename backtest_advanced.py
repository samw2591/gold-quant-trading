"""
Advanced Backtest Suite
========================
4 tests on the C12 configuration (Trail 0.8/0.25 + SL 3.5 + TP 5.0 + ADX 18):

  Test 1: Monte Carlo Simulation       (~5 min)
  Test 2: K-Fold Cross Validation      (~1.5 hours)
  Test 3: Regime Adaptive Parameters   (~1.5 hours)
  Test 4: New Parameter Exploration    (~4-5 hours)
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from strategies.signals import prepare_indicators
from backtest import DataBundle, run_variant
from backtest.runner import (
    C12_KWARGS,
    load_m15,
    load_h1_aligned,
    add_atr_percentile,
    prepare_indicators_custom,
    H1_CSV_PATH,
)

RESULTS = {}


def run_engine(m15_df, h1_df, label, **kwargs):
    stats = run_variant(DataBundle(m15_df, h1_df), label, verbose=False, **kwargs)
    trades = stats['_trades']
    eq = stats['_equity_curve']
    return stats, trades, eq


# ══════════════════════════════════════════════════════════════
# TEST 1: Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════

def test_monte_carlo(m15_df, h1_df, n_simulations=1000):
    print("\n" + "=" * 90)
    print("  TEST 1: Monte Carlo Simulation")
    print(f"  {n_simulations} random permutations of trade order")
    print("=" * 90)

    print("\n  Running C12 to get trade list...", flush=True)
    t0 = time.time()
    stats, trades, eq = run_engine(m15_df, h1_df, "C12 for MC", **C12_KWARGS)
    print(f"  Got {len(trades)} trades in {time.time()-t0:.0f}s")
    print(f"  Original: Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, MaxDD=${stats['max_dd']:.0f}")

    trade_pnls = [t.pnl for t in trades]

    sharpes = []
    max_dds = []
    final_pnls = []

    print(f"\n  Running {n_simulations} simulations...", flush=True)
    t0 = time.time()
    for i in range(n_simulations):
        shuffled = np.random.permutation(trade_pnls)
        equity = [config.CAPITAL]
        for pnl in shuffled:
            equity.append(equity[-1] + pnl)
        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = eq_arr - peak
        max_dd = abs(dd.min())

        daily_chunks = [shuffled[j:j+6] for j in range(0, len(shuffled), 6)]
        daily_pnl = [sum(chunk) for chunk in daily_chunks]
        if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
            sh = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
        else:
            sh = 0

        sharpes.append(sh)
        max_dds.append(max_dd)
        final_pnls.append(eq_arr[-1] - config.CAPITAL)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_simulations}...", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    pnl_arr = np.array(final_pnls)
    dd_arr = np.array(max_dds)
    sh_arr = np.array(sharpes)

    print(f"\n  --- Monte Carlo Results ({n_simulations} simulations) ---")
    print(f"  {'Metric':<20} {'P5':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P95':>10} {'Mean':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, arr in [("PnL ($)", pnl_arr), ("MaxDD ($)", dd_arr), ("Sharpe", sh_arr)]:
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
        fmt = ".0f" if name != "Sharpe" else ".2f"
        print(f"  {name:<20} {p5:>10{fmt}} {p25:>10{fmt}} {p50:>10{fmt}} "
              f"{p75:>10{fmt}} {p95:>10{fmt}} {np.mean(arr):>10{fmt}}")

    prob_positive = np.mean(pnl_arr > 0) * 100
    prob_dd_under_1k = np.mean(dd_arr < 1000) * 100
    prob_dd_under_2k = np.mean(dd_arr < 2000) * 100
    worst_dd = np.max(dd_arr)
    worst_pnl = np.min(pnl_arr)

    print(f"\n  Probability PnL > 0: {prob_positive:.1f}%")
    print(f"  Probability MaxDD < $1,000: {prob_dd_under_1k:.1f}%")
    print(f"  Probability MaxDD < $2,000: {prob_dd_under_2k:.1f}%")
    print(f"  Worst case PnL: ${worst_pnl:.0f}")
    print(f"  Worst case MaxDD: ${worst_dd:.0f}")

    result = {
        'n_simulations': n_simulations,
        'n_trades': len(trades),
        'original_sharpe': float(stats['sharpe']),
        'original_pnl': float(stats['total_pnl']),
        'original_maxdd': float(stats['max_dd']),
        'pnl_percentiles': {f'p{p}': float(np.percentile(pnl_arr, p)) for p in [5, 25, 50, 75, 95]},
        'dd_percentiles': {f'p{p}': float(np.percentile(dd_arr, p)) for p in [5, 25, 50, 75, 95]},
        'sharpe_percentiles': {f'p{p}': float(np.percentile(sh_arr, p)) for p in [5, 25, 50, 75, 95]},
        'prob_positive_pnl': float(prob_positive),
        'prob_dd_under_1k': float(prob_dd_under_1k),
        'worst_dd': float(worst_dd),
        'worst_pnl': float(worst_pnl),
    }
    RESULTS['monte_carlo'] = result
    return result


# ══════════════════════════════════════════════════════════════
# TEST 2: K-Fold Cross Validation
# ══════════════════════════════════════════════════════════════

def run_on_window(data: DataBundle, start: str, end: str, label, **kwargs):
    sliced = data.slice(start, end)
    if len(sliced.m15_df) < 1000 or len(sliced.h1_df) < 200:
        return None
    return run_variant(sliced, label, verbose=False, **kwargs)


CONFIGS = {
    "Baseline": {},
    "Trail-only": {"trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25},
    "C12 Full": C12_KWARGS,
}


def test_kfold(data: DataBundle):
    print("\n" + "=" * 90)
    print("  TEST 2: K-Fold Cross Validation (6 folds)")
    print("=" * 90)

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    results = []
    for fold_name, test_start, test_end in folds:
        print(f"\n  {fold_name}: Test={test_start}~{test_end}", flush=True)

        for cfg_name, cfg_kwargs in CONFIGS.items():
            stats = run_on_window(
                data, test_start, test_end,
                f"{cfg_name} [{fold_name}]", **cfg_kwargs,
            )
            if stats:
                results.append({
                    'fold': fold_name, 'test_start': test_start, 'test_end': test_end,
                    'config': cfg_name,
                    'sharpe': float(stats['sharpe']), 'pnl': float(stats['total_pnl']),
                    'wr': float(stats['win_rate']), 'rr': float(stats['rr']),
                    'dd': float(stats['max_dd']), 'n': int(stats['n']),
                })

    print(f"\n  {'Fold':<8} {'Config':<15} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'RR':>5} {'MaxDD':>8} {'N':>6}")
    print(f"  {'-'*8} {'-'*15} {'-'*8} {'-'*10} {'-'*6} {'-'*5} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['fold']:<8} {r['config']:<15} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} "
              f"{r['wr']:>5.1f}% {r['rr']:>4.2f} ${r['dd']:>7.0f} {r['n']:>6}")

    print(f"\n  --- Cross-Fold Summary ---")
    for cfg_name in CONFIGS:
        rows = [r for r in results if r['config'] == cfg_name]
        if not rows:
            continue
        sharpes = [r['sharpe'] for r in rows]
        pnls = [r['pnl'] for r in rows]
        all_pos = all(s > 0 for s in sharpes)
        print(f"  {cfg_name:<15}: Avg Sharpe={np.mean(sharpes):.2f}, Std={np.std(sharpes):.2f}, "
              f"Min={min(sharpes):.2f}, Avg PnL=${np.mean(pnls):.0f}, "
              f"All folds positive={all_pos}")

    c12_rows = [r for r in results if r['config'] == 'C12 Full']
    base_rows = [r for r in results if r['config'] == 'Baseline']
    if c12_rows and base_rows:
        c12_wins = sum(1 for c, b in zip(c12_rows, base_rows) if c['sharpe'] > b['sharpe'])
        print(f"\n  C12 beats Baseline in {c12_wins}/{len(c12_rows)} folds")

    RESULTS['kfold'] = results
    return results


# ══════════════════════════════════════════════════════════════
# TEST 3: Regime Adaptive Parameters
# ══════════════════════════════════════════════════════════════

def test_regime_adaptive(m15_df, h1_df):
    print("\n" + "=" * 90)
    print("  TEST 3: Regime Adaptive Parameters")
    print("=" * 90)

    variants = [
        ("V1: C12 Fixed (baseline)", None, C12_KWARGS),
        ("V2: Adaptive SL", {
            'low': {'sl': 2.5}, 'normal': {'sl': 3.5}, 'high': {'sl': 4.5},
        }, C12_KWARGS),
        ("V3: Adaptive Trail", {
            'low': {'trail_act': 1.0, 'trail_dist': 0.35},
            'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
            'high': {'trail_act': 0.6, 'trail_dist': 0.20},
        }, C12_KWARGS),
        ("V4: Adaptive SL+Trail", {
            'low': {'sl': 2.5, 'trail_act': 1.0, 'trail_dist': 0.35},
            'normal': {'sl': 3.5, 'trail_act': 0.8, 'trail_dist': 0.25},
            'high': {'sl': 4.5, 'trail_act': 0.6, 'trail_dist': 0.20},
        }, C12_KWARGS),
        ("V5: Strategy Switch", {
            'low': {'disable_keltner': True},
            'high': {'disable_rsi': True},
        }, C12_KWARGS),
    ]

    bundle = DataBundle(m15_df, h1_df)
    results = []
    for i, (label, regime_cfg, base_kwargs) in enumerate(variants, 1):
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        t0 = time.time()
        if regime_cfg is None:
            stats, _, _ = run_engine(m15_df, h1_df, label, **base_kwargs)
        else:
            stats = run_variant(
                bundle, label, verbose=False,
                regime_config=regime_cfg, **base_kwargs,
            )
        elapsed = time.time() - t0
        print(f"    {stats['n']} trades, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"MaxDD=${stats['max_dd']:.0f}, {elapsed:.0f}s")
        results.append({
            'label': label, 'sharpe': float(stats['sharpe']),
            'pnl': float(stats['total_pnl']), 'wr': float(stats['win_rate']),
            'rr': float(stats['rr']), 'dd': float(stats['max_dd']),
            'dd_pct': float(stats['max_dd_pct']), 'n': int(stats['n']),
        })

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    base_sh = [r['sharpe'] for r in results if 'Fixed' in r['label']]
    base_sh = base_sh[0] if base_sh else 0

    print(f"\n  {'Rank':<5} {'Variant':<30} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} {'MaxDD':>8} {'DD%':>6} {'WR%':>6}")
    print(f"  {'-'*5} {'-'*30} {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
    for rank, r in enumerate(results, 1):
        ds = r['sharpe'] - base_sh
        print(f"  {rank:<5} {r['label']:<30} {r['sharpe']:>8.2f} {ds:>+5.2f} ${r['pnl']:>9.0f} "
              f"${r['dd']:>7.0f} {r['dd_pct']:>5.1f}% {r['wr']:>5.1f}%")

    RESULTS['regime_adaptive'] = results
    return results


# ══════════════════════════════════════════════════════════════
# TEST 4: New Parameter Exploration
# ══════════════════════════════════════════════════════════════

def run_param_explore(raw: DataBundle, label, kc_ema=20, kc_mult=1.5, ema_trend=100,
                      rsi_buy=15, rsi_sell=85, **engine_kwargs):
    t0 = time.time()
    print(f"    Preparing indicators (KC_ema={kc_ema}, KC_mult={kc_mult}, EMA={ema_trend})...", end='', flush=True)
    m15_df = prepare_indicators_custom(
        raw.m15_df, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=ema_trend,
    )
    h1_df = prepare_indicators_custom(
        raw.h1_df, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=ema_trend,
    )
    h1_df = add_atr_percentile(h1_df)
    print(f" done ({time.time()-t0:.0f}s)")

    data = DataBundle(m15_df, h1_df)
    stats = run_variant(
        data, label, verbose=False,
        rsi_buy_threshold=rsi_buy, rsi_sell_threshold=rsi_sell, **engine_kwargs,
    )
    elapsed = time.time() - t0
    print(f"    {stats['n']} trades (H1={stats['h1_entries']}, M15={stats['m15_entries']}), "
          f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, {elapsed:.0f}s")
    return stats


def test_new_params(raw: DataBundle):
    print("\n" + "=" * 90)
    print("  TEST 4: New Parameter Exploration")
    print("=" * 90)

    variants = []

    variants.append(("N00: C12 Baseline", {"kc_ema": 20, "kc_mult": 1.5, "ema_trend": 100,
                                         "rsi_buy": 15, "rsi_sell": 85}))

    for mult in [1.0, 1.25, 1.75, 2.0]:
        variants.append((f"N01: KC mult {mult}", {"kc_ema": 20, "kc_mult": mult, "ema_trend": 100,
                                                  "rsi_buy": 15, "rsi_sell": 85}))

    for buy, sell in [(10, 90), (20, 80), (25, 75)]:
        variants.append((f"N02: RSI {buy}/{sell}", {"kc_ema": 20, "kc_mult": 1.5, "ema_trend": 100,
                                                     "rsi_buy": buy, "rsi_sell": sell}))

    for ema in [50, 150, 200]:
        variants.append((f"N03: EMA{ema}", {"kc_ema": 20, "kc_mult": 1.5, "ema_trend": ema,
                                             "rsi_buy": 15, "rsi_sell": 85}))

    for kc_ema in [15, 25, 30]:
        variants.append((f"N04: KC_EMA{kc_ema}", {"kc_ema": kc_ema, "kc_mult": 1.5, "ema_trend": 100,
                                                  "rsi_buy": 15, "rsi_sell": 85}))

    variants.append(("N05: KC1.25+RSI20/80", {"kc_ema": 20, "kc_mult": 1.25, "ema_trend": 100,
                                                "rsi_buy": 20, "rsi_sell": 80}))
    variants.append(("N06: KC1.75+EMA50", {"kc_ema": 20, "kc_mult": 1.75, "ema_trend": 50,
                                           "rsi_buy": 15, "rsi_sell": 85}))

    results = []
    for i, (label, params) in enumerate(variants, 1):
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        kw = {k: v for k, v in params.items() if k in ('kc_ema', 'kc_mult', 'ema_trend', 'rsi_buy', 'rsi_sell')}
        stats = run_param_explore(raw, label, **kw, **C12_KWARGS)
        results.append({
            'label': label, 'sharpe': float(stats['sharpe']),
            'pnl': float(stats['total_pnl']), 'wr': float(stats['win_rate']),
            'rr': float(stats['rr']), 'dd': float(stats['max_dd']),
            'dd_pct': float(stats['max_dd_pct']), 'n': int(stats['n']),
            'h1_entries': int(stats.get('h1_entries', 0)),
            'm15_entries': int(stats.get('m15_entries', 0)),
            'params': params,
        })

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    base_sh = [r['sharpe'] for r in results if 'Baseline' in r['label']]
    base_sh = base_sh[0] if base_sh else 0

    print(f"\n  {'Rank':<5} {'Variant':<30} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} "
          f"{'MaxDD':>8} {'DD%':>6} {'WR%':>6} {'RR':>5} {'H1':>5} {'M15':>6}")
    print(f"  {'-'*5} {'-'*30} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*6}")
    for rank, r in enumerate(results, 1):
        ds = r['sharpe'] - base_sh
        print(f"  {rank:<5} {r['label']:<30} {r['n']:>6} {r['sharpe']:>8.2f} {ds:>+5.2f} ${r['pnl']:>9.0f} "
              f"${r['dd']:>7.0f} {r['dd_pct']:>5.1f}% {r['wr']:>5.1f}% {r['rr']:>4.2f} "
              f"{r['h1_entries']:>5} {r['m15_entries']:>6}")

    RESULTS['new_params'] = results
    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def save_results():
    out = Path("data/advanced_backtest_results.json")
    safe = {}
    for k, v in RESULTS.items():
        if isinstance(v, list):
            safe[k] = [{sk: (int(sv) if isinstance(sv, np.integer) else
                             float(sv) if isinstance(sv, np.floating) else sv)
                         for sk, sv in item.items()} for item in v]
        elif isinstance(v, dict):
            safe[k] = {sk: (int(sv) if isinstance(sv, np.integer) else
                            float(sv) if isinstance(sv, np.floating) else sv)
                        for sk, sv in v.items()}
        else:
            safe[k] = v
    out.write_text(json.dumps(safe, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")


def main():
    t_start = time.time()
    print("=" * 90)
    print("  ADVANCED BACKTEST SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  C12 Config: {C12_KWARGS}")
    print("=" * 90)

    print("\nLoading data...")
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    print("Preparing default indicators...")
    m15_df = prepare_indicators(m15_raw)
    h1_df = prepare_indicators(h1_raw)
    h1_df = add_atr_percentile(h1_df)

    print(f"M15: {len(m15_df)} bars, H1: {len(h1_df)} bars\n")

    full_data = DataBundle(m15_df, h1_df)
    raw_bundle = DataBundle(m15_raw, h1_raw)

    test_monte_carlo(m15_df, h1_df)
    save_results()

    test_kfold(full_data)
    save_results()

    test_regime_adaptive(m15_df, h1_df)
    save_results()

    test_new_params(raw_bundle)
    save_results()

    elapsed = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"  TOTAL ELAPSED: {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
