"""
Overfitting Detection Suite
============================
Three independent tests to check if C12 (T0.8/0.25+SL3.5+TP5+ADX18) is overfit:

1. Walk-Forward Analysis: Train on 4yr, test on 2yr, roll forward
2. Half-Sample Stability: Odd years vs Even years
3. Regime Stress Test: Performance in distinct market regimes
4. Parameter Sensitivity: Small perturbations around the "optimal" point
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from strategies.signals import prepare_indicators, get_orb_strategy
import strategies.signals as signals_mod
from backtest_m15 import (
    load_m15, load_h1_aligned, calc_stats,
    M15_CSV_PATH, H1_CSV_PATH,
)
from backtest_round2 import Round2Engine


CONFIGS = {
    "Baseline": {},
    "Trail-only (0.8/0.25)": {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25},
    "C12 Full Combo": {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25,
        "sl_atr_mult": 3.5, "tp_atr_mult": 5.0, "keltner_adx_threshold": 18},
}


def run_on_window(m15_full, h1_full, start, end, label, **kwargs):
    """Run a config on a date-filtered sub-window."""
    m15 = m15_full[(m15_full.index >= start) & (m15_full.index < end)]
    h1 = h1_full[(h1_full.index >= start) & (h1_full.index < end)]
    if len(m15) < 1000 or len(h1) < 200:
        return None

    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    engine = Round2Engine(m15, h1, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['n_trades'] = stats['n']
    stats['start'] = str(start)
    stats['end'] = str(end)
    return stats


def test_walk_forward(m15_df, h1_df):
    """Walk-Forward: Train 4yr, Test 2yr, rolling windows."""
    print("\n" + "=" * 90)
    print("  TEST 1: Walk-Forward Analysis (4yr train / 2yr test)")
    print("=" * 90)

    windows = [
        ("2015-2018 → 2019-2020", "2015-01-01", "2019-01-01", "2019-01-01", "2021-01-01"),
        ("2017-2020 → 2021-2022", "2017-01-01", "2021-01-01", "2021-01-01", "2023-01-01"),
        ("2019-2022 → 2023-2024", "2019-01-01", "2023-01-01", "2023-01-01", "2025-01-01"),
        ("2021-2024 → 2025-2026", "2021-01-01", "2025-01-01", "2025-01-01", "2026-04-01"),
    ]

    results = []
    for name, tr_s, tr_e, te_s, te_e in windows:
        print(f"\n  Window: {name}")
        for cfg_name, cfg_kwargs in CONFIGS.items():
            train = run_on_window(m15_df, h1_df,
                pd.Timestamp(tr_s, tz='UTC'), pd.Timestamp(tr_e, tz='UTC'),
                f"{cfg_name} [TRAIN]", **cfg_kwargs)
            test = run_on_window(m15_df, h1_df,
                pd.Timestamp(te_s, tz='UTC'), pd.Timestamp(te_e, tz='UTC'),
                f"{cfg_name} [TEST]", **cfg_kwargs)
            if train and test:
                results.append({
                    'window': name, 'config': cfg_name,
                    'train_sharpe': train['sharpe'], 'test_sharpe': test['sharpe'],
                    'train_pnl': train['total_pnl'], 'test_pnl': test['total_pnl'],
                    'train_wr': train['win_rate'], 'test_wr': test['win_rate'],
                    'train_rr': train['rr'], 'test_rr': test['rr'],
                    'train_n': train['n'], 'test_n': test['n'],
                    'train_dd': train['max_dd'], 'test_dd': test['max_dd'],
                })

    print(f"\n  {'Window':<25} {'Config':<25} {'Train Sh':>9} {'Test Sh':>9} {'Decay%':>7} "
          f"{'Train PnL':>10} {'Test PnL':>10} {'Train WR':>9} {'Test WR':>9}")
    print(f"  {'-'*25} {'-'*25} {'-'*9} {'-'*9} {'-'*7} {'-'*10} {'-'*10} {'-'*9} {'-'*9}")
    for r in results:
        decay = ((r['test_sharpe'] - r['train_sharpe']) / abs(r['train_sharpe']) * 100
                 if r['train_sharpe'] != 0 else 0)
        print(f"  {r['window']:<25} {r['config']:<25} {r['train_sharpe']:>9.2f} {r['test_sharpe']:>9.2f} "
              f"{decay:>+6.0f}% ${r['train_pnl']:>9.0f} ${r['test_pnl']:>9.0f} "
              f"{r['train_wr']:>8.1f}% {r['test_wr']:>8.1f}%")

    # Overfit ratio: how often does test Sharpe < 50% of train Sharpe?
    print(f"\n  --- Overfit Summary ---")
    for cfg_name in CONFIGS:
        cfg_rows = [r for r in results if r['config'] == cfg_name]
        if not cfg_rows:
            continue
        train_sharpes = [r['train_sharpe'] for r in cfg_rows]
        test_sharpes = [r['test_sharpe'] for r in cfg_rows]
        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)
        decay_pct = (avg_test - avg_train) / abs(avg_train) * 100 if avg_train != 0 else 0
        all_positive = all(s > 0 for s in test_sharpes)
        print(f"  {cfg_name:<25}: Avg Train Sharpe={avg_train:.2f}, Avg Test Sharpe={avg_test:.2f}, "
              f"Decay={decay_pct:+.0f}%, OOS all positive={all_positive}")

    return results


def test_half_sample(m15_df, h1_df):
    """Split into odd years (2015,2017,...) vs even years (2016,2018,...) and compare."""
    print("\n" + "=" * 90)
    print("  TEST 2: Half-Sample Stability (Odd vs Even years)")
    print("=" * 90)

    results = []
    for cfg_name, cfg_kwargs in CONFIGS.items():
        for year in range(2015, 2026):
            start = pd.Timestamp(f"{year}-01-01", tz='UTC')
            end = pd.Timestamp(f"{year+1}-01-01", tz='UTC')
            stats = run_on_window(m15_df, h1_df, start, end,
                                  f"{cfg_name} {year}", **cfg_kwargs)
            if stats:
                results.append({
                    'config': cfg_name, 'year': year,
                    'group': 'odd' if year % 2 == 1 else 'even',
                    'sharpe': stats['sharpe'], 'pnl': stats['total_pnl'],
                    'wr': stats['win_rate'], 'rr': stats['rr'],
                    'n': stats['n'], 'dd': stats['max_dd'],
                })

    print(f"\n  {'Config':<25} {'Group':<6} {'Years':>6} {'Avg Sharpe':>11} {'Avg PnL':>10} "
          f"{'Avg WR':>7} {'Avg RR':>7} {'Min Sharpe':>11} {'Yrs>0':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*11} {'-'*10} {'-'*7} {'-'*7} {'-'*11} {'-'*6}")
    for cfg_name in CONFIGS:
        for group in ['odd', 'even']:
            rows = [r for r in results if r['config'] == cfg_name and r['group'] == group]
            if not rows:
                continue
            avg_sh = np.mean([r['sharpe'] for r in rows])
            avg_pnl = np.mean([r['pnl'] for r in rows])
            avg_wr = np.mean([r['wr'] for r in rows])
            avg_rr = np.mean([r['rr'] for r in rows])
            min_sh = min([r['sharpe'] for r in rows])
            yrs_pos = sum(1 for r in rows if r['pnl'] > 0)
            print(f"  {cfg_name:<25} {group:<6} {len(rows):>6} {avg_sh:>11.2f} ${avg_pnl:>9.0f} "
                  f"{avg_wr:>6.1f}% {avg_rr:>6.2f} {min_sh:>11.2f} {yrs_pos:>4}/{len(rows)}")

    # Year-by-year detail
    print(f"\n  --- Year-by-Year Detail ---")
    print(f"  {'Year':<6}", end='')
    for cfg_name in CONFIGS:
        print(f" {cfg_name[:18]:>20}", end='')
    print()
    for year in range(2015, 2026):
        print(f"  {year:<6}", end='')
        for cfg_name in CONFIGS:
            row = [r for r in results if r['config'] == cfg_name and r['year'] == year]
            if row:
                sh = row[0]['sharpe']
                pnl = row[0]['pnl']
                print(f" Sh{sh:>5.2f} ${pnl:>7.0f}", end='')
            else:
                print(f" {'N/A':>20}", end='')
        print()

    return results


def test_parameter_sensitivity(m15_df, h1_df):
    """Perturb each parameter ±1 step from C12 optimal. If Sharpe drops sharply, it's a fragile peak."""
    print("\n" + "=" * 90)
    print("  TEST 3: Parameter Sensitivity (±1 step perturbation from C12)")
    print("=" * 90)

    base = {"trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25,
            "sl_atr_mult": 3.5, "tp_atr_mult": 5.0, "keltner_adx_threshold": 18}

    perturbations = [
        ("C12 Optimal",           base),
        ("Trail Act 0.6",         {**base, "trailing_activate_atr": 0.6}),
        ("Trail Act 1.0",         {**base, "trailing_activate_atr": 1.0}),
        ("Trail Dist 0.20",       {**base, "trailing_distance_atr": 0.20}),
        ("Trail Dist 0.30",       {**base, "trailing_distance_atr": 0.30}),
        ("SL 3.0ATR",             {**base, "sl_atr_mult": 3.0}),
        ("SL 4.0ATR",             {**base, "sl_atr_mult": 4.0}),
        ("TP 4.0ATR",             {**base, "tp_atr_mult": 4.0}),
        ("TP 6.0ATR",             {**base, "tp_atr_mult": 6.0}),
        ("ADX 16",                {**base, "keltner_adx_threshold": 16}),
        ("ADX 20",                {**base, "keltner_adx_threshold": 20}),
    ]

    start = pd.Timestamp('2015-01-01', tz='UTC')
    end = pd.Timestamp('2026-04-01', tz='UTC')

    results = []
    for label, kwargs in perturbations:
        print(f"  Running: {label}...", flush=True)
        stats = run_on_window(m15_df, h1_df, start, end, label, **kwargs)
        if stats:
            results.append({
                'label': label, 'sharpe': stats['sharpe'],
                'pnl': stats['total_pnl'], 'wr': stats['win_rate'],
                'rr': stats['rr'], 'dd': stats['max_dd'],
                'dd_pct': stats['max_dd_pct'], 'n': stats['n'],
            })

    base_sharpe = results[0]['sharpe'] if results else 1
    print(f"\n  {'Perturbation':<25} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} {'MaxDD':>8} "
          f"{'DD%':>6} {'WR%':>6} {'RR':>5}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*5}")
    for r in results:
        ds = r['sharpe'] - base_sharpe
        print(f"  {r['label']:<25} {r['sharpe']:>8.2f} {ds:>+5.2f} ${r['pnl']:>9.0f} "
              f"${r['dd']:>7.0f} {r['dd_pct']:>5.1f}% {r['wr']:>5.1f}% {r['rr']:>4.2f}")

    sharpes = [r['sharpe'] for r in results]
    print(f"\n  Sensitivity: min={min(sharpes):.2f}, max={max(sharpes):.2f}, "
          f"range={max(sharpes)-min(sharpes):.2f}, "
          f"all > 1.5? {'YES' if all(s > 1.5 for s in sharpes) else 'NO'}, "
          f"all > 2.0? {'YES' if all(s > 2.0 for s in sharpes) else 'NO'}")

    return results


def test_regime_stress(m15_df, h1_df):
    """Test in distinct market regimes."""
    print("\n" + "=" * 90)
    print("  TEST 4: Regime Stress Test")
    print("=" * 90)

    regimes = [
        ("2015-2016 Low Vol Range",   "2015-01-01", "2017-01-01"),
        ("2017-2018 Gradual Rise",    "2017-01-01", "2019-01-01"),
        ("2019-2020 COVID Spike",     "2019-01-01", "2021-01-01"),
        ("2021-2022 Rate Hike Crash", "2021-01-01", "2023-01-01"),
        ("2023-2024 Recovery Rally",  "2023-01-01", "2025-01-01"),
        ("2025-2026 Trump 2.0",       "2025-01-01", "2026-04-01"),
    ]

    results = []
    for regime_name, start, end in regimes:
        print(f"\n  Regime: {regime_name}")
        for cfg_name, cfg_kwargs in CONFIGS.items():
            stats = run_on_window(m15_df, h1_df,
                pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC'),
                f"{cfg_name}", **cfg_kwargs)
            if stats:
                results.append({
                    'regime': regime_name, 'config': cfg_name,
                    'sharpe': stats['sharpe'], 'pnl': stats['total_pnl'],
                    'wr': stats['win_rate'], 'rr': stats['rr'],
                    'dd': stats['max_dd'], 'n': stats['n'],
                })

    print(f"\n  {'Regime':<30} {'Config':<25} {'Sharpe':>8} {'PnL':>10} {'WR%':>6} {'RR':>5} {'MaxDD':>8}")
    print(f"  {'-'*30} {'-'*25} {'-'*8} {'-'*10} {'-'*6} {'-'*5} {'-'*8}")
    for r in results:
        print(f"  {r['regime']:<30} {r['config']:<25} {r['sharpe']:>8.2f} ${r['pnl']:>9.0f} "
              f"{r['wr']:>5.1f}% {r['rr']:>4.2f} ${r['dd']:>7.0f}")

    # Check: does C12 beat baseline in every regime?
    print(f"\n  --- C12 vs Baseline by Regime ---")
    for regime_name, _, _ in regimes:
        base = [r for r in results if r['regime'] == regime_name and r['config'] == 'Baseline']
        c12 = [r for r in results if r['regime'] == regime_name and r['config'] == 'C12 Full Combo']
        if base and c12:
            b, c = base[0], c12[0]
            verdict = "C12 WINS" if c['sharpe'] > b['sharpe'] else "BASELINE WINS"
            print(f"  {regime_name:<30}: Baseline Sh={b['sharpe']:.2f}, C12 Sh={c['sharpe']:.2f} → {verdict}")

    return results


def main():
    t_start = time.time()
    print("=" * 90)
    print("  OVERFITTING DETECTION SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Configs under test:")
    for name in CONFIGS:
        print(f"    - {name}")
    print("=" * 90)

    print("\nLoading data...")
    m15_df = load_m15()
    m15_df = m15_df[m15_df.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_df.index[0])

    print("Preparing indicators...")
    m15_df = prepare_indicators(m15_df)
    h1_df = prepare_indicators(h1_df)

    if 'atr_percentile' not in h1_df.columns:
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        h1_df['atr_percentile'] = h1_df['atr_percentile'].fillna(0.5)

    print(f"M15: {len(m15_df)} bars, H1: {len(h1_df)} bars\n")

    all_results = {}

    all_results['walk_forward'] = test_walk_forward(m15_df, h1_df)
    all_results['half_sample'] = test_half_sample(m15_df, h1_df)
    all_results['sensitivity'] = test_parameter_sensitivity(m15_df, h1_df)
    all_results['regime'] = test_regime_stress(m15_df, h1_df)

    elapsed = time.time() - t_start
    print(f"\n{'='*90}")
    print(f"  TOTAL ELAPSED: {elapsed/60:.0f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}")

    # Final verdict
    print(f"\n  === OVERFIT VERDICT ===")
    wf = all_results['walk_forward']
    c12_wf = [r for r in wf if r['config'] == 'C12 Full Combo']
    if c12_wf:
        oos_sharpes = [r['test_sharpe'] for r in c12_wf]
        oos_all_pos = all(s > 0 for s in oos_sharpes)
        avg_decay = np.mean([(r['test_sharpe']-r['train_sharpe'])/abs(r['train_sharpe'])*100
                             for r in c12_wf if r['train_sharpe'] != 0])
        print(f"  Walk-Forward OOS Sharpes: {[round(s,2) for s in oos_sharpes]}")
        print(f"  All OOS positive? {'YES' if oos_all_pos else 'NO'}")
        print(f"  Avg Sharpe Decay: {avg_decay:+.0f}%")
        if oos_all_pos and avg_decay > -50:
            print(f"  → LOW overfit risk (OOS consistently positive, decay < 50%)")
        elif oos_all_pos:
            print(f"  → MODERATE overfit risk (OOS positive but significant decay)")
        else:
            print(f"  → HIGH overfit risk (some OOS periods negative)")

    sens = all_results['sensitivity']
    if sens:
        sens_sharpes = [r['sharpe'] for r in sens]
        sens_range = max(sens_sharpes) - min(sens_sharpes)
        print(f"  Parameter Sensitivity Range: {sens_range:.2f}")
        if sens_range < 0.5:
            print(f"  → ROBUST (small perturbations have little effect)")
        elif sens_range < 1.0:
            print(f"  → MODERATE sensitivity")
        else:
            print(f"  → FRAGILE (performance highly sensitive to parameters)")

    out = Path("data/overfit_test_results.json")
    safe_results = {}
    for k, v in all_results.items():
        safe = []
        for r in v:
            sr = {}
            for rk, rv in r.items():
                if isinstance(rv, (np.integer,)):
                    sr[rk] = int(rv)
                elif isinstance(rv, (np.floating,)):
                    sr[rk] = float(rv)
                else:
                    sr[rk] = rv
            safe.append(sr)
        safe_results[k] = safe
    out.write_text(json.dumps(safe_results, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
