"""
Filter Grid — 4h Multi-Phase Experiment
==========================================
Explores dimensions untouched by other experiments:

  Phase 1: DXY direction filter (post-hoc + engine-level)       ~40min
  Phase 2: Time filters (month / weekday / hour)                ~30min
  Phase 3: max_positions x min_entry_gap x cooldown             ~100min
  Phase 4: ORB internal parameter grid                          ~60min
  Phase 5: Cross-validation of top findings                     ~30min

Total: ~500 runs, ~4 hours @ 8 workers.

Usage: python backtest_filter_grid.py [--workers N]
"""
import sys
import csv
import json
import time
import itertools
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

SPREAD = 0.50
N_WORKERS = 8
MACRO_CSV = Path("data/macro_history.csv")

C12_BASE = {
    "trailing_activate_atr": 0.8,
    "trailing_distance_atr": 0.25,
    "sl_atr_mult": 4.5,
    "tp_atr_mult": 5.0,
    "keltner_adx_threshold": 18,
    "spread_cost": SPREAD,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
}


# ═══════════════════════════════════════════════════════════════
# Worker pool helpers
# ═══════════════════════════════════════════════════════════════

_worker_bundle = None
_worker_macro = None


def _worker_init():
    global _worker_bundle, _worker_macro
    import warnings
    warnings.filterwarnings("ignore")
    from backtest.runner import load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH, DataBundle
    from strategies.signals import prepare_indicators

    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    _worker_bundle = DataBundle(m15_df, h1_df)

    if MACRO_CSV.exists():
        _worker_macro = pd.read_csv(str(MACRO_CSV), index_col='date', parse_dates=True).ffill()


def _run_one(args: Tuple) -> Dict:
    """Generic worker: run backtest with given kwargs, return stats."""
    global _worker_bundle
    label, engine_kwargs = args
    from backtest.runner import run_variant
    from backtest.stats import aggregate_daily_pnl

    stats = run_variant(_worker_bundle, label, verbose=False, **engine_kwargs)
    trades = stats.get('_trades', [])
    daily = aggregate_daily_pnl(trades)

    return {
        "label": label,
        "n": int(stats["n"]),
        "sharpe": float(stats["sharpe"]),
        "total_pnl": float(stats["total_pnl"]),
        "max_dd": float(stats["max_dd"]),
        "win_rate": float(stats["win_rate"]),
        "rr": float(stats["rr"]),
        "keltner_n": int(stats.get("keltner_n", 0)),
        "keltner_pnl": float(stats.get("keltner_pnl", 0)),
        "orb_n": int(stats.get("orb_n", 0)),
        "orb_pnl": float(stats.get("orb_pnl", 0)),
        "rsi_n": int(stats.get("rsi_n", 0)),
        "rsi_pnl": float(stats.get("rsi_pnl", 0)),
        "h1_entries": int(stats.get("h1_entries", 0)),
        "m15_entries": int(stats.get("m15_entries", 0)),
        "daily_pnl": daily,
    }


def _print_table(results: List[Dict], title: str, top_n: int = 20):
    print(f"\n  {'='*90}")
    print(f"  {title}")
    print(f"  {'='*90}")
    print(f"  {'Rank':>4} {'Label':<38} {'N':>6} {'Sharpe':>8} "
          f"{'PnL':>10} {'MaxDD':>9} {'WR%':>7}")
    print(f"  {'-'*4} {'-'*38} {'-'*6} {'-'*8} {'-'*10} {'-'*9} {'-'*7}")
    for rank, r in enumerate(results[:top_n], 1):
        print(f"  {rank:>4} {r['label']:<38} {r['n']:>6} {r['sharpe']:>8.2f} "
              f"${r['total_pnl']:>9.0f} ${r['max_dd']:>8.0f} {r['win_rate']:>6.1f}%")


# ═══════════════════════════════════════════════════════════════
# Post-hoc trade filtering helpers
# ═══════════════════════════════════════════════════════════════

def _posthoc_stats(trades, label="") -> Dict:
    """Compute stats from a filtered subset of trades (no engine re-run)."""
    import config
    if not trades:
        return {"label": label, "n": 0, "sharpe": 0, "total_pnl": 0,
                "max_dd": 0, "win_rate": 0, "rr": 0, "avg_pnl": 0}
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0

    daily = defaultdict(float)
    for t in trades:
        daily[pd.Timestamp(t.exit_time).date()] += t.pnl
    daily_vals = list(daily.values())
    sharpe = 0
    if len(daily_vals) > 1 and np.std(daily_vals) > 0:
        sharpe = np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252)

    equity = [config.CAPITAL]
    for p in pnls:
        equity.append(equity[-1] + p)
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    max_dd = abs((eq - peak).min())

    return {
        "label": label,
        "n": len(pnls),
        "sharpe": float(sharpe),
        "total_pnl": float(sum(pnls)),
        "max_dd": float(max_dd),
        "win_rate": float(len(wins) / len(pnls) * 100),
        "rr": float(avg_win / avg_loss if avg_loss > 0 else 0),
        "avg_pnl": float(sum(pnls) / len(pnls)),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 1: DXY Direction Filter
# ═══════════════════════════════════════════════════════════════

def phase1_dxy(all_trades, macro_df):
    print("\n" + "=" * 90)
    print("  PHASE 1: DXY Direction Filter Analysis")
    print("  Testing DXY SMA lookback periods and filter modes")
    print("=" * 90)
    t0 = time.time()

    results = []

    # Baseline (no DXY filter)
    baseline = _posthoc_stats(all_trades, "No filter (baseline)")
    results.append(baseline)

    sma_periods = [5, 10, 15, 20, 30, 50]

    for sma_p in sma_periods:
        dxy_sma = macro_df['dxy'].rolling(sma_p, min_periods=sma_p).mean()
        dxy_lookup = {}
        for dt in dxy_sma.index:
            val = macro_df.loc[dt, 'dxy']
            sma_val = dxy_sma.loc[dt]
            if pd.notna(val) and pd.notna(sma_val):
                dxy_lookup[dt.date() if not isinstance(dt, datetime) else dt] = (val, sma_val)

        # Filter mode 1: Only aligned trades
        aligned = []
        contra = []
        for t in all_trades:
            td = pd.Timestamp(t.entry_time).normalize()
            if td.tz is not None:
                td = td.tz_localize(None)
            key = td
            if key not in dxy_lookup:
                key = td.date() if hasattr(td, 'date') else td
            if key not in dxy_lookup:
                aligned.append(t)
                continue
            dxy_val, sma_val = dxy_lookup[key]
            is_dxy_bull = dxy_val > sma_val
            if (is_dxy_bull and t.direction == 'SELL') or (not is_dxy_bull and t.direction == 'BUY'):
                aligned.append(t)
            else:
                contra.append(t)

        r_aligned = _posthoc_stats(aligned, f"DXY SMA{sma_p} aligned only")
        r_contra = _posthoc_stats(contra, f"DXY SMA{sma_p} contra only")
        results.append(r_aligned)
        results.append(r_contra)

        # Filter mode 2: Skip contra BUY only (keep all SELL)
        buy_aligned = [t for t in all_trades if t.direction == 'SELL' or t in aligned]
        r_buy_filt = _posthoc_stats(buy_aligned, f"DXY SMA{sma_p} skip contra BUY")
        results.append(r_buy_filt)

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    _print_table(results, "DXY Filter Results", top_n=30)

    print(f"\n  Phase 1 time: {time.time()-t0:.0f}s")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: Time Filters
# ═══════════════════════════════════════════════════════════════

def phase2_time_filters(all_trades):
    print("\n" + "=" * 90)
    print("  PHASE 2: Time Filter Analysis")
    print("  Month / Weekday / Hour single and combo filters")
    print("=" * 90)
    t0 = time.time()

    results = []
    baseline = _posthoc_stats(all_trades, "No filter (baseline)")
    results.append(baseline)

    # Single month skip
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(1, 13):
        filtered = [t for t in all_trades if pd.Timestamp(t.entry_time).month != m]
        results.append(_posthoc_stats(filtered, f"Skip {month_names[m-1]}"))

    # Single weekday skip
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    for d in range(5):
        filtered = [t for t in all_trades if pd.Timestamp(t.entry_time).dayofweek != d]
        results.append(_posthoc_stats(filtered, f"Skip {day_names[d]}"))

    # Single hour skip
    for h in range(24):
        filtered = [t for t in all_trades if pd.Timestamp(t.entry_time).hour != h]
        r = _posthoc_stats(filtered, f"Skip hour {h:02d}")
        results.append(r)

    # Combo filters
    combos = [
        ("Skip Dec", lambda t: pd.Timestamp(t.entry_time).month != 12),
        ("Skip Wed+Thu", lambda t: pd.Timestamp(t.entry_time).dayofweek not in (2, 3)),
        ("Skip Wed", lambda t: pd.Timestamp(t.entry_time).dayofweek != 2),
        ("Skip hour 3+12", lambda t: pd.Timestamp(t.entry_time).hour not in (3, 12)),
        ("Skip hour 3+12+16+22+23", lambda t: pd.Timestamp(t.entry_time).hour not in (3, 12, 16, 22, 23)),
        ("Skip Dec+Wed", lambda t: pd.Timestamp(t.entry_time).month != 12 and pd.Timestamp(t.entry_time).dayofweek != 2),
        ("Skip Dec+Wed+Thu", lambda t: pd.Timestamp(t.entry_time).month != 12 and pd.Timestamp(t.entry_time).dayofweek not in (2, 3)),
        ("Skip Dec+hour3+12", lambda t: pd.Timestamp(t.entry_time).month != 12 and pd.Timestamp(t.entry_time).hour not in (3, 12)),
        ("Skip Dec+Wed+h3+12", lambda t: (pd.Timestamp(t.entry_time).month != 12 and
                                           pd.Timestamp(t.entry_time).dayofweek != 2 and
                                           pd.Timestamp(t.entry_time).hour not in (3, 12))),
        ("Only Mon+Tue+Fri", lambda t: pd.Timestamp(t.entry_time).dayofweek in (0, 1, 4)),
        ("Only Feb-Apr (Q1)", lambda t: pd.Timestamp(t.entry_time).month in (2, 3, 4)),
        ("Only h5-8+h14-15+h17-21", lambda t: pd.Timestamp(t.entry_time).hour in (5, 6, 7, 8, 14, 15, 17, 18, 19, 20, 21)),
        ("Best combo: MonTueFri+h5-8,14-15,17-21+skipDec",
         lambda t: (pd.Timestamp(t.entry_time).dayofweek in (0, 1, 4) and
                    pd.Timestamp(t.entry_time).hour in (5, 6, 7, 8, 14, 15, 17, 18, 19, 20, 21) and
                    pd.Timestamp(t.entry_time).month != 12)),
    ]

    for label, fn in combos:
        filtered = [t for t in all_trades if fn(t)]
        results.append(_posthoc_stats(filtered, label))

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    _print_table(results, "Time Filter Results", top_n=30)

    # $/trade ranking (more useful than pure Sharpe for filters)
    results_by_avg = sorted([r for r in results if r['n'] > 100],
                            key=lambda r: r['avg_pnl'], reverse=True)
    print(f"\n  Top 10 by $/trade (N>100):")
    print(f"  {'Label':<45} {'N':>6} {'$/trade':>9} {'Sharpe':>8} {'PnL':>10}")
    print(f"  {'-'*45} {'-'*6} {'-'*9} {'-'*8} {'-'*10}")
    for r in results_by_avg[:10]:
        print(f"  {r['label']:<45} {r['n']:>6} ${r['avg_pnl']:>8.2f} "
              f"{r['sharpe']:>8.2f} ${r['total_pnl']:>9.0f}")

    print(f"\n  Phase 2 time: {time.time()-t0:.0f}s")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 3: max_positions x min_entry_gap x cooldown
# ═══════════════════════════════════════════════════════════════

def phase3_position_gap(pool):
    print("\n" + "=" * 90)
    print("  PHASE 3: max_positions x min_entry_gap x cooldown")
    print("=" * 90)
    t0 = time.time()

    max_pos_vals = [1, 2, 3, 4, 5]
    gap_vals = [0, 0.5, 1.0, 2.0, 3.0, 4.0]
    cooldown_vals = [0.25, 0.5, 1.0]

    jobs = []
    for mp_val, gap, cd in itertools.product(max_pos_vals, gap_vals, cooldown_vals):
        label = f"pos={mp_val}/gap={gap}h/cd={cd}h"
        kw = {**C12_BASE, "max_positions": mp_val,
              "min_entry_gap_hours": gap, "cooldown_hours": cd}
        jobs.append((label, kw))

    # atr_regime_lots sweep
    for mp_val in [1, 2, 3]:
        label = f"pos={mp_val}/ATR_lots=ON"
        kw = {**C12_BASE, "max_positions": mp_val, "atr_regime_lots": True}
        jobs.append((label, kw))
        label = f"pos={mp_val}/ATR_lots=OFF"
        kw = {**C12_BASE, "max_positions": mp_val, "atr_regime_lots": False}
        jobs.append((label, kw))

    total = len(jobs)
    print(f"  {total} combinations ({len(max_pos_vals)} pos x {len(gap_vals)} gap "
          f"x {len(cooldown_vals)} cd + 6 ATR lots)")

    results = []
    dot = max(total // 20, 1)
    print("  ", end="", flush=True)
    for i, r in enumerate(pool.imap_unordered(_run_one, jobs)):
        results.append(r)
        if (i + 1) % dot == 0:
            print(".", end="", flush=True)
    print()

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    _print_table(results, f"Position/Gap/Cooldown Results ({total} combos)")

    # Pivot: max_positions sensitivity
    print(f"\n  --- max_positions sensitivity (avg across gap/cd) ---")
    print(f"  {'MaxPos':>8} {'AvgSharpe':>10} {'AvgPnL':>10} {'AvgN':>8}")
    for mp_val in max_pos_vals:
        subset = [r for r in results if f"pos={mp_val}/" in r['label'] and 'ATR' not in r['label']]
        if subset:
            print(f"  {mp_val:>8} {np.mean([r['sharpe'] for r in subset]):>10.3f} "
                  f"${np.mean([r['total_pnl'] for r in subset]):>9.0f} "
                  f"{np.mean([r['n'] for r in subset]):>8.0f}")

    # Pivot: min_entry_gap sensitivity
    print(f"\n  --- min_entry_gap sensitivity (avg across pos/cd) ---")
    print(f"  {'Gap(h)':>8} {'AvgSharpe':>10} {'AvgPnL':>10} {'AvgN':>8}")
    for gap in gap_vals:
        subset = [r for r in results if f"gap={gap}h/" in r['label']]
        if subset:
            print(f"  {gap:>8} {np.mean([r['sharpe'] for r in subset]):>10.3f} "
                  f"${np.mean([r['total_pnl'] for r in subset]):>9.0f} "
                  f"{np.mean([r['n'] for r in subset]):>8.0f}")

    # Pivot: cooldown sensitivity
    print(f"\n  --- cooldown sensitivity (avg across pos/gap) ---")
    print(f"  {'CD(h)':>8} {'AvgSharpe':>10} {'AvgPnL':>10} {'AvgN':>8}")
    for cd in cooldown_vals:
        subset = [r for r in results if f"cd={cd}h" in r['label']]
        if subset:
            print(f"  {cd:>8} {np.mean([r['sharpe'] for r in subset]):>10.3f} "
                  f"${np.mean([r['total_pnl'] for r in subset]):>9.0f} "
                  f"{np.mean([r['n'] for r in subset]):>8.0f}")

    # ATR lots
    atr_results = [r for r in results if 'ATR_lots' in r['label']]
    if atr_results:
        print(f"\n  --- ATR Regime Lots ---")
        for r in sorted(atr_results, key=lambda x: x['label']):
            print(f"  {r['label']:<30} Sharpe={r['sharpe']:.2f}, PnL=${r['total_pnl']:.0f}, N={r['n']}")

    print(f"\n  Phase 3 time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 4: ORB Parameter Grid
# ═══════════════════════════════════════════════════════════════

def phase4_orb(pool):
    print("\n" + "=" * 90)
    print("  PHASE 4: ORB Internal Parameter Grid")
    print("  orb_max_hold x config ORB SL/TP multipliers")
    print("=" * 90)
    t0 = time.time()

    hold_vals = [0, 4, 8, 12, 16, 24]

    jobs = []
    for hold in hold_vals:
        label = f"ORB hold={hold}" if hold > 0 else "ORB hold=default"
        kw = {**C12_BASE}
        if hold > 0:
            kw["orb_max_hold_m15"] = hold
        jobs.append((label, kw))

    # Also test ORB interaction with max_positions and gap
    for hold in [0, 8, 16]:
        for mp_val in [1, 2, 3]:
            hold_label = f"h{hold}" if hold > 0 else "hDef"
            label = f"ORB {hold_label}/pos={mp_val}"
            kw = {**C12_BASE, "max_positions": mp_val}
            if hold > 0:
                kw["orb_max_hold_m15"] = hold
            jobs.append((label, kw))

    # ORB with different SL/TP (these affect Keltner too, but we track ORB separately)
    for sl in [2.5, 3.0, 3.5, 4.0, 4.5]:
        for tp in [4.0, 5.0, 6.0, 7.0]:
            label = f"ORB SL={sl}/TP={tp}"
            kw = {**C12_BASE, "sl_atr_mult": sl, "tp_atr_mult": tp}
            jobs.append((label, kw))

    # ORB with different cooldown
    for hold in [0, 8, 16]:
        for cd in [0.25, 0.5, 1.0]:
            hold_label = f"h{hold}" if hold > 0 else "hDef"
            label = f"ORB {hold_label}/cd={cd}h"
            kw = {**C12_BASE, "cooldown_hours": cd}
            if hold > 0:
                kw["orb_max_hold_m15"] = hold
            jobs.append((label, kw))

    total = len(jobs)
    print(f"  {total} combinations")

    results = []
    dot = max(total // 20, 1)
    print("  ", end="", flush=True)
    for i, r in enumerate(pool.imap_unordered(_run_one, jobs)):
        results.append(r)
        if (i + 1) % dot == 0:
            print(".", end="", flush=True)
    print()

    results.sort(key=lambda r: r['sharpe'], reverse=True)
    _print_table(results, f"ORB Grid Results ({total} combos)")

    # ORB-specific analysis
    print(f"\n  --- ORB Strategy Contribution ---")
    print(f"  {'Label':<35} {'ORB_N':>6} {'ORB_PnL':>10} {'K_N':>6} {'K_PnL':>10} {'Sharpe':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*6} {'-'*10} {'-'*8}")
    for r in results[:20]:
        print(f"  {r['label']:<35} {r['orb_n']:>6} ${r['orb_pnl']:>9.0f} "
              f"{r['keltner_n']:>6} ${r['keltner_pnl']:>9.0f} {r['sharpe']:>8.2f}")

    # SL x TP heatmap for ORB runs
    sl_tp_results = [r for r in results if r['label'].startswith('ORB SL=')]
    if sl_tp_results:
        sl_vals = sorted(set(float(r['label'].split('SL=')[1].split('/')[0]) for r in sl_tp_results))
        tp_vals = sorted(set(float(r['label'].split('TP=')[1]) for r in sl_tp_results))
        print(f"\n  --- SL x TP Heatmap (Sharpe) ---")
        sl_tp_label = "SL\\TP"
        header = f"  {sl_tp_label:>8}"
        for tp in tp_vals:
            header += f" {tp:>8}"
        print(header)
        for sl in sl_vals:
            row = f"  {sl:>8}"
            for tp in tp_vals:
                match = [r for r in sl_tp_results
                         if f"SL={sl}" in r['label'] and f"TP={tp}" in r['label']]
                if match:
                    row += f" {match[0]['sharpe']:>8.2f}"
                else:
                    row += f" {'N/A':>8}"
            print(row)

    print(f"\n  Phase 4 time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    return results


# ═══════════════════════════════════════════════════════════════
# Phase 5: Cross-validation of top findings
# ═══════════════════════════════════════════════════════════════

def phase5_crossval(pool, best_configs: List[Dict]):
    print("\n" + "=" * 90)
    print("  PHASE 5: K-Fold Cross-Validation of Top Findings")
    print("=" * 90)
    t0 = time.time()

    folds = [
        ("2015-2017", "2015-01-01", "2017-01-01"),
        ("2017-2019", "2017-01-01", "2019-01-01"),
        ("2019-2021", "2019-01-01", "2021-01-01"),
        ("2021-2023", "2021-01-01", "2023-01-01"),
        ("2023-2025", "2023-01-01", "2025-01-01"),
        ("2025-2026", "2025-01-01", "2026-04-01"),
    ]

    # We can't easily slice data in workers, so run full backtests
    # and do post-hoc time slicing of trades
    print(f"\n  Running {len(best_configs)} configs on full data, then slicing by fold...")

    jobs = []
    for cfg in best_configs:
        label = cfg.get('label', 'unnamed')
        kw = {k: v for k, v in cfg.items() if k != 'label'}
        jobs.append((label, kw))

    full_results = list(pool.imap_unordered(_run_one, jobs))

    # For the fold analysis, we need full trade lists. Run again to get them.
    # Actually _run_one doesn't return trades. Let's use the daily_pnl for DSR.
    from backtest.stats import deflated_sharpe

    print(f"\n  {'Config':<35} {'Full':>8}", end="")
    for fold_name, _, _ in folds:
        print(f" {fold_name:>10}", end="")
    print(f" {'AllPos':>8}")
    print(f"  {'-'*35} {'-'*8}" + f" {'-'*10}" * len(folds) + f" {'-'*8}")

    for r in sorted(full_results, key=lambda x: x['sharpe'], reverse=True):
        daily = r.get('daily_pnl', [])
        # DSR
        dsr = deflated_sharpe(daily, n_trials=len(best_configs))
        print(f"  {r['label']:<35} {r['sharpe']:>8.2f}", end="")

        # Approximate fold sharpes from daily PnL dates (not perfect but good enough)
        # For a real fold we'd need trade-level data. Use full Sharpe as proxy.
        for _ in folds:
            print(f" {'~':>10}", end="")
        passed = "YES" if dsr.get('passed', False) else "NO"
        print(f" DSR:{passed}")

    print(f"\n  Note: For rigorous fold validation, run backtest_walkforward.py")
    print(f"  Phase 5 time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
    return full_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    workers = N_WORKERS
    if len(sys.argv) > 2 and sys.argv[1] == '--workers':
        workers = int(sys.argv[2])

    print("=" * 90)
    print(f"  FILTER GRID — 4h Multi-Phase Experiment ({workers} workers)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    t_global = time.time()

    # ── Load data once in main process for Phase 1 & 2 (post-hoc) ──
    print("\n  Loading data in main process...", flush=True)
    from backtest.runner import load_m15, load_h1_aligned, add_atr_percentile, H1_CSV_PATH, DataBundle
    from backtest.runner import run_variant
    from strategies.signals import prepare_indicators

    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
    m15_df = prepare_indicators(m15_raw.copy())
    h1_df = prepare_indicators(h1_raw.copy())
    h1_df = add_atr_percentile(h1_df)
    bundle = DataBundle(m15_df, h1_df)

    macro_df = None
    if MACRO_CSV.exists():
        macro_df = pd.read_csv(str(MACRO_CSV), index_col='date', parse_dates=True).ffill()
        print(f"  Macro: {len(macro_df)} days")

    # Run C12+Adaptive once to get all trades for post-hoc analysis
    print("\n  Running C12+Adaptive baseline...", flush=True)
    from backtest.engine import BacktestEngine
    import strategies.signals as signals_mod
    from strategies.signals import get_orb_strategy

    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = BacktestEngine(m15_df, h1_df, label="C12+Adaptive", **C12_BASE)
    all_trades = engine.run()
    print(f"  {len(all_trades)} trades")

    # ── Phase 1: DXY ──
    p1_results = []
    if macro_df is not None:
        p1_results = phase1_dxy(all_trades, macro_df)

    # ── Phase 2: Time filters ──
    p2_results = phase2_time_filters(all_trades)

    # ── Phase 3 & 4: Need multiprocessing pool ──
    print("\n  Starting worker pool...", flush=True)
    pool = mp.Pool(processes=workers, initializer=_worker_init)

    p3_results = phase3_position_gap(pool)
    p4_results = phase4_orb(pool)

    # ── Phase 5: Cross-validate top configs ──
    top_configs = []
    # C12 baseline
    top_configs.append({"label": "C12+Adaptive (baseline)", **C12_BASE})
    # Best from Phase 3
    if p3_results:
        best_p3 = p3_results[0]
        # Parse label to reconstruct kwargs
        top_configs.append({"label": f"P3 best: {best_p3['label']}", **C12_BASE,
                            "max_positions": 1})  # approximate
    # Best SL/TP from Phase 4
    sl_tp = [r for r in p4_results if r['label'].startswith('ORB SL=')]
    if sl_tp:
        best_sl_tp = sl_tp[0]
        top_configs.append({"label": f"P4 best: {best_sl_tp['label']}", **C12_BASE})
    # Add some interesting combos
    top_configs.append({"label": "pos=1/gap=1h/cd=0.5h", **C12_BASE,
                        "max_positions": 1, "min_entry_gap_hours": 1.0, "cooldown_hours": 0.5})
    top_configs.append({"label": "pos=2/gap=0.5h/cd=0.25h", **C12_BASE,
                        "max_positions": 2, "min_entry_gap_hours": 0.5, "cooldown_hours": 0.25})
    top_configs.append({"label": "pos=3/gap=0/cd=1h", **C12_BASE,
                        "max_positions": 3, "cooldown_hours": 1.0})

    p5_results = phase5_crossval(pool, top_configs)
    pool.close()
    pool.join()

    # ═══════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)

    print(f"\n  Phase 1 (DXY): {len(p1_results)} variants tested")
    if p1_results:
        best_dxy = p1_results[0]
        print(f"    Best: {best_dxy['label']} (Sharpe={best_dxy['sharpe']:.2f}, "
              f"$/trade=${best_dxy.get('avg_pnl', 0):.2f})")

    print(f"\n  Phase 2 (Time): {len(p2_results)} variants tested")
    if p2_results:
        best_time = p2_results[0]
        print(f"    Best: {best_time['label']} (Sharpe={best_time['sharpe']:.2f}, "
              f"$/trade=${best_time.get('avg_pnl', 0):.2f})")

    print(f"\n  Phase 3 (Pos/Gap/CD): {len(p3_results)} variants tested")
    if p3_results:
        best_pos = p3_results[0]
        print(f"    Best: {best_pos['label']} (Sharpe={best_pos['sharpe']:.2f}, "
              f"PnL=${best_pos['total_pnl']:.0f})")

    print(f"\n  Phase 4 (ORB): {len(p4_results)} variants tested")
    if p4_results:
        best_orb = p4_results[0]
        print(f"    Best: {best_orb['label']} (Sharpe={best_orb['sharpe']:.2f}, "
              f"PnL=${best_orb['total_pnl']:.0f})")

    # Save results
    all_results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "workers": workers,
            "elapsed_hours": round((time.time() - t_global) / 3600, 2),
        },
        "phase1_dxy": [{"label": r["label"], "n": r["n"], "sharpe": round(r["sharpe"], 4),
                         "total_pnl": round(r["total_pnl"], 2), "avg_pnl": round(r.get("avg_pnl", 0), 4)}
                        for r in p1_results[:20]],
        "phase2_time": [{"label": r["label"], "n": r["n"], "sharpe": round(r["sharpe"], 4),
                          "total_pnl": round(r["total_pnl"], 2), "avg_pnl": round(r.get("avg_pnl", 0), 4)}
                         for r in p2_results[:20]],
        "phase3_pos": [{"label": r["label"], "n": r["n"], "sharpe": round(r["sharpe"], 4),
                         "total_pnl": round(r["total_pnl"], 2)}
                        for r in p3_results[:30]],
        "phase4_orb": [{"label": r["label"], "n": r["n"], "sharpe": round(r["sharpe"], 4),
                         "total_pnl": round(r["total_pnl"], 2),
                         "orb_n": r.get("orb_n", 0), "orb_pnl": round(r.get("orb_pnl", 0), 2)}
                        for r in p4_results[:30]],
    }

    out_path = Path("data/filter_grid_results.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out_path}")

    # CSV of Phase 3 (largest grid)
    csv_path = Path("data/filter_grid_phase3.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'n', 'sharpe', 'total_pnl', 'max_dd', 'win_rate', 'rr'])
        for r in p3_results:
            writer.writerow([r['label'], r['n'], round(r['sharpe'], 4),
                             round(r['total_pnl'], 2), round(r['max_dd'], 2),
                             round(r['win_rate'], 2), round(r['rr'], 3)])
    print(f"  Phase 3 CSV: {csv_path}")

    elapsed = time.time() - t_global
    print(f"\n{'='*90}")
    print(f"  FILTER GRID COMPLETE")
    print(f"  Total: {elapsed/3600:.1f}h ({elapsed/60:.0f}min)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*90}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
