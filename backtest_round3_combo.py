"""
Round 3 Combo Test — Cross-Dimension Combinations
===================================================
Combine the top performers from R2 across different dimensions:
  - Trail: 0.8/0.25, 1.0/0.2
  - SL: 3.0ATR, 3.5ATR
  - TP: 4.0ATR, 5.0ATR
  - KC ADX threshold: 18, 20
  
~16 variants, estimated ~50 minutes.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from strategies.signals import prepare_indicators, get_orb_strategy
import strategies.signals as signals_mod
from backtest_m15 import (
    load_m15, load_h1_aligned, calc_stats,
    M15_CSV_PATH, H1_CSV_PATH,
)
from backtest_round2 import Round2Engine, print_results


def run_variant(m15_df, h1_df, label, **kwargs):
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    t0 = time.time()
    engine = Round2Engine(m15_df, h1_df, label=label, **kwargs)
    trades = engine.run()
    elapsed = time.time() - t0
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['rsi_filtered'] = engine.rsi_filtered_count
    stats['rsi_total'] = engine.rsi_total_signals
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    stats['elapsed_s'] = round(elapsed, 1)
    print(f"    {stats['n']} trades (H1={engine.h1_entry_count}, M15={engine.m15_entry_count}), "
          f"RSI filt={engine.rsi_filtered_count}/{engine.rsi_total_signals}, {elapsed:.0f}s")
    return stats


def build_variants():
    V = []

    # References from R2
    V.append(("C00: Baseline", {}))
    V.append(("C01: R2#1 SL3.5+T1.0/0.3", {
        "sl_atr_mult": 3.5, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("C02: R2#2 Trail 0.8/0.25", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25}))
    V.append(("C03: R2#3 Trail 1.0/0.2", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2}))

    # ── Core combos: Trail + SL ──
    V.append(("C04: T0.8/0.25 + SL3.0", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25, "sl_atr_mult": 3.0}))
    V.append(("C05: T0.8/0.25 + SL3.5", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25, "sl_atr_mult": 3.5}))
    V.append(("C06: T1.0/0.2 + SL3.0", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2, "sl_atr_mult": 3.0}))
    V.append(("C07: T1.0/0.2 + SL3.5", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2, "sl_atr_mult": 3.5}))

    # ── Trail + SL + TP ──
    V.append(("C08: T0.8/0.25+SL3.5+TP5", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25,
        "sl_atr_mult": 3.5, "tp_atr_mult": 5.0}))
    V.append(("C09: T1.0/0.2+SL3.5+TP5", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2,
        "sl_atr_mult": 3.5, "tp_atr_mult": 5.0}))

    # ── Trail + SL + KC ADX ──
    V.append(("C10: T0.8/0.25+SL3.5+ADX18", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25,
        "sl_atr_mult": 3.5, "keltner_adx_threshold": 18}))
    V.append(("C11: T1.0/0.2+SL3.5+ADX18", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2,
        "sl_atr_mult": 3.5, "keltner_adx_threshold": 18}))

    # ── Full combo: Trail + SL + TP + ADX ──
    V.append(("C12: T0.8/0.25+SL3.5+TP5+ADX18", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25,
        "sl_atr_mult": 3.5, "tp_atr_mult": 5.0, "keltner_adx_threshold": 18}))
    V.append(("C13: T1.0/0.2+SL3.5+TP5+ADX18", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2,
        "sl_atr_mult": 3.5, "tp_atr_mult": 5.0, "keltner_adx_threshold": 18}))

    # ── Top combo + SL 3.0 variants (slightly tighter SL) ──
    V.append(("C14: T0.8/0.25+SL3.0+TP5+ADX18", {
        "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25,
        "sl_atr_mult": 3.0, "tp_atr_mult": 5.0, "keltner_adx_threshold": 18}))
    V.append(("C15: T1.0/0.2+SL3.0+TP5+ADX18", {
        "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2,
        "sl_atr_mult": 3.0, "tp_atr_mult": 5.0, "keltner_adx_threshold": 18}))

    return V


def main():
    t_start = time.time()
    print("=" * 70)
    print("  ROUND 3 COMBO TEST — Cross-Dimension Combinations")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\nLoading M15 data...")
    m15_df = load_m15()
    m15_df = m15_df[m15_df.index >= pd.Timestamp('2015-01-01', tz='UTC')]

    print("\nLoading H1 data...")
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_df.index[0])

    print("\nPreparing indicators...")
    print("  M15...", end='', flush=True)
    m15_df = prepare_indicators(m15_df)
    print(" done")
    print("  H1...", end='', flush=True)
    h1_df = prepare_indicators(h1_df)
    print(" done")

    if 'atr_percentile' not in h1_df.columns:
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        h1_df['atr_percentile'] = h1_df['atr_percentile'].fillna(0.5)

    print(f"\nM15: {len(m15_df)} bars, H1: {len(h1_df)} bars")

    variants = build_variants()
    print(f"\nTotal variants: {len(variants)}")
    print(f"Estimated runtime: {len(variants) * 5 / 60:.1f} hours\n")

    results = []
    for i, (label, kwargs) in enumerate(variants, 1):
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        stats = run_variant(m15_df, h1_df, label, **kwargs)
        results.append(stats)

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed/60:.0f} minutes ({elapsed:.0f}s)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_results(results)

    out = Path("data/round3_combo_results.json")
    safe = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
            else:
                sr[k] = v
        safe.append(sr)
    out.write_text(json.dumps(safe, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
