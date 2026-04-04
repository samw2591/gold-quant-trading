"""
Overnight Comprehensive M15 Multi-Timeframe Backtest
=====================================================
Exhaustive A/B test across all strategy dimensions on 11 years of data (2015-2026).
~50 variants, estimated runtime ~6 hours.

Dimensions tested:
  1. RSI ADX filter thresholds (25, 30, 35, 40, 45, 50)
  2. RSI direction (BUY+SELL, BUY-only)
  3. RSI ATR percentile min filter (0, 0.20, 0.30, 0.40)
  4. RSI hold time (10, 12, 15, 20, 25 M15 bars)
  5. ORB hold time (8, 12, 16, 20, 24 M15 bars)
  6. Keltner hold time (40, 48, 60, 72, 80 M15 bars)
  7. Trailing stop activate (1.0, 1.25, 1.5, 1.75, 2.0 ATR)
  8. Trailing stop distance (0.3, 0.5, 0.7, 1.0 ATR)
  9. ATR regime lot sizing
 10. Combined best-of-each
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from strategies.signals import (
    prepare_indicators,
    scan_all_signals,
    check_exit_signal,
    calc_auto_lot_size,
    get_orb_strategy,
)
import strategies.signals as signals_mod
from backtest import load_csv, Position, TradeRecord, _aggregate_daily_pnl
from backtest_m15 import (
    load_m15,
    load_h1_aligned,
    build_h1_lookup,
    MultiTimeframeEngine,
    calc_stats,
    M15_CSV_PATH,
    H1_CSV_PATH,
)


class FlexEngine(MultiTimeframeEngine):
    """Extended engine with additional tunable parameters."""

    def __init__(self, m15_df, h1_df,
                 rsi_adx_filter=0,
                 rsi_atr_pct_filter=0,
                 rsi_sell_enabled=True,
                 rsi_atr_pct_min_filter=0,
                 rsi_max_hold_m15=15,
                 orb_max_hold_m15=0,
                 keltner_max_hold_m15=0,
                 trailing_activate_atr=0,
                 trailing_distance_atr=0,
                 atr_regime_lots=False,
                 sell_lot_scale=1.0,
                 label=""):
        super().__init__(
            m15_df, h1_df,
            rsi_adx_filter=rsi_adx_filter,
            rsi_atr_pct_filter=rsi_atr_pct_filter,
            rsi_sell_enabled=rsi_sell_enabled,
            rsi_atr_pct_min_filter=rsi_atr_pct_min_filter,
            orb_max_hold_m15=orb_max_hold_m15,
            atr_regime_lots=atr_regime_lots,
            label=label,
        )
        self.rsi_max_hold_m15 = rsi_max_hold_m15
        self.keltner_max_hold_m15 = keltner_max_hold_m15
        self.trailing_activate_atr_override = trailing_activate_atr
        self.trailing_distance_atr_override = trailing_distance_atr
        self.sell_lot_scale = sell_lot_scale

    def _check_exits(self, m15_window, h1_window, bar, bar_time):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None
            exit_price = close

            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = f"SL"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = f"TP"
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = f"SL"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = f"TP"
                    exit_price = pos.tp_price

            # Trailing stop with overridable parameters
            if not reason and pos.strategy == 'keltner':
                trailing_enabled = config.TRAILING_STOP_ENABLED
                act_atr = self.trailing_activate_atr_override or config.TRAILING_ACTIVATE_ATR
                dist_atr = self.trailing_distance_atr_override or config.TRAILING_DISTANCE_ATR

                if trailing_enabled:
                    atr = self._get_h1_atr(h1_window)
                    if atr > 0:
                        if pos.direction == 'BUY':
                            float_profit = high - pos.entry_price
                            pos.extreme_price = max(pos.extreme_price, high)
                        else:
                            float_profit = pos.entry_price - low
                            pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                        if float_profit >= atr * act_atr:
                            trail_distance = atr * dist_atr
                            if pos.direction == 'BUY':
                                new_trail = pos.extreme_price - trail_distance
                                pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                                if low <= pos.trailing_stop_price:
                                    reason = "Trailing"
                                    exit_price = pos.trailing_stop_price
                            else:
                                new_trail = pos.extreme_price + trail_distance
                                if pos.trailing_stop_price <= 0:
                                    pos.trailing_stop_price = new_trail
                                else:
                                    pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                                if high >= pos.trailing_stop_price:
                                    reason = "Trailing"
                                    exit_price = pos.trailing_stop_price

            # Signal exit
            if not reason and pos.strategy == 'keltner':
                pass
            elif not reason and pos.strategy in ('m15_rsi', 'm5_rsi'):
                if pos.bars_held > 1:
                    exit_sig = check_exit_signal(m15_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close
            elif not reason and pos.strategy not in ('keltner',):
                if h1_window is not None and len(h1_window) > 2:
                    exit_sig = check_exit_signal(h1_window, pos.strategy, pos.direction)
                    if exit_sig:
                        reason = exit_sig
                        exit_price = close

            # Time stop with overridable hold times
            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = self.rsi_max_hold_m15
                elif pos.strategy == 'orb' and self.orb_max_hold_m15 > 0:
                    max_hold = self.orb_max_hold_m15
                elif pos.strategy == 'keltner' and self.keltner_max_hold_m15 > 0:
                    max_hold = self.keltner_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4
                if pos.bars_held >= max_hold:
                    reason = f"Timeout:{pos.bars_held}>={max_hold}"
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    def _process_signals(self, signals, bar_time, source):
        active_strategies = {p.strategy for p in self.positions}
        current_dir = self.positions[0].direction if self.positions else None

        slots = config.MAX_POSITIONS - len(self.positions)
        for sig in signals[:slots]:
            strategy = sig['strategy']
            direction = sig['signal']
            close = sig['close']
            sl = sig.get('sl', config.STOP_LOSS_PIPS)
            tp = sig.get('tp', 0)
            if tp <= 0:
                tp = sl * 2

            cooldown = self.cooldown_until.get(strategy)
            if cooldown and bar_time <= cooldown:
                continue
            if strategy in active_strategies:
                continue
            if current_dir and direction != current_dir:
                continue

            lots = calc_auto_lot_size(0, sl)
            lots = max(config.MIN_LOT_SIZE, lots)

            if self.atr_regime_lots:
                h1_window = self._get_h1_window(bar_time)
                if h1_window is not None and len(h1_window) > 0:
                    atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
                    if not pd.isna(atr_pct):
                        if atr_pct > 0.70:
                            lots = round(lots * 1.2, 2)
                        elif atr_pct < 0.30:
                            lots = round(lots * 0.7, 2)
                lots = max(config.MIN_LOT_SIZE, lots)

            if direction == 'SELL' and self.sell_lot_scale != 1.0:
                lots = round(lots * self.sell_lot_scale, 2)
                lots = max(config.MIN_LOT_SIZE, lots)

            pos = Position(
                strategy=strategy, direction=direction,
                entry_price=close, entry_time=bar_time,
                lots=lots, sl_distance=sl, tp_distance=tp,
            )
            self.positions.append(pos)
            active_strategies.add(strategy)
            if current_dir is None:
                current_dir = direction
            if source == 'H1':
                self.h1_entry_count += 1
            else:
                self.m15_entry_count += 1


def run_variant(m15_df, h1_df, label, **kwargs):
    """Run a single variant and return stats dict."""
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    t0 = time.time()
    engine = FlexEngine(m15_df, h1_df, label=label, **kwargs)
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
    """Define all ~50 variants to test."""
    V = []

    # ════════════════════════════════════════════════
    # Group 0: Baseline
    # ════════════════════════════════════════════════
    V.append(("A00: Baseline", {}))

    # ════════════════════════════════════════════════
    # Group 1: RSI ADX filter sweep (6 variants)
    # ════════════════════════════════════════════════
    for adx in [25, 30, 35, 40, 45, 50]:
        V.append((f"A01: RSI ADX>{adx}", {"rsi_adx_filter": adx}))

    # ════════════════════════════════════════════════
    # Group 2: RSI direction (2 variants)
    # ════════════════════════════════════════════════
    V.append(("A02: RSI BUY-only", {"rsi_sell_enabled": False}))
    V.append(("A03: RSI BUY-only+ADX>40", {"rsi_sell_enabled": False, "rsi_adx_filter": 40}))

    # ════════════════════════════════════════════════
    # Group 3: RSI ATR percentile min (4 variants)
    # ════════════════════════════════════════════════
    for pct in [0.20, 0.30, 0.40, 0.50]:
        V.append((f"A04: ADX>40+ATR_min>{pct:.2f}", {"rsi_adx_filter": 40, "rsi_atr_pct_min_filter": pct}))

    # ════════════════════════════════════════════════
    # Group 4: RSI hold time sweep (5 variants)
    # ════════════════════════════════════════════════
    for hold in [8, 10, 12, 15, 20, 25]:
        V.append((f"A05: RSI hold {hold} bars", {"rsi_max_hold_m15": hold}))

    # ════════════════════════════════════════════════
    # Group 5: ORB hold time sweep (6 variants)
    # ════════════════════════════════════════════════
    for hold in [8, 12, 16, 20, 24, 32]:
        V.append((f"A06: ORB hold {hold} bars", {"orb_max_hold_m15": hold}))

    # ════════════════════════════════════════════════
    # Group 6: Keltner hold time sweep (5 variants)
    # ════════════════════════════════════════════════
    for hold in [40, 48, 60, 72, 80]:
        lbl = f"A07: KC hold {hold} bars ({hold*15/60:.0f}h)"
        V.append((lbl, {"keltner_max_hold_m15": hold}))

    # ════════════════════════════════════════════════
    # Group 7: Trailing stop activate sweep (5 variants)
    # ════════════════════════════════════════════════
    for act in [1.0, 1.25, 1.5, 1.75, 2.0]:
        V.append((f"A08: Trail act {act}ATR", {"trailing_activate_atr": act}))

    # ════════════════════════════════════════════════
    # Group 8: Trailing stop distance sweep (4 variants)
    # ════════════════════════════════════════════════
    for dist in [0.3, 0.5, 0.7, 1.0]:
        V.append((f"A09: Trail dist {dist}ATR", {"trailing_distance_atr": dist}))

    # ════════════════════════════════════════════════
    # Group 9: Trailing combos (act x dist) (4 variants)
    # ════════════════════════════════════════════════
    for act, dist in [(1.0, 0.3), (1.25, 0.5), (1.5, 0.3), (1.0, 0.5)]:
        V.append((f"A10: Trail {act}/{dist}", {"trailing_activate_atr": act, "trailing_distance_atr": dist}))

    # ════════════════════════════════════════════════
    # Group 10: ATR regime lots (2 variants)
    # ════════════════════════════════════════════════
    V.append(("A11: ATR regime lots", {"atr_regime_lots": True}))
    V.append(("A12: ADX>40+ATR regime", {"rsi_adx_filter": 40, "atr_regime_lots": True}))

    # ════════════════════════════════════════════════
    # Group 11: SELL lot scale (3 variants)
    # ════════════════════════════════════════════════
    for scale in [0.5, 0.7, 0.3]:
        V.append((f"A13: SELL lots x{scale}", {"sell_lot_scale": scale}))

    # ════════════════════════════════════════════════
    # Group 12: Combined best-of-each candidates
    # ════════════════════════════════════════════════
    V.append(("B01: ADX>40+ORB12", {"rsi_adx_filter": 40, "orb_max_hold_m15": 12}))
    V.append(("B02: ADX>40+ORB12+regime", {"rsi_adx_filter": 40, "orb_max_hold_m15": 12, "atr_regime_lots": True}))
    V.append(("B03: ADX>40+SELL0.5", {"rsi_adx_filter": 40, "sell_lot_scale": 0.5}))
    V.append(("B04: ADX>40+ORB12+SELL0.5", {"rsi_adx_filter": 40, "orb_max_hold_m15": 12, "sell_lot_scale": 0.5}))
    V.append(("B05: ADX>40+Trail1.25/0.5", {"rsi_adx_filter": 40, "trailing_activate_atr": 1.25, "trailing_distance_atr": 0.5}))
    V.append(("B06: ADX>40+Trail1.0/0.3", {"rsi_adx_filter": 40, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("B07: ADX>40+ORB12+Trail1.25", {
        "rsi_adx_filter": 40, "orb_max_hold_m15": 12,
        "trailing_activate_atr": 1.25, "trailing_distance_atr": 0.5,
    }))
    V.append(("B08: Full combo", {
        "rsi_adx_filter": 40, "orb_max_hold_m15": 12,
        "trailing_activate_atr": 1.25, "trailing_distance_atr": 0.5,
        "sell_lot_scale": 0.5,
    }))
    V.append(("B09: Full+regime", {
        "rsi_adx_filter": 40, "orb_max_hold_m15": 12,
        "trailing_activate_atr": 1.25, "trailing_distance_atr": 0.5,
        "sell_lot_scale": 0.5, "atr_regime_lots": True,
    }))
    V.append(("B10: RSI10+ADX40+ORB12", {
        "rsi_adx_filter": 40, "rsi_max_hold_m15": 10, "orb_max_hold_m15": 12,
    }))

    return V


def print_results(results: List[Dict]):
    results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
    baseline = [r for r in results if r['label'] == 'A00: Baseline']
    base_sharpe = baseline[0]['sharpe'] if baseline else 0

    print("\n")
    print("=" * 130)
    print("  OVERNIGHT COMPREHENSIVE BACKTEST RESULTS (2015-2026, M15+H1)")
    print("=" * 130)

    print(f"\n  {'Rank':<5} {'Variant':<35} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} "
          f"{'MaxDD':>10} {'DD%':>6} {'WR%':>6} {'RR':>5} {'Time':>5}")
    print(f"  {'-'*5} {'-'*35} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*5} {'-'*5}")

    for rank, v in enumerate(results, 1):
        if v['n'] == 0:
            print(f"  {rank:<5} {v['label']:<35}   (no trades)")
            continue
        ds = v['sharpe'] - base_sharpe
        print(f"  {rank:<5} {v['label']:<35} {v['n']:>6} {v['sharpe']:>8.2f} {ds:>+5.2f} "
              f"${v['total_pnl']:>9.0f} ${v['max_dd']:>9.0f} {v['max_dd_pct']:>5.1f}% "
              f"{v['win_rate']:>5.1f}% {v['rr']:>4.2f} {v.get('elapsed_s', 0):>4.0f}s")

    # Strategy breakdown for top 15
    print(f"\n  --- Strategy Breakdown (Top 15) ---")
    print(f"  {'Variant':<35} {'KC_N':>5} {'KC_PnL':>9} {'ORB_N':>5} {'ORB_PnL':>9} "
          f"{'RSI_N':>5} {'RSI_PnL':>9} {'RSI_BUY':>9} {'RSI_SEL':>9} {'Filt':>5}")
    print(f"  {'-'*35} {'-'*5} {'-'*9} {'-'*5} {'-'*9} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*5}")
    for v in results[:15]:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<35} {v['keltner_n']:>5} ${v['keltner_pnl']:>8.0f} "
              f"{v.get('orb_n', 0):>5} ${v.get('orb_pnl', 0):>8.0f} "
              f"{v['rsi_n']:>5} ${v['rsi_pnl']:>8.0f} "
              f"${v.get('rsi_buy_pnl', 0):>8.0f} ${v.get('rsi_sell_pnl', 0):>8.0f} "
              f"{v.get('rsi_filtered', 0):>5}")

    # Year-by-year for top 10
    top10 = results[:10]
    all_years = set()
    for v in top10:
        all_years.update(v.get('year_pnl', {}).keys())
    years = sorted(all_years)

    if years:
        print(f"\n  --- Year-by-Year PnL (Top 10) ---")
        print(f"  {'Year':<6}", end='')
        for v in top10:
            print(f" {v['label'][:14]:>15}", end='')
        print()
        for y in years:
            print(f"  {y:<6}", end='')
            for v in top10:
                pnl = v.get('year_pnl', {}).get(y, 0)
                print(f" ${pnl:>14.0f}", end='')
            print()

    print("\n" + "=" * 130)


def main():
    t_start = time.time()
    print("=" * 70)
    print("  OVERNIGHT COMPREHENSIVE M15 BACKTEST")
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
    print(f"Estimated runtime: {len(variants) * 7 / 60:.0f} hours\n")

    results = []
    for i, (label, kwargs) in enumerate(variants, 1):
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        stats = run_variant(m15_df, h1_df, label, **kwargs)
        results.append(stats)

        # Save intermediate results every 5 variants
        if i % 5 == 0 or i == len(variants):
            out = Path("data/overnight_backtest_results.json")
            safe_results = []
            for r in results:
                sr = {}
                for k, v in r.items():
                    if isinstance(v, (np.integer,)):
                        sr[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        sr[k] = float(v)
                    else:
                        sr[k] = v
                safe_results.append(sr)
            out.write_text(json.dumps(safe_results, indent=2, default=str), encoding='utf-8')

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_results(results)

    # Save final results
    out = Path("data/overnight_backtest_results.json")
    safe_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
            else:
                sr[k] = v
        safe_results.append(sr)
    out.write_text(json.dumps(safe_results, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")


if __name__ == '__main__':
    main()
