"""
Round 2 Backtest — Unexplored Dimensions
==========================================
Focus on parameters NOT covered in the overnight run:
  1. Trail 1.0/0.3 fine grid (confirm the champion is a true peak)
  2. SL/TP ATR multipliers (never tested)
  3. Keltner ADX entry threshold (never tested)
  4. MAX_POSITIONS (never tested)
  5. Cooldown hours (never tested)
  6. Champion + stacking combos

~17 variants, estimated ~2 hours on 11yr data.
"""
import json
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
    _calc_atr_stop,
    _calc_atr_tp,
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


class Round2Engine(MultiTimeframeEngine):
    """Engine with SL/TP multiplier overrides, ADX threshold, max positions, cooldown."""

    def __init__(self, m15_df, h1_df,
                 trailing_activate_atr=0,
                 trailing_distance_atr=0,
                 sl_atr_mult=0,
                 tp_atr_mult=0,
                 keltner_adx_threshold=0,
                 max_positions_override=0,
                 cooldown_hours=0,
                 rsi_adx_filter=0,
                 spread_cost=0.0,
                 label=""):
        super().__init__(
            m15_df, h1_df,
            rsi_adx_filter=rsi_adx_filter,
            spread_cost=spread_cost,
            label=label,
        )
        self.trailing_activate_atr_override = trailing_activate_atr
        self.trailing_distance_atr_override = trailing_distance_atr
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.keltner_adx_threshold = keltner_adx_threshold
        self.max_positions_override = max_positions_override
        self.cooldown_hours_override = cooldown_hours

    @property
    def _max_pos(self):
        return self.max_positions_override or config.MAX_POSITIONS

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
                    reason = "SL"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = "TP"
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = "SL"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = "TP"
                    exit_price = pos.tp_price

            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                act_atr = self.trailing_activate_atr_override or config.TRAILING_ACTIVATE_ATR
                dist_atr = self.trailing_distance_atr_override or config.TRAILING_DISTANCE_ATR
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

            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = 15
                elif pos.strategy == 'orb' and self.orb_max_hold_m15 > 0:
                    max_hold = self.orb_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4
                if pos.bars_held >= max_hold:
                    reason = f"Timeout:{pos.bars_held}>={max_hold}"
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    def _close_position(self, pos, exit_price, exit_time, reason):
        strategy = pos.strategy
        super()._close_position(pos, exit_price, exit_time, reason)
        if self.cooldown_hours_override > 0 and self.trades:
            pnl = self.trades[-1].pnl
            if pnl < 0:
                self.cooldown_until[strategy] = (
                    pd.Timestamp(exit_time) + pd.Timedelta(hours=self.cooldown_hours_override)
                )

    def _check_h1_entries(self, h1_window, bar_time):
        if len(self.positions) >= self._max_pos:
            return
        if self.keltner_adx_threshold > 0:
            adx_val = float(h1_window.iloc[-1].get('ADX', 0))
            if not pd.isna(adx_val):
                old_threshold = signals_mod.ADX_TREND_THRESHOLD
                signals_mod.ADX_TREND_THRESHOLD = self.keltner_adx_threshold
                signals = scan_all_signals(h1_window, 'H1')
                signals_mod.ADX_TREND_THRESHOLD = old_threshold
            else:
                signals = scan_all_signals(h1_window, 'H1')
        else:
            signals = scan_all_signals(h1_window, 'H1')
        if not signals:
            return
        self._process_signals(signals, bar_time, source='H1')

    def _check_m15_entries(self, m15_window, h1_window, bar_time):
        if len(self.positions) >= self._max_pos:
            return
        signals = scan_all_signals(m15_window, 'M15')
        if not signals:
            return
        filtered = []
        for sig in signals:
            self.rsi_total_signals += 1
            blocked = False
            if self.rsi_adx_filter > 0 and h1_window is not None and len(h1_window) > 0:
                adx_val = float(h1_window.iloc[-1].get('ADX', 0))
                if not pd.isna(adx_val) and adx_val > self.rsi_adx_filter:
                    self.rsi_filtered_count += 1
                    blocked = True
            if not blocked:
                filtered.append(sig)
        if filtered:
            self._process_signals(filtered, bar_time, source='M15')

    def _process_signals(self, signals, bar_time, source):
        active_strategies = {p.strategy for p in self.positions}
        current_dir = self.positions[0].direction if self.positions else None
        slots = self._max_pos - len(self.positions)

        for sig in signals[:slots]:
            strategy = sig['strategy']
            direction = sig['signal']
            close = sig['close']
            sl = sig.get('sl', config.STOP_LOSS_PIPS)
            tp = sig.get('tp', 0)

            if self.sl_atr_mult > 0:
                atr = self._get_h1_atr(self._get_h1_window(bar_time))
                if atr > 0:
                    sl = round(atr * self.sl_atr_mult, 2)
                    sl = max(signals_mod.ATR_SL_MIN, min(signals_mod.ATR_SL_MAX, sl))
            if self.tp_atr_mult > 0:
                atr = self._get_h1_atr(self._get_h1_window(bar_time))
                if atr > 0:
                    tp = round(atr * self.tp_atr_mult, 2)

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

    # ── Baseline + R1 champion for comparison ──
    V.append(("R2-00: Baseline", {}))
    V.append(("R2-01: R1 champ Trail1.0/0.3", {"trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))

    # ── G1: Trail fine grid around 1.0/0.3 (confirm peak) ──
    V.append(("R2-02: Trail 0.8/0.3", {"trailing_activate_atr": 0.8, "trailing_distance_atr": 0.3}))
    V.append(("R2-03: Trail 0.8/0.25", {"trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25}))
    V.append(("R2-04: Trail 1.0/0.25", {"trailing_activate_atr": 1.0, "trailing_distance_atr": 0.25}))
    V.append(("R2-05: Trail 1.0/0.2", {"trailing_activate_atr": 1.0, "trailing_distance_atr": 0.2}))
    V.append(("R2-06: Trail 1.2/0.3", {"trailing_activate_atr": 1.2, "trailing_distance_atr": 0.3}))
    V.append(("R2-07: Trail 0.8/0.4", {"trailing_activate_atr": 0.8, "trailing_distance_atr": 0.4}))

    # ── G2: SL ATR multiplier (current=2.5) ──
    V.append(("R2-08: SL 2.0ATR", {"sl_atr_mult": 2.0, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-09: SL 3.0ATR", {"sl_atr_mult": 3.0, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-10: SL 3.5ATR", {"sl_atr_mult": 3.5, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))

    # ── G3: TP ATR multiplier (current=3.0) ──
    V.append(("R2-11: TP 2.0ATR", {"tp_atr_mult": 2.0, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-12: TP 4.0ATR", {"tp_atr_mult": 4.0, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-13: TP 5.0ATR", {"tp_atr_mult": 5.0, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))

    # ── G4: Keltner ADX entry threshold (current=24) ──
    V.append(("R2-14: KC ADX>20", {"keltner_adx_threshold": 20, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-15: KC ADX>28", {"keltner_adx_threshold": 28, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-16: KC ADX>18", {"keltner_adx_threshold": 18, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))

    # ── G5: Max positions (current=2) ──
    V.append(("R2-17: MaxPos 1", {"max_positions_override": 1, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-18: MaxPos 3", {"max_positions_override": 3, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))

    # ── G6: Cooldown hours (current=3h) ──
    V.append(("R2-19: Cooldown 1h", {"cooldown_hours": 1, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-20: Cooldown 5h", {"cooldown_hours": 5, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))
    V.append(("R2-21: Cooldown 8h", {"cooldown_hours": 8, "trailing_activate_atr": 1.0, "trailing_distance_atr": 0.3}))

    return V


def print_results(results):
    results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)
    baseline = [r for r in results if r['label'] == 'R2-00: Baseline']
    base_sharpe = baseline[0]['sharpe'] if baseline else 0

    print("\n")
    print("=" * 130)
    print("  ROUND 2 BACKTEST RESULTS — Unexplored Dimensions (2015-2026, M15+H1)")
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

    # Strategy breakdown for all
    print(f"\n  --- Strategy Breakdown ---")
    print(f"  {'Variant':<35} {'KC_N':>5} {'KC_PnL':>9} {'ORB_N':>5} {'ORB_PnL':>9} "
          f"{'RSI_N':>5} {'RSI_PnL':>9} {'RSI_BUY':>9} {'RSI_SEL':>9}")
    print(f"  {'-'*35} {'-'*5} {'-'*9} {'-'*5} {'-'*9} {'-'*5} {'-'*9} {'-'*9} {'-'*9}")
    for v in results:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<35} {v['keltner_n']:>5} ${v['keltner_pnl']:>8.0f} "
              f"{v.get('orb_n', 0):>5} ${v.get('orb_pnl', 0):>8.0f} "
              f"{v['rsi_n']:>5} ${v['rsi_pnl']:>8.0f} "
              f"${v.get('rsi_buy_pnl', 0):>8.0f} ${v.get('rsi_sell_pnl', 0):>8.0f}")

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
    print("  ROUND 2 BACKTEST — Unexplored Dimensions")
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
    print(f"Estimated runtime: {len(variants) * 7 / 60:.1f} hours\n")

    results = []
    for i, (label, kwargs) in enumerate(variants, 1):
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        stats = run_variant(m15_df, h1_df, label, **kwargs)
        results.append(stats)

        if i % 5 == 0 or i == len(variants):
            out = Path("data/round2_backtest_results.json")
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

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_results(results)

    out = Path("data/round2_backtest_results.json")
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
