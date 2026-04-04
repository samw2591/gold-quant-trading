"""
M15 Multi-Timeframe Backtest Framework
=======================================
Replays M15 bars as the primary loop while aligning H1 bars for
Keltner/ORB signal generation. Faithfully reproduces the live system's
dual-timeframe architecture (gold_trader.py).

At each M15 bar:
  - Check exits for all positions (SL/TP/trailing/signal/time)
  - Run M15 RSI signals
At H1 boundaries (minute == 0):
  - Also run H1 signals (Keltner, ORB, gap fill)

Includes A/B test variants for M15 RSI ADX filtering.

Usage:
    python backtest_m15.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""
import sys
import glob as glob_mod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
from backtest import (
    load_csv,
    Position,
    TradeRecord,
    _aggregate_daily_pnl,
)

# Default file paths
M15_CSV_PATH = Path("data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv")
H1_CSV_PATH = Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv")


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_m15(csv_path: Path = M15_CSV_PATH) -> pd.DataFrame:
    """Load the merged M15 CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"M15 CSV not found: {csv_path}")
    df = load_csv(str(csv_path))
    print(f"  M15: {csv_path} ({len(df)} bars, {df.index[0]} -> {df.index[-1]})")
    return df


def load_h1_aligned(h1_path: Path, m15_start: pd.Timestamp) -> pd.DataFrame:
    """Load H1 data, keep from before the M15 start (for indicator warmup)."""
    df = load_csv(str(h1_path))
    warmup_start = m15_start - pd.Timedelta(hours=200)
    df = df[df.index >= warmup_start]
    print(f"  H1: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    return df


def build_h1_lookup(h1_df: pd.DataFrame) -> Dict[pd.Timestamp, int]:
    """Build a timestamp -> iloc index lookup for H1 bars."""
    return {ts: i for i, ts in enumerate(h1_df.index)}


# ═══════════════════════════════════════════════════════════════
# Multi-Timeframe Engine
# ═══════════════════════════════════════════════════════════════

class MultiTimeframeEngine:
    """Dual-timeframe backtest: loop M15, scan H1 at hourly boundaries."""

    M15_WINDOW = 150
    H1_WINDOW = 150

    def __init__(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame,
                 rsi_adx_filter: float = 0,
                 rsi_atr_pct_filter: float = 0,
                 rsi_sell_enabled: bool = True,
                 rsi_atr_pct_min_filter: float = 0,
                 orb_max_hold_m15: int = 0,
                 atr_regime_lots: bool = False,
                 spread_cost: float = 0.0,
                 label: str = ""):
        self.m15_df = m15_df
        self.h1_df = h1_df
        self.h1_lookup = build_h1_lookup(h1_df)
        self.rsi_adx_filter = rsi_adx_filter
        self.rsi_atr_pct_filter = rsi_atr_pct_filter
        self.rsi_sell_enabled = rsi_sell_enabled
        self.rsi_atr_pct_min_filter = rsi_atr_pct_min_filter
        self.orb_max_hold_m15 = orb_max_hold_m15
        self.atr_regime_lots = atr_regime_lots
        self.spread_cost = spread_cost
        self.label = label

        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []

        self.daily_loss_count = 0
        self.current_date = None
        self.cooldown_until: Dict[str, datetime] = {}

        # Stats
        self.rsi_filtered_count = 0
        self.rsi_total_signals = 0
        self.h1_entry_count = 0
        self.m15_entry_count = 0

    def run(self) -> List[TradeRecord]:
        self._reset_global_state()

        total_bars = len(self.m15_df)
        lookback = self.M15_WINDOW
        realized_pnl = 0.0

        m15_start = self.m15_df.index[lookback]
        m15_end = self.m15_df.index[-1]
        print(f"  Backtest: {m15_start.strftime('%Y-%m-%d')} -> {m15_end.strftime('%Y-%m-%d')}")
        print(f"  M15 bars: {total_bars}, H1 bars: {len(self.h1_df)}")

        last_pct = 0
        for i in range(lookback, total_bars):
            pct = int((i - lookback) / (total_bars - lookback) * 100) // 10 * 10
            if pct > last_pct:
                print(f"    {pct}%...", end='', flush=True)
                last_pct = pct

            bar = self.m15_df.iloc[i]
            bar_time = self.m15_df.index[i]
            bar_date = bar_time.date()

            # Daily reset
            if bar_date != self.current_date:
                self.current_date = bar_date
                self.daily_loss_count = 0

            is_flat = bool(bar.get('is_flat', False))
            if is_flat:
                unrealized = self._calc_unrealized(float(bar['Close']))
                self.equity_curve.append(config.CAPITAL + realized_pnl + unrealized)
                continue

            # M15 window for exits and M15 signals
            m15_start_idx = max(0, i - self.M15_WINDOW + 1)
            m15_window = self.m15_df.iloc[m15_start_idx:i + 1]

            # Find corresponding H1 window
            h1_window = self._get_h1_window(bar_time)

            # 1. Check exits (on M15 bar granularity)
            self._check_exits(m15_window, h1_window, bar, bar_time)

            # 2. Check entries (if within daily loss limit)
            if self.daily_loss_count < config.DAILY_MAX_LOSSES:
                is_h1_boundary = (bar_time.minute == 0)

                # H1 signals at hourly boundaries
                if is_h1_boundary and h1_window is not None and len(h1_window) >= 50:
                    self._check_h1_entries(h1_window, bar_time)

                # M15 RSI signals every bar
                if len(m15_window) >= 105:
                    self._check_m15_entries(m15_window, h1_window, bar_time)

            # Track equity
            realized_pnl = sum(t.pnl for t in self.trades)
            unrealized = self._calc_unrealized(float(bar['Close']))
            self.equity_curve.append(config.CAPITAL + realized_pnl + unrealized)

        # Force close remaining positions
        if self.positions:
            last_bar = self.m15_df.iloc[-1]
            last_time = self.m15_df.index[-1]
            for pos in list(self.positions):
                self._close_position(pos, float(last_bar['Close']), last_time, "backtest_end")

        print(f" done!")
        return self.trades

    def _get_h1_window(self, m15_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Find the H1 window for a given M15 timestamp."""
        # H1 bar timestamp is the hour start (minute=0)
        h1_time = m15_time.floor('h')

        if h1_time in self.h1_lookup:
            h1_idx = self.h1_lookup[h1_time]
        else:
            # Find the most recent H1 bar before this M15 time
            h1_times = self.h1_df.index
            mask = h1_times <= m15_time
            if not mask.any():
                return None
            h1_idx = mask.sum() - 1

        start = max(0, h1_idx - self.H1_WINDOW + 1)
        return self.h1_df.iloc[start:h1_idx + 1]

    def _check_exits(self, m15_window: pd.DataFrame, h1_window: Optional[pd.DataFrame],
                     bar: pd.Series, bar_time):
        """Check exit conditions using M15 bar granularity."""
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None
            exit_price = close

            # 1. SL/TP
            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = f"SL: {low:.2f} <= {pos.sl_price:.2f}"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = f"TP: {high:.2f} >= {pos.tp_price:.2f}"
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = f"SL: {high:.2f} >= {pos.sl_price:.2f}"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = f"TP: {low:.2f} <= {pos.tp_price:.2f}"
                    exit_price = pos.tp_price

            # 2. Keltner trailing stop (uses H1 ATR)
            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                atr = self._get_h1_atr(h1_window)
                if atr > 0:
                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        float_profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                    activate_threshold = atr * config.TRAILING_ACTIVATE_ATR
                    if float_profit >= activate_threshold:
                        trail_distance = atr * config.TRAILING_DISTANCE_ATR
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_distance
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason = f"Trailing: extreme={pos.extreme_price:.2f}, trail={pos.trailing_stop_price:.2f}"
                                exit_price = pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = f"Trailing: extreme={pos.extreme_price:.2f}, trail={pos.trailing_stop_price:.2f}"
                                exit_price = pos.trailing_stop_price

            # 3. Signal exit
            if not reason and pos.strategy == 'keltner':
                pass  # Keltner exits via SL/TP/trailing/timeout only
            elif not reason and pos.strategy in ('m15_rsi', 'm5_rsi'):
                # Skip exit signal check for first 1 bar (~15 min) like live system
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

            # 4. Time stop
            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = 15
                elif pos.strategy == 'orb' and self.orb_max_hold_m15 > 0:
                    max_hold = self.orb_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4
                if pos.bars_held >= max_hold:
                    reason = f"Timeout: {pos.bars_held} M15 bars >= {max_hold}"
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    def _check_h1_entries(self, h1_window: pd.DataFrame, bar_time):
        """Check H1 signals (Keltner, ORB, gap fill)."""
        if len(self.positions) >= config.MAX_POSITIONS:
            return

        signals = scan_all_signals(h1_window, 'H1')
        if not signals:
            return

        self._process_signals(signals, bar_time, source='H1')

    def _check_m15_entries(self, m15_window: pd.DataFrame, h1_window: Optional[pd.DataFrame],
                           bar_time):
        """Check M15 RSI signals with optional ADX/ATR filtering."""
        if len(self.positions) >= config.MAX_POSITIONS:
            return

        signals = scan_all_signals(m15_window, 'M15')
        if not signals:
            return

        filtered = []
        for sig in signals:
            self.rsi_total_signals += 1
            blocked = False

            # Direction filter: block SELL signals
            if not self.rsi_sell_enabled and sig.get('signal') == 'SELL':
                self.rsi_filtered_count += 1
                blocked = True

            # ADX filter: block M15 RSI when H1 ADX is too high (strong trend)
            if not blocked and self.rsi_adx_filter > 0 and h1_window is not None and len(h1_window) > 0:
                adx_val = float(h1_window.iloc[-1].get('ADX', 0))
                if not pd.isna(adx_val) and adx_val > self.rsi_adx_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            # ATR percentile max filter (block in extreme volatility)
            if not blocked and self.rsi_atr_pct_filter > 0 and h1_window is not None and len(h1_window) > 0:
                atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
                if not pd.isna(atr_pct) and atr_pct > self.rsi_atr_pct_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            # ATR percentile min filter (block in low volatility)
            if not blocked and self.rsi_atr_pct_min_filter > 0 and h1_window is not None and len(h1_window) > 0:
                atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
                if not pd.isna(atr_pct) and atr_pct < self.rsi_atr_pct_min_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked:
                filtered.append(sig)

        if filtered:
            self._process_signals(filtered, bar_time, source='M15')

    def _process_signals(self, signals: List[Dict], bar_time, source: str):
        """Shared entry processing for both H1 and M15 signals."""
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

            # ATR regime lot sizing
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

            pos = Position(
                strategy=strategy,
                direction=direction,
                entry_price=close,
                entry_time=bar_time,
                lots=lots,
                sl_distance=sl,
                tp_distance=tp,
            )
            self.positions.append(pos)
            active_strategies.add(strategy)
            if current_dir is None:
                current_dir = direction
            if source == 'H1':
                self.h1_entry_count += 1
            else:
                self.m15_entry_count += 1

    def _close_position(self, pos: Position, exit_price: float, exit_time, reason: str):
        if pos.direction == 'BUY':
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price
        pnl = round(pnl_points * pos.lots * config.POINT_VALUE_PER_LOT, 2)
        if self.spread_cost > 0:
            pnl -= round(self.spread_cost * pos.lots * config.POINT_VALUE_PER_LOT, 2)

        trade = TradeRecord(
            strategy=pos.strategy,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            lots=pos.lots,
            pnl=pnl,
            exit_reason=reason,
            bars_held=pos.bars_held,
        )
        self.trades.append(trade)
        self.positions.remove(pos)

        if pnl < 0:
            self.daily_loss_count += 1
            cooldown_hours = config.COOLDOWN_BARS
            self.cooldown_until[pos.strategy] = (
                pd.Timestamp(exit_time) + pd.Timedelta(hours=cooldown_hours)
            )

    def _calc_unrealized(self, current_price: float) -> float:
        total = 0.0
        for pos in self.positions:
            if pos.direction == 'BUY':
                pnl = (current_price - pos.entry_price) * pos.lots * config.POINT_VALUE_PER_LOT
            else:
                pnl = (pos.entry_price - current_price) * pos.lots * config.POINT_VALUE_PER_LOT
            total += pnl
        return total

    @staticmethod
    def _get_h1_atr(h1_window: Optional[pd.DataFrame]) -> float:
        if h1_window is None or len(h1_window) == 0:
            return 0
        atr = float(h1_window.iloc[-1].get('ATR', 0))
        return atr if not np.isnan(atr) else 0

    def _reset_global_state(self):
        orb = get_orb_strategy()
        orb.reset_daily()
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False


# ═══════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════

def calc_stats(trades: List[TradeRecord], equity_curve: List[float]) -> Dict:
    if not trades:
        return {'n': 0, 'total_pnl': 0, 'sharpe': 0, 'win_rate': 0,
                'max_dd': 0, 'max_dd_pct': 0, 'rr': 0,
                'rsi_n': 0, 'rsi_pnl': 0, 'rsi_wr': 0,
                'keltner_n': 0, 'keltner_pnl': 0, 'keltner_wr': 0,
                'avg_win': 0, 'avg_loss': 0, 'year_pnl': {}}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    daily_pnl = _aggregate_daily_pnl(trades)
    sharpe = 0.0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)

    eq = np.array(equity_curve) if equity_curve else np.array([config.CAPITAL])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    rsi_trades = [t for t in trades if t.strategy == 'm15_rsi']
    keltner_trades = [t for t in trades if t.strategy == 'keltner']
    orb_trades = [t for t in trades if t.strategy == 'orb']
    rsi_pnl = sum(t.pnl for t in rsi_trades)
    rsi_wr = len([t for t in rsi_trades if t.pnl > 0]) / len(rsi_trades) * 100 if rsi_trades else 0
    keltner_pnl = sum(t.pnl for t in keltner_trades)
    keltner_wr = len([t for t in keltner_trades if t.pnl > 0]) / len(keltner_trades) * 100 if keltner_trades else 0
    orb_pnl = sum(t.pnl for t in orb_trades)
    orb_wr = len([t for t in orb_trades if t.pnl > 0]) / len(orb_trades) * 100 if orb_trades else 0

    # RSI BUY/SELL breakdown
    rsi_buy = [t for t in rsi_trades if t.direction == 'BUY']
    rsi_sell = [t for t in rsi_trades if t.direction == 'SELL']
    rsi_buy_pnl = sum(t.pnl for t in rsi_buy)
    rsi_sell_pnl = sum(t.pnl for t in rsi_sell)

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] = year_pnl.get(y, 0) + t.pnl

    return {
        'n': len(pnls), 'total_pnl': total_pnl, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'sharpe': sharpe, 'max_dd': max_dd, 'max_dd_pct': max_dd_pct,
        'rsi_n': len(rsi_trades), 'rsi_pnl': rsi_pnl, 'rsi_wr': rsi_wr,
        'rsi_buy_n': len(rsi_buy), 'rsi_buy_pnl': rsi_buy_pnl,
        'rsi_sell_n': len(rsi_sell), 'rsi_sell_pnl': rsi_sell_pnl,
        'keltner_n': len(keltner_trades), 'keltner_pnl': keltner_pnl, 'keltner_wr': keltner_wr,
        'orb_n': len(orb_trades), 'orb_pnl': orb_pnl, 'orb_wr': orb_wr,
        'year_pnl': year_pnl,
    }


# ═══════════════════════════════════════════════════════════════
# A/B Test Runner
# ═══════════════════════════════════════════════════════════════

def run_variant(m15_df: pd.DataFrame, h1_df: pd.DataFrame,
                label: str, **kwargs) -> Dict:
    print(f"\n  [{label}]", flush=True)

    # Reset global state for each variant
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    engine = MultiTimeframeEngine(m15_df, h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['rsi_filtered'] = engine.rsi_filtered_count
    stats['rsi_total'] = engine.rsi_total_signals
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    print(f"    {stats['n']} trades (H1={engine.h1_entry_count}, M15={engine.m15_entry_count}), "
          f"RSI filtered={engine.rsi_filtered_count}/{engine.rsi_total_signals}")
    return stats


def print_comparison(variants: List[Dict]):
    print("\n")
    print("=" * 120)
    print("  M15 Multi-Timeframe Backtest: Strategy Optimization A/B Test")
    print("=" * 120)

    header = (f"  {'Variant':<40} {'Trades':>6} {'Sharpe':>8} {'PnL':>10} "
              f"{'MaxDD':>10} {'DD%':>6} {'WinR%':>7} {'RR':>6}")
    print(header)
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    baseline = variants[0]
    for v in variants:
        if v['n'] == 0:
            print(f"  {v['label']:<40}   (no trades)")
            continue
        sharpe_diff = v['sharpe'] - baseline['sharpe']
        marker = f"({sharpe_diff:+.2f})" if v != baseline else ""
        print(f"  {v['label']:<40} {v['n']:>6} {v['sharpe']:>8.2f} ${v['total_pnl']:>9.2f} "
              f"${v['max_dd']:>9.2f} {v['max_dd_pct']:>5.1f}% {v['win_rate']:>6.1f}% {v['rr']:>5.2f} {marker}")

    # Strategy breakdown
    print(f"\n  --- Strategy Breakdown ---")
    print(f"  {'Variant':<40} {'K_N':>5} {'K_PnL':>10} {'ORB_N':>5} {'ORB_PnL':>10} "
          f"{'RSI_N':>5} {'RSI_PnL':>10} {'RSI_WR':>6} {'Filt':>5}")
    print(f"  {'-'*40} {'-'*5} {'-'*10} {'-'*5} {'-'*10} "
          f"{'-'*5} {'-'*10} {'-'*6} {'-'*5}")
    for v in variants:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<40} {v['keltner_n']:>5} ${v['keltner_pnl']:>9.2f} "
              f"{v.get('orb_n', 0):>5} ${v.get('orb_pnl', 0):>9.2f} "
              f"{v['rsi_n']:>5} ${v['rsi_pnl']:>9.2f} {v['rsi_wr']:>5.1f}% {v.get('rsi_filtered', 0):>5}")

    # RSI BUY/SELL breakdown
    print(f"\n  --- RSI Direction Breakdown ---")
    print(f"  {'Variant':<40} {'BUY_N':>6} {'BUY_PnL':>10} {'SELL_N':>6} {'SELL_PnL':>10}")
    print(f"  {'-'*40} {'-'*6} {'-'*10} {'-'*6} {'-'*10}")
    for v in variants:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<40} {v.get('rsi_buy_n', 0):>6} ${v.get('rsi_buy_pnl', 0):>9.2f} "
              f"{v.get('rsi_sell_n', 0):>6} ${v.get('rsi_sell_pnl', 0):>9.2f}")

    # Year-by-year
    all_years = set()
    for v in variants:
        all_years.update(v.get('year_pnl', {}).keys())
    years = sorted(all_years)

    if years:
        print(f"\n  --- Year-by-Year PnL ($) ---")
        print(f"  {'Year':<6}", end='')
        for v in variants:
            print(f"  {v['label'][:18]:>20}", end='')
        print()

        for y in years:
            print(f"  {y:<6}", end='')
            for v in variants:
                pnl = v.get('year_pnl', {}).get(y, 0)
                print(f"  ${pnl:>19.2f}", end='')
            print()

    print("\n" + "=" * 120)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='M15 Multi-TF Backtest')
    parser.add_argument('--start', default='2020-01-01', help='Start date')
    parser.add_argument('--end', default=None, help='End date')
    args = parser.parse_args()

    print("=" * 60)
    print("  M15 Multi-Timeframe Backtest")
    print("=" * 60)

    # Load M15 data
    print("\nLoading M15 data...")
    m15_df = load_m15()

    # Filter date range
    if args.start:
        m15_df = m15_df[m15_df.index >= pd.Timestamp(args.start, tz='UTC')]
    if args.end:
        m15_df = m15_df[m15_df.index <= pd.Timestamp(args.end, tz='UTC')]

    # Load H1 data
    print("\nLoading H1 data...")
    h1_df = load_h1_aligned(H1_CSV_PATH, m15_df.index[0])

    # Prepare indicators
    print("\nPreparing indicators...")
    print("  M15 indicators...", end='', flush=True)
    m15_df = prepare_indicators(m15_df)
    print(" done")
    print("  H1 indicators...", end='', flush=True)
    h1_df = prepare_indicators(h1_df)
    print(" done")

    # Compute ATR percentile on H1
    if 'atr_percentile' not in h1_df.columns:
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        h1_df['atr_percentile'] = h1_df['atr_percentile'].fillna(0.5)

    # Rebuild H1 lookup after indicator prep
    print(f"\nM15: {len(m15_df)} bars, H1: {len(h1_df)} bars")

    variants = []

    # A: Baseline
    variants.append(run_variant(m15_df, h1_df, "A: Baseline (H1+M15)"))

    # B1: RSI block when H1 ADX > 40 (proven winner from prior test)
    variants.append(run_variant(m15_df, h1_df, "B1: RSI ADX>40",
                                rsi_adx_filter=40))

    # C1: ADX>40 + RSI BUY-only
    variants.append(run_variant(m15_df, h1_df, "C1: ADX>40 + RSI BUY-only",
                                rsi_adx_filter=40, rsi_sell_enabled=False))

    # C2: RSI BUY-only (no ADX filter)
    variants.append(run_variant(m15_df, h1_df, "C2: RSI BUY-only",
                                rsi_sell_enabled=False))

    # D1: ADX>40 + block RSI when ATR_pct < 0.30
    variants.append(run_variant(m15_df, h1_df, "D1: ADX>40 + ATR_pct>=0.30",
                                rsi_adx_filter=40, rsi_atr_pct_min_filter=0.30))

    # D2: ADX>40 + block RSI when ATR_pct < 0.20
    variants.append(run_variant(m15_df, h1_df, "D2: ADX>40 + ATR_pct>=0.20",
                                rsi_adx_filter=40, rsi_atr_pct_min_filter=0.20))

    # E1: ORB max_hold = 16 M15 bars (~4 hours)
    variants.append(run_variant(m15_df, h1_df, "E1: ORB hold 16 bars(4h)",
                                orb_max_hold_m15=16))

    # E2: ORB max_hold = 12 M15 bars (~3 hours)
    variants.append(run_variant(m15_df, h1_df, "E2: ORB hold 12 bars(3h)",
                                orb_max_hold_m15=12))

    # F1: ATR regime lots (all strategies)
    variants.append(run_variant(m15_df, h1_df, "F1: ATR regime lots",
                                atr_regime_lots=True))

    # F2: ADX>40 + ATR regime lots
    variants.append(run_variant(m15_df, h1_df, "F2: ADX>40 + ATR regime",
                                rsi_adx_filter=40, atr_regime_lots=True))

    print_comparison(variants)


if __name__ == '__main__':
    main()
