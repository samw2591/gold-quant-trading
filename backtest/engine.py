"""
Unified Backtest Engine
========================
Single parameterized engine that replaces the previous inheritance chain:
  MultiTimeframeEngine → Round2Engine → RegimeEngine
  → IntradayAdaptiveEngine / CooldownEngine / ParamExploreEngine

All behavior is controlled via constructor parameters — no subclassing needed.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

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


# ═══════════════════════════════════════════════════════════════
# Data types (shared across the package)
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    strategy: str
    direction: str        # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    lots: float
    sl_distance: float
    tp_distance: float
    sl_price: float = 0.0
    tp_price: float = 0.0
    extreme_price: float = 0.0
    trailing_stop_price: float = 0.0
    bars_held: int = 0

    def __post_init__(self):
        if self.direction == 'BUY':
            self.sl_price = self.entry_price - self.sl_distance
            self.tp_price = self.entry_price + self.tp_distance
            self.extreme_price = self.entry_price
        else:
            self.sl_price = self.entry_price + self.sl_distance
            self.tp_price = self.entry_price - self.tp_distance
            self.extreme_price = self.entry_price


@dataclass
class TradeRecord:
    strategy: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lots: float
    pnl: float
    exit_reason: str
    bars_held: int


# ═══════════════════════════════════════════════════════════════
# Unified Engine
# ═══════════════════════════════════════════════════════════════

class BacktestEngine:
    """Dual-timeframe (M15 primary + H1) backtest engine.

    All previously separate engine behaviors are controlled by parameters:
      - trailing_activate_atr / trailing_distance_atr: trailing stop tuning
      - sl_atr_mult / tp_atr_mult: override signal SL/TP with ATR multiples
      - keltner_adx_threshold: override Keltner ADX entry filter
      - max_positions: override config.MAX_POSITIONS
      - cooldown_hours: override config.COOLDOWN_MINUTES (in hours)
      - regime_config: ATR-percentile based parameter adaptation
      - intraday_adaptive / choppy_threshold / kc_only_threshold: trend gating
      - min_entry_gap_hours: global entry cooldown
      - rsi_adx_filter / rsi_atr_pct_filter / etc.: M15 RSI filters
      - rsi_buy_threshold / rsi_sell_threshold: custom RSI thresholds
      - spread_cost: per-trade transaction cost
      - atr_regime_lots: ATR-based lot sizing
    """

    M15_WINDOW = 150
    H1_WINDOW = 150

    def __init__(
        self,
        m15_df: pd.DataFrame,
        h1_df: pd.DataFrame,
        *,
        # Trailing stop
        trailing_activate_atr: float = 0,
        trailing_distance_atr: float = 0,
        # SL/TP overrides
        sl_atr_mult: float = 0,
        tp_atr_mult: float = 0,
        # Keltner ADX threshold override
        keltner_adx_threshold: float = 0,
        # Position / cooldown overrides
        max_positions: int = 0,
        cooldown_hours: float = 0,
        # Regime-adaptive (replaces RegimeEngine)
        regime_config: Optional[Dict] = None,
        # Intraday trend gating (replaces IntradayAdaptiveEngine)
        intraday_adaptive: bool = False,
        choppy_threshold: float = 0.35,
        kc_only_threshold: float = 0.60,
        # Global entry gap (replaces CooldownEngine)
        min_entry_gap_hours: float = 0,
        # M15 RSI filters
        rsi_adx_filter: float = 0,
        rsi_atr_pct_filter: float = 0,
        rsi_sell_enabled: bool = True,
        rsi_atr_pct_min_filter: float = 0,
        # Custom RSI thresholds (replaces ParamExploreEngine)
        rsi_buy_threshold: float = 0,
        rsi_sell_threshold: float = 0,
        # ORB
        orb_max_hold_m15: int = 0,
        # Lot sizing
        atr_regime_lots: bool = False,
        # Transaction cost
        spread_cost: float = 0.0,
        spread_model: str = "fixed",        # "fixed" | "atr_scaled" | "session_aware"
        spread_base: float = 0.30,          # base spread for dynamic models
        spread_max: float = 3.0,            # max spread cap
        # Macro regime (P4)
        macro_df: Optional[pd.DataFrame] = None,
        macro_regime_enabled: bool = False,
        # Label
        label: str = "",
    ):
        self.m15_df = m15_df
        self.h1_df = h1_df
        self.h1_lookup = self._build_h1_lookup(h1_df)
        self.label = label

        # Trailing stop params
        self._trail_act = trailing_activate_atr
        self._trail_dist = trailing_distance_atr
        # Originals saved for regime reset
        self._trail_act_base = trailing_activate_atr
        self._trail_dist_base = trailing_distance_atr

        # SL/TP
        self._sl_atr_mult = sl_atr_mult
        self._sl_atr_mult_base = sl_atr_mult
        self._tp_atr_mult = tp_atr_mult

        # Keltner
        self._kc_adx_threshold = keltner_adx_threshold

        # Positions / cooldown
        self._max_pos = max_positions or config.MAX_POSITIONS
        self._cooldown_hours_override = cooldown_hours

        # Regime
        self._regime_config = regime_config

        # Intraday adaptive
        self._intraday_adaptive = intraday_adaptive
        self._choppy_threshold = choppy_threshold
        self._kc_only_threshold = kc_only_threshold
        self._current_score = 0.5
        self._current_regime = 'neutral'
        self._cached_date = None
        self._cached_h1_count = 0
        self._h1_date_map: Dict = {}
        if intraday_adaptive:
            self._precompute_h1_dates()

        # Global entry gap
        self._min_entry_gap_hours = min_entry_gap_hours
        self._last_entry_time = None

        # RSI filters
        self._rsi_adx_filter = rsi_adx_filter
        self._rsi_atr_pct_filter = rsi_atr_pct_filter
        self._rsi_sell_enabled = rsi_sell_enabled
        self._rsi_atr_pct_min_filter = rsi_atr_pct_min_filter
        self._rsi_buy_threshold = rsi_buy_threshold
        self._rsi_sell_threshold = rsi_sell_threshold

        # ORB
        self._orb_max_hold_m15 = orb_max_hold_m15

        # Lots
        self._atr_regime_lots = atr_regime_lots

        # Cost
        self._spread_cost = spread_cost
        self._spread_model = spread_model
        self._spread_base = spread_base
        self._spread_max = spread_max

        # Macro regime
        self._macro_df = macro_df
        self._macro_regime_enabled = macro_regime_enabled
        self._macro_regime_detector = None
        if macro_regime_enabled and macro_df is not None:
            try:
                from macro.regime_detector import MacroRegimeDetector
                self._macro_regime_detector = MacroRegimeDetector()
            except ImportError:
                pass

        # State
        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.daily_loss_count = 0
        self.current_date = None
        self.cooldown_until: Dict[str, datetime] = {}

        # Counters
        self.rsi_filtered_count = 0
        self.rsi_total_signals = 0
        self.h1_entry_count = 0
        self.m15_entry_count = 0
        self.skipped_choppy = 0
        self.skipped_neutral_m15 = 0

    # ── Main loop ─────────────────────────────────────────────

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

            if bar_date != self.current_date:
                self.current_date = bar_date
                self.daily_loss_count = 0

            is_flat = bool(bar.get('is_flat', False))
            if is_flat:
                unrealized = self._calc_unrealized(float(bar['Close']))
                self.equity_curve.append(config.CAPITAL + realized_pnl + unrealized)
                continue

            m15_start_idx = max(0, i - self.M15_WINDOW + 1)
            m15_window = self.m15_df.iloc[m15_start_idx:i + 1]
            h1_window = self._get_h1_window(bar_time)

            # 1. Check exits
            self._check_exits(m15_window, h1_window, bar, bar_time)

            # 2. Check entries
            if self.daily_loss_count < config.DAILY_MAX_LOSSES:
                is_h1_boundary = (bar_time.minute == 0)

                if is_h1_boundary and h1_window is not None and len(h1_window) >= 50:
                    self._check_h1_entries(h1_window, bar_time)

                if len(m15_window) >= 105:
                    self._check_m15_entries(m15_window, h1_window, bar_time)

            realized_pnl = sum(t.pnl for t in self.trades)
            unrealized = self._calc_unrealized(float(bar['Close']))
            self.equity_curve.append(config.CAPITAL + realized_pnl + unrealized)

        if self.positions:
            last_bar = self.m15_df.iloc[-1]
            last_time = self.m15_df.index[-1]
            for pos in list(self.positions):
                self._close_position(pos, float(last_bar['Close']), last_time, "backtest_end")

        print(f" done!")
        return self.trades

    # ── Exits ─────────────────────────────────────────────────

    def _check_exits(self, m15_window, h1_window, bar, bar_time):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        # Apply regime-adaptive parameters if configured
        if self._regime_config and h1_window is not None and len(h1_window) > 0:
            atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
            if not pd.isna(atr_pct):
                regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
                rc = self._regime_config.get(regime, {})
                self._trail_act = rc.get('trail_act', self._trail_act_base)
                self._trail_dist = rc.get('trail_dist', self._trail_dist_base)
                self._sl_atr_mult = rc.get('sl', self._sl_atr_mult_base)

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None
            exit_price = close

            # 1. SL/TP
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

            # 2. Keltner trailing stop
            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                act_atr = self._trail_act or config.TRAILING_ACTIVATE_ATR
                dist_atr = self._trail_dist or config.TRAILING_DISTANCE_ATR
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

            # 3. Signal exit
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

            # 4. Time stop
            if not reason:
                if pos.strategy == 'm15_rsi':
                    max_hold = 15
                elif pos.strategy == 'orb' and self._orb_max_hold_m15 > 0:
                    max_hold = self._orb_max_hold_m15
                else:
                    max_hold_h1 = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
                    max_hold = max_hold_h1 * 4
                if pos.bars_held >= max_hold:
                    reason = f"Timeout:{pos.bars_held}>={max_hold}"
                    exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    # ── H1 Entries ────────────────────────────────────────────

    def _check_h1_entries(self, h1_window, bar_time):
        if len(self.positions) >= self._max_pos:
            return

        # Intraday trend gating
        if self._intraday_adaptive:
            self._update_intraday_score(h1_window, bar_time)
            if self._current_regime == 'choppy':
                self.skipped_choppy += 1
                return

        # Macro regime gating
        if self._macro_regime_enabled and self._macro_regime_detector and self._macro_df is not None:
            try:
                bar_date = pd.Timestamp(bar_time).normalize()
                if bar_date.tz is not None:
                    bar_date = bar_date.tz_localize(None)
                if bar_date in self._macro_df.index:
                    macro_row = self._macro_df.loc[bar_date]
                    m_regime = self._macro_regime_detector.detect_from_row(macro_row)
                    m_weights = self._macro_regime_detector.get_strategy_weights(m_regime)
                    if not m_weights.get('allow_trading', True):
                        self.skipped_choppy += 1
                        return
            except Exception:
                pass

        # Regime-based disable
        if self._regime_config and h1_window is not None and len(h1_window) > 0:
            atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
            if not pd.isna(atr_pct):
                regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
                rc = self._regime_config.get(regime, {})
                if rc.get('disable_keltner', False):
                    return
                if rc.get('keltner_adx', 0) > 0:
                    self._kc_adx_threshold = rc['keltner_adx']

        # Keltner ADX threshold override
        if self._kc_adx_threshold > 0:
            old_threshold = signals_mod.ADX_TREND_THRESHOLD
            signals_mod.ADX_TREND_THRESHOLD = self._kc_adx_threshold
            signals = scan_all_signals(h1_window, 'H1')
            signals_mod.ADX_TREND_THRESHOLD = old_threshold
        else:
            signals = scan_all_signals(h1_window, 'H1')

        if not signals:
            return
        self._process_signals(signals, bar_time, source='H1')

    # ── M15 Entries ───────────────────────────────────────────

    def _check_m15_entries(self, m15_window, h1_window, bar_time):
        if len(self.positions) >= self._max_pos:
            return

        # Intraday trend gating — skip M15 in neutral regime
        if self._intraday_adaptive:
            if self._current_regime == 'choppy':
                self.skipped_choppy += 1
                return
            if self._current_regime == 'neutral':
                self.skipped_neutral_m15 += 1
                return

        # Regime-based disable
        if self._regime_config and h1_window is not None and len(h1_window) > 0:
            atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
            if not pd.isna(atr_pct):
                regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
                rc = self._regime_config.get(regime, {})
                if rc.get('disable_rsi', False):
                    return

        # Custom RSI thresholds — bypass scan_all_signals
        if self._rsi_buy_threshold > 0 or self._rsi_sell_threshold > 0:
            self._check_m15_custom_rsi(m15_window, h1_window, bar_time)
            return

        signals = scan_all_signals(m15_window, 'M15')
        if not signals:
            return

        filtered = []
        for sig in signals:
            self.rsi_total_signals += 1
            blocked = False

            if not self._rsi_sell_enabled and sig.get('signal') == 'SELL':
                self.rsi_filtered_count += 1
                blocked = True

            if not blocked and self._rsi_adx_filter > 0 and h1_window is not None and len(h1_window) > 0:
                adx_val = float(h1_window.iloc[-1].get('ADX', 0))
                if not pd.isna(adx_val) and adx_val > self._rsi_adx_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked and self._rsi_atr_pct_filter > 0 and h1_window is not None and len(h1_window) > 0:
                atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
                if not pd.isna(atr_pct) and atr_pct > self._rsi_atr_pct_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked and self._rsi_atr_pct_min_filter > 0 and h1_window is not None and len(h1_window) > 0:
                atr_pct = float(h1_window.iloc[-1].get('atr_percentile', 0.5))
                if not pd.isna(atr_pct) and atr_pct < self._rsi_atr_pct_min_filter:
                    self.rsi_filtered_count += 1
                    blocked = True

            if not blocked:
                filtered.append(sig)

        if filtered:
            self._process_signals(filtered, bar_time, source='M15')

    def _check_m15_custom_rsi(self, m15_window, h1_window, bar_time):
        """Custom RSI threshold logic (replaces ParamExploreEngine)."""
        latest = m15_window.iloc[-1]
        close = float(latest['Close'])
        rsi2 = float(latest['RSI2'])
        sma50 = float(latest['SMA50'])
        ema100 = float(latest['EMA100'])
        if pd.isna(rsi2) or pd.isna(sma50) or pd.isna(ema100):
            return

        h1_adx_val = 0
        if h1_window is not None and len(h1_window) > 0:
            h1_adx_val = float(h1_window.iloc[-1].get('ADX', 0))
            if pd.isna(h1_adx_val):
                h1_adx_val = 0

        self.rsi_total_signals += 1

        if self._rsi_adx_filter > 0 and h1_adx_val > self._rsi_adx_filter:
            self.rsi_filtered_count += 1
            return

        atr_val = float(latest['ATR']) if not pd.isna(latest['ATR']) else 0
        sl = round(atr_val * signals_mod.ATR_SL_MULTIPLIER, 2) if atr_val > 0 else 15
        sl = max(signals_mod.ATR_SL_MIN, min(signals_mod.ATR_SL_MAX, sl))

        buy_th = self._rsi_buy_threshold or 15
        sell_th = self._rsi_sell_threshold or 85

        sig = None
        if rsi2 < buy_th and close > sma50 and close > ema100:
            sig = {'strategy': 'm15_rsi', 'signal': 'BUY', 'close': close, 'sl': sl, 'tp': 0,
                   'reason': f"RSI BUY: RSI2={rsi2:.1f}<{buy_th}"}
        elif rsi2 > sell_th and close < sma50 and close < ema100:
            sig = {'strategy': 'm15_rsi', 'signal': 'SELL', 'close': close, 'sl': sl, 'tp': 0,
                   'reason': f"RSI SELL: RSI2={rsi2:.1f}>{sell_th}"}

        if sig:
            self._process_signals([sig], bar_time, source='M15')

    # ── Signal processing ─────────────────────────────────────

    def _process_signals(self, signals: List[Dict], bar_time, source: str):
        # Global entry gap check
        if self._min_entry_gap_hours > 0 and self._last_entry_time is not None:
            gap = (pd.Timestamp(bar_time) - self._last_entry_time).total_seconds() / 3600
            if gap < self._min_entry_gap_hours:
                return

        active_strategies = {p.strategy for p in self.positions}
        current_dir = self.positions[0].direction if self.positions else None
        slots = self._max_pos - len(self.positions)
        entered = False

        for sig in signals[:slots]:
            strategy = sig['strategy']
            direction = sig['signal']
            close = sig['close']
            sl = sig.get('sl', config.STOP_LOSS_PIPS)
            tp = sig.get('tp', 0)

            # SL/TP ATR overrides
            if self._sl_atr_mult > 0:
                atr = self._get_h1_atr(self._get_h1_window(bar_time))
                if atr > 0:
                    sl = round(atr * self._sl_atr_mult, 2)
                    sl = max(signals_mod.ATR_SL_MIN, min(signals_mod.ATR_SL_MAX, sl))
            if self._tp_atr_mult > 0:
                atr = self._get_h1_atr(self._get_h1_window(bar_time))
                if atr > 0:
                    tp = round(atr * self._tp_atr_mult, 2)

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

            if self._atr_regime_lots:
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
                strategy=strategy, direction=direction,
                entry_price=close, entry_time=bar_time,
                lots=lots, sl_distance=sl, tp_distance=tp,
            )
            self.positions.append(pos)
            active_strategies.add(strategy)
            if current_dir is None:
                current_dir = direction
            entered = True

            if source == 'H1':
                self.h1_entry_count += 1
            else:
                self.m15_entry_count += 1

        if entered and self._min_entry_gap_hours > 0:
            self._last_entry_time = pd.Timestamp(bar_time)

    # ── Close position ────────────────────────────────────────

    def _calc_dynamic_spread(self, bar_time, h1_atr: float = 0) -> float:
        """Calculate spread cost based on the active model."""
        if self._spread_model == "fixed":
            return self._spread_cost

        if self._spread_model == "atr_scaled":
            if h1_atr <= 0:
                return self._spread_base
            h1_window = self._get_h1_window(bar_time)
            atr_pct = 0.5
            if h1_window is not None and len(h1_window) > 0:
                val = h1_window.iloc[-1].get('atr_percentile', 0.5)
                if not pd.isna(val):
                    atr_pct = float(val)
            scaled = self._spread_base * (1 + atr_pct)
            return min(scaled, self._spread_max)

        if self._spread_model == "session_aware":
            hour = pd.Timestamp(bar_time).hour
            if 0 <= hour < 8:       # Asia session
                mult = 1.5
            elif 8 <= hour < 14:    # London session
                mult = 1.0
            elif 14 <= hour < 21:   # NY session (tightest)
                mult = 0.8
            else:                   # Late/close
                mult = 2.0
            return min(self._spread_base * mult, self._spread_max)

        return self._spread_cost

    def _close_position(self, pos: Position, exit_price: float, exit_time, reason: str):
        if pos.direction == 'BUY':
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price
        pnl = round(pnl_points * pos.lots * config.POINT_VALUE_PER_LOT, 2)

        h1_atr = 0
        h1w = self._get_h1_window(exit_time)
        if h1w is not None and len(h1w) > 0:
            atr_val = h1w.iloc[-1].get('ATR', 0)
            if not pd.isna(atr_val):
                h1_atr = float(atr_val)

        spread = self._calc_dynamic_spread(exit_time, h1_atr)
        if spread > 0:
            pnl -= round(spread * pos.lots * config.POINT_VALUE_PER_LOT, 2)

        trade = TradeRecord(
            strategy=pos.strategy, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time, exit_time=exit_time,
            lots=pos.lots, pnl=pnl, exit_reason=reason,
            bars_held=pos.bars_held,
        )
        self.trades.append(trade)
        self.positions.remove(pos)

        if pnl < 0:
            self.daily_loss_count += 1
            hours = self._cooldown_hours_override or (config.COOLDOWN_MINUTES / 60)
            self.cooldown_until[pos.strategy] = (
                pd.Timestamp(exit_time) + pd.Timedelta(hours=hours)
            )

    # ── Intraday adaptive ─────────────────────────────────────

    def _precompute_h1_dates(self):
        self._h1_date_map = {}
        if self.h1_df is not None:
            dates = self.h1_df.index.date
            for i, d in enumerate(dates):
                if d not in self._h1_date_map:
                    self._h1_date_map[d] = []
                self._h1_date_map[d].append(i)

    def _update_intraday_score(self, h1_window, bar_time):
        if h1_window is None or len(h1_window) < 2:
            return
        bar_date = pd.Timestamp(bar_time).date()
        h1_len = len(h1_window)
        if bar_date == self._cached_date and h1_len == self._cached_h1_count:
            return

        indices = self._h1_date_map.get(bar_date)
        if indices:
            valid = [i for i in indices if i < h1_len]
            if len(valid) >= 2:
                today_bars = self.h1_df.iloc[valid]
                self._current_score = self._calc_realtime_score(today_bars)
                if self._current_score >= self._kc_only_threshold:
                    self._current_regime = 'trending'
                elif self._current_score >= self._choppy_threshold:
                    self._current_regime = 'neutral'
                else:
                    self._current_regime = 'choppy'
        elif bar_date != self._cached_date:
            self._current_score = 0.5
            self._current_regime = 'neutral'

        self._cached_date = bar_date
        self._cached_h1_count = h1_len

    @staticmethod
    def _calc_realtime_score(today_bars: pd.DataFrame) -> float:
        if len(today_bars) < 2:
            return 0.5
        latest = today_bars.iloc[-1]

        adx = float(latest.get('ADX', 20))
        if np.isnan(adx):
            adx = 20
        adx_score = min(adx / 40.0, 1.0)

        kc_upper = today_bars.get('KC_upper')
        kc_lower = today_bars.get('KC_lower')
        if kc_upper is not None and kc_lower is not None:
            breaks = ((today_bars['Close'] > kc_upper) | (today_bars['Close'] < kc_lower)).sum()
            kc_score = min(float(breaks) / len(today_bars), 1.0)
        else:
            kc_score = 0.0

        ema9 = today_bars.get('EMA9')
        ema21 = today_bars.get('EMA21')
        ema100 = today_bars.get('EMA100')
        if ema9 is not None and ema21 is not None and ema100 is not None:
            bullish = (ema9 > ema21) & (ema21 > ema100)
            bearish = (ema9 < ema21) & (ema21 < ema100)
            aligned = (bullish | bearish).sum()
            ema_score = float(aligned) / len(today_bars)
        else:
            ema_score = 0.0

        day_open = float(today_bars.iloc[0]['Open'])
        day_close = float(latest['Close'])
        day_high = float(today_bars['High'].max())
        day_low = float(today_bars['Low'].min())
        day_range = day_high - day_low
        ti = abs(day_close - day_open) / day_range if day_range > 0.01 else 0.0

        return round(0.30 * adx_score + 0.25 * kc_score + 0.25 * ema_score + 0.20 * ti, 3)

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_h1_lookup(h1_df: pd.DataFrame) -> Dict[pd.Timestamp, int]:
        return {ts: i for i, ts in enumerate(h1_df.index)}

    def _get_h1_window(self, m15_time: pd.Timestamp) -> Optional[pd.DataFrame]:
        h1_time = m15_time.floor('h')
        if h1_time in self.h1_lookup:
            h1_idx = self.h1_lookup[h1_time]
        else:
            h1_times = self.h1_df.index
            mask = h1_times <= m15_time
            if not mask.any():
                return None
            h1_idx = mask.sum() - 1
        h1_len = len(self.h1_df)
        if h1_idx >= h1_len:
            h1_idx = h1_len - 1
        start = max(0, h1_idx - self.H1_WINDOW + 1)
        return self.h1_df.iloc[start:h1_idx + 1]

    @staticmethod
    def _get_h1_atr(h1_window: Optional[pd.DataFrame]) -> float:
        if h1_window is None or len(h1_window) == 0:
            return 0
        atr = float(h1_window.iloc[-1].get('ATR', 0))
        return atr if not np.isnan(atr) else 0

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
    def _reset_global_state():
        orb = get_orb_strategy()
        orb.reset_daily()
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False
