"""
止盈/追踪止盈 A/B 对比回测
===========================
测试 3 个优化方案:

  A : 基准 (当前: 激活=2.5ATR, 距离=0.5ATR, 无保本)
  B1: 降低激活门槛 (1.5ATR)
  B2: 高波动自适应 (ATR%>=90时 激活=1.5ATR, 否则2.5ATR)
  B3: 保本止损 (浮盈>=1.0ATR → SL移到入场价)
  B4: B2+B3 组合 (自适应激活 + 保本)
  B5: 降低激活+保本 (1.5ATR激活 + 保本)

用法:
    python backtest_trailing_test.py <csv_path>
"""
import sys
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional

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
    BacktestEngine,
    LOT_SCALE_BY_LOSSES,
    _aggregate_daily_pnl,
)


class TrailingTestEngine(BacktestEngine):
    """支持多种追踪止盈变体的回测引擎。"""

    def __init__(self, df: pd.DataFrame,
                 trailing_activate_atr: float = 2.5,
                 trailing_distance_atr: float = 0.5,
                 adaptive_trailing: bool = False,
                 adaptive_high_vol_threshold: float = 0.90,
                 adaptive_activate_atr: float = 1.5,
                 breakeven_enabled: bool = False,
                 breakeven_activate_atr: float = 1.0,
                 label: str = ""):
        super().__init__(df, adaptive_lots=False)
        self.trailing_activate_atr = trailing_activate_atr
        self.trailing_distance_atr = trailing_distance_atr
        self.adaptive_trailing = adaptive_trailing
        self.adaptive_high_vol_threshold = adaptive_high_vol_threshold
        self.adaptive_activate_atr = adaptive_activate_atr
        self.breakeven_enabled = breakeven_enabled
        self.breakeven_activate_atr = breakeven_activate_atr
        self.label = label

        # 统计追踪止盈触发次数
        self.trailing_exits = 0
        self.breakeven_exits = 0
        self.sl_exits = 0
        self.tp_exits = 0
        self.timeout_exits = 0

    def _check_exits(self, window: pd.DataFrame, bar: pd.Series, bar_time):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None

            # 1. SL/TP (bar 内 High/Low)
            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = f"止损: {low:.2f} <= SL{pos.sl_price:.2f}"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = f"止盈: {high:.2f} >= TP{pos.tp_price:.2f}"
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = f"止损: {high:.2f} >= SL{pos.sl_price:.2f}"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = f"止盈: {low:.2f} <= TP{pos.tp_price:.2f}"
                    exit_price = pos.tp_price

            if reason:
                if "止损" in reason:
                    self.sl_exits += 1
                else:
                    self.tp_exits += 1

            # 2. 保本止损 (在 SL/TP 之后、trailing 之前检查)
            if not reason and self.breakeven_enabled and pos.strategy == 'keltner':
                atr = float(window.iloc[-1]['ATR'])
                if not np.isnan(atr) and atr > 0:
                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                    else:
                        float_profit = pos.entry_price - low

                    be_threshold = atr * self.breakeven_activate_atr
                    if float_profit >= be_threshold:
                        # 把 SL 移到入场价 (保本), 只升不降
                        if pos.direction == 'BUY':
                            new_sl = pos.entry_price
                            if new_sl > pos.sl_price:
                                pos.sl_price = new_sl
                        else:
                            new_sl = pos.entry_price
                            if new_sl < pos.sl_price:
                                pos.sl_price = new_sl

                        # 检查保本是否当 bar 就被触发
                        if pos.direction == 'BUY' and low <= pos.sl_price:
                            reason = f"保本止损: 浮盈>={be_threshold:.1f}, SL移至{pos.sl_price:.2f}"
                            exit_price = pos.sl_price
                            self.breakeven_exits += 1
                        elif pos.direction == 'SELL' and high >= pos.sl_price:
                            reason = f"保本止损: 浮盈>={be_threshold:.1f}, SL移至{pos.sl_price:.2f}"
                            exit_price = pos.sl_price
                            self.breakeven_exits += 1

            # 3. Keltner 追踪止盈
            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                atr = float(window.iloc[-1]['ATR'])
                if not np.isnan(atr) and atr > 0:
                    if pos.direction == 'BUY':
                        float_profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        float_profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                    # 确定激活门槛
                    if self.adaptive_trailing:
                        atr_pct = float(window.iloc[-1].get('atr_percentile', 0.5))
                        if atr_pct >= self.adaptive_high_vol_threshold:
                            activate_mult = self.adaptive_activate_atr
                        else:
                            activate_mult = self.trailing_activate_atr
                    else:
                        activate_mult = self.trailing_activate_atr

                    activate_threshold = atr * activate_mult
                    if float_profit >= activate_threshold:
                        trail_distance = atr * self.trailing_distance_atr
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_distance
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason = f"追踪止盈: 极值{pos.extreme_price:.2f}, 追踪价{pos.trailing_stop_price:.2f}"
                                exit_price = pos.trailing_stop_price
                                self.trailing_exits += 1
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = f"追踪止盈: 极值{pos.extreme_price:.2f}, 追踪价{pos.trailing_stop_price:.2f}"
                                exit_price = pos.trailing_stop_price
                                self.trailing_exits += 1

            # 4. 信号出场 (非 Keltner)
            if not reason and pos.strategy != 'keltner':
                exit_sig = check_exit_signal(window, pos.strategy, pos.direction)
                if exit_sig:
                    reason = exit_sig
                    exit_price = close

            # 5. 时间止损
            max_hold = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
            if not reason and pos.bars_held >= max_hold:
                reason = f"超时平仓: {pos.bars_held} bars >= {max_hold}"
                exit_price = close
                self.timeout_exits += 1

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)


def calc_stats(trades: List[TradeRecord], equity_curve: List[float]) -> Dict:
    if not trades:
        return {'n': 0}

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    daily_pnl = _aggregate_daily_pnl(trades)
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    else:
        sharpe = 0.0

    eq = np.array(equity_curve) if equity_curve else np.array([config.CAPITAL])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    # 分策略统计
    keltner_trades = [t for t in trades if t.strategy == 'keltner']
    other_trades = [t for t in trades if t.strategy != 'keltner']
    keltner_pnl = sum(t.pnl for t in keltner_trades)
    keltner_wr = len([t for t in keltner_trades if t.pnl > 0]) / len(keltner_trades) * 100 if keltner_trades else 0
    other_pnl = sum(t.pnl for t in other_trades)

    # 分年度
    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] = year_pnl.get(y, 0) + t.pnl

    return {
        'n': len(pnls),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'rr': rr,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'max_dd_pct': max_dd_pct,
        'keltner_n': len(keltner_trades),
        'keltner_pnl': keltner_pnl,
        'keltner_wr': keltner_wr,
        'other_n': len(other_trades),
        'other_pnl': other_pnl,
        'year_pnl': year_pnl,
    }


def run_variant(df: pd.DataFrame, label: str, **kwargs) -> Dict:
    print(f"\n  [{label}] ...", end='', flush=True)

    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    engine = TrailingTestEngine(df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['trailing_exits'] = engine.trailing_exits
    stats['breakeven_exits'] = engine.breakeven_exits
    stats['sl_exits'] = engine.sl_exits
    stats['tp_exits'] = engine.tp_exits
    stats['timeout_exits'] = engine.timeout_exits
    print(f" done ({stats['n']} trades)")
    return stats


def print_comparison(variants: List[Dict]):
    print("\n")
    print("=" * 100)
    print("  Trailing Stop A/B Test Report")
    print("=" * 100)

    header = (f"  {'Variant':<35} {'Trades':>6} {'Sharpe':>8} {'PnL':>10} "
              f"{'MaxDD':>10} {'DD%':>6} {'WinR%':>7} {'RR':>6}")
    print(header)
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    baseline = variants[0]
    for v in variants:
        if v['n'] == 0:
            continue
        sharpe_diff = v['sharpe'] - baseline['sharpe']
        marker = f"({sharpe_diff:+.2f})" if v != baseline else ""
        print(f"  {v['label']:<35} {v['n']:>6} {v['sharpe']:>8.2f} ${v['total_pnl']:>9.2f} "
              f"${v['max_dd']:>9.2f} {v['max_dd_pct']:>5.1f}% {v['win_rate']:>6.1f}% {v['rr']:>5.2f} {marker}")

    # Keltner-only stats
    print(f"\n  --- Keltner Strategy Stats ---")
    print(f"  {'Variant':<35} {'K_Trades':>8} {'K_PnL':>10} {'K_WR%':>7}")
    print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*7}")
    for v in variants:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<35} {v['keltner_n']:>8} ${v['keltner_pnl']:>9.2f} {v['keltner_wr']:>6.1f}%")

    # Exit reason breakdown
    print(f"\n  --- Exit Reason Breakdown (Keltner) ---")
    print(f"  {'Variant':<35} {'SL':>6} {'TP':>6} {'Trail':>6} {'BE':>6} {'Time':>6}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for v in variants:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<35} {v['sl_exits']:>6} {v['tp_exits']:>6} "
              f"{v['trailing_exits']:>6} {v['breakeven_exits']:>6} {v['timeout_exits']:>6}")

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

    print("\n" + "=" * 100)


def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_trailing_test.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    print("=" * 60)
    print("  Trailing Stop Optimization A/B Test")
    print("=" * 60)

    print("\nLoading data...")
    df = load_csv(csv_path)
    print(f"Bars: {len(df)}")

    print("Preparing indicators...")
    df = prepare_indicators(df)

    # 计算 ATR percentile (B2 方案需要)
    if 'atr_percentile' not in df.columns:
        df['atr_percentile'] = df['ATR'].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df['atr_percentile'] = df['atr_percentile'].fillna(0.5)

    variants = []

    # A: Baseline (current: activate=2.5, distance=0.5, no breakeven)
    variants.append(run_variant(df, "A: Baseline (2.5/0.5)"))

    # B1: Lower activation threshold
    variants.append(run_variant(df, "B1: Low activate (1.5/0.5)",
                                trailing_activate_atr=1.5))

    # B2: Adaptive trailing (high vol -> tighter)
    variants.append(run_variant(df, "B2: Adaptive (2.5->1.5 hi-vol)",
                                adaptive_trailing=True,
                                adaptive_high_vol_threshold=0.90,
                                adaptive_activate_atr=1.5))

    # B3: Breakeven stop (move SL to entry after 1.0 ATR profit)
    variants.append(run_variant(df, "B3: Breakeven (1.0ATR)",
                                breakeven_enabled=True,
                                breakeven_activate_atr=1.0))

    # B4: Adaptive + Breakeven
    variants.append(run_variant(df, "B4: Adaptive + Breakeven",
                                adaptive_trailing=True,
                                adaptive_high_vol_threshold=0.90,
                                adaptive_activate_atr=1.5,
                                breakeven_enabled=True,
                                breakeven_activate_atr=1.0))

    # B5: Low activate + Breakeven
    variants.append(run_variant(df, "B5: 1.5ATR + Breakeven",
                                trailing_activate_atr=1.5,
                                breakeven_enabled=True,
                                breakeven_activate_atr=1.0))

    # B6: Even lower activation (2.0 ATR)
    variants.append(run_variant(df, "B6: Mid activate (2.0/0.5)",
                                trailing_activate_atr=2.0))

    print_comparison(variants)


if __name__ == '__main__':
    main()
