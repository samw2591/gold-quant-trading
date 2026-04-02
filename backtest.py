"""
XAUUSD H1 回测框架
==================
在历史 H1 数据上逐 bar 回放，模拟 Keltner / ORB / 周一跳空回补策略的开平仓。
直接复用 strategies/signals.py 的信号函数和 config.py 的参数。

用法:
    python backtest.py <csv_path> [--start YYYY-MM-DD] [--end YYYY-MM-DD]

CSV 格式: timestamp(ms), open, high, low, close
"""
import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from strategies.signals import (
    prepare_indicators,
    scan_all_signals,
    check_exit_signal,
    calc_auto_lot_size,
    _calc_atr_stop,
    _calc_atr_tp,
    get_orb_strategy,
)
import strategies.signals as signals_mod


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_csv(path: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # 标记休市填充 bar（OHLC 全部相同），保留用于指标计算，但回测时跳过信号扫描
    df['is_flat'] = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])

    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]

    return df


# ═══════════════════════════════════════════════════════════════
# 持仓 & 交易记录
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    strategy: str
    direction: str        # 'BUY' or 'SELL'
    entry_price: float
    entry_time: datetime
    lots: float
    sl_distance: float    # dollar distance
    tp_distance: float    # dollar distance
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
# 回测引擎
# ═══════════════════════════════════════════════════════════════

class BacktestEngine:
    WINDOW_SIZE = 150  # rolling window rows passed to signal functions

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []

        # 风控状态
        self.daily_loss_count = 0
        self.current_date = None
        self.cooldown_until: Dict[str, datetime] = {}

    def run(self) -> List[TradeRecord]:
        print(f"回测区间: {self.df.index[0].strftime('%Y-%m-%d')} → {self.df.index[-1].strftime('%Y-%m-%d')}")
        print(f"总 K 线数: {len(self.df)}")
        print(f"策略: Keltner + ORB + 周一跳空回补")
        print(f"回测中...", end='', flush=True)

        self._reset_global_state()

        lookback = self.WINDOW_SIZE
        total_bars = len(self.df)
        progress_step = total_bars // 10

        for i in range(lookback, total_bars):
            # Fixed-size window view — no copy, O(1)
            start = max(0, i - self.WINDOW_SIZE + 1)
            window = self.df.iloc[start:i + 1]
            bar = self.df.iloc[i]
            bar_time = self.df.index[i]

            # 日切重置
            bar_date = bar_time.date()
            if bar_date != self.current_date:
                self.current_date = bar_date
                self.daily_loss_count = 0

            is_flat = bool(bar.get('is_flat', False))

            if not is_flat:
                # 1. 检查出场（仅交易时段）
                self._check_exits(window, bar, bar_time)

                # 2. 检查入场（仅交易时段）
                if self.daily_loss_count < config.DAILY_MAX_LOSSES:
                    self._check_entries(window, bar_time)

            # 3. 记录净值（所有 bar 都记录）
            unrealized = self._calc_unrealized_pnl(float(bar['Close']))
            realized = sum(t.pnl for t in self.trades)
            self.equity_curve.append(config.CAPITAL + realized + unrealized)

            if progress_step > 0 and (i - lookback) % progress_step == 0:
                print('.', end='', flush=True)

        # 强制平掉剩余持仓
        if self.positions:
            last_bar = self.df.iloc[-1]
            last_time = self.df.index[-1]
            for pos in list(self.positions):
                self._close_position(pos, float(last_bar['Close']), last_time, "回测结束强制平仓")

        print(" 完成!")
        return self.trades

    def _reset_global_state(self):
        """重置 signals.py 中的全局状态"""
        orb = get_orb_strategy()
        orb.reset_daily()
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False

    def _check_exits(self, window: pd.DataFrame, bar: pd.Series, bar_time: datetime):
        high = float(bar['High'])
        low = float(bar['Low'])
        close = float(bar['Close'])

        for pos in list(self.positions):
            pos.bars_held += 1
            reason = None

            # 1. SL/TP 检查 (用 bar 内 High/Low)
            if pos.direction == 'BUY':
                if low <= pos.sl_price:
                    reason = f"止损: 价格{low:.2f} <= SL{pos.sl_price:.2f}"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    reason = f"止盈: 价格{high:.2f} >= TP{pos.tp_price:.2f}"
                    exit_price = pos.tp_price
            else:
                if high >= pos.sl_price:
                    reason = f"止损: 价格{high:.2f} >= SL{pos.sl_price:.2f}"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    reason = f"止盈: 价格{low:.2f} <= TP{pos.tp_price:.2f}"
                    exit_price = pos.tp_price

            # 2. Keltner 追踪止盈
            if not reason and pos.strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                atr = float(window.iloc[-1]['ATR'])
                if not np.isnan(atr) and atr > 0:
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
                                reason = f"追踪止盈: 极值{pos.extreme_price:.2f}, 追踪价{pos.trailing_stop_price:.2f}"
                                exit_price = pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_distance
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason = f"追踪止盈: 极值{pos.extreme_price:.2f}, 追踪价{pos.trailing_stop_price:.2f}"
                                exit_price = pos.trailing_stop_price

            # 3. 信号出场 (非 Keltner — Keltner 靠 SL/TP/trailing/超时)
            if not reason and pos.strategy != 'keltner':
                exit_sig = check_exit_signal(window, pos.strategy, pos.direction)
                if exit_sig:
                    reason = exit_sig
                    exit_price = close

            # 4. 时间止损
            max_hold = config.STRATEGIES.get(pos.strategy, {}).get('max_hold_bars', 15)
            if not reason and pos.bars_held >= max_hold:
                reason = f"超时平仓: {pos.bars_held} bars >= {max_hold}"
                exit_price = close

            if reason:
                self._close_position(pos, exit_price, bar_time, reason)

    def _check_entries(self, window: pd.DataFrame, bar_time: datetime):
        if len(self.positions) >= config.MAX_POSITIONS:
            return

        signals = scan_all_signals(window, 'H1')
        if not signals:
            return

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

            # 冷却期
            cooldown = self.cooldown_until.get(strategy)
            if cooldown and bar_time <= cooldown:
                continue

            # 同策略重复
            if strategy in active_strategies:
                continue

            # 方向冲突
            if current_dir and direction != current_dir:
                continue

            lots = calc_auto_lot_size(0, sl)

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

    def _close_position(self, pos: Position, exit_price: float, exit_time, reason: str):
        if pos.direction == 'BUY':
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price

        pnl = round(pnl_points * pos.lots * config.POINT_VALUE_PER_LOT, 2)

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
            self.cooldown_until[pos.strategy] = pd.Timestamp(exit_time) + pd.Timedelta(hours=cooldown_hours)

    def _calc_unrealized_pnl(self, current_price: float) -> float:
        total = 0.0
        for pos in self.positions:
            if pos.direction == 'BUY':
                pnl_points = current_price - pos.entry_price
            else:
                pnl_points = pos.entry_price - current_price
            total += pnl_points * pos.lots * config.POINT_VALUE_PER_LOT
        return total


# ═══════════════════════════════════════════════════════════════
# 统计报告
# ═══════════════════════════════════════════════════════════════

def print_report(trades: List[TradeRecord], equity_curve: List[float]):
    if not trades:
        print("\n没有交易记录。")
        return

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # Sharpe (日频)
    daily_pnl = _aggregate_daily_pnl(trades)
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    else:
        sharpe = 0.0

    # MaxDD
    eq = np.array(equity_curve) if equity_curve else np.array([config.CAPITAL])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    # 分策略统计
    strat_stats = {}
    for t in trades:
        s = t.strategy
        if s not in strat_stats:
            strat_stats[s] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pnls': []}
        strat_stats[s]['trades'] += 1
        strat_stats[s]['pnl'] += t.pnl
        strat_stats[s]['pnls'].append(t.pnl)
        if t.pnl > 0:
            strat_stats[s]['wins'] += 1

    # 分年度统计
    year_stats = {}
    for t in trades:
        y = t.entry_time.year if hasattr(t.entry_time, 'year') else pd.Timestamp(t.entry_time).year
        if y not in year_stats:
            year_stats[y] = {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pnls': []}
        year_stats[y]['trades'] += 1
        year_stats[y]['pnl'] += t.pnl
        year_stats[y]['pnls'].append(t.pnl)
        if t.pnl > 0:
            year_stats[y]['wins'] += 1

    # 打印
    print("\n" + "=" * 60)
    print("  XAUUSD H1 回测报告")
    print("=" * 60)

    print(f"\n  总交易数:    {len(pnls)}")
    print(f"  盈利笔数:    {len(wins)}  |  亏损笔数:  {len(losses)}")
    print(f"  胜率:        {win_rate:.1f}%")
    print(f"  平均盈利:    ${avg_win:.2f}")
    print(f"  平均亏损:    ${avg_loss:.2f}")
    print(f"  盈亏比 (RR): {rr_ratio:.2f}")
    print(f"  利润因子:    {profit_factor:.2f}")
    print(f"  总盈亏:      ${total_pnl:.2f}")
    print(f"  Sharpe:      {sharpe:.2f}")
    print(f"  MaxDD:       ${max_dd:.2f} ({max_dd_pct:.1f}%)")

    print(f"\n  {'策略':<12} {'笔数':>6} {'胜率':>8} {'总PnL':>10} {'Sharpe':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for s, st in sorted(strat_stats.items()):
        wr = st['wins'] / st['trades'] * 100 if st['trades'] > 0 else 0
        daily = _aggregate_daily_pnl_list(st['pnls'], trades, s)
        s_sharpe = np.mean(daily) / np.std(daily) * np.sqrt(252) if len(daily) > 1 and np.std(daily) > 0 else 0
        print(f"  {s:<12} {st['trades']:>6} {wr:>7.1f}% ${st['pnl']:>9.2f} {s_sharpe:>8.2f}")

    print(f"\n  {'年份':<6} {'笔数':>6} {'胜率':>8} {'总PnL':>10} {'Sharpe':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")
    for y in sorted(year_stats.keys()):
        ys = year_stats[y]
        wr = ys['wins'] / ys['trades'] * 100 if ys['trades'] > 0 else 0
        daily = _aggregate_daily_pnl_list(ys['pnls'], trades, year=y)
        y_sharpe = np.mean(daily) / np.std(daily) * np.sqrt(252) if len(daily) > 1 and np.std(daily) > 0 else 0
        print(f"  {y:<6} {ys['trades']:>6} {wr:>7.1f}% ${ys['pnl']:>9.2f} {y_sharpe:>8.2f}")

    print("\n" + "=" * 60)


def _aggregate_daily_pnl(trades: List[TradeRecord]) -> List[float]:
    """按日聚合 PnL"""
    daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl
    return list(daily.values())


def _aggregate_daily_pnl_list(pnls: List[float], all_trades: List[TradeRecord],
                               strategy: Optional[str] = None, year: Optional[int] = None) -> List[float]:
    """按日聚合特定策略/年份的 PnL"""
    daily = {}
    for t in all_trades:
        if strategy and t.strategy != strategy:
            continue
        if year:
            t_year = t.exit_time.year if hasattr(t.exit_time, 'year') else pd.Timestamp(t.exit_time).year
            if t_year != year:
                continue
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl
    return list(daily.values()) if daily else [0]


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='XAUUSD H1 回测')
    parser.add_argument('csv_path', help='历史数据 CSV 路径')
    parser.add_argument('--start', default=None, help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='结束日期 (YYYY-MM-DD)')
    args = parser.parse_args()

    print("加载数据...")
    df = load_csv(args.csv_path, args.start, args.end)
    print(f"有效 K 线: {len(df)}")

    print("计算指标...")
    df = prepare_indicators(df)

    engine = BacktestEngine(df)
    trades = engine.run()
    print_report(trades, engine.equity_curve)


if __name__ == '__main__':
    main()
