"""
回测 A/B 对比实验
=================
在 backtest.py 基础上，对比：
  A: 基准 (当前策略)
  B1: 周一仓位×0.7
  B2: SELL 仓位×0.5
  B3: SELL ADX门槛提高到 28
  B4: B1+B2 组合 (周一降仓 + 做空缩仓)

用法:
    python backtest_ab_test.py <csv_path>
"""
import sys
import copy
from dataclasses import dataclass
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


class ABTestEngine(BacktestEngine):
    """可配置的回测引擎，支持 A/B 测试参数。"""

    def __init__(self, df: pd.DataFrame,
                 monday_lot_scale: float = 1.0,
                 sell_lot_scale: float = 1.0,
                 sell_adx_min: float = 24.0,
                 label: str = ""):
        super().__init__(df, adaptive_lots=False)
        self.monday_lot_scale = monday_lot_scale
        self.sell_lot_scale = sell_lot_scale
        self.sell_adx_min = sell_adx_min
        self.label = label

    def _check_entries(self, window: pd.DataFrame, bar_time):
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

            cooldown = self.cooldown_until.get(strategy)
            if cooldown and bar_time <= cooldown:
                continue

            if strategy in active_strategies:
                continue

            if current_dir and direction != current_dir:
                continue

            # SELL 信号的额外 ADX 门槛
            if direction == 'SELL' and self.sell_adx_min > 24.0:
                adx_val = float(window.iloc[-1]['ADX'])
                if pd.notna(adx_val) and adx_val < self.sell_adx_min:
                    continue

            lots = calc_auto_lot_size(0, sl)

            # 周一降仓
            if hasattr(bar_time, 'weekday') and bar_time.weekday() == 0:
                lots = round(lots * self.monday_lot_scale, 2)

            # 做空缩仓
            if direction == 'SELL':
                lots = round(lots * self.sell_lot_scale, 2)

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


def calc_stats(trades: List[TradeRecord], equity_curve: List[float]) -> Dict:
    """计算回测统计指标。"""
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

    # 分方向统计
    buy_trades = [t for t in trades if t.direction == 'BUY']
    sell_trades = [t for t in trades if t.direction == 'SELL']
    buy_pnl = sum(t.pnl for t in buy_trades)
    sell_pnl = sum(t.pnl for t in sell_trades)
    buy_wr = len([t for t in buy_trades if t.pnl > 0]) / len(buy_trades) * 100 if buy_trades else 0
    sell_wr = len([t for t in sell_trades if t.pnl > 0]) / len(sell_trades) * 100 if sell_trades else 0

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
        'buy_n': len(buy_trades),
        'buy_pnl': buy_pnl,
        'buy_wr': buy_wr,
        'sell_n': len(sell_trades),
        'sell_pnl': sell_pnl,
        'sell_wr': sell_wr,
        'year_pnl': year_pnl,
    }


def run_variant(df: pd.DataFrame, label: str, **kwargs) -> Dict:
    """运行一个变体并返回统计。"""
    print(f"\n  [{label}] 运行中...", end='', flush=True)

    # 重置全局状态
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    engine = ABTestEngine(df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['trades'] = trades
    return stats


def print_comparison(variants: List[Dict]):
    """打印对比表格。"""
    print("\n")
    print("=" * 90)
    print("  A/B 回测对比报告")
    print("=" * 90)

    # 主指标
    header = f"  {'变体':<30} {'笔数':>6} {'Sharpe':>8} {'总PnL':>10} {'MaxDD':>10} {'DD%':>6} {'胜率':>7} {'RR':>6}"
    print(header)
    print(f"  {'-'*30} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    baseline = variants[0]
    for v in variants:
        sharpe_diff = v['sharpe'] - baseline['sharpe']
        marker = f"({sharpe_diff:+.2f})" if v != baseline else ""
        print(f"  {v['label']:<30} {v['n']:>6} {v['sharpe']:>8.2f} ${v['total_pnl']:>9.2f} "
              f"${v['max_dd']:>9.2f} {v['max_dd_pct']:>5.1f}% {v['win_rate']:>6.1f}% {v['rr']:>5.2f} {marker}")

    # BUY vs SELL 分解
    print(f"\n  --- BUY vs SELL 分解 ---")
    print(f"  {'变体':<30} {'BUY笔':>6} {'BUY PnL':>10} {'BUY胜率':>8} "
          f"{'SELL笔':>6} {'SELL PnL':>10} {'SELL胜率':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*10} {'-'*8}")

    for v in variants:
        print(f"  {v['label']:<30} {v['buy_n']:>6} ${v['buy_pnl']:>9.2f} {v['buy_wr']:>7.1f}% "
              f"{v['sell_n']:>6} ${v['sell_pnl']:>9.2f} {v['sell_wr']:>7.1f}%")

    # 分年度对比 (Sharpe 太难按年算, 用年度PnL)
    all_years = set()
    for v in variants:
        all_years.update(v.get('year_pnl', {}).keys())
    years = sorted(all_years)

    if years:
        print(f"\n  --- 分年度 PnL ($) ---")
        print(f"  {'年份':<6}", end='')
        for v in variants:
            print(f"  {v['label'][:15]:>16}", end='')
        print()

        wins_count = {v['label']: 0 for v in variants}

        for y in years:
            print(f"  {y:<6}", end='')
            year_vals = []
            for v in variants:
                pnl = v.get('year_pnl', {}).get(y, 0)
                year_vals.append((v['label'], pnl))
                print(f"  ${pnl:>15.2f}", end='')
            print()

            # 哪个变体在这一年最好
            if len(year_vals) > 1:
                best_label = max(year_vals, key=lambda x: x[1])[0]
                wins_count[best_label] = wins_count.get(best_label, 0) + 1

        print(f"\n  年度最优次数: ", end='')
        for label, count in wins_count.items():
            print(f"{label[:15]}={count}  ", end='')
        print()

    print("\n" + "=" * 90)


def main():
    if len(sys.argv) < 2:
        print("用法: python backtest_ab_test.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    print("=" * 60)
    print("  A/B 回测对比实验")
    print("=" * 60)

    print("\n加载数据...")
    df = load_csv(csv_path)
    print(f"K 线: {len(df)}")

    print("计算指标...")
    df = prepare_indicators(df)

    variants = []

    # A: 基准
    variants.append(run_variant(df, "A: 基准 (当前策略)"))

    # B1: 周一仓位×0.7
    variants.append(run_variant(df, "B1: 周一仓位x0.7",
                                monday_lot_scale=0.7))

    # B2: SELL 仓位×0.5
    variants.append(run_variant(df, "B2: SELL仓位x0.5",
                                sell_lot_scale=0.5))

    # B3: SELL ADX门槛 28
    variants.append(run_variant(df, "B3: SELL ADX>=28",
                                sell_adx_min=28.0))

    # B4: 周一降仓 + 做空缩仓
    variants.append(run_variant(df, "B4: 周一x0.7 + SELLx0.5",
                                monday_lot_scale=0.7,
                                sell_lot_scale=0.5))

    # B5: 周一降仓 + SELL ADX>=28
    variants.append(run_variant(df, "B5: 周一x0.7 + SELL ADX>=28",
                                monday_lot_scale=0.7,
                                sell_adx_min=28.0))

    print_comparison(variants)


if __name__ == '__main__':
    main()
