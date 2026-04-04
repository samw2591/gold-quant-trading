"""
M15 RSI 逆势过滤 + 连续亏损冷却 A/B 回测
==========================================
测试方案:

  A : 基准 (当前策略, TRAILING_ACTIVATE_ATR=1.5 已更新)
  B1: M15 RSI ADX>40 禁入
  B2: M15 RSI ADX>35 禁入
  B3: M15 RSI ATR_percentile>0.90 禁入
  B4: 连续同向亏损 2 笔 → 该方向冷却 8 小时
  B5: B1 + B4 组合
  B6: B2 + B4 组合

用法:
    python backtest_rsi_cooldown_test.py <csv_path>
"""
import sys
from collections import defaultdict
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
    _aggregate_daily_pnl,
)


class RSICooldownEngine(BacktestEngine):
    """支持 M15 RSI 逆势过滤和连续亏损冷却的回测引擎。"""

    def __init__(self, df: pd.DataFrame,
                 rsi_adx_filter: float = 0,
                 rsi_atr_pct_filter: float = 0,
                 consecutive_loss_cooldown: bool = False,
                 consecutive_loss_threshold: int = 2,
                 consecutive_cooldown_hours: int = 8,
                 label: str = ""):
        super().__init__(df, adaptive_lots=False)
        self.rsi_adx_filter = rsi_adx_filter
        self.rsi_atr_pct_filter = rsi_atr_pct_filter
        self.consecutive_loss_cooldown = consecutive_loss_cooldown
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.consecutive_cooldown_hours = consecutive_cooldown_hours
        self.label = label

        self.direction_losses: Dict[str, int] = defaultdict(int)
        self.direction_cooldown_until: Dict[str, pd.Timestamp] = {}

        self.rsi_filtered_count = 0
        self.consecutive_filtered_count = 0

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

            # ── M15 RSI: ADX 过滤 (ADX高=趋势极强, RSI均值回归失效) ──
            if strategy == 'm15_rsi' and self.rsi_adx_filter > 0:
                adx_val = float(window.iloc[-1].get('ADX', 0))
                if not pd.isna(adx_val) and adx_val > self.rsi_adx_filter:
                    self.rsi_filtered_count += 1
                    continue

            # ── M15 RSI: ATR percentile 过滤 (极端波动=趋势行情概率高) ──
            if strategy == 'm15_rsi' and self.rsi_atr_pct_filter > 0:
                atr_pct = float(window.iloc[-1].get('atr_percentile', 0.5))
                if not pd.isna(atr_pct) and atr_pct > self.rsi_atr_pct_filter:
                    self.rsi_filtered_count += 1
                    continue

            # ── 连续同向亏损冷却 ──
            if self.consecutive_loss_cooldown:
                dir_cd = self.direction_cooldown_until.get(direction)
                if dir_cd and bar_time <= dir_cd:
                    self.consecutive_filtered_count += 1
                    continue

            lots = calc_auto_lot_size(0, sl)
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

    def _close_position(self, pos, exit_price, exit_time, reason):
        if pos.direction == 'BUY':
            pnl_check = exit_price - pos.entry_price
        else:
            pnl_check = pos.entry_price - exit_price

        super()._close_position(pos, exit_price, exit_time, reason)

        if pnl_check < 0:
            self.direction_losses[pos.direction] += 1
            if (self.consecutive_loss_cooldown and
                    self.direction_losses[pos.direction] >= self.consecutive_loss_threshold):
                self.direction_cooldown_until[pos.direction] = (
                    pd.Timestamp(exit_time) + pd.Timedelta(hours=self.consecutive_cooldown_hours)
                )
        else:
            self.direction_losses[pos.direction] = 0


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
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
    else:
        sharpe = 0.0

    eq = np.array(equity_curve) if equity_curve else np.array([config.CAPITAL])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    rsi_trades = [t for t in trades if t.strategy == 'm15_rsi']
    keltner_trades = [t for t in trades if t.strategy == 'keltner']
    rsi_pnl = sum(t.pnl for t in rsi_trades)
    rsi_wr = len([t for t in rsi_trades if t.pnl > 0]) / len(rsi_trades) * 100 if rsi_trades else 0
    keltner_pnl = sum(t.pnl for t in keltner_trades)
    keltner_wr = len([t for t in keltner_trades if t.pnl > 0]) / len(keltner_trades) * 100 if keltner_trades else 0

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] = year_pnl.get(y, 0) + t.pnl

    return {
        'n': len(pnls), 'total_pnl': total_pnl, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'sharpe': sharpe, 'max_dd': max_dd, 'max_dd_pct': max_dd_pct,
        'rsi_n': len(rsi_trades), 'rsi_pnl': rsi_pnl, 'rsi_wr': rsi_wr,
        'keltner_n': len(keltner_trades), 'keltner_pnl': keltner_pnl, 'keltner_wr': keltner_wr,
        'year_pnl': year_pnl,
    }


def run_variant(df: pd.DataFrame, label: str, **kwargs) -> Dict:
    print(f"\n  [{label}] ...", end='', flush=True)

    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    engine = RSICooldownEngine(df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['rsi_filtered'] = engine.rsi_filtered_count
    stats['consec_filtered'] = engine.consecutive_filtered_count
    print(f" done ({stats['n']} trades, RSI_filt={engine.rsi_filtered_count}, "
          f"consec_filt={engine.consecutive_filtered_count})")
    return stats


def print_comparison(variants: List[Dict]):
    print("\n")
    print("=" * 105)
    print("  M15 RSI Filter + Consecutive Loss Cooldown A/B Test")
    print("=" * 105)

    header = (f"  {'Variant':<38} {'Trades':>6} {'Sharpe':>8} {'PnL':>10} "
              f"{'MaxDD':>10} {'DD%':>6} {'WinR%':>7} {'RR':>6}")
    print(header)
    print(f"  {'-'*38} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    baseline = variants[0]
    for v in variants:
        if v['n'] == 0:
            print(f"  {v['label']:<38}   (no trades)")
            continue
        sharpe_diff = v['sharpe'] - baseline['sharpe']
        marker = f"({sharpe_diff:+.2f})" if v != baseline else ""
        print(f"  {v['label']:<38} {v['n']:>6} {v['sharpe']:>8.2f} ${v['total_pnl']:>9.2f} "
              f"${v['max_dd']:>9.2f} {v['max_dd_pct']:>5.1f}% {v['win_rate']:>6.1f}% {v['rr']:>5.2f} {marker}")

    print(f"\n  --- M15 RSI Strategy Only ---")
    print(f"  {'Variant':<38} {'RSI_N':>6} {'RSI_PnL':>10} {'RSI_WR%':>8} {'Filtered':>8}")
    print(f"  {'-'*38} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    for v in variants:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<38} {v['rsi_n']:>6} ${v['rsi_pnl']:>9.2f} {v['rsi_wr']:>7.1f}% "
              f"{v.get('rsi_filtered', 0):>8}")

    print(f"\n  --- Keltner Strategy Only ---")
    print(f"  {'Variant':<38} {'K_N':>6} {'K_PnL':>10} {'K_WR%':>8} {'Consec_F':>8}")
    print(f"  {'-'*38} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")
    for v in variants:
        if v['n'] == 0:
            continue
        print(f"  {v['label']:<38} {v['keltner_n']:>6} ${v['keltner_pnl']:>9.2f} {v['keltner_wr']:>7.1f}% "
              f"{v.get('consec_filtered', 0):>8}")

    all_years = set()
    for v in variants:
        all_years.update(v.get('year_pnl', {}).keys())
    years = sorted(all_years)

    if years:
        show = variants[:4]
        print(f"\n  --- Year-by-Year PnL ($) [first 4 variants] ---")
        print(f"  {'Year':<6}", end='')
        for v in show:
            print(f"  {v['label'][:20]:>22}", end='')
        print()

        for y in years:
            print(f"  {y:<6}", end='')
            for v in show:
                pnl = v.get('year_pnl', {}).get(y, 0)
                print(f"  ${pnl:>21.2f}", end='')
            print()

    print("\n" + "=" * 105)


def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_rsi_cooldown_test.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    print("=" * 60)
    print("  M15 RSI Filter + Cooldown A/B Test")
    print("=" * 60)

    print("\nLoading data...")
    df = load_csv(csv_path)
    print(f"Bars: {len(df)}")

    print("Preparing indicators...")
    df = prepare_indicators(df)

    if 'atr_percentile' not in df.columns:
        df['atr_percentile'] = df['ATR'].rolling(500, min_periods=50).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        df['atr_percentile'] = df['atr_percentile'].fillna(0.5)

    variants = []

    # A: Baseline (with new trailing 1.5 ATR)
    variants.append(run_variant(df, "A: Baseline (trailing 1.5ATR)"))

    # B1: M15 RSI: ADX > 40 -> block all RSI signals
    variants.append(run_variant(df, "B1: RSI block ADX>40",
                                rsi_adx_filter=40))

    # B2: M15 RSI: ADX > 35 -> block all RSI signals
    variants.append(run_variant(df, "B2: RSI block ADX>35",
                                rsi_adx_filter=35))

    # B3: M15 RSI: ATR percentile > 0.90 -> block
    variants.append(run_variant(df, "B3: RSI block ATR_pct>0.90",
                                rsi_atr_pct_filter=0.90))

    # B4: Consecutive same-direction loss cooldown (2 losses -> 8h cooldown)
    variants.append(run_variant(df, "B4: ConsecLoss 2x -> 8h CD",
                                consecutive_loss_cooldown=True,
                                consecutive_loss_threshold=2,
                                consecutive_cooldown_hours=8))

    # B5: B1 + B4
    variants.append(run_variant(df, "B5: RSI ADX>40 + ConsecLoss",
                                rsi_adx_filter=40,
                                consecutive_loss_cooldown=True,
                                consecutive_loss_threshold=2,
                                consecutive_cooldown_hours=8))

    # B6: B2 + B4
    variants.append(run_variant(df, "B6: RSI ADX>35 + ConsecLoss",
                                rsi_adx_filter=35,
                                consecutive_loss_cooldown=True,
                                consecutive_loss_threshold=2,
                                consecutive_cooldown_hours=8))

    print_comparison(variants)


if __name__ == '__main__':
    main()
