"""
历史行情回放测试器
==================
回放 3/26 12:00 ~ 3/27 12:00 UTC 的 M5 K 线数据,
验证 Keltner Trailing Stop + 顺势加仓逻辑.

独立运行, 不连接 MT4, 日志写入 data/test_trade_log.json.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import config
from strategies.signals import prepare_indicators, check_keltner_signal, calc_auto_lot_size

TEST_LOG_FILE = config.DATA_DIR / "test_trade_log.json"

REPLAY_START = pd.Timestamp("2026-03-26 12:00", tz="UTC")
REPLAY_END = pd.Timestamp("2026-03-27 12:00", tz="UTC")
DATA_START = "2026-03-24"
DATA_END = "2026-03-28"


class SimTrader:
    """内存中的虚拟交易器, 复用实盘的参数和算法."""

    def __init__(self):
        self.positions: List[Dict] = []
        self.tracking: Dict[str, Dict] = {}
        self.trade_log: List[Dict] = []
        self.total_pnl = 0.0
        self._next_ticket = 90000001

    def open_position(self, direction: str, price: float, sl: float, tp: float,
                      lots: float, atr: float, adx: float, bar_time: pd.Timestamp,
                      is_pyramid: bool = False):
        ticket = self._next_ticket
        self._next_ticket += 1
        tk = str(ticket)

        pos = {
            "ticket": ticket, "direction": direction,
            "open_price": price, "current_price": price,
            "lots": lots, "sl": sl, "tp": tp,
            "profit": 0.0, "open_time": str(bar_time),
        }
        self.positions.append(pos)

        self.tracking[tk] = {
            "strategy": "keltner", "direction": direction,
            "entry_price": price, "entry_date": str(bar_time),
            "lots": lots, "sl": sl,
            "trailing_stop_price": 0,
            "extreme_price": price,
            "is_pyramid": is_pyramid,
            "last_profit": 0.0, "last_price": price,
        }

        tag = "加仓" if is_pyramid else "入场"
        print(f"\n{'='*70}")
        print(f"  >>> 空单{tag} #{ticket}  @ {bar_time}")
        print(f"      价格: {price:.2f}  |  ATR: {atr:.2f}  |  ADX: {adx:.1f}")
        print(f"      手数: {lots}  |  SL: {sl:.2f}  |  TP: {tp:.2f}")
        if is_pyramid:
            first = self.positions[0]
            fp = first["open_price"] - price
            print(f"      首仓浮盈: ${fp:.2f}  (首仓入场价 {first['open_price']:.2f})")
        print(f"{'='*70}")

        self.trade_log.append({
            "action": "OPEN", "ticket": ticket,
            "direction": direction, "price": price,
            "lots": lots, "sl": sl, "tp": tp,
            "atr": round(atr, 2), "adx": round(adx, 1),
            "is_pyramid": is_pyramid,
            "time": str(bar_time),
        })

    def update_positions(self, current_price: float):
        """更新所有持仓的浮动盈亏和极值价格."""
        for pos in self.positions:
            if pos["direction"] == "SELL":
                pos["profit"] = round(
                    (pos["open_price"] - current_price) * pos["lots"] * config.POINT_VALUE_PER_LOT, 2
                )
            else:
                pos["profit"] = round(
                    (current_price - pos["open_price"]) * pos["lots"] * config.POINT_VALUE_PER_LOT, 2
                )
            pos["current_price"] = current_price

            tk = str(pos["ticket"])
            if tk in self.tracking:
                self.tracking[tk]["last_profit"] = pos["profit"]
                self.tracking[tk]["last_price"] = current_price
                direction = self.tracking[tk]["direction"]
                old_extreme = self.tracking[tk].get("extreme_price", current_price)
                if direction == "BUY":
                    self.tracking[tk]["extreme_price"] = max(old_extreme, current_price)
                else:
                    self.tracking[tk]["extreme_price"] = (
                        min(old_extreme, current_price) if old_extreme > 0 else current_price
                    )

    def check_trailing_stop(self, atr: float, bar_time: pd.Timestamp) -> List[int]:
        """检查 Trailing Stop, 返回需要平仓的 ticket 列表."""
        if not config.TRAILING_STOP_ENABLED or atr <= 0:
            return []

        activate_threshold = atr * config.TRAILING_ACTIVATE_ATR
        trail_distance = atr * config.TRAILING_DISTANCE_ATR
        to_close = []

        for pos in self.positions:
            tk = str(pos["ticket"])
            track = self.tracking.get(tk, {})
            direction = track.get("direction", "BUY")
            open_price = pos["open_price"]
            current_price = pos["current_price"]

            if direction == "SELL":
                float_profit_points = open_price - current_price
            else:
                float_profit_points = current_price - open_price

            if float_profit_points < activate_threshold:
                continue

            extreme = track.get("extreme_price", current_price)
            if direction == "BUY":
                extreme = max(extreme, current_price)
                new_trail = round(extreme - trail_distance, 2)
                old_trail = track.get("trailing_stop_price", 0)
                trail_price = max(new_trail, old_trail)
            else:
                extreme = min(extreme, current_price) if extreme > 0 else current_price
                new_trail = round(extreme + trail_distance, 2)
                old_trail = track.get("trailing_stop_price", 0)
                trail_price = min(new_trail, old_trail) if old_trail > 0 else new_trail

            self.tracking[tk]["trailing_stop_price"] = trail_price
            self.tracking[tk]["extreme_price"] = extreme

            triggered = (
                (direction == "BUY" and current_price <= trail_price)
                or (direction == "SELL" and current_price >= trail_price)
            )

            if triggered:
                print(f"  !!! Trailing Stop 触发 #{pos['ticket']}  @ {bar_time}")
                print(f"      价格 {current_price:.2f} 触及追踪防线 {trail_price:.2f}")
                print(f"      极值价: {extreme:.2f}  浮盈点: ${float_profit_points:.2f}")
                to_close.append(pos["ticket"])
            else:
                print(f"      Trailing Stop 激活 #{pos['ticket']}: "
                      f"极值 {extreme:.2f}  追踪防线 {trail_price:.2f}  "
                      f"当前价 {current_price:.2f}  浮盈 ${float_profit_points:.2f}")

        return to_close

    def close_position(self, ticket: int, bar_time: pd.Timestamp, reason: str):
        """平仓指定 ticket."""
        pos = next((p for p in self.positions if p["ticket"] == ticket), None)
        if not pos:
            return
        profit = pos["profit"]
        self.total_pnl = round(self.total_pnl + profit, 2)

        tk = str(ticket)
        is_pyr = self.tracking.get(tk, {}).get("is_pyramid", False)
        tag = "加仓单" if is_pyr else "原始单"

        print(f"\n{'='*70}")
        print(f"  <<< 平仓 #{ticket} ({tag})  @ {bar_time}")
        print(f"      入场: {pos['open_price']:.2f}  平仓: {pos['current_price']:.2f}")
        print(f"      手数: {pos['lots']}  盈亏: ${profit:+.2f}")
        print(f"      原因: {reason}")
        print(f"      累计总盈亏: ${self.total_pnl:+.2f}")
        print(f"{'='*70}")

        self.trade_log.append({
            "action": "CLOSE", "ticket": ticket,
            "direction": pos["direction"],
            "open_price": pos["open_price"],
            "close_price": pos["current_price"],
            "lots": pos["lots"],
            "profit": profit,
            "is_pyramid": is_pyr,
            "reason": reason,
            "time": str(bar_time),
        })

        self.positions = [p for p in self.positions if p["ticket"] != ticket]
        self.tracking.pop(tk, None)

    def can_add_position(self, signal_direction: str, df_window: pd.DataFrame) -> bool:
        """复用实盘 _can_add_position 的判断逻辑."""
        if not config.ADD_POSITION_ENABLED:
            return False
        if not self.positions:
            return False

        latest = df_window.iloc[-1]
        adx = float(latest["ADX"]) if not pd.isna(latest.get("ADX", float("nan"))) else 0
        atr = float(latest["ATR"]) if not pd.isna(latest.get("ATR", float("nan"))) else 0
        if adx < config.ADD_POSITION_MIN_ADX or atr <= 0:
            return False

        same_count = len(self.positions)
        if same_count >= config.KELTNER_MAX_SAME_STRATEGY:
            return False

        existing_dir = self.positions[0]["direction"]
        if signal_direction != existing_dir:
            return False

        profit_threshold = atr * config.ADD_POSITION_MIN_PROFIT_ATR
        min_profit = min(p["profit"] for p in self.positions)
        lots = self.positions[0]["lots"]
        float_profit_points = min_profit / (lots * config.POINT_VALUE_PER_LOT) if lots > 0 else 0

        if float_profit_points < profit_threshold:
            return False

        return True

    @property
    def keltner_count(self) -> int:
        return len(self.positions)

    def save_log(self):
        with open(TEST_LOG_FILE, "w") as f:
            json.dump(self.trade_log, f, indent=2, default=str, ensure_ascii=False)
        print(f"\n  测试日志已保存: {TEST_LOG_FILE}")


def load_m5_data() -> pd.DataFrame:
    """下载 M5 数据并计算指标."""
    print(f"  下载 GC=F M5 数据 ({DATA_START} ~ {DATA_END}) ...")
    df = yf.download("GC=F", start=DATA_START, end=DATA_END, interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if len(df) < 110:
        print(f"  数据不足: {len(df)} 根 K 线 (需要至少 110)")
        sys.exit(1)
    print(f"  获取 {len(df)} 根 M5 K 线  ({df.index[0]} ~ {df.index[-1]})")
    df = prepare_indicators(df)
    return df


def main():
    print("=" * 70)
    print("  黄金量化 — 历史行情回放测试")
    print(f"  回放区间: {REPLAY_START} ~ {REPLAY_END}")
    print(f"  目标行情: 4445 -> 4353 暴跌 (Keltner SELL)")
    print("=" * 70)

    df = load_m5_data()
    trader = SimTrader()

    replay_mask = (df.index >= REPLAY_START) & (df.index <= REPLAY_END)
    replay_indices = df.index[replay_mask]
    print(f"  回放 K 线数: {len(replay_indices)}")
    if len(replay_indices) == 0:
        print("  没有符合回放区间的数据, 退出.")
        sys.exit(1)

    min_window = 105
    bar_count = 0

    for bar_time in replay_indices:
        bar_idx = df.index.get_loc(bar_time)
        if bar_idx < min_window:
            continue

        df_window = df.iloc[:bar_idx + 1]
        latest = df_window.iloc[-1]
        close = float(latest["Close"])
        atr = float(latest["ATR"]) if not pd.isna(latest.get("ATR", float("nan"))) else 0
        adx = float(latest["ADX"]) if not pd.isna(latest.get("ADX", float("nan"))) else 0

        bar_count += 1

        trader.update_positions(close)

        # --- Trailing Stop 检查 ---
        if trader.positions:
            to_close = trader.check_trailing_stop(atr, bar_time)
            for ticket in to_close:
                trader.close_position(ticket, bar_time, "Trailing Stop 触发")
            if not trader.positions:
                continue

        # --- Keltner 信号检测 ---
        sig = check_keltner_signal(df_window)
        if sig and sig["signal"] == "SELL":
            if trader.keltner_count == 0:
                lots = calc_auto_lot_size(0, sig["sl"])
                lots = max(config.MIN_LOT_SIZE, min(config.MAX_LOT_SIZE, lots))
                trader.open_position(
                    direction="SELL", price=close,
                    sl=sig["sl"], tp=sig["tp"], lots=lots,
                    atr=atr, adx=adx, bar_time=bar_time,
                    is_pyramid=False,
                )
            elif trader.can_add_position("SELL", df_window):
                lots = calc_auto_lot_size(0, sig["sl"])
                lots = max(config.MIN_LOT_SIZE, min(config.MAX_LOT_SIZE, lots))
                trader.open_position(
                    direction="SELL", price=close,
                    sl=sig["sl"], tp=sig["tp"], lots=lots,
                    atr=atr, adx=adx, bar_time=bar_time,
                    is_pyramid=True,
                )

    # --- 回放结束, 强制平掉剩余持仓 ---
    if trader.positions:
        last_close = float(df.loc[replay_indices[-1], "Close"])
        trader.update_positions(last_close)
        print(f"\n  回放结束, 强制平仓剩余 {len(trader.positions)} 笔:")
        for pos in list(trader.positions):
            trader.close_position(pos["ticket"], replay_indices[-1], "回放结束强制平仓")

    # --- 汇总 ---
    print(f"\n{'='*70}")
    print(f"  回放完成")
    print(f"  处理 K 线: {bar_count} 根")
    print(f"  总交易笔数: {len(trader.trade_log)}")
    print(f"  总盈亏: ${trader.total_pnl:+.2f}")
    print(f"{'='*70}")

    trader.save_log()


if __name__ == "__main__":
    main()
