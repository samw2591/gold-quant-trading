"""
共享出场逻辑
============
Trailing Stop（ATR Regime 自适应）和时间衰减止盈的核心计算逻辑。
实盘 (gold_trader.py) 和模拟盘 (paper_trader.py) 共用此模块。
"""
from typing import Optional, Tuple


def calc_trailing_params(atr: float, atr_percentile: float) -> Tuple[float, float]:
    """
    根据 ATR 百分位计算 Mega Trail 追踪止盈参数。

    Returns:
        (activate_threshold, trail_distance) — 均为价格距离（非 ATR 倍数）
    """
    if atr_percentile > 0.70:
        activate = atr * 0.4
        distance = atr * 0.10
    elif atr_percentile < 0.30:
        activate = atr * 0.7
        distance = atr * 0.25
    else:
        activate = atr * 0.5
        distance = atr * 0.15
    return round(activate, 2), round(distance, 2)


def check_trailing_exit(
    direction: str,
    current_price: float,
    open_price: float,
    atr: float,
    atr_percentile: float,
    extreme_price: float,
    prev_trail_price: float,
) -> Tuple[Optional[str], float, float, float]:
    """
    检查 Trailing Stop 是否触发。

    Returns:
        (reason_or_None, new_extreme, new_trail_price, activate_threshold)
        reason 为 None 表示未触发平仓（可能已激活追踪但价格未触及）。
    """
    if atr <= 0 or open_price <= 0:
        return None, extreme_price, prev_trail_price, 0

    activate_threshold, trail_distance = calc_trailing_params(atr, atr_percentile)

    if direction == 'BUY':
        float_profit = current_price - open_price
    else:
        float_profit = open_price - current_price

    if float_profit < activate_threshold:
        return None, extreme_price, prev_trail_price, activate_threshold

    if direction == 'BUY':
        extreme = max(extreme_price, current_price)
        new_trail = round(extreme - trail_distance, 2)
        trail_price = max(new_trail, prev_trail_price)
        triggered = current_price <= trail_price
    else:
        extreme = min(extreme_price, current_price) if extreme_price > 0 else current_price
        new_trail = round(extreme + trail_distance, 2)
        trail_price = min(new_trail, prev_trail_price) if prev_trail_price > 0 else new_trail
        triggered = current_price >= trail_price

    if triggered:
        reason = (
            f"📈 Trailing Stop: 浮盈${float_profit:.2f} "
            f"(激活阈值${activate_threshold:.2f}), "
            f"价格{current_price:.2f}触及追踪止盈{trail_price:.2f}"
        )
        return reason, extreme, trail_price, activate_threshold

    return None, extreme, trail_price, activate_threshold


def check_time_decay_tp(
    direction: str,
    current_price: float,
    open_price: float,
    hold_hours: float,
    atr: float,
    trailing_active: bool,
    start_hour: float = 1.0,
    atr_start: float = 0.30,
    atr_step: float = 0.10,
) -> Optional[str]:
    """
    时间衰减止盈：盈利持仓长时间未触发 trailing 时，逐步降低锁利门槛。

    只在 trailing 尚未激活时生效，避免与 trailing stop 冲突。

    Returns:
        平仓理由字符串，或 None（未触发）
    """
    if trailing_active or hold_hours < start_hour or atr <= 0:
        return None

    decay_hours = hold_hours - start_hour
    min_profit_atr = max(0.0, atr_start - decay_hours * atr_step)
    min_profit = atr * min_profit_atr

    if direction == 'BUY':
        float_pnl = current_price - open_price
    else:
        float_pnl = open_price - current_price

    if float_pnl > 0 and float_pnl >= min_profit:
        return (
            f"⏳ 时间衰减止盈: 持仓{hold_hours:.1f}h, "
            f"浮盈${float_pnl:.2f} >= 门槛${min_profit:.2f} "
            f"({min_profit_atr:.2f}×ATR)"
        )

    return None
