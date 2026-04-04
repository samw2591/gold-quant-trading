"""
Strategy Change Monitor — C12 观察期追踪
==========================================
自动追踪策略调整后的实盘表现，对比回测预期。
每日日切时由 gold_runner.py 调用 update()。

观察期：2026-04-03 起，至少 2 周（30+ 笔交易）
预警条件：
  - 连续 3 天亏损
  - 累计胜率 < 60%（回测预期 72%）
  - 累计 RR < 0.35（回测预期 0.51）
  - 单日亏损 > $200（≈10% 本金）
"""
import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import config

MONITOR_FILE = config.DATA_DIR / "strategy_monitor.json"

CHANGE_DATE = "2026-04-03"
BACKTEST_EXPECTATIONS = {
    "sharpe": 2.54,
    "win_rate": 72.0,
    "rr": 0.51,
    "config": "Trail 0.8/0.25 + SL 3.5ATR + TP 5.0ATR + ADX 18",
}

ALERT_THRESHOLDS = {
    "consecutive_loss_days": 3,
    "min_win_rate": 60.0,
    "min_rr": 0.35,
    "max_daily_loss": -200.0,
}


def _load() -> dict:
    if MONITOR_FILE.exists():
        return json.loads(MONITOR_FILE.read_text(encoding='utf-8'))
    return {
        "change_date": CHANGE_DATE,
        "config": BACKTEST_EXPECTATIONS["config"],
        "backtest": BACKTEST_EXPECTATIONS,
        "thresholds": ALERT_THRESHOLDS,
        "days": [],
    }


def _save(data: dict):
    tmp = MONITOR_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(MONITOR_FILE)


def update(equity_record: dict) -> Optional[str]:
    """
    每日日切时调用。传入当天的 equity_curve 记录。
    返回告警消息（如有），否则返回 None。
    """
    data = _load()
    record_date = equity_record.get("date", str(date.today()))

    if record_date < CHANGE_DATE:
        return None

    if any(d["date"] == record_date for d in data["days"]):
        return None

    daily_pnl = equity_record.get("daily_pnl", 0)
    trades = equity_record.get("daily_trades", 0)
    wins = equity_record.get("daily_wins", 0)
    losses = equity_record.get("daily_losses", 0)

    strats = equity_record.get("strategies", {})
    kc_pnl = strats.get("keltner", {}).get("pnl", 0)
    kc_n = strats.get("keltner", {}).get("trades", 0)
    rsi_pnl = strats.get("m15_rsi", {}).get("pnl", 0)
    rsi_n = strats.get("m15_rsi", {}).get("trades", 0)
    orb_pnl = strats.get("orb", {}).get("pnl", 0)
    orb_n = strats.get("orb", {}).get("trades", 0)

    day = {
        "date": record_date,
        "pnl": daily_pnl,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "kc_n": kc_n, "kc_pnl": kc_pnl,
        "rsi_n": rsi_n, "rsi_pnl": rsi_pnl,
        "orb_n": orb_n, "orb_pnl": orb_pnl,
        "equity": equity_record.get("equity", 0),
    }
    data["days"].append(day)

    total_trades = sum(d["trades"] for d in data["days"])
    total_wins = sum(d["wins"] for d in data["days"])
    total_losses = sum(d["losses"] for d in data["days"])
    cum_pnl = sum(d["pnl"] for d in data["days"])
    cum_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0

    win_pnls = []
    loss_pnls = []
    for d in data["days"]:
        if d["trades"] > 0:
            avg = d["pnl"] / d["trades"]
            for _ in range(d["wins"]):
                win_pnls.append(abs(avg) if avg > 0 else abs(d["pnl"] / max(d["wins"], 1)))
            for _ in range(d["losses"]):
                loss_pnls.append(abs(avg) if avg < 0 else abs(d["pnl"] / max(d["losses"], 1)))

    avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
    avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 1
    cum_rr = avg_win / avg_loss if avg_loss > 0 else 0

    data["summary"] = {
        "observation_days": len(data["days"]),
        "total_trades": total_trades,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "cum_pnl": round(cum_pnl, 2),
        "cum_win_rate": round(cum_wr, 1),
        "cum_rr": round(cum_rr, 2),
        "cum_kc_pnl": round(sum(d["kc_pnl"] for d in data["days"]), 2),
        "cum_rsi_pnl": round(sum(d["rsi_pnl"] for d in data["days"]), 2),
        "cum_orb_pnl": round(sum(d["orb_pnl"] for d in data["days"]), 2),
        "last_updated": str(datetime.now()),
    }

    alerts = []

    recent = data["days"][-3:]
    if len(recent) >= 3 and all(d["pnl"] < 0 for d in recent):
        loss_str = ', '.join(f'${d["pnl"]:+.0f}' for d in recent)
        alerts.append(f"连续 3 天亏损 ({loss_str})")

    if total_trades >= 10 and cum_wr < ALERT_THRESHOLDS["min_win_rate"]:
        alerts.append(f"累计胜率 {cum_wr:.1f}% < {ALERT_THRESHOLDS['min_win_rate']}% 预警线 (回测预期 {BACKTEST_EXPECTATIONS['win_rate']}%)")

    if total_trades >= 10 and cum_rr < ALERT_THRESHOLDS["min_rr"]:
        alerts.append(f"累计 RR {cum_rr:.2f} < {ALERT_THRESHOLDS['min_rr']} 预警线 (回测预期 {BACKTEST_EXPECTATIONS['rr']})")

    if daily_pnl < ALERT_THRESHOLDS["max_daily_loss"]:
        alerts.append(f"单日亏损 ${daily_pnl:.0f} 超过 ${ALERT_THRESHOLDS['max_daily_loss']:.0f} 预警线")

    data["alerts"] = alerts
    _save(data)

    if not alerts:
        return None

    msg_lines = [
        f"⚠️ <b>策略观察期预警</b>",
        f"",
        f"配置: {BACKTEST_EXPECTATIONS['config']}",
        f"观察天数: {len(data['days'])} / 14",
        f"累计: {total_trades}笔, 胜率{cum_wr:.1f}%, RR {cum_rr:.2f}, PnL ${cum_pnl:+.0f}",
        f"",
    ]
    for a in alerts:
        msg_lines.append(f"🚨 {a}")
    msg_lines.append(f"")
    msg_lines.append(f"回测预期: 胜率{BACKTEST_EXPECTATIONS['win_rate']}%, RR {BACKTEST_EXPECTATIONS['rr']}")
    msg_lines.append(f"建议: 如持续偏离预期，考虑回退 SL/TP/ADX 到旧值")

    return "\n".join(msg_lines)


def get_report() -> str:
    """生成当前观察期的文本摘要，供 Telegram 日报使用。"""
    data = _load()
    if not data.get("days"):
        return ""

    s = data.get("summary", {})
    days = len(data["days"])
    target = 14

    lines = [
        f"📋 <b>C12 观察期 ({days}/{target} 天)</b>",
        f"  累计: {s.get('total_trades', 0)}笔, 胜率{s.get('cum_win_rate', 0):.1f}%, RR {s.get('cum_rr', 0):.2f}",
        f"  PnL: ${s.get('cum_pnl', 0):+.0f} (KC ${s.get('cum_kc_pnl', 0):+.0f} / RSI ${s.get('cum_rsi_pnl', 0):+.0f} / ORB ${s.get('cum_orb_pnl', 0):+.0f})",
        f"  回测预期: 胜率{BACKTEST_EXPECTATIONS['win_rate']}%, RR {BACKTEST_EXPECTATIONS['rr']}",
    ]

    if days >= target:
        lines.append(f"  ✅ 观察期已结束，请评估是否保留当前配置")

    return "\n".join(lines)
