"""
Telegram 通知模块
================
在关键交易事件时推送消息到手机
"""
import logging
import requests
import config

log = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"


def send_telegram(message: str):
    """发送 Telegram 消息"""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        resp = requests.post(TELEGRAM_API, data={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
        if not resp.json().get("ok"):
            log.debug(f"Telegram发送失败: {resp.text}")
    except Exception as e:
        log.debug(f"Telegram异常: {e}")


def notify_open(strategy: str, direction: str, lots: float, price: float, sl: float, reason: str):
    """开仓通知"""
    emoji = "📈" if direction == "BUY" else "📉"
    send_telegram(
        f"{emoji} <b>开仓 {direction}</b>\n"
        f"策略: {strategy}\n"
        f"手数: {lots}  价格: ${price:.2f}\n"
        f"止损: ${sl:.2f}\n"
        f"原因: {reason}"
    )


def notify_close(ticket: int, strategy: str, profit: float, reason: str):
    """平仓通知"""
    emoji = "✅" if profit >= 0 else "❌"
    send_telegram(
        f"{emoji} <b>平仓 #{ticket}</b>\n"
        f"策略: {strategy}\n"
        f"盈亏: ${profit:+.2f}\n"
        f"原因: {reason}"
    )


def notify_stop_review(daily_pnl: float):
    """日内限亏停止通知"""
    send_telegram(
        f"🚨🚨🚨 <b>系统已停止</b> 🚨🚨🚨\n\n"
        f"日内亏损: ${daily_pnl:.2f}\n"
        f"已达日限亏{config.DAILY_MAX_LOSSES}笔\n\n"
        f"⚠️ 明日自动恢复"
    )


def notify_daily_report(total_pnl: float, daily_pnl: float, trade_count: int,
                        equity_record: dict = None):
    """每日绩效报告（含净值曲线 + 分策略明细）"""
    emoji = "🟢" if daily_pnl >= 0 else "🔴"
    lines = [
        f"📊 <b>每日绩效报告</b>",
        "",
        f"{emoji} 当日盈亏: ${daily_pnl:+.2f}",
        f"💰 累计盈亏: ${total_pnl:+.2f}",
        f"📊 总交易笔数: {trade_count}",
        f"🛡️ 止损余量: ${config.MAX_TOTAL_LOSS + total_pnl:.2f}",
    ]

    if equity_record:
        equity = equity_record.get("equity", 0)
        d_trades = equity_record.get("daily_trades", 0)
        d_wins = equity_record.get("daily_wins", 0)
        d_losses = equity_record.get("daily_losses", 0)
        win_rate = f"{d_wins / d_trades * 100:.0f}%" if d_trades > 0 else "N/A"

        lines.append("")
        lines.append(f"💎 账户净值: ${equity:,.2f}")
        lines.append(f"📈 当日: {d_trades}笔 (胜{d_wins}/负{d_losses}) 胜率{win_rate}")

        strats = equity_record.get("strategies", {})
        if strats:
            lines.append("")
            lines.append("<b>分策略明细:</b>")
            for name, s in strats.items():
                s_emoji = "✅" if s["pnl"] >= 0 else "❌"
                lines.append(f"  {s_emoji} {name}: {s['trades']}笔 ${s['pnl']:+.2f}")

    send_telegram("\n".join(lines))


def notify_system_start():
    """系统启动通知"""
    send_telegram(
        f"🥇 <b>黄金量化系统启动</b>\n\n"
        f"品种: {config.SYMBOL}\n"
        f"风险/笔: ${config.RISK_PER_TRADE}\n"
        f"日限亏: {config.DAILY_MAX_LOSSES}笔\n"
        f"总限亏: ${config.MAX_TOTAL_LOSS}"
    )


def notify_error(error_msg: str):
    """系统异常通知"""
    send_telegram(f"⚠️ <b>系统异常</b>\n\n{error_msg}")
