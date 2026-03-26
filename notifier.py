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


def notify_daily_report(total_pnl: float, daily_pnl: float, trade_count: int):
    """每日绩效报告"""
    emoji = "🟢" if daily_pnl >= 0 else "🔴"
    send_telegram(
        f"📊 <b>每日绩效报告</b>\n\n"
        f"{emoji} 当日盈亏: ${daily_pnl:+.2f}\n"
        f"💰 累计盈亏: ${total_pnl:+.2f}\n"
        f"📊 总交易笔数: {trade_count}\n"
        f"🛡️ 止损余量: ${config.MAX_TOTAL_LOSS + total_pnl:.2f}"
    )


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
