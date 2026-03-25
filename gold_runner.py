"""
黄金量化交易本地运行器 (Windows)
================================
24/5 持续运行，每分钟扫描一次

使用方法:
  1. 先配置 config.py 中的 METATRADER_DIR_PATH
  2. 在MT4上加载 mt4_ea/GoldBridge_EA.mq4
  3. python gold_runner.py

按 Ctrl+C 停止
"""
import sys
import time
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from gold_trader import GoldTrader

# ============================================================
# 时区
# ============================================================
ET = ZoneInfo("America/New_York")  # 用于判断周末
LOCAL_TZ = ZoneInfo("Asia/Singapore")

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / 'gold_runner.log', encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)


def is_market_open():
    """
    黄金交易时间: 周日22:00 UTC - 周五21:00 UTC (几乎24/5)
    简化判断: 周六全天 + 周日早些时候 = 休市
    """
    now_utc = datetime.now(ZoneInfo("UTC"))
    weekday = now_utc.weekday()  # 0=Mon, 6=Sun

    if weekday == 5:  # 周六
        return False, "周六休市"
    if weekday == 6 and now_utc.hour < 22:  # 周日22:00 UTC前
        return False, "周日尚未开市 (22:00 UTC开市)"
    if weekday == 4 and now_utc.hour >= 21:  # 周五21:00 UTC后
        return False, "周五已收市"

    return True, f"交易中 ({now_utc.strftime('%H:%M')} UTC)"


def main():
    log.info("🥇 黄金量化交易系统启动")
    log.info(f"   品种: {config.SYMBOL}")
    log.info(f"   手数: {config.LOT_SIZE}")
    log.info(f"   本金: ${config.CAPITAL}  止损上限: ${config.MAX_TOTAL_LOSS}")
    log.info(f"   扫描频率: 每{config.SCAN_INTERVAL_SECONDS}秒")
    log.info(f"   策略: H1 Keltner+MACD(做多做空) + M15 RSI均值回归")

    trader = GoldTrader()
    signal_scanned_today = False
    last_date = None
    scan_count = 0
    daily_start_pnl = trader.total_pnl.get('total_pnl', 0)
    daily_trades = 0

    while True:
        try:
            now = datetime.now(LOCAL_TZ)
            today = now.date()

            # 新的一天重置 + 打印前一天绩效报告
            if today != last_date:
                if last_date is not None:
                    # 前一天绩效报告
                    current_pnl = trader.total_pnl.get('total_pnl', 0)
                    day_pnl = round(current_pnl - daily_start_pnl, 2)
                    total_trades = trader.total_pnl.get('trade_count', 0)
                    emoji = '🟢' if day_pnl >= 0 else '🔴'
                    log.info(f"\n{'='*60}")
                    log.info(f"📊 每日绩效报告 — {last_date}")
                    log.info(f"  {emoji} 当日盈亏: ${day_pnl:+.2f}")
                    log.info(f"  💰 累计盈亏: ${current_pnl:+.2f}")
                    log.info(f"  📊 总交易笔数: {total_trades}")
                    log.info(f"  🛡️ 止损余量: ${config.MAX_TOTAL_LOSS + current_pnl:.2f}")
                    log.info(f"{'='*60}")
                
                signal_scanned_today = False
                last_date = today
                scan_count = 0
                daily_start_pnl = trader.total_pnl.get('total_pnl', 0)
                daily_trades = 0
                log.info(f"\n📅 {today} ({now.strftime('%A')})")
                log.info(f"  💰 今日起始盈亏: ${daily_start_pnl:+.2f}")

            is_open, status = is_market_open()

            if not is_open:
                log.info(f"💤 {status} | 等待5分钟...")
                time.sleep(300)
                continue

            scan_count += 1

            # 每次扫描都检查持仓出场
            if scan_count % 10 == 1:  # 每10次(约10分钟)打印一次状态
                log.info(f"\n🔄 扫描 #{scan_count} | {status}")

            # 检查出场
            try:
                trader.check_exits_only()
            except Exception as e:
                log.error(f"出场检查出错: {e}")

            # 每5分钟做一次完整信号扫描 (M15策略需要, 每10次循环×30秒≈5分钟)
            if scan_count == 1 or scan_count % 10 == 0:
                log.info(f"\n📊 完整信号扫描 (#{scan_count})")
                try:
                    result = trader.scan_and_trade()
                    entries = result.get('entries', [])
                    exits = result.get('exits', [])

                    for e in entries:
                        log.info(f"📈 买入: {e.get('reason', '')}")
                    for e in exits:
                        log.info(f"📉 平仓: {e.get('reason', '')}")

                except Exception as e:
                    log.error(f"信号扫描出错: {e}")

            # 等待下一次扫描
            time.sleep(config.SCAN_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log.info("\n⏹️ 用户中断，停止运行")
            break
        except Exception as e:
            log.error(f"主循环异常: {e}")
            time.sleep(60)


if __name__ == '__main__':
    main()
