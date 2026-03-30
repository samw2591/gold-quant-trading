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
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from gold_trader import GoldTrader
from paper_trader import PaperTrader, setup_paper_strategies

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


def sync_data_to_github():
    """同步交易数据到GitHub (供云端日报cron读取)"""
    try:
        repo_dir = str(Path(__file__).parent)
        # 先拉取远程更新，避免冲突
        subprocess.run(['git', 'pull', 'origin', 'main', '--no-rebase'],
                       cwd=repo_dir, capture_output=True, timeout=30)
        # 推送data目录下的所有JSON文件 (包括data/paper/子目录)
        subprocess.run(['git', 'add', 'data/'], cwd=repo_dir, 
                       capture_output=True, timeout=10)
        result = subprocess.run(
            ['git', 'commit', '-m', f'data sync {datetime.now().strftime("%Y-%m-%d %H:%M")}'],
            cwd=repo_dir, capture_output=True, timeout=10
        )
        if result.returncode == 0:
            push_result = subprocess.run(
                ['git', 'push'], cwd=repo_dir, capture_output=True, timeout=30
            )
            if push_result.returncode == 0:
                log.info("📤 交易数据已同步到GitHub")
            else:
                log.warning(f"git push失败: {push_result.stderr.decode()[:200]}")
        else:
            # 没有变化，不需要推送
            pass
    except Exception as e:
        log.debug(f"数据同步失败: {e}")


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
    log.info(f"   策略: H1 Keltner(状态机)+MACD+ORB(NY开盘突破) + M15 RSI")

    trader = GoldTrader()
    
    # 模拟盘初始化
    paper = PaperTrader()
    setup_paper_strategies(paper)
    
    # 启动时先同步MT4持仓状态，避免重启后重复开仓
    try:
        trader._sync_positions_tracking()
        positions = trader.get_strategy_positions()
        if positions:
            log.info(f"📊 启动时检测到 {len(positions)} 笔持仓:")
            for p in positions:
                tk = str(p['ticket'])
                track = trader.tracking.get(tk, {})
                log.info(f"   #{tk} {track.get('strategy','?')} {track.get('direction','?')} @ {p.get('open_price',0)}")
        else:
            log.info("📊 启动时无持仓")
    except Exception as e:
        log.warning(f"启动同步失败: {e}")
    
    # Telegram启动通知
    try:
        import notifier
        notifier.notify_system_start()
    except:
        pass
    
    signal_scanned_today = False
    last_date = None
    scan_count = 0
    daily_start_pnl = trader.total_pnl.get('total_pnl', 0)
    daily_trades = 0
    last_data_sync_hour = -1  # 数据同步跟踪
    
    # ── 心跳检测状态 ──
    consecutive_disconnect = 0       # 连续掉线次数
    DISCONNECT_ALERT_THRESHOLD = 6   # 连续6次掉线(~3分钟) → 报警
    last_heartbeat_alert = None      # 上次报警时间 (避免刷屏)
    STARTUP_GRACE_SCANS = 6          # 启动后前6次扫描(~3分钟)不检测心跳，给EA启动时间

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
                    try:
                        import json
                        eq_file = config.DATA_DIR / "equity_curve.json"
                        eq_record = None
                        if eq_file.exists():
                            with open(eq_file, "r", encoding="utf-8") as f:
                                eq_data = json.load(f)
                            if eq_data and isinstance(eq_data, list):
                                eq_record = eq_data[-1]
                        notifier.notify_daily_report(current_pnl, day_pnl, total_trades, eq_record)
                    except:
                        pass
                
                signal_scanned_today = False
                last_date = today
                scan_count = 0
                daily_start_pnl = trader.total_pnl.get('total_pnl', 0)
                daily_trades = 0
                log.info(f"\n📅 {today} ({now.strftime('%A')})")
                log.info(f"  💰 今日起始盈亏: ${daily_start_pnl:+.2f}")

            is_open, status = is_market_open()

            # ── 周五收盘前自动清仓 (UTC 20:50, 收盘前10分钟) ──
            now_utc = datetime.now(ZoneInfo('UTC'))
            is_friday_close = (now_utc.weekday() == 4 and now_utc.hour == 20 and now_utc.minute >= 50)
            if is_friday_close and is_open:
                positions = trader.get_strategy_positions()
                if positions:
                    log.warning(f"\n🚨 周五收盘前10分钟，强制清仓 {len(positions)} 笔持仓")
                    for pos in positions:
                        ticket = pos.get('ticket', 0)
                        try:
                            success = trader.bridge.close_order(ticket)
                            profit = pos.get('profit', 0)
                            emoji = '✅' if success else '❌'
                            status_text = '成功' if success else '失败'
                            log.info(f"  {emoji} #{ticket} 平仓 {status_text} (当前浮盈${profit:.2f})")
                            if success:
                                notifier.notify_close(ticket, 'weekend_close', profit, '周五收盘前强制平仓')
                        except Exception as e:
                            log.error(f"  ❌ #{ticket} 平仓异常: {e}")
                    notifier.send_telegram(
                        f"🚨 <b>周五收盘前强制清仓</b>\n\n"
                        f"平仓 {len(positions)} 笔持仓\n"
                        f"原因: 避免周末持仓风险"
                    )

            if not is_open:
                log.info(f"💤 {status} | 等待5分钟...")
                time.sleep(300)
                continue

            scan_count += 1

            # ── 心跳检测: EA是否在线 ──
            # 跳过条件: 启动宽限期 / 周五收盘前1小时(UTC 20:00+)
            now_utc = datetime.now(ZoneInfo('UTC'))
            skip_heartbeat = (
                scan_count <= STARTUP_GRACE_SCANS
                or (now_utc.weekday() == 4 and now_utc.hour >= 20)  # 周五UTC 20:00后
            )
            if skip_heartbeat:
                pass
            elif trader.bridge.is_connected():
                if consecutive_disconnect > 0:
                    log.info(f"❤️ EA已恢复连接 (此前断开{consecutive_disconnect}次)")
                consecutive_disconnect = 0
            else:
                consecutive_disconnect += 1
                log.warning(f"📡 EA心跳丢失 (连续{consecutive_disconnect}次)")
                
                if consecutive_disconnect >= DISCONNECT_ALERT_THRESHOLD:
                    # 每30分钟最多报警一次，避免刷屏
                    should_alert = True
                    if last_heartbeat_alert:
                        elapsed = (now - last_heartbeat_alert).total_seconds()
                        if elapsed < 1800:  # 30分钟
                            should_alert = False
                    
                    if should_alert:
                        last_heartbeat_alert = now
                        positions = trader.get_strategy_positions()
                        pos_count = len(positions) if positions else 0
                        alert_msg = (
                            f"🚨 <b>MT4 EA可能宕机</b>\n\n"
                            f"连续{consecutive_disconnect}次心跳检测失败\n"
                            f"当前持仓: {pos_count}笔\n\n"
                            f"❗ 请检查:\n"
                            f"1. MT4是否运行\n"
                            f"2. EA是否加载\n"
                            f"3. 网络是否正常"
                        )
                        notifier.notify_error(alert_msg)
                        log.warning(f"🚨 已发送宕机警报 (Telegram)")

            # 每次扫描都检查持仓出场
            if scan_count % 10 == 1:  # 每10次(约10分钟)打印一次状态
                log.info(f"\n🔄 扫描 #{scan_count} | {status}")

            # 检查出场
            try:
                trader.check_exits_only()
            except Exception as e:
                log.error(f"出场检查出错: {e}")

            # 模拟盘扫描 (和实盘共享行情数据)
            try:
                df_h1 = trader.get_hourly_data()
                df_m15 = trader.get_m15_data()
                paper.scan(df_h1, df_m15)
            except Exception as e:
                log.debug(f"模拟盘扫描出错: {e}")

            # 每5分钟做一次完整信号扫描 (M15策略需要, 每10次循环×30秒≈5分钟)
            # 周五收盘前30分钟(UTC 20:30+)停止开新仓，只检查出场
            is_friday_no_new = (now_utc.weekday() == 4 and now_utc.hour >= 20 and now_utc.minute >= 30)
            if is_friday_no_new:
                if scan_count % 20 == 0:
                    log.info(f"🚧 周五收盘前30分钟，停止开新仓，只监控出场")
            elif scan_count == 1 or scan_count % 10 == 0:
                log.info(f"\n📊 完整信号扫描 (#{scan_count})")
                try:
                    result = trader.scan_and_trade()
                    
                    # 检查是否触发停止复盘
                    if result.get('status') == 'STOP_REVIEW':
                        log.warning("\n" + "=" * 60)
                        log.warning("🚨 系统已停止 — 日内亏损超限")
                        log.warning("请复盘后手动重启: python gold_runner.py")
                        log.warning("=" * 60)
                        break
                    
                    entries = result.get('entries', [])
                    exits = result.get('exits', [])

                    for e in entries:
                        log.info(f"📈 买入: {e.get('reason', '')}")
                    for e in exits:
                        log.info(f"📉 平仓: {e.get('reason', '')}")

                except Exception as e:
                    log.error(f"信号扫描出错: {e}")

            # 每小时同步交易数据到GitHub (供日报cron读取)
            current_hour = now.hour
            if current_hour != last_data_sync_hour and scan_count > 1:
                last_data_sync_hour = current_hour
                sync_data_to_github()

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
