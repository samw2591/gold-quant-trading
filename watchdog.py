"""
交易系统看门狗
==============
监控 gold_runner.py 进程，如果崩溃则自动重启。
在 Windows 上通过 Task Scheduler 或直接运行:
  python watchdog.py

特性:
  - 每30秒检查一次 gold_runner.py 是否存活
  - 崩溃后自动重启，附带指数退避（30s → 60s → 120s → ... 最大300s）
  - 重启事件通过 Telegram 通知
  - 连续重启超过5次发送紧急告警
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

RUNNER_SCRIPT = str(Path(__file__).parent / "gold_runner.py")
PYTHON = sys.executable
CHECK_INTERVAL = 30
MAX_BACKOFF = 300

_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    handlers=[
        _stream_handler,
        logging.FileHandler(Path(__file__).parent / "logs" / "watchdog.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("watchdog")


def _notify(msg: str):
    try:
        import notifier
        notifier.send_telegram(f"🐕 <b>Watchdog</b>\n{msg}")
    except Exception:
        pass


def main():
    log.info("Watchdog 启动 — 监控 gold_runner.py")
    restart_count = 0
    backoff = CHECK_INTERVAL

    while True:
        log.info(f"启动 gold_runner.py (第{restart_count + 1}次)")
        _notify(f"启动 gold_runner.py (第{restart_count + 1}次)")

        proc = subprocess.Popen(
            [PYTHON, RUNNER_SCRIPT],
            cwd=str(Path(__file__).parent),
        )

        try:
            exit_code = proc.wait()
        except KeyboardInterrupt:
            log.info("Watchdog 收到 Ctrl+C，终止 gold_runner")
            proc.terminate()
            proc.wait(timeout=10)
            break

        restart_count += 1
        log.warning(f"gold_runner.py 退出 (code={exit_code})，准备重启 (第{restart_count}次)")
        _notify(f"⚠️ gold_runner.py 退出 (code={exit_code})\n即将第{restart_count}次重启 (等待{backoff}s)")

        if restart_count >= 5:
            _notify(f"🚨 <b>紧急</b>: gold_runner.py 已连续崩溃{restart_count}次!\n请立即检查系统")
            log.error(f"连续崩溃{restart_count}次，请检查系统!")

        time.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF)

        if restart_count >= 20:
            log.critical("连续重启超过20次，Watchdog 停止")
            _notify("🛑 连续重启超过20次，Watchdog 已停止。请人工干预！")
            break


if __name__ == "__main__":
    main()
