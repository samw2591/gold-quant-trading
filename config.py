"""
黄金量化交易系统 — 配置文件
============================
所有参数集中管理，修改这里即可
"""
from pathlib import Path

# ============================================================
# MT4 连接配置
# ============================================================
# MT4 数据文件夹路径 (在MT4中 File → Open Data Folder 获取)
# 例如: C:\Users\hlin2\AppData\Roaming\MetaQuotes\Terminal\XXXXXXXX
METATRADER_DIR_PATH = r"C:\Users\hlin2\AppData\Roaming\MetaQuotes\Terminal\35EEC3EFDB656AF6FC775F21FEAD053B"

# MT4 文件桥接目录 (EA和Python通过这个目录通信)
BRIDGE_DIR = Path(METATRADER_DIR_PATH) / "MQL4" / "Files" / "DWX"

# ============================================================
# 交易账户参数
# ============================================================
SYMBOL = "XAUUSD.mx"      # EMX Pro Limited 的黄金品种名称
CAPITAL = 3000            # 本金 (USD)
MAX_TOTAL_LOSS = 1500     # 最大总亏损 (USD)，达到后停止交易
LOT_SIZE = 0.01           # 手数 (0.01手 = 1盎司 = $1/点)
MAX_POSITIONS = 2         # 最大同时持仓数
STOP_LOSS_PIPS = 50       # 止损距离 (点/$)
MAGIC_NUMBER = 20260325   # EA魔术号 (区分手动单和策略单)
SLIPPAGE = 5              # 最大滑点 (点)

# ============================================================
# 策略参数
# ============================================================
STRATEGIES = {
    "bollinger": {
        "enabled": True,
        "name": "布林带均值回归",
        "bb_period": 20,
        "bb_std": 2.0,
        "ma_trend": 200,
        "exit_target": "bb_mid",
        "stop_loss": 50,
        "max_hold_bars": 15,
        # GC=F回测: Sharpe 2.21, 胜率75%, 均收+$9.6/笔, 回撤-8.9%
    },
    "range_breakout": {
        "enabled": True,
        "name": "窄幅突破",
        "range_pct": 0.6,
        "lookback": 5,
        "ma_trend": 200,
        "ma_exit": 10,
        "stop_loss": 50,
        "max_hold_bars": 15,
        # GC=F回测: Sharpe 1.27, 胜率43.2%, 均收+$13.6/笔, 回撤-9.9%
    },
    "atr_squeeze": {
        "enabled": True,
        "name": "ATR收缩突破",
        "atr_squeeze_mult": 1.3,   # ATR < 近50日最低值×1.3
        "lookback": 5,
        "ma_trend": 200,
        "ma_exit": 10,
        "stop_loss": 50,
        "max_hold_bars": 15,
        # GC=F回测: Sharpe 1.19, 胜率43%, 均收+$21.8/笔, 回撤-20%
    },
}

# ============================================================
# 扫描频率
# ============================================================
SCAN_INTERVAL_SECONDS = 60    # 每60秒扫描一次 (盘中)
SIGNAL_CHECK_TIMEFRAME = "D1"  # 日线级别信号

# ============================================================
# 通知
# ============================================================
NOTIFY_METHOD = "console"      # "console" | "telegram"
# TELEGRAM_BOT_TOKEN = ""
# TELEGRAM_CHAT_ID = ""

# ============================================================
# 路径
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
