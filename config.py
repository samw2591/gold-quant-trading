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
LOT_SIZE = 0.03           # 手数 (0.03手 = 3盎司 = $3/点) — 先用模拟盘验证，确认后可加0.05
MAX_POSITIONS = 2         # 最大同时持仓数
STOP_LOSS_PIPS = 50       # 止损距离 (点/$)
MAGIC_NUMBER = 20260325   # EA魔术号 (区分手动单和策略单)
SLIPPAGE = 5              # 最大滑点 (点)

# ============================================================
# 策略参数
# ============================================================
STRATEGIES = {
    "keltner": {
        "enabled": True,
        "name": "Keltner通道突破",
        "stop_loss": 20,
        "take_profit": 35,
        "max_hold_bars": 15,
        # 11年回测: Sharpe 0.92, 年均260笔, 胜率49%
        # 特朗普2: 年化+51.7%, 回撤-17.5%
        # 支持做多+做空
    },
    "macd": {
        "enabled": True,
        "name": "MACD+SMA50趋势",
        "stop_loss": 20,
        "take_profit": 50,
        "max_hold_bars": 20,
        # 11年回测: Sharpe 1.14, 年均123笔, 盈亏比2.46
        # 特朗普2: 年化+24.7%, 回撤仅-4.5%
        # 支持做多+做空
    },
}

# ============================================================
# 扫描频率
# ============================================================
SCAN_INTERVAL_SECONDS = 30    # 每30秒扫描一次 (M5策略需要高频扫描)
SIGNAL_CHECK_TIMEFRAME = "MULTI"  # 多时间框架: H1 + M5

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
