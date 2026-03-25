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
LOT_SIZE = 0.03           # 手数 (0.03手 = 3盎司)
# 每点价值: 0.03手 × $100/点/标准手 = $3/点 (价格每变动$1 = 盈亏$3)
POINT_VALUE_PER_LOT = 100  # 标准手每点价值 ($100/点)
MAX_POSITIONS = 2         # 最大同时持仓数
STOP_LOSS_PIPS = 20       # 默认止损距离 ($20 = 0.03手亏$60)
MAGIC_NUMBER = 20260325   # EA魔术号 (区分手动单和策略单)
SLIPPAGE = 5              # 最大滑点 (点)
DAILY_MAX_LOSS = 200      # 单日最大亏损 ($200，达到后停止交易，需要复盘后手动重启)
COOLDOWN_BARS = 3         # 止损后冷却期 (3根H1 K线 = 3小时)

# ── ATR自动调仓 ──
RISK_PER_TRADE = 100      # 每笔交易最大风险金额 ($100)
AUTO_LOT_SIZING = True    # 是否启用ATR自动调仓 (True=根据ATR调整手数, False=固定LOT_SIZE)
MIN_LOT_SIZE = 0.01       # 最小手数
MAX_LOT_SIZE = 0.03       # 最大手数 (本金$2000, 保守控制)

# ── ORB策略参数 ──
ORB_ENABLED = True                # 是否启用NY开盘区间突破策略
ORB_NY_OPEN_HOUR_UTC = 14         # 纽约开盘时间 UTC (14:30 = 纽约9:30, 用14近似)
ORB_RANGE_MINUTES = 15            # 开盘后前15分钟的高低点作为区间
ORB_EXPIRY_MINUTES = 120          # 突破窗口有效期 (2小时)
ORB_SL_MULTIPLIER = 0.75          # 止损 = 0.75 × 区间宽度 (v5优化: 1.0→0.75, Sharpe+0.36)
ORB_TP_MULTIPLIER = 3.0           # 止盈 = 3.0 × 区间宽度 (v5优化: 2.2→3.0, 特朗普2期Sharpe+0.62)

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
    },
    "macd": {
        "enabled": True,
        "name": "MACD+EMA100趋势",
        "stop_loss": 20,
        "take_profit": 50,
        "max_hold_bars": 20,
    },
    "orb": {
        "enabled": True,
        "name": "NY开盘区间突破",
        "max_hold_bars": 6,  # v5优化: 8→6根K线 (~6小时, Sharpe 1.31→1.51)
    },
}

# ============================================================
# 扫描频率
# ============================================================
SCAN_INTERVAL_SECONDS = 30    # 每30秒扫描一次 (M15策略)
SIGNAL_CHECK_TIMEFRAME = "MULTI"  # 多时间框架: H1 + M15

# ============================================================
# 通知
# ============================================================
NOTIFY_METHOD = "telegram"     # "console" | "telegram"
TELEGRAM_BOT_TOKEN = "8646871612:AAFzMhC_4-rh7_f2E47ankyh45IxFczmVw8"
TELEGRAM_CHAT_ID = "8531960227"

# ============================================================
# 路径
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
