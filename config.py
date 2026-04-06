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
CAPITAL = 2000            # 本金 (USD)
MAX_TOTAL_LOSS = 1500     # 最大总亏损 (USD)，达到后停止交易
LOT_SIZE = 0.03           # 手数 (0.03手 = 3盎司)
# 每点价值: 0.03手 × $100/点/标准手 = $3/点 (价格每变动$1 = 盈亏$3)
POINT_VALUE_PER_LOT = 100  # 标准手每点价值 ($100/点)
MAX_POSITIONS = 2         # 最大同时持仓数
STOP_LOSS_PIPS = 20       # 默认止损距离 ($20 = 0.03手亏$60)
MAGIC_NUMBER = 20260325   # EA魔术号 (区分手动单和策略单)
SLIPPAGE = 5              # 最大滑点 (点)
DAILY_MAX_LOSS = 9999     # 单日最大亏损金额 (已改用笔数控制，此项保留作极端保护)
DAILY_MAX_LOSSES = 5      # 单日最大亏损笔数
COOLDOWN_MINUTES = 30     # 止损后冷却期 (30分钟, 回测最优: Sharpe 1.03→1.17, 3h→30min)
RSI_ADX_BLOCK_THRESHOLD = 40  # H1 ADX > 40 时禁止 M15 RSI 开仓 (M15回测: Sharpe +0.21, 过滤强趋势逆势信号)

# ── ATR自动调仓 ──
RISK_PER_TRADE = 50       # 每笔交易最大风险金额 (2.5%×$2000=$50)
AUTO_LOT_SIZING = True    # 是否启用ATR自动调仓 (True=根据ATR调整手数, False=固定LOT_SIZE)
MIN_LOT_SIZE = 0.01       # 最小手数
MAX_LOT_SIZE = 0.05       # 绝对安全上限 (防止极端 ATR 下手数失控)
# 日内亏损递减手数上限: 0笔亏损→ATR自由计算(上限0.05), 1笔→0.03, 2笔→0.02, 3笔+→0.01
MAX_LOT_CAP_BY_LOSSES = {0: 0.05, 1: 0.03, 2: 0.02, 3: 0.01}

# ── ORB策略参数 ──
ORB_ENABLED = True                # 是否启用NY开盘区间突破策略
ORB_NY_OPEN_HOUR_UTC = 14         # 纽约开盘时间 UTC (14:30 = 纽约9:30, 用14近似)
ORB_RANGE_MINUTES = 15            # 开盘后前15分钟的高低点作为区间
ORB_EXPIRY_MINUTES = 120          # 突破窗口有效期 (2小时)
ORB_SL_MULTIPLIER = 0.75          # 止损 = 0.75 × 区间宽度 (v5优化: 1.0→0.75, Sharpe+0.36)
ORB_TP_MULTIPLIER = 3.0           # 止盈 = 3.0 × 区间宽度 (v5优化: 2.2→3.0, 特朗普2期Sharpe+0.62)
ORB_SL_MIN_ATR_MULTIPLIER = 1.5   # ORB止损下限 = 1.5×ATR (防止区间太窄导致SL过小)

# ── Trailing Stop (追踪止盈) ──
TRAILING_STOP_ENABLED = True      # 是否启用Keltner追踪止盈
TRAILING_ACTIVATE_ATR = 0.8       # 默认激活倍数 (V3 Regime 会动态覆盖)
TRAILING_DISTANCE_ATR = 0.25      # 默认追踪距离 (V3 Regime 会动态覆盖)
V3_ATR_REGIME_ENABLED = True      # V3 波动率自适应: 高波动紧追踪(0.6/0.20), 低波动松追踪(1.0/0.35), K-Fold 6/6折全赢, Sharpe+0.33

# ── 顺势加仓 ──
ADD_POSITION_ENABLED = False      # 关闭加仓 (回测最优: 无加仓Sharpe更高)
ADD_POSITION_MIN_ADX = 28         # 加仓要求: ADX >= 28 (极强趋势)
ADD_POSITION_MIN_PROFIT_ATR = 1.0 # 加仓要求: 已有持仓浮盈 > 1.0×ATR
ADD_POSITION_MIN_HOLD_MINUTES = 30  # 加仓要求: 距上一笔同方向入场至少30分钟
ADD_POSITION_MIN_DISTANCE_ATR = 0.5 # 加仓要求: 价格距上一笔入场价至少0.5×ATR
KELTNER_MAX_SAME_STRATEGY = 2     # Keltner同策略最大持仓数

# ── 趋势耗尽熔断 ──
KELTNER_EXHAUSTION_RSI_LOW = 0    # SELL熔断: 关闭 (回测最优: 无熔断Sharpe更高, SL兜底)
KELTNER_EXHAUSTION_RSI_HIGH = 100 # BUY熔断: 关闭

# ── 盘中趋势自适应 ──
INTRADAY_TREND_ENABLED = True         # 启用盘中趋势评分门控
INTRADAY_TREND_THRESHOLD = 0.35       # < 此值 = CHOPPY，禁止所有新开仓
INTRADAY_TREND_KC_ONLY_THRESHOLD = 0.60  # < 此值 = NEUTRAL，仅允许 H1 策略 (Keltner/ORB)

# ── 宏观数据管道 ──
import os as _os
FRED_API_KEY = _os.environ.get("FRED_API_KEY", "")  # FRED API key (免费注册: https://fred.stlouisfed.org/docs/api/api_key.html)
MACRO_ENABLED = True              # 启用宏观数据采集 (DXY/VIX/TIPS/国债/BEI)
MACRO_CACHE_TTL = 3600            # 宏观快照缓存时间 (秒), 盘中每小时更新

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
        "enabled": False,  # v5禁用: 11年Sharpe=-0.36, 71%信号与Keltner重叠, 拉低组合Sharpe
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
    "gap_fill": {
        "enabled": False,  # 禁用: 两次独立回测Sharpe均为负(-1.25/-1.71), 实盘0次触发
        "name": "周一跳空回补",
        "max_hold_bars": 8,
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
MACRO_CACHE_PATH = DATA_DIR / "macro_snapshot.json"
