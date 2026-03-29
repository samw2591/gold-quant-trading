# 黄金 XAU/USD 量化交易系统 v4

基于 Python + MT4 EA 的黄金自动化交易系统。通过文件桥接实现 Python 信号分析与 MT4 交易执行的闭环。

## 架构

```
Python (信号分析)                    MT4 EA (交易执行)
┌─────────────────┐    文件桥接     ┌─────────────────┐
│ gold_runner.py  │◄──────────────►│ GoldBridge_EA   │
│ gold_trader.py  │  DWX/*.json    │                 │
│ strategies/     │                │ XAUUSD.mx 实时  │
│ sentiment/      │                │ H1/M15 K线数据  │
└─────────────────┘                └─────────────────┘
```

- **Python 端**：每30秒扫描信号，生成交易指令
- **MT4 EA**：每500ms检查指令并执行，每30秒回写K线数据
- **通信**：通过 `MQL4/Files/DWX/` 目录下的 JSON 文件

## 策略组合

### 1. Keltner 通道突破（H1 主力策略）
- 突破上轨做多 / 跌破下轨做空
- ADX > 25 过滤（震荡市不交易）
- EMA100 趋势过滤（只顺大趋势方向交易）
- 11年回测：Sharpe 0.52，年化 +$119/0.01手
- 特朗普2期：Sharpe 1.47，年化 +$861/0.01手

### 2. MACD + EMA100 趋势（H1 补充策略）
- MACD 柱状图转正做多 / 转负做空
- ADX > 25 + 价格 vs EMA100 过滤

### 3. NY 开盘区间突破 — ORB（H1 时段策略）
- 纽约开盘后第1根H1 K线的高低点作为当日区间
- 突破上沿做多 / 跌破下沿做空
- 止损 = 区间宽度，止盈 = 2.2× 区间宽度
- 窗口有效2小时，每日只交易1次

### 4. M15 RSI 均值回归（M15 低风险补充）
- RSI(2) < 15 超卖做多 / RSI(2) > 85 超买做空
- 不受 ADX 过滤（震荡市反而有效）

## 风控体系

| 参数 | 值 | 说明 |
|---|---|---|
| ATR 自动调仓 | 0.01-0.05手 | 每笔风险固定 $100 |
| 止损 | 2.5 × ATR | 自适应市场波动 |
| 止盈 | 6.5 × ATR | 盈亏比约 1:2.6 |
| 最大持仓 | 2笔 | 同方向，不对冲 |
| 日内限亏 | $100/天 | 达到后暂停当日交易 |
| 总止损 | $1500 | 达到后系统停止 |
| 冷却期 | 3小时 | 止损后同策略暂停开仓 |
| 方向冲突 | 禁止 | 有持仓时不开反向单 |

## 舆情分析系统

后台线程每5分钟自动采集分析，独立于交易扫描循环运行。

- **数据源**：Google News RSS（6个频道：黄金×2、宏观×2、特朗普×2）
- **NLP 模型**：VADER（快速）+ FinBERT（精准），双模型加权
- **经济日历**：FOMC/NFP/CPI 等60+事件，高风险事件前后自动暂停交易

**当前模式：记录观察**

舆情模块持续在后台运行并记录情感数据，但**不影响下单决策**（经济日历暂停除外）。方向过滤与仓位调整功能已实现，当前处于观察阶段——先积累数据验证准确率，待验证通过后再接入交易链路。

## 数据源

优先使用 MT4 本地K线数据（EA每30秒写入），yfinance 作为 fallback。

| 来源 | 数据 | 优先级 |
|---|---|---|
| MT4 EA | XAUUSD.mx H1/M15 实时K线 | 主要 |
| yfinance | GC=F COMEX 期货 | 备用 |

## 快速开始

### 1. 环境准备
```bash
git clone https://github.com/linhuang1313/gold-quant-trading.git
cd gold-quant-trading
pip install -r requirements.txt
```

### 2. 配置
编辑 `config.py`：
- `METATRADER_DIR_PATH`：MT4 数据文件夹路径
- `SYMBOL`：交易品种名称（如 `XAUUSD.mx`）
- `CAPITAL`：本金金额

### 3. MT4 EA 安装
1. 将 `mt4_ea/GoldBridge_EA.mq4` 复制到 `MQL4/Experts/`
2. MetaEditor 打开 → F7 编译
3. 拖到 XAUUSD 图表上，勾选"允许实时交易"
4. 确认 AutoTrading 按钮为绿色

### 4. 启动
```bash
python gold_runner.py
```

## 项目结构

```
gold-quant-trading/
├── gold_runner.py          # 主运行器（24/5循环）
├── gold_trader.py          # 交易引擎（持仓管理+风控）
├── config.py               # 集中配置
├── mt4_bridge.py           # MT4文件桥接通信
├── strategies/
│   └── signals.py          # 信号引擎（Keltner/MACD/ORB/RSI）
├── sentiment/
│   ├── news_collector.py   # 新闻采集（RSS）
│   ├── analyzer.py         # NLP情绪分析（VADER+FinBERT）
│   ├── calendar_guard.py   # 经济日历避险
│   └── sentiment_engine.py # 舆情主引擎（后台线程）
├── mt4_ea/
│   └── GoldBridge_EA.mq4   # MT4 EA（交易执行+K线数据）
├── data/                   # 交易日志、持仓跟踪、历史数据
├── logs/                   # 运行日志
└── research/               # 策略研究报告
```

## 回测数据

- Dukascopy XAU/USD H1：98,424根K线（2015-2026，11年）
- Dukascopy XAU/USD M15：30,879根K线（2020-2025，5年）
- Dukascopy XAU/USD M5：357,770根K线（2020-2025，5年）

## 免责声明

本项目仅供学习研究，不构成投资建议。量化交易有风险，过去的回测表现不代表未来收益。
