# XAU/USD 量化交易系统 — 规划路线图

> 最后更新: 2026-03-30
> 系统上线日期: 2026-03-25

---

## 一、已完成的修改 (2026-03-25 ~ 2026-03-30)

### 核心架构
- [x] Keltner 通道突破策略 + M15 RSI 均值回归 + ORB 纽约开盘突破
- [x] MT4 JSON 文件桥接通信
- [x] Telegram 实时通知 (开仓/平仓/日报/异常)
- [x] ATR 自动调仓 (动态手数)

### Trailing Stop & 加仓 (已优化)
- [x] Keltner 追踪止盈: 激活阈值 2.5×ATR，追踪距离 0.5×ATR
- [x] 顺势加仓: **已关闭** (回测证明无加仓 Sharpe 更高)
- [x] 状态持久化: `trailing_stop_price`, `extreme_price`, `is_pyramid` 写入 tracking.json

### 舆情分析系统
- [x] 多模型融合: FinBERT + VADER + 关键词打分
- [x] 主语识别: 区分黄金直接新闻 vs 宏观新闻，反转评分
- [x] 高影响力关键词 3 倍加权
- [x] 跨资产监控: 布伦特原油 + 美10年期国债收益率 (观察模式)
- [x] 舆情状态落盘: `gold_daily_state.json` 含 macro_sentiment + macro_cross_assets
- [x] 历史记录: `sentiment_history.json` 按日去重，含各子模型分数

### 数据记录 (2026-03-30 新增)
- [x] **每日净值曲线**: `equity_curve.json` — 净值/日盈亏/分策略交易笔数+盈亏
- [x] **Telegram 日报增强**: 推送净值、胜率、分策略明细
- [x] 舆情子模型分数拆分记录 (keyword_score, finbert_score, vader_score)

### 风控修复 (2026-03-30 新增)
- [x] **P0: ORB 区间宽度动态化** — `range_width < 8` 改为 `max(8, 0.5×ATR)`，ATR 无效时拒绝开仓
- [x] **P1: M15 RSI 最小持仓 15 分钟** — 防止 RSI(2) K 线内闪烁导致 30 秒开平仓

---

## 二、待观察项 (需积累数据后决策)

### 2.1 舆情"一票否决权"
- **内容**: direction_bias 为 BUY 时拦截所有 SELL，反之亦然
- **代码状态**: 已注释，随时可启用
- **触发条件**: 积累 100 笔交易或 2 周后，对 `gold_trade_log.json` vs `gold_daily_state.json` 做对账
  - 如果舆情"记功"次数 >> "记过"次数 → 启用
  - 如果舆情经常瞎指挥 → 永远当日志插件
- **当前评估 (3/30)**: 3/25-3/30 对账显示舆情在 3/26 暴跌中仍给出 BULLISH，**不适合启用**

### 2.2 "古法黄金"现货防线
- **内容**: 价格 <= $4100 时禁止做空 (`GUFA_GOLD_HARD_FLOOR`)
- **代码状态**: 未实现，config 常量未添加
- **触发条件**: 用户明确要求实施

### 2.3 极端波动率降杠杆
- **内容**: ATR > `NORMAL_ATR_MAX` 时按比例缩小手数
- **代码状态**: 未实现
- **触发条件**: 用户明确要求实施

### 2.4 舆情权重科学回测 (50%/35%/15%)
- **内容**: 当前关键词/FinBERT/VADER 权重是经验值，需要回测验证
- **前置条件**: `sentiment_history.json` 积累 30 天以上数据
- **预计可回测时间**: 2026-04-25 之后

### 2.5 石油冲击悖论 — 跨资产翻转逻辑
- **内容**: 战争+油价飙升时强制翻转舆情 direction_bias 为 SELL
- **代码状态**: `MacroMonitor` 已在观察模式收集数据，翻转逻辑未实现
- **触发条件**: 跨资产数据积累足够后设计阈值

---

## 三、中期规划 (数据驱动，按触发条件执行)

### 3.1 时段分析 — Keltner 交易窗口
- **发现**: 3/25 欧洲盘 13:55-17:35 UTC Keltner 做多亏损 $71 (6笔仅1盈)
- **假设**: Keltner 在纽约盘 14:00-20:00 UTC 表现更好
- **所需数据**: 至少 2-4 周实盘，按时段分组统计胜率和盈亏比
- **数据来源**: `equity_curve.json` + `gold_trade_log.json`
- **可能的行动**: 限制 Keltner 只在特定时段开仓，或引入时段加权系数

### 3.2 M15 RSI 亚洲盘优化
- **发现**: 3/28 凌晨 00:00-05:00 UTC RSI 信号频繁但利润极薄
- **假设**: 亚洲盘波动率不足，RSI 均值回归无意义
- **可能的行动**: 亚洲盘时段提高 RSI 入场阈值 (如 RSI<10 才做多)

### 3.3 Missed Signals 加仓机会成本量化
- **数据**: 3/26-3/27 有 60+ 条 Keltner SELL 被"同策略已持仓"拦截
- **初步估算**: 额外加仓理论利润约 $42，但回测证明无加仓长期 Sharpe 更高
- **结论**: 暂不改变，继续观察

---

## 四、关键数据积累里程碑

| 里程碑 | 预计达成 | 解锁能力 |
|--------|---------|----------|
| 实盘运行 2 周 / 100 笔交易 | ~2026-04-08 | 舆情对账 → 决定是否启用一票否决权 |
| equity_curve.json 积累 30 天 | ~2026-04-25 | 计算实盘 Sharpe / 最大回撤 / 盈亏比 |
| sentiment_history.json 30 天 | ~2026-04-25 | 舆情权重回测 (FinBERT/VADER/关键词最优配比) |
| 实盘运行 2 个月 | ~2026-05-25 | 时段分析统计显著 / 策略贡献度归因 |

---

## 五、当前最优配置参数 (回测最优)

```python
TRAILING_ACTIVATE_ATR = 2.5       # 浮盈 > 2.5×ATR 激活追踪
TRAILING_DISTANCE_ATR = 0.5       # 追踪距离 0.5×ATR
ADD_POSITION_ENABLED = False      # 关闭加仓
KELTNER_EXHAUSTION_RSI_LOW = 0    # SELL 熔断关闭
KELTNER_EXHAUSTION_RSI_HIGH = 100 # BUY 熔断关闭
ORB_SL_MULTIPLIER = 0.75          # ORB 止损 = 0.75×区间
ORB_TP_MULTIPLIER = 3.0           # ORB 止盈 = 3.0×区间
```

> 来源: 11 年回测 + Trump 2.0 时期回测，Sharpe 0.98 / Trump 2.0 Sharpe 2.85

---

## 六、系统架构备忘

```
gold_runner.py          # 主循环 (24/5运行, 30秒扫描)
gold_trader.py          # 交易引擎 (信号评估/出场/风控/状态管理)
strategies/signals.py   # 策略信号生成 (Keltner/ORB/M15 RSI/MACD/GapFill)
mt4_bridge.py           # MT4 JSON 文件桥接
notifier.py             # Telegram 通知
config.py               # 集中配置
sentiment/              # 舆情分析系统
  analyzer.py           # 多模型融合打分
  sentiment_engine.py   # 后台线程编排
  news_collector.py     # RSS新闻抓取 + 经济日历
  macro_monitor.py      # 跨资产监控 (原油/国债)
data/
  gold_trade_log.json        # 交易记录
  gold_missed_signals.json   # 被拦截信号
  gold_position_tracking.json # 持仓状态
  gold_daily_state.json      # 当日状态 (含舆情)
  gold_total_pnl.json        # 累计盈亏
  equity_curve.json          # 净值曲线 (新增)
  sentiment_history.json     # 舆情历史 (新增)
```
