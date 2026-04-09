# 当前上下文 (Active Context)

> **读取频率: 每次对话必读**
> 当前焦点、进行中的实验、待办事项、最近系统状态

---

## 当前实盘状态 (2026-04-09)

- 实盘运行中: `gold_runner.py` → MT4 桥接
- 本金 $2,000，最大总亏损保护 $1,500
- 最近连续亏损: 4/6-4/8 共 5 笔止损（~$163），4/6 两笔 Keltner + 4/8 三笔(Keltner+RSI)

## 进行中的实验

- **EXP-EQ (Entry Quality Filters)**: 服务器运行 `run_exp_entry_quality.py` 中
  - 测试 3 个过滤器: min_h1_bars_today, adx_gray_zone, escalating_cooldown
  - ⚠️ **注意**: 这三个过滤器是在未经回测验证的情况下写入 engine 的（见 constraints.md 2026-04-09 方法论错误记录），等结果出来后严格按数据判断是否保留
- **等待结果**: session_filter / sl_optimization / tp_atr_sweep / trailing_evolution / exit_combo_matrix（服务器运行中）

## 模拟盘策略

| 策略 | 状态 | 观察重点 |
|---|---|---|
| P4_atr_regime | 运行中 | 不同波动率环境下胜率差异 |
| P5_volume_breakout | 运行中 | volume_ratio IC≈0，预测力存疑 |
| P6_dxy_filtered | 运行中 | DXY 与金价负相关是否稳定 |
| P7_mega_trail | 运行中 | T0.5/D0.15 实盘 trailing 触发频率 |
| P8_mega_h20 | 运行中 | 短持仓(5h)是否减少SL损失 |
| P9_eurusd_keltner | 运行中 | EUR/USD KC mult=2.0 实盘验证 |

## 待办事项

### 高优先级
- [ ] **EXP-EQ 回测结果分析**: 等服务器跑完，严格按数据决定三个过滤器去留
- [ ] **Trail Momentum (+50%) K-Fold 验证后考虑实装** — 12/12 年一致, Sharpe +0.44
- [ ] 等待 exit_combo_matrix / sl_optimization / tp_atr_sweep / trailing_evolution / session_filter 结果并分析

### 中期
- [ ] 测试缩短最大持仓时间从 60 bars 到 24-32 bars（Timeout 60 bars 亏损 $12,287，占总损失最大项）
- [ ] 测试 Mega Grid 最优 T0.5/D0.15 在 $0.30/$0.50 带成本下的表现
- [ ] 重新评估 ORB 策略: 去掉 ORB 后 Sharpe +0.24，可能拖累组合
- [ ] 探索 M15 级别均值回归策略（H1 不可行，M15 可能更适合）
- [ ] 将 squeeze-to-expansion 信号作为 Keltner 入场 confidence score 测试
- [ ] EUR/USD paper trade 至少 20 笔后评估实盘切换

### 低优先级
- [ ] 考虑让 MT4 EA 在 positions.json 中包含已平仓订单历史
- [ ] Telegram Token 从代码移到 .env 文件
- [ ] 舆情系统 30 天后（~4/30）做第一次正式评估
- [ ] Polymarket 地缘风险指数观察 2 周后评估其与金价的实际相关性
- [ ] Stochastic 10/90 作为 RSI 替代候选 — $/t=$27.66 但 11 年仅 103 笔
- [ ] 考虑连续 5+ SELL 后减仓（均值 -$0.07/笔）
- [ ] DXY 日线作为日级别方向过滤

### 已完成 (最近)
- [x] Bug 修复: backtest engine IntradayTrendMeter 索引 bug（choppy 门控之前未生效）
- [x] Bug 修复: monkey-patch 信号注入机制（Strategy A/C/D）
- [x] ATR spike protection 验证: 真实引擎仅 +0.03 Sharpe，不值得增加复杂度
- [x] KC Bandwidth 过滤: 弊大于利，加入否决列表
- [x] Strategy A/C/D 三策略 596 种组合: 全部无法超越 Keltner 基线
- [x] EXP28 事件日防御 "带伞策略": 全部无效或有害
- [x] 宏观 Regime 过滤: 第三次确认无效
- [x] EUR/USD Keltner 策略通过 11 年回测验证，已注册 P9 模拟盘
