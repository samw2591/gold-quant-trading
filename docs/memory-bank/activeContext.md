# 当前上下文 (Active Context)

> **读取频率: 每次对话必读**
> 当前焦点、进行中的实验、待办事项、最近系统状态

---

## 当前实盘状态 (2026-04-09)

- 实盘运行中: `gold_runner.py` → MT4 桥接
- 本金 $2,000，最大总亏损保护 $1,500
- 最近连续亏损: 4/6-4/8 共 5 笔止损（~$163），4/6 两笔 Keltner + 4/8 三笔(Keltner+RSI)

## 🚨 回测引擎重大修复 (2026-04-09)

**H1 look-ahead bias + 入场价 look-ahead 已修复**（commit 7f02772）

修复内容：
1. `_get_h1_window(closed_only=True)` — 入场信号只使用已收盘的 H1 bar
2. pending signal 队列 — 信号在下一根 M15 bar 的 Open 执行，不用当前 bar Close

修复前后初步对比（Current 无成本）：
- 修复前: N=18,544 Sharpe=5.06 PnL=$35,251
- 修复后: N=25,677 Sharpe=3.18 PnL=$26,206
- Sharpe 下降 37%，但仍为强正值

**⚠️ 重要**: 之前所有实验的 Sharpe 数字都基于有 look-ahead 的旧引擎，不再可信。
在服务器运行 `run_lookahead_fix_verify.py` 获取 6 配置完整基准线后，
之前发现的优化（T7 ExtremeRegime、Trail Momentum 1.5x 等）需要在修复后的引擎上重新验证。

## 进行中的实验

- **`run_lookahead_fix_verify.py`** — 已推送，待服务器运行（最高优先级）
  - Current/Mega × $0/$0.30/$0.50 = 6 个配置的修复后基准线
- **`run_t7_extreme_validation.py`** — 已推送，T7 K-Fold + 带成本 + 消融测试
  - **注意**: 此脚本基于修复后引擎，结果将是可信的
- **batch2 / exit_combo_matrix** — 服务器后台运行中（基于旧引擎，结果仅供参考）

## 之前实验结论（基于旧引擎，仅供参考）

以下结论基于有 look-ahead 的旧引擎。方向性判断可能仍然有效，但具体 Sharpe 数字不可信：
- Trail Momentum 1.5x: K-Fold 6/6 通过（需重新验证）
- T7 ExtremeRegime: Sharpe +1.20（需重新验证）
- Entry Quality Filters: 全部无效（否决结论应仍有效）
- Session/SL/TP 优化: 全部无效（否决结论应仍有效）

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
- [ ] **等 `run_lookahead_fix_verify.py` 结果** — 修复后 6 配置基准线（决定一切后续行动的基础）
- [ ] **看 `run_t7_extreme_validation.py` 结果** — T7 在修复后引擎上是否仍有效
- [ ] 基准线确认后，决定是否需要重新跑 Trail Momentum 1.5x K-Fold
- [x] ~~H1 look-ahead + 入场价 look-ahead 修复~~ → commit 7f02772
- [x] ~~EXP-EQ 回测结果分析~~ → 全部无效，不采纳
- [x] ~~session_filter / sl_optimization / tp_atr_sweep / trailing_evolution 分析~~ → 见 backtestArchive

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
