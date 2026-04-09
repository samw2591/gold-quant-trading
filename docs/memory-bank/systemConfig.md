# 系统配置与核心认知 (System Config)

> **读取频率: 每次对话必读**
> 当前生效参数、参数来源、核心策略认知

---

## 当前实盘参数 (2026-04-09)

| 参数 | 值 | 来源回测 |
|------|-----|------|
| KC EMA | 25 | EXP13/EXP20 K-Fold 4/6折赢 |
| KC Multiplier | 1.2 | EXP13/EXP20 |
| TrailAct | 由 V3 ATR Regime 控制 | EXP07 K-Fold 6/6折全赢 |
| TrailDist | 由 V3 ATR Regime 控制 | EXP07 |
| V3 ATR Regime | 启用 (低波T1.0/D0.35, 正常T0.8/D0.25, 高波T0.6/D0.20) | EXP07 |
| ADX_TREND_THRESHOLD | 18 | EXP09/ADX阈值测试 确认最优 |
| ATR_SL_MULTIPLIER | 4.5 | Mega Grid + 带成本精调 |
| ATR_TP_MULTIPLIER | 8.0 | EXP06: TP5.0仅1.4%触发, TP8.0 Sharpe+0.04 |
| COOLDOWN_MINUTES | 30 | 20变体交叉测试, 所有点差下均优于3h |
| INTRADAY_TREND_ENABLED | True | Phase 5 回测 |
| INTRADAY_TREND_THRESHOLD | 0.35 (choppy) | Phase 5 |
| INTRADAY_TREND_KC_ONLY_THRESHOLD | 0.60 (trending) | Phase 5 |
| RSI_ADX_BLOCK_THRESHOLD | 40 | M15 回测 Sharpe +0.21 |
| DAILY_MAX_LOSSES | 5 | 风控 |
| MAX_LOT_SIZE | 0.05 | 绝对安全上限 |
| 亏损递减上限 | 0笔→0.05, 1笔→0.03, 2笔→0.02, 3+→0.01 | 替代旧lot_scale |
| Max Positions | 2 | 风控 |
| Risk per Trade | $50 (2.5%) | 风控 |

## 实盘策略组合

| 策略 | 时间框架 | 状态 | 核心逻辑 |
|---|---|---|---|
| Keltner通道突破 | H1 | 启用（主力） | 4阶段状态机, EMA100趋势过滤, ADX>18, ATR自适应止损, V3追踪止盈 |
| NY开盘区间突破(ORB) | M15 | 启用 | 纽约开盘前15分钟高低点突破, 2小时窗口 |
| M15 RSI均值回归 | M15 | 启用（实质消亡） | RSI2超卖/超买, Adaptive下仅6笔/11年 |
| 周一跳空回补 | H1 | 启用 | 周一开盘跳空后回补方向交易 |

## 核心认知 (2026-04-09)

1. **交易成本是策略杀手**: 无成本 Sharpe 3.46 加 $0.50 点差后仅 0.35。所有回测必须包含成本
2. **少交易比好参数更重要**: C12 无 Adaptive 15,770 笔 Sharpe -0.53; 加 Adaptive 7,365 笔 Sharpe 1.03。砍掉一半交易反而从亏到赚
3. **避开震荡比抓住趋势更重要**: 利润主要来自大量小额追踪止盈（11,611 次 trail 触发，97.7%WR，中位持仓 1.5h）。震荡时段仍是亏损源，Adaptive 门控砍掉震荡时段交易仍然关键
4. **追踪止盈是核心 alpha**: 5,542 笔 trailing +$41,088 (97.7%WR), 赢家快进快出(中位 1.5h), 输家拖很久(中位 11.5h)
5. **过拟合风险低, 结构风险高**: PBO=0.00, 参数平滑, DSR 通过。真正风险是低波动盘整期持续数月小额亏损(1/3 半年窗口为负)
6. **Keltner+ADX+EMA100 信号集已饱和**: Strategy A/C/D 共 596 种组合全部无法超越基线。改善方向应聚焦于出场机制和仓位管理
7. **策略 alpha 极其鲁棒**: VIX/DXY/日线趋势/K线形态/ATR regime/时段/整数关口 — 都无法显著改善
8. **宏观 Regime 过滤无效（三次确认）**: 策略在所有 6 种 regime 下都盈利，是 regime-agnostic
9. **V3 ATR Regime 是唯一有效的 Regime 类调节**: 调整止盈参数而非过滤信号方向，本质是波动率自适应

## 因子有效性摘要

### 有效因子 (IC 显著且稳定)
- `RSI2 × ret_1`: IC=-0.0314, WF=100% — M15 RSI 策略的因子基础
- `day_of_week × ret_4/8`: IC=+0.033, WF=100% — 但回测显示跳过任何一天都更差
- `ATR × ret_4/8`: IC=+0.032~0.036, WF=60% — 高波动后正收益（趋势延续）
- `momentum_5/10 × ret_1`: IC=-0.019~0.021, WF=100% — 短期动量反转
- `KC_position/breakout_strength × ret_1`: IC=-0.016, WF=100% — KC 位置短期反转
- `MACD_hist_change × ret_1`: IC=-0.025, WF=100% — MACD 加速度反转

### 无效因子 (|IC| < 0.005)
- `ADX`, `close_ema100_dist`, `ema9_ema21_cross`, `volume_ratio`
- ADX 作为线性预测因子无效，但作为条件筛选器（ADX>18 门槛）仍有用

### Gradient Boosting 因子重要性
- ATR(22.9%) > EMA100_dist(16.8%) > KC_pos(14.5%) > RSI14(11.5%) > KC_bw(8.8%)

## IntradayTrendMeter

- 4 个子因子: ADX(30%) + KC突破比例(25%) + EMA排列一致性(25%) + 趋势强度(20%)
- 三级门控: ≥0.60 TRENDING (全策略), 0.35-0.60 NEUTRAL (仅H1), <0.35 CHOPPY (禁止开仓)
- 27.2% 的交易日至少出现一次 choppy 窗口
- choppy 门控有价值: 无门控 Sharpe 从 4.46 降到 4.05

## EUR/USD 策略配置

| 参数 | 值 | 与黄金差异 |
|---|---|---|
| KC mult | 2.0 | 黄金 1.2 (EUR/USD 更宽通道最优) |
| MaxHold | 20 bars | 黄金 60 |
| lots | 0.05 | 匹配 $50 风险 |
| point_value | 100,000 | 外汇标准手 |
| 点差 | 1.8 pips | — |
| 11年 Sharpe | 1.91 | 含点差，12/12年全正 |
