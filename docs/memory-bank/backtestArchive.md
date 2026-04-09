# 回测详细数据归档 (Backtest Archive)

> **读取频率: 按需搜索，不需每次读全文**
> 包含所有回测的详细数据表格、变体对比、K-Fold 结果
> 使用 Grep 搜索关键词定位相关段落

---

## 索引

| 实验 | 关键词 | 结论 |
|---|---|---|
| 首次全量回测 | H1策略组合 2015-2026 | Keltner主力, gap_fill 禁用 |
| 边际改进 A/B | 周一降仓/SELL缩半/ADX28 | 周一不采纳, SELL缩半谨慎, ADX28不采纳 |
| 追踪止盈 R1/R2/R3 | Trail激活/距离, SL/TP | C12冠军 T0.8/0.25+SL3.5+TP5+ADX18 |
| 过拟合检测 | PBO/PSR/DSR/WF | LOW overfit risk, PBO=0.00 |
| 高级回测套件 | Monte Carlo/K-Fold/Regime/参数探索 | V3 ATR Regime Sharpe+0.37 |
| 组合验证 | Combo三合一 | 通过但无成本虚假繁荣 |
| 压力测试 | 交易成本/信号聚集/参数悬崖 | Sharpe 3.46 无成本=虚假 |
| 成本调整 | 点差敏感度/频率降低 | $0.50下仅Combo存活(0.35) |
| 大趋势日 | Oracle/早盘预判/K-Fold | Oracle 6.44, 技术预判失败 |
| Phase 5 盘中自适应 | IntradayTrendMeter | C12 Adaptive Sharpe -0.53→1.03 |
| Mega Grid | 1440组合 TrailAct/Dist/ADX/SL/Choppy | T0.5/D0.15 最优(无成本) |
| EXP01-26 大规模并行 | 参数/时间/风控/策略/综合 | V3 唯一有效Regime调节 |
| EXP27 情景压力 | 伊朗/流动性/央行/关税 | S1/S6盈利, S4/S5亏损 |
| EXP28 带伞策略 | 事件日防御 | 全部无效或有害 |
| EXP30-32 | 缩短持仓/EMA斜率/Mega Trail | Mega Trail 7.66 压倒性 |
| EXP33-35 | Mega+Hold/周内择时/连续亏损 | Mega H12 Sharpe 9.03 |
| EXP36-41 | 时段/分批止盈/RSI背离/D1/K线/ATR反波 | 仅分批止盈微弱正效果 |
| EXP42-47 | 宏观DXY VIX/动量/波动率聚集/整数关口/隔夜 | DXY最强相关但不可预测 |
| EXP48-53 | KC均值回归/波动率clustering/BW变化率/ATR spike/组合 | ATR spike +0.03实际, BW最佳因子 |
| EXP-MOM 动量旁路 | choppy bypass | 效果微弱, bug修复是最大收获 |
| Strategy A/C/D | 动量/D1过滤/回调 | 596种组合全部不如基线 |
| ADX阈值测试 | ADX 18-28 | ADX=18 确认最优 |
| D1+3h 过拟合检测 | PBO/PSR/DSR/参数敏感性 | 5/5 通过 |
| EUR/USD 深度回测 | 11年 Dukascopy | KC2.0 MH20 Sharpe 1.91, 12/12年全正 |
| 第四批实验 | 因子重要性/BW/Kelly/London/Stoch/Regime | Trail Momentum 唯一可操作 |
| Trail Momentum K-Fold | 1.5x/2.0x Current+Mega 6折 | **1.5x 6/6折全通过 ✅ 可实装** |
| T7 ExtremeRegime | 高波动 trail_act=0.25/dist=0.05 | Sharpe+1.20, 12/12年，待K-Fold+成本 |
| EXP-EQ Entry Quality | min_bars/ADX gray zone | 全部无效否决 |
| EXP-B ORB K-Fold | NoORB monkey-patch | patch失败，ORB影响<2%保留 |
| EXP-C Stochastic K-Fold | 10/90 H6 带成本 | 5/6折正但样本极小，待观察 |
| EXP-D BW Confidence | squeeze-to-expansion sizing | 效果太弱否决 |
| Session/SL/TP 扫描 | 时段/SL 3-6/TP 4-12 | 全部无效否决 |

---

## 详细数据

(以下为从 trading_journal.md 迁移的完整回测数据，保留原始格式供搜索查阅)

### 首次全量回测 (H1, 2015-2026, 66376根K线)
- 总体: 3161笔, 胜率49.6%, Sharpe 0.50, MaxDD $835(26%), 总PnL $2461, RR 1.09
- Keltner: 2449笔, 胜率49.9%, Sharpe 0.54, PnL $2165
- ORB: 364笔, 胜率49.5%, Sharpe 1.29, PnL $1014
- gap_fill: 348笔, 胜率47.7%, Sharpe -1.71, PnL -$718 → 已禁用

### 边际改进 A/B (H1, 2015-2026)
- 基准 A: Sharpe 1.12, PnL $6498, MaxDD $696
- B1 周一仓位×0.7: Sharpe 1.05(-0.07) → 不采纳
- B2 SELL仓位×0.5: Sharpe 1.28(+0.16) → 谨慎考虑
- B3 SELL ADX>=28: Sharpe 1.10(-0.02) → 不采纳
- B4 周一×0.7+SELL×0.5: Sharpe 1.20 → 不如单独B2
- B5 周一×0.7+SELL ADX>=28: Sharpe 1.00(-0.12) → 最差

### M15 RSI ADX 过滤 (2020-2026, M15+H1)
- A 基准: 1666笔, Sharpe 0.76, PnL $718
- B1 RSI block H1_ADX>40: Sharpe **0.97** (+0.21) → 已实装
- B2 RSI block H1_ADX>35: Sharpe 0.84 → 过滤过多
- 关键发现: M15 RSI 是主力利润来源(PnL $530 vs Keltner $327)

### 追踪止盈 R2/R3 (2015-2026)
- R2冠军 Trail0.8/0.25: Sharpe 1.74, PnL $13,149, MaxDD $741, 11年全正
- C12冠军: T0.8/0.25+SL3.5+TP5+ADX18, Sharpe **2.54**, PnL $18,147, MaxDD $553

### 成本调整回测
| 配置 | sp=$0 | sp=$0.30 | sp=$0.50 | sp=$0.80 |
|---|---|---|---|---|
| True Baseline | 0.53 | -1.04 | -2.14 | -3.67 |
| C12 | 2.54 | 0.73 | -0.53 | -2.37 |
| C12 + Adaptive Trail | 2.90 | 1.10 | -0.17 | -2.03 |
| Combo | 3.46 | 1.65 | 0.35 | -1.56 |

### Phase 5 盘中自适应 (C12 config)
| 配置 | N | Sharpe | PnL | MaxDD |
|---|---|---|---|---|
| C12 Baseline | 15,770 | -0.53 | -$3,656 | $4,243 |
| **C12 Adaptive** | **7,365** | **1.03** | **$5,497** | **$732** |

### SL 精调 ($0.50 + Adaptive)
- SL=4.5: Sharpe **1.35**, MaxDD $559 → 已实装
- SL=4.0: Sharpe 1.34, MaxDD $647
- SL=5.0: Sharpe 1.29, MaxDD $553

### 点差×冷却期交叉 (20 变体)
- 最优: $0.30/cd=30min/Adaptive, Sharpe **1.93**
- 30min 在所有点差下均优于 3h

### EXP20 K-Fold 最终决策
- KC25/M1.2: K-Fold 4/6折赢 → 已实装
- EMA150: Fold4崩溃 → 否决

### EXP07 V3 ATR Regime
- $0.50: Sharpe 1.52→1.85 (+0.33)
- K-Fold 6/6 折全赢 → 已实装

### EUR/USD 11年回测
- KC mult=2.0 MaxHold=20: Sharpe 1.91, 12/12年全正
- K-Fold 6/6折全正面, Avg Sharpe 1.71

### D1+3h 过拟合检测
- K-Fold 6折全正(Avg=8.52, Min=5.14) PASS
- PBO=0.41 < 0.50 PASS
- PSR p=0.000000 PASS
- DSR > 0.95 PASS
- 参数敏感性 Sharpe 8.01-9.41 无悬崖 PASS

### Trail Momentum (+50%) — K-Fold 验证通过 ✅
- 全样本: Sharpe 8.45→8.89 (+0.44), PnL $59K→$81K, MaxDD +7%
- K-Fold Current 1.5x: 6/6折全正 delta (+0.44~+0.73), MaxDD最坏+17%
- K-Fold Mega 1.5x: 6/6折全正 delta (+0.36~+0.55), MaxDD最坏+35%
- **结论: 1.5x 可实装，2.0x 过激**

### T7 ExtremeRegime (待验证)
- 全样本 Mega: Sharpe 8.39→9.59 (+1.20), 12/12年全赢
- 高波动档: trail_act=0.25, trail_dist=0.05（当前 0.4/0.10）
- **需要 K-Fold + 带成本验证**

### KC Bandwidth Filter (否决)
- BW3→BW12 全部降低 Sharpe (-0.40 to -1.60)
- 12/12年BW3均不如基线

### BW Confidence Sizing (否决)
- Conservative K-Fold: 3正2负（不稳定）
- Mega 全样本仅 +0.02 Sharpe
- 效果太弱不采纳

### ATR Spike Protection (真实引擎, 否决)
- post-hoc +0.48 Sharpe → 真实引擎仅 +0.03
- K-Fold 6/6折正面但每折仅 +0.02~0.04
- 不值得增加复杂度

### Entry Quality Filters (否决)
- min_h1_bars: 每增1bar限制 Sharpe降~0.10（过滤好信号）
- ADX gray zone: 最好 gray=7/score≥0.50 也仅 -0.03 Sharpe
- **全部无效，不采纳**

### Session Filter / SL优化 / TP Sweep (否决)
- Session Filter: 所有时段过滤 Sharpe 均下降 (-0.96 to -3.71)
- SL=5.0: 仅 +0.06 Sharpe，PnL 反降 → 保持 SL=4.5
- TP Sweep 4.0-12.0: 全部 ±0.01 → 保持 TP=8.0

### Stochastic 10/90 (待观察)
- 5/6 K-Fold 折正 Sharpe，全样本 $/t=$28.53
- 但样本量极小（10-21笔/折），不足以决策
- **待 paper trade 观察**

### ORB K-Fold (NoORB 测试失败)
- DisabledORB monkey-patch 未生效，Full=NoORB 完全一致
- ORB 交易仅占 <2%，影响可忽略
- **暂时保留**

### 第五批否决列表补充
- Strategy A/C/D: 596种组合全部不如基线
- London Breakout: 全部无效
- Regime Validation: 确认V3有效，其他无用
- Factor Importance: 确认 BW/ATR/ADX 为前三因子
- Trump Factor: 2024后+$1107但样本太小
