# 系统变更记录 (Changelog)

> **读取频率: 需要时查阅，不需每次读**

---

## 2026-04-09
- 新增 entry quality filters 到 backtest engine (min_h1_bars_today, adx_gray_zone, escalating_cooldown) — 待回测验证
- 新增 run_exp_entry_quality.py 和 test_entry_quality.py
- Memory Bank 迁移: trading_journal.md → docs/memory-bank/ 6 文件分离

## 2026-04-08
- 修复 backtest engine IntradayTrendMeter 索引 bug — choppy 门控之前在回测中从未生效
- 修复 monkey-patch 信号注入 bug — Strategy A/C/D 的信号注入之前无效
- EXP-MOM 短窗口动量旁路回测: 效果微弱(±0.01 Sharpe), 不上线

## 2026-04-07
- Strategy A/C/D 三策略 596 种组合回测完成, 全部无法超越基线
- EUR/USD Keltner 深度回测(11年), P9_eurusd_keltner 注册模拟盘
- P7_mega_trail, P8_mega_h20 注册模拟盘
- EXP36-47 实验完成(时段/分批止盈/RSI背离/D1方向/K线形态/ATR反波动率/宏观/波动率聚集/整数关口/隔夜收益)

## 2026-04-05
- **SL 4.5 ATR + Cooldown 30min 实装** (commit 379cb17)
- **日内亏损递减手数上限** — MAX_LOT_CAP_BY_LOSSES 替代旧 lot_scale
- **KC EMA 25 + KC mult 1.2 实装** (EXP13/EXP20)
- **V3 ATR Regime 启用** (EXP07 K-Fold 6/6 折全赢)
- **TP 8.0 ATR 实装** (EXP06)
- Mega Grid Search 1440 组合 + Walk-Forward 18 窗口验证
- EXP28 事件日防御全部否决
- Polymarket 监控 v2 (pizzint.watch NEH API)
- 旧引擎 7 个文件删除 (~3055 行)

## 2026-04-04
- backtest/ 统一回测包重构 (engine.py + stats.py + runner.py)
- 4 个核心实验脚本迁移至统一包
- P2 动态点差模型, P3 高级统计检验, P4 宏观Regime自动识别
- macro/ 宏观数据管道 (yfinance + FRED)
- 成本调整回测: $0.50 点差下当时仅 Combo 勉强存活
- 大趋势日识别研究: Oracle Sharpe 6.44 但技术指标无法预判
- 盘中自适应系统设计 + Phase 5 回测

## 2026-04-03
- 追踪止盈 Round 2 精调 (22 变体)
- R3 组合测试 — C12 冠军: Trail0.8/0.25 + SL3.5 + TP5 + ADX18, Sharpe 2.54
- 过拟合检测 4 项全通过
- 高级回测套件: 蒙特卡洛/K-Fold/Regime自适应/参数探索
- 组合验证: Adaptive Trail + KC1.25 + KC_EMA30 三合一通过
- 策略压力测试: **发现无成本回测的致命缺陷** (18,606笔交易成本 $25,043)

## 2026-04-02
- 架构大重构: gold_trader.py → data_provider + risk_manager + position_tracker + gold_trader
- IC 监控上线, 42 个单元测试
- Cursor Rule 创建, trading_journal.md 创建
- 首次全量因子 IC 扫描 (27因子×3窗口=81次检验)
- M15 RSI ADX>40 过滤实装
- 修复 CLOSE_DETECTED 重复 PnL bug
- 新增 backtest.py, factor_scanner.py, factor_deep_dive.py, backtest_ab_test.py
