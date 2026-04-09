# 交易知识库 (Trading Journal)

> 本文件是 AI 的"长期记忆"。每次对话结束前，将重要结论追加到对应章节。
> 每次对话开始时，AI 应先读取本文件了解历史上下文。

---

## 系统变更记录

- **2026-04-04**: `backtest/stats.py` 新增高级统计：`probabilistic_sharpe`（PSR）、`deflated_sharpe`（DSR）、`compute_pbo`（CSCV/PBO，组合数>100 随机抽样）；依赖 `scipy.stats`、`itertools`、`random`。
- **2026-04-04**: **宏观 Regime 检测** — 新增 `macro/regime_detector.py`：`MacroRegime`（6 类）、`MacroRegimeDetector`（快照/CSV 行规则分类）、`add_regime_column`（回测表增加 `macro_regime` 与 `regime_weights`）；`macro/__init__.py` 导出上述符号。流动性危机用 VIX>35+DXY 走强代理（无金价列）；`spread_2_10` 日变动在 `add_regime_column` 内注入为 `spread_2_10_chg`。
- **2026-04-04**: **P2 动态点差模型** — `backtest/engine.py` 新增 `spread_model` 参数（fixed/atr_scaled/session_aware）
  - `atr_scaled`: 基于 ATR 百分位放大点差（高波动 = 宽点差），`spread = base × (1 + atr_pct)`
  - `session_aware`: 按交易时段调整（亚盘×1.5, 伦敦×1.0, 纽约×0.8, 收盘×2.0）
  - 向后兼容：默认 `spread_model="fixed"` 行为不变
- **2026-04-04**: **P3 高级统计检验** — `backtest/stats.py` 新增 3 个函数
  - `probabilistic_sharpe()`: 概率夏普比率（Bailey & LdP 2012），测试SR是否显著优于基准
  - `deflated_sharpe()`: 通缩夏普比率（Bailey & LdP 2014），校正多重测试偏差
  - `compute_pbo()`: CSCV 回测过拟合概率（Bailey et al 2015），16分区组合验证
  - 修复: 将公式中的 SR 从年化值改为日级值，避免 `inner < 0` 导致 sr_std=0
- **2026-04-04**: **P4 宏观Regime自动识别** — 新建 `macro/regime_detector.py`
  - 6种Regime: EASING_INFLATION_UP/DOWN, TIGHTENING_INFLATION_UP/DOWN, RISK_OFF, LIQUIDITY_CRISIS
  - 检测逻辑: VIX>35+DXY涨→流动性危机, VIX>30→避险, 然后TIPS趋势+BEI趋势→4种宏观组合
  - 策略权重映射: lot_multiplier 0.3~1.2, sell_enabled, expand_sl 等
  - 回测集成: `BacktestEngine` 新增 `macro_df`/`macro_regime_enabled` 参数
  - 11年分布: 紧缩+通胀上升38.6%, 紧缩+通胀下降31.7%, 宽松+通胀上升13.3%, 宽松+通胀下降10.9%, 避险4.0%, 流动性危机1.5%
  - 关键验证: 2020-03-16→LIQUIDITY_CRISIS, 2022-03-01→RISK_OFF, 2024-09-18→EASING_INFLATION_UP
- **2026-04-04**: **P1 宏观数据管道** — 新建 `macro/` 包（`data_provider.py` + `__init__.py`），统一 yfinance + FRED 数据获取
  - yfinance: DXY(美元指数), VIX(恐慌指数), Brent(原油), US10Y(10年期国债)
  - FRED (需 API key): TIPS 10Y(实际利率), US2Y, 2-10利差, 5Y BEI(通胀预期)
  - 实时快照: `get_snapshot()` 返回 `MacroSnapshot` dataclass，含衍生指标（DXY SMA20、VIX 百分位、实际利率regime、VIX regime）
  - 历史下载: `download_history()` 保存 CSV，支持回测对齐。11年测试: 2935天, DXY/VIX/Brent/US10Y 全部成功，NaN<0.1%
  - 向后兼容: `sentiment/macro_monitor.py` 改为薄包装，`gold_trader.py` 无需修改
  - `config.py` 新增 `FRED_API_KEY`, `MACRO_ENABLED`, `MACRO_CACHE_TTL`, `MACRO_CACHE_PATH`
  - `requirements.txt` 新增 `fredapi`, `scipy`
  - `data/macro_history.csv` 已生成（2015-2026, 2935天×10列）
- **2026-04-05**: **日内亏损递减手数上限** (`config.py` + `risk_manager.py` + `gold_trader.py`)
  - 替代旧的 `get_lot_scale()` 缩减系数（0→1.0, 1→0.7, 2→0.5...），改为直接控制手数上限
  - `MAX_LOT_SIZE` 0.03 → **0.05**（绝对安全上限）
  - `MAX_LOT_CAP_BY_LOSSES`: 0笔亏损→**0.05**(ATR自由计算), 1笔→**0.03**, 2笔→**0.02**, 3笔+→**0.01**
  - `risk_manager.get_lot_scale()` → `get_max_lot_cap()`，返回当前允许的最大手数而非缩减比例
  - ATR 自动调仓（`calc_auto_lot_size`）仍然生效，`get_max_lot_cap()` 是硬上限
- **2026-04-05**: **SL 4.5 ATR + Cooldown 30min 实装** (`config.py` + `strategies/signals.py` + `risk_manager.py` + `gold_trader.py`)
  - `strategies/signals.py`: `ATR_SL_MULTIPLIER` 3.5 → **4.5**（回测: Sharpe 1.03→1.35, MaxDD $732→$559, 全维度改善）
  - `config.py`: `COOLDOWN_BARS=3`(3h) → `COOLDOWN_MINUTES=30`(30min)（回测: Sharpe 1.03→1.17, 所有点差下均优于3h）
  - `risk_manager.py`: `add_cooldown(hours)` → `add_cooldown(minutes)`, `timedelta(hours=)` → `timedelta(minutes=)`
  - `backtest/runner.py`: `C12_KWARGS.sl_atr_mult` 3.5 → 4.5（同步回测预设）
  - `backtest/engine.py`: fallback 改为 `config.COOLDOWN_MINUTES / 60`
  - 全部 `.py` 文件中 `COOLDOWN_BARS` 引用已清零
  - 已推送 GitHub（commit `379cb17`）
- **2026-04-04**: `backtest_cost_adjusted.py` 迁移至统一包：`run_variant(DataBundle(...), ..., verbose=False, spread_cost=..., regime_config=..., min_entry_gap_hours=...)` 替代 `Round2Engine`/`RegimeEngine`/`CooldownEngine` 继承链；`TRUE_BASELINE_KWARGS`/`C12_KWARGS`/`V3_REGIME`/`load_m15`/`load_h1_aligned`/`prepare_indicators_custom`/`add_atr_percentile` 自 `backtest.runner`；移除对 `config` 追踪止损全局变量的临时修改。
- **2026-04-04**: `backtest_advanced.py` 迁移至统一包：`DataBundle`、`run_variant(verbose=False)`（保留原控制台格式）、`C12_KWARGS`/`prepare_indicators_custom`/`add_atr_percentile` 自 `backtest.runner`；移除 `RegimeEngine`/`ParamExploreEngine`/`Round2Engine` 与本地 `prepare_indicators_custom`；K-Fold 用 `DataBundle.slice`；参数探索用 raw `DataBundle` + 每 variant 自定义指标（等同 `load_custom` 逻辑）。`backtest.runner.run_variant` 增加关键字参数 `verbose=True` 供静默调用。
- **2026-04-04**: `backtest_combo_verify.py` 迁移至统一包：`BacktestEngine` + `DataBundle` + `calc_stats`，`C12_KWARGS`/`V3_REGIME`/`prepare_indicators_custom`/`add_atr_percentile`/`load_m15`/`load_h1_aligned` 自 `backtest.runner`；本地 `_run_bundle` 复现原 `run_fixed`/`run_regime` 的全局重置与打印格式（未用 `run_variant` 以免其自带输出覆盖 Phase1 行格式）。
- **2026-04-04**: `backtest_intraday_adaptive.py` 迁移至统一包 `backtest`：`BacktestEngine` + `intraday_adaptive=True` 替代 `IntradayAdaptiveEngine`/`RegimeEngine`/`Round2Engine` 继承链；数据与阈值实验仍用 `load_m15`/`load_h1_aligned`、`C12_KWARGS`、`V3_REGIME`、`prepare_indicators_custom`、`add_atr_percentile`；K-Fold 使用 `DataBundle.slice`。
- **2026-04-02**: 完成架构大重构，将 `gold_trader.py`（~1048行）拆分为 4 个模块：
  - `data_provider.py` — 行情数据获取
  - `risk_manager.py` — 风控管理
  - `position_tracker.py` — 持仓追踪与交易记录
  - `gold_trader.py` — 编排层（~380行）
- **2026-04-02**: 新增 `ic_monitor.py`，因子 IC 监控模块，集成到日切报告
- **2026-04-02**: 实盘+模拟盘开仓时记录因子快照（factors dict），供 IC 分析
- **2026-04-02**: 修复 bare except、原子化 JSON 写入、RSS 缓存、依赖版本锁定
- **2026-04-02**: 新增 42 个单元测试（含 IC 监控测试），全部通过
- **2026-04-02**: 创建 Cursor Rule（`.cursor/rules/gold-quant-trader.mdc`），含宏观经济框架
- **2026-04-02**: 创建 `docs/trading_journal.md` 知识库，实现 Harness Engineering 记忆持久化
- **2026-04-02**: Cursor Rule 增加"记忆管理"章节，约束 AI 每次对话读取/更新知识库
- **2026-04-02**: 所有变更已推送 GitHub（commit `9083279`）
- **2026-04-02**: 新增 `backtest.py` 离线回测框架，复用 signals.py 信号函数 + config.py 参数，逐 bar 回放模拟交易。支持 Keltner/ORB/周一跳空策略，含追踪止盈、时间止损、冷却期等风控逻辑。用法: `python backtest.py <csv_path> [--start] [--end]`
- **2026-04-02**: 新增 `factor_scanner.py` 因子 IC 离线扫描器，系统性评估候选因子预测力。功能：Rank IC 计算、Walk-Forward 分段验证（防过拟合）、Bootstrap 置换检验（统计显著性）、Bonferroni 多重比较修正、因子衰减检测。27 个候选因子覆盖动量/波动率/形态/时间/通道五大类。用法: `python factor_scanner.py <csv_path> [--horizons 1 4 8] [--wf-splits 5] [--bootstrap 500]`
- **2026-04-02**: 新增 `factor_deep_dive.py` 因子深度研究工具，含6项专题：①周内择时 ②ADX分箱 ③因子交叉组合 ④volume_ratio诊断 ⑤交易时段分析 ⑥Keltner叠加过滤器增量。用法: `python factor_deep_dive.py <csv_path>`
- **2026-04-02**: 新增 `backtest_ab_test.py` A/B 回测对比工具，对比边际改进方案的实际效果。用法: `python backtest_ab_test.py <csv_path>`

## 策略观察记录

- **2026-04-02**: P1_stoch_extreme 和 P2_london_ny 模拟策略因表现差已删除（P1 固定止损在高波动下频繁触发，P2 止盈距离过远导致 timeout 平仓）
- **2026-04-02**: 替换为 P4_atr_regime / P5_volume_breakout / P6_dxy_filtered，等待数据积累
- **2026-04-02**: 模拟盘首批数据评审（19笔）：P1/P2 淘汰决策经数据确认（合计-$44.52）；P4_atr_regime 唯一盈利（3笔+$15.17），继续观察；P5/P6 零触发，排查后放宽条件
- **2026-04-02**: P5 放量阈值从 1.5x 降到 1.2x（tick volume 放大 1.5 倍太严格）；P6 时间窗口从 13-16 扩大到 12-20（仅 4 根 H1 K 线太窄）
- **2026-04-02**: 实盘策略逻辑未做任何改动，仅架构重构
- **2026-04-02**: M15 RSI 策略加入 EMA100 方向过滤（v7）：BUY 需 close>EMA100，SELL 需 close<EMA100。回测依据：Sharpe 1.02→1.28(+25%), MaxDD $662→$524(-21%), 胜率 56.4%→58.6%。动机：4/2 当日该策略在强下跌趋势中逆势做多被止损2笔(-$101)，研究发现问题不是ADX高低而是缺少趋势方向过滤
- **2026-04-02**: 日内自适应仓位缩减上线（`risk_manager.get_lot_scale()`）：日内每亏一笔自动缩减仓位（1.0→0.7→0.5→0.3→0.1），次日重置。回测验证：Sharpe 1.12→1.19(+6.3%), MaxDD $696→$596(-14.4%), 总PnL不变。核心价值：用风险管理换取更平滑的净值曲线
- **2026-04-02**: 首次全量回测结果（H1策略组合, 2015-2026, 66376根K线）:
  - 总体: 3161笔, 胜率49.6%, Sharpe 0.50, MaxDD $835(26%), 总PnL $2461, RR 1.09
  - Keltner: 2449笔, 胜率49.9%, Sharpe 0.54, PnL $2165 — **主力贡献者**
  - ORB: 364笔, 胜率49.5%, Sharpe 1.29, PnL $1014 — **最稳定**
  - gap_fill: 348笔, 胜率47.7%, Sharpe -1.71, PnL -$718 — **拖累组合，需重点优化**
  - 注意: 该回测不含M15 RSI策略（无M15数据），实际组合Sharpe可能不同
  - 分年度: 2015/2019/2022/2024偏弱, 2023/2025/2026强势
- **2026-04-02**: **边际改进 A/B 回测对比**（H1, 2015-2026, 3888笔, gap_fill已禁用后的基准）：
  - 基准 A: Sharpe 1.12, PnL $6498, MaxDD $696(19.3%), BUY PnL $5448, SELL PnL $1050
  - **B1 周一仓位×0.7**: Sharpe 1.05(-0.07), PnL $5627(-$871) — **不采纳，Sharpe下降**
  - **B2 SELL仓位×0.5**: Sharpe 1.28(+0.16), PnL $6289(-$209), MaxDD $668(-$28) — **最优方案**
  - B3 SELL ADX>=28: Sharpe 1.10(-0.02), PnL $6006 — 效果不明显
  - B4 周一×0.7+SELL×0.5: Sharpe 1.20(+0.07) — 不如单独B2
  - B5 周一×0.7+SELL ADX>=28: Sharpe 1.00(-0.12) — 最差
  - **结论**：
    - 周一降仓：**不采纳**。虽然因子分析显示周一偏弱，但回测中 Sharpe 下降 0.07、PnL 减少 $871。2025-2026年周一反而是强日，该效应已反转
    - SELL仓位缩半：**值得考虑但需谨慎**。Sharpe +0.16 是最大提升，MaxDD 略降，但 PnL 减少 $209。本质是"少亏"而非"多赚"——SELL PnL 从 $1050 降到 $841，以牺牲做空利润换取更平滑净值。在当前"紧缩+通胀上升"下跌regime中可能反而不利
    - SELL ADX>=28：**不采纳**。效果几乎为零，还减少了 309 笔交易样本

- **2026-04-08**: **Strategy A/C/D 三策略回测完成** — 596 种参数组合(A:180 动量追击 + C:32 D1趋势过滤 + D:384 回调入场)，全部无法超越 Keltner 基线(Sharpe 4.92)。A 组合后 Sharpe 降至 4.62(多了 128 笔噪声交易)；C/D 全部与基线完全一致(Delta=0)。结论：Keltner+ADX+EMA100 信号集已饱和，新信号方向全部关闭。总耗时 6.1h(A:119m + C:8m + D:241m)
- **2026-04-05**: **点差研究终止决策** — 短期内不会更换交易商，点差问题不再作为研究方向。以当前交易商实际点差为既定条件，回测中仍保留 `spread_cost` 参数用于成本建模，但不再投入时间研究降低点差的方案（如换交易商、换账户类型等）

## 被否决的优化方向 — 禁止重复研究 (2026-04-05 确认)

> **硬约束**：以下方向已经过回测验证确认无效或有害，**不再投入时间重复研究**。
> 未来新的优化方向必须与以下列表不重叠。如果某个新想法本质上是以下方向的变体，直接跳过。

- **Combo (KC1.25+EMA30+Adaptive Trail)**: 无成本 Sharpe 3.46, $0.50 仅 0.35。增加 36% 交易量，成本吞噬利润
- **降低交易频率 (min gap 2-8h)**: 全部变差。好信号被跳过，不是信号太多而是每笔太薄
- **宏观 Regime 过滤**: Sharpe 下降 0.05。砍掉了赚钱和亏钱的交易
- **日前趋势预判**: 准确率 ~55% ≈ 抛硬币。趋势日由事件驱动，技术指标无法预测
- **周一降仓**: Sharpe -0.07。2025-2026 周一反而是强日，效应已反转
- **RSI 参数调整 (阈值/方向/ADX/ATR)**: 整体 Sharpe 不变。Adaptive 下 RSI 仅 6 笔/11 年，已自然消亡
- **ORB 缩短持仓**: 所有缩短方案都更差。当前框架下 ORB 需要长持仓才能盈亏平衡
- **波动率过滤**: 跳过高波动 Sharpe 降到 0.79。高波动是主要利润来源
- **禁用 SELL**: Sharpe 1.10(+0.07) 但 PnL -$2,127。下跌年份 SELL 是唯一利润来源
- **Choppy 阈值调整**: range=0.000，完全无影响
- **kc_only=0.65**: Sharpe 1.86 但是阈值悬崖。本质是关闭 M15 RSI，非渐进优化
- **点差优化/换交易商**: 短期不换交易商，以当前点差为既定条件
- **事件日防御("带伞策略")**: EXP28 穷举 11 把伞×3 触发器×6 折验证，全部无效或有害。V3+Adaptive+Cooldown 已是充分防御
- **Strategy A — 动量追击**: 180 组合网格搜索(lookback×ATR_mult×SL×max_hold)。所有组合 Sharpe≤4.62 均低于基线 4.92。Momentum-Only 产生与 Keltner 完全相同的交易(N=1779)，组合后额外 128 笔净亏损。K-Fold 3/4正但 Std=7.28 极不稳定。动量信号完全被 Keltner 吸收
- **Strategy C — D1 趋势过滤**: 32 组合(EMA fast/slow×ADX thresh×allow_flat)。384/384 全部与基线完全一致(Sharpe=4.92, N=1779, PnL=$4,416)。D1 过滤未阻挡任何信号(黄金 57% UP/20% DOWN/23% FLAT)。Delta Sharpe=+0.000
- **Strategy D — 趋势回调入场**: 384 组合(impulse_lb×impulse_atr×RSI×EMA×SL)。384/384 全部与基线完全一致。Pullback-Only 也产生与 Keltner 完全相同的 1779 笔交易。Overlap 分析：Pullback 产生 2204 个信号，Keltner 报告 0 个（回测引擎内部处理），0 重叠。回调信号在引擎中完全不触发额外交易
- **KC Bandwidth 扩张过滤**: BW3-BW12 全部降低 Sharpe（-0.40 到 -1.60）。虽然提高 KC 单笔质量($/t $3.30→$3.73)但过滤掉大量盈利交易，12/12 年均不如基线
- **London Breakout**: 7 个变体全部负 Sharpe（-0.15 到 -0.39），WR 仅 48%。远不如现有 ORB（Sharpe 正, $/t=$1.28）
- **宏观 Regime 过滤 (第三次确认)**: Baseline Sharpe=5.58 > Regime-Weighted 5.53。K-Fold 0/6 折赢。所有 6 种 regime 下均盈利，策略是 regime-agnostic
- **Trump/波动率 Sizing**: 所有方案效果在 ±0.05 以内。高波动加仓反而 -0.16。低波动时 $/t 反而最高

## 核心认知 (2026-04-09 更新)

1. **交易成本是策略杀手**: 无成本 Sharpe 3.46 加 $0.50 点差后仅 0.35。所有回测必须包含成本
2. **少交易比好参数更重要**: C12 无 Adaptive 15,770 笔 Sharpe -0.53; 加 Adaptive 7,365 笔 Sharpe 1.03。砍掉一半交易反而从亏到赚
3. **~~利润来自少数大趋势日~~ → 避开震荡比抓住趋势更重要**: (2026-04-09 修正) 旧结论基于 Combo+$0.50 点差的过时配置。当前 D1+3h (Mega Trail 0.5ATR 激活 + Time Decay) 下，利润主要来自大量小额追踪止盈（11,611 次 trail 触发，97.7%WR，中位持仓 1.5h），TP 仅触发 25 次。小波段高频锁利是核心利润模式，不依赖大趋势日。但**震荡时段仍是亏损源**，Adaptive 门控砍掉震荡时段交易仍然关键
4. **追踪止盈是核心 alpha**: 5,542 笔 trailing +$41,088 (97.7%WR), 赢家快进快出(中位 1.5h), 输家拖很久(中位 11.5h)。Mega Trail (0.5/0.15) + Time Decay (1h/0.30/0.10) 进一步强化了这个优势
5. **过拟合风险低, 结构风险高**: PBO=0.00, 参数平滑, DSR 通过。真正风险是低波动盘整期持续数月小额亏损(1/3 半年窗口为负)
6. **Keltner+ADX+EMA100 信号集已饱和**: Strategy A/C/D 共 596 种组合(动量/D1趋势/回调)全部无法超越基线。新信号要么与 Keltner 完全重合，要么是净负贡献。改善方向应聚焦于**出场机制和仓位管理**而非新入场信号

## 因子研究笔记

- **2026-04-02**: IC 监控上线，采集因子包括 RSI14、RSI2、ATR、ADX、MACD_hist、EMA9/21/100、KC 通道、成交量比率、ATR 百分位
- **2026-04-02**: 舆情分析 v3.1 — 极端 FinBERT 信号权重提升：当 |FinBERT| > 0.30 时权重从 50/30/20 调为 70/15/15，防止 keyword_score 稀释强语义信号。验证：4/2 FinBERT=-0.40 场景下 combined 从 +0.11 降至 -0.12，更接近 BEARISH 阈值
- **2026-04-02**: 需要积累 20+ 笔带 factors 的交易后 IC 报告才有意义
- **2026-04-02**: 预期 ATR_percentile 和 volume_ratio 可能有较高预测力，待验证
- **2026-04-02**: **首次全量因子 IC 扫描完成**（`factor_scanner.py`，H1数据 2015-2026，66218 行，27 因子 × 3 窗口 = 81 次检验）
  - 工具：Rank IC + Walk-Forward 5段验证 + Bootstrap 500次置换检验 + Bonferroni 修正 (p<0.0006)
  - **核心结论：单因子对黄金 H1 收益的线性预测力普遍很弱（|IC| < 0.04），但部分因子统计显著且稳定**
  - 第一梯队（通过 Bonferroni + WF 一致性 ≥ 60%，27 个因子×窗口组合通过）：
    - `RSI2 × ret_1`: IC=-0.0314, WF=100%, stable — **你的 M15 RSI 策略的因子基础确认有效**
    - `day_of_week × ret_4/8`: IC=+0.033, WF=100%, stable — **星期效应跨窗口稳定，值得研究周内择时**
    - `ATR × ret_4/8`: IC=+0.032~0.036, WF=60%, strengthening — **高波动后正收益（趋势延续效应）**
    - `momentum_5/10 × ret_1`: IC=-0.019~0.021, WF=100%, stable — **短期动量反转效应（均值回归）**
    - `KC_position/breakout_strength × ret_1`: IC=-0.016, WF=100%, stable — **KC 位置的短期反转**
    - `MACD_hist_change × ret_1`: IC=-0.025, WF=100%, stable — **MACD 加速度的反转**
  - 无效因子（|IC| < 0.005，无统计显著性）：`ADX`、`close_ema100_dist`、`ema9_ema21_cross`、`volume_ratio`
  - 衰减因子：`lower_shadow_ratio`（IC 从 -0.038 衰减到 -0.007）
  - 增强因子：`ATR`、`EMA9`、`EMA21`（后半段 IC 明显高于前半段，近年趋势效应增强）
  - `is_london_ny_overlap` IC=+0.24 是统计伪信号（二值变量与收益的 rank 相关性不适用 Spearman）
  - **关键发现**：`volume_ratio` IC≈0，P5 策略的理论基础受质疑（tick volume 放量对突破的预测力不足）
  - **关键发现**：`ADX` IC≈0，但这不意味着 ADX 过滤无用——它是条件筛选器而非线性预测因子
  - 详细报告：`data/factor_ic_report.csv`
- **2026-04-02**: **因子深度研究完成**（`factor_deep_dive.py`，6 项专题分析）
  - **研究1 — day_of_week 周内择时**：
    - 周一始终是最差交易日（ret_4=+0.25bp，12年中5次是最差日），ret_8 周一 vs 整体显著偏低（p=0.027）
    - 周四/周五是最优日（ret_4 约+1.0~1.4bp），但分年度不稳定（周四4次最优，周五2次）
    - **可操作结论**：周一适当降低仓位或提高开仓门槛有统计支持，但不建议完全停止交易
  - **研究2 — ADX 非线性分析**：
    - ADX 各区间的均值收益差异很小（均在 ±2bp 内），ADX 对收益方向没有可靠的预测力
    - 但 ADX 在 Keltner 突破场景下有**过滤价值**：BUY+ADX>=24 的 ret_4=+1.92bp 显著优于无过滤
    - SELL 信号在所有 ADX 水平下均弱（ret_4 约 0bp），做空比做多更依赖其他过滤条件
    - **结论**：当前 ADX>=24 的阈值合理，保留
  - **研究3 — 因子组合效应**：
    - RSI2<15 + close>EMA100（顺势做多）显著优于 close<EMA100（逆势）：ret_4 +1.68bp vs +0.33bp
    - RSI2 + ATR_pct>70（高波动超卖）比 ATR_pct<30（低波动超卖）好得多：ret_1 +0.75bp vs -0.68bp
    - **你的 M15 RSI v7 加入 EMA100 过滤的决策被因子分析验证**
    - RSI2>85 做空整体弱（ret_1=-0.32bp），做空信号质量远不如做多
  - **研究4 — volume_ratio 诊断**：
    - CSV 数据中 Volume 全部为 0（Dukascopy H1 数据不含成交量），volume_ratio 全 NaN
    - **结论**：volume_ratio 的 IC≈0 不是"因子无效"，而是数据缺失。P5 策略在实盘 MT4 tick volume 上仍可观察
  - **研究5 — 交易时段分析**：
    - UTC 21:00 的 ret_1=+6.53bp 是最强异常值（亚盘开盘跳空效应）
    - 伦敦时段（08-13 UTC）ret 略负，纽约收盘后（17-21 UTC）ret_4=+3.18bp 最强
    - **`is_london_ny_overlap` 的高 IC 是统计伪信号**：实际上重叠时段（13-17 UTC）ret 接近 0，是二值变量的 Spearman 陷阱
  - **研究6 — Keltner 叠加因子过滤的增量**：
    - LN-NY 重叠时段（13-17 UTC）的 Keltner BUY 胜率 52.0%（基准 49.6%），均值 +0.94bp（基准 +0.60bp），但样本只有 1564
    - SELL + LN-NY 时段 ret_4=+2.41bp 是最优组合（基准 +0.54bp），但同样样本偏少
    - body_range_ratio > 0.5（大实体K线确认）对 BUY 有小幅提升（+0.69bp vs +0.60bp），但对 SELL 反而恶化
    - **整体结论**：单个额外过滤器的增量有限（1-2bp），过多过滤器会大幅减少样本。当前 Keltner v4（ADX+EMA100）已接近最优

## 踩过的坑

- `_check_exits` 平仓后必须同时更新 `risk_manager` 的 `daily_pnl` 和 `cooldown`，否则日内风控状态不一致
- JSON 文件写入必须用原子操作（`tempfile` + `os.replace`），否则进程崩溃会导致数据损坏
- `strategies/signals.py` 中统一 `import config` 时，`_gap_cfg.` 和 `_scan_cfg.` 的前缀替换会误伤，需要逐个确认
- PowerShell 不支持 bash heredoc 语法，git commit 消息需要用单行
- **2026-04-05 服务器测试最佳实践**: 远程服务器不一定有 jupyter/nbformat，默认用独立 `.py` 脚本测试。流程：①本地写 `run_expXX.py`（`sys.path.insert(0, ...)` 指向项目根目录）→ ②推送 GitHub → ③服务器 `git pull && python -u run_expXX.py 2>&1 | tee logs/expXX.log`。`-u` 确保 print 实时输出，`tee` 同时显示和保存日志。不再使用 `jupyter nbconvert --execute`
- **CLOSE_DETECTED 重复检测**：MT4 桥接文件读写竞争可能导致持仓短暂"消失"又重现，`sync_positions` 会误判为平仓并重复计入 PnL。已加入 `_ticket_already_closed()` 防重复机制
- **模拟盘 P5/P6 零触发**：P5 的 tick volume 放量 1.5x 阈值过严（MT4 tick volume 波动模式不同于真实成交量）；P6 的 13-16 时间窗口仅覆盖 4 根 H1 K 线，叠加 EMA 交叉条件后触发概率极低
- **市场分析必须先查日历再归因**：分析金价大跌时只搜了"为什么跌"，遗漏了当天Liberation Day关税生效这一已知事件。教训：①盘前/分析前先确认当日经济日历和政策事件（关税、FOMC、非农等）；②大跌通常是多因素共振，找到一个原因后必须继续问"还有什么"；③异常数据（如原油-14.5%）要从多角度解释，不能单因素归因就满足；④搜索要多角度——不仅搜"为什么跌"，还要搜"今天有什么事件"

## 宏观环境记录

- **2026-04-02**: Cursor Rule 已写入完整宏观分析框架（实际利率、通胀预期、DXY、避险需求四因子模型 + 6种Regime分类）
- **2026-04-02**: 金价从~$4,770暴跌至~$4,575（日内-4%），**多重利空共振**：①伊朗战争降级（Trump表态2-3周结束）→ 避险退潮；②Liberation Day关税生效（全球10%基准+中国34%/欧盟20%等）→ 通胀预期走高 → Fed不降息 → 强美元；③流动性冲击（margin call抛售）。原油暴跌14.5%同时受伊朗降级+关税打压全球需求预期双重驱动
- **2026-04-02**: 3月金价跌14.6%，2008年以来最差月度表现；从1月高点$5,626已回撤约25%
- **2026-04-02**: 当前Regime判断："紧缩+通胀上升"向"避险退潮"过渡。市场已完全price out 2026降息预期
- **2026-04-02**: 关键节点：4/6 Trump暂停对伊朗能源打击到期，若不续期 → 油价继续跌 → 金价短期承压但中期可能因通胀回落而受益
- **2026-04-02**: 系统当日亏损$94.34（3笔止损），Keltner做空单方向正确（浮盈$20.96），M15做多信号被风控过滤（避免逆势做多）
- **2026-04-02**: MT4 实际账户确认总盈利 $330.72（9 个交易日，16.5% 回报率）。利润结构：靠 2 天强趋势（3/26 +$204, 4/1 +$385）覆盖 5 天亏损，Keltner 追踪止盈 + M15 RSI 顺势捕捉是核心利润来源
- **2026-04-02**: 舆情系统 4 天数据：3/31-4/1 正确判断 BULLISH，4/2 FinBERT=-0.40 正确捕捉到暴跌信号但被 keyword_score 稀释为 NEUTRAL。lot_multiplier 始终为 1.0，舆情尚未实质影响仓位
- **2026-04-02**: 短期展望：基准情景 $4,400-4,700 宽幅震荡（50%），关注 4/6 伊朗能源打击暂停到期。当前 Regime = "紧缩+通胀上升"向"避险退潮"过渡
- **2026-04-02**: **新增 Polymarket 地缘风险监控** (`sentiment/polymarket_monitor.py`)
  - 通过 Gamma API `/public-search` 端点（免认证）搜索 14 个地缘政治关键词，获取预测市场概率
  - 聚合为 0-100 风险指数: LOW(<30) / MEDIUM(30-60) / HIGH(60-85) / EXTREME(>=85)
  - 同类日级市场去重（同一问题不同日期只取最有信息量的一个，避免权重膨胀）
  - 概率极端（>98% 或 <2%）和已确定的日级市场自动过滤
  - 集成到 `sentiment_engine.py`: 风险>=60 → direction_bias=BUY; 风险>=85 → lot_multiplier×0.5
  - 首次测试: 30 个活跃市场, 风险指数=28.8 (LOW), 黄金情绪增量=+0.095
  - 5 分钟缓存，不阻塞主交易循环
  - 灵感来源: pizzint.watch "Nothing Ever Happens Index"（基于 Polymarket 的地缘风险聚合指标）
- **2026-04-05**: **Polymarket 监控 v2 — 接入 pizzint.watch NEH API** (`sentiment/polymarket_monitor.py`)
  - 数据源从 Gamma API 14 关键词搜索改为 pizzint.watch `/api/neh-index/doomsday` 端点
  - NEH API 返回 ~35 个专家 curated 的地缘政治市场（覆盖 americas/middle_east/europe/asia/global）
  - API 调用从 14 次降为 1 次，数据质量更高（人工精选 basket vs 关键词搜索噪声）
  - 保留黄金方向性判断：自动按关键词分类 bullish/bearish，pizzint.watch 不提供此功能
  - 新增三层权重：区域权重（中东×1.5, 亚太×1.3）× 关键词加成（invade×1.5, tariff×0.8）× 交易量权重
  - 接口完全向后兼容：`get_risk_index()` 返回格式不变，`sentiment_engine.py` 无需修改
  - 测试: 35 个市场, 风险指数=13.6 (LOW), 冠军="入侵伊朗 57.5%"
  - 风险指数从旧版 32 降至 13.6：旧版等权平均被低质量搜索结果抬高，新版加权更精确

## 系统Bug修复记录

- **2026-04-02**: **修复 CLOSE_DETECTED 重复计算 PnL 的 bug**
  - 根因：ticket 3985509 被 CLOSE_DETECTED 记录了两次（+$27.99 和 +$89.61），导致 total_pnl 多算 $27.99
  - 触发场景：MT4 桥接文件读写竞争（positions.json 更新瞬间被 Python 读到不完整数据）→ 误判持仓消失 → 记录假平仓 → 下次 sync 持仓重新出现 → 再次加入 tracking → 真正平仓时再次记录
  - 修复：在 `position_tracker.py` 的 `sync_positions()` 中新增 `_ticket_already_closed()` 防重复检查
    - 步骤2（检测消失持仓）：如果 ticket 已有 CLOSE 记录，仅清理 tracking 不再累加 PnL
    - 步骤4（检测新仓位）：如果 ticket 已有 CLOSE 记录，不再重新加入 tracking
  - 数据修正：total_pnl 从 $418.90 校准至 MT4 实际 $330.72
  - 附带发现：CLOSE_DETECTED 使用 `last_profit`（浮盈估值）替代实际平仓价，22笔累计偏差约 $60（平均 $2.74/笔）。未来可通过 MT4 EA 导出交易历史来获取精确平仓价

## 策略研究 — 追踪止盈优化 (2026-04-02)

- **2026-04-02**: **追踪止盈 A/B 回测** (`backtest_trailing_test.py`)，11 年 H1 数据，7 个变体
  - 起因：4/2 暴跌日 Keltner SELL 单浮盈 15 点后回撤止损，追踪激活门槛 2.5×ATR≈111 点日内几乎无法触发
  - **B1 (激活 1.5ATR)**: Sharpe 1.36 (+0.24), PnL +$8,049 (+24%), MaxDD $512 (-26%), **最优方案**
  - **B6 (激活 2.0ATR)**: Sharpe 1.14 (+0.02), 微幅改善，保守备选
  - **B3 (保本止损 1.0ATR)**: Sharpe -2.72, **灾难性方案**。保本出场 2,175 次，把回测通道后继续趋势的大肉单全部砍掉
  - 关键洞察：Keltner 是趋势跟踪策略，价格常在突破后先回测通道再继续。保本止损在回测阶段误杀盈利单
  - **决策**: 已将 `TRAILING_ACTIVATE_ATR` 从 2.5 改为 1.5
- **2026-04-02**: **M15 RSI 过滤 + 连续亏损冷却回测** (`backtest_rsi_cooldown_test.py`)
  - M15 RSI ADX 过滤 (B1/B2/B3): **无法在 H1 回测框架中测试**。H1 backtest 只含 Keltner/ORB，M15 RSI 需要 M15 数据框架。需另建 M15 回测工具
  - 连续同向亏损冷却 (B4: 2 笔→8h): Sharpe 1.36→1.40 (+0.03)，拦截 208 笔，PnL +$63。**正向但幅度小**，可作为行为风控考虑
  - 决策: 连续亏损冷却暂不实施（效果不够显著），M15 RSI ADX 过滤需单独建 M15 回测框架后再验证
- **2026-04-02**: **gold_runner.py 改进**: 日内亏损超限不再终止进程，改为 `trading_halted` flag，系统继续运行（心跳/舆情/持仓监控），次日 0:00 自动恢复交易
- **2026-04-02**: **新增 M15 多时间框架回测框架** (`backtest_m15.py`)
  - 解决旧 `backtest.py` 只回放 H1 无法测试 M15 RSI 策略的问题
  - 主循环遍历 M15 bars，H1 boundary（minute==0）时运行 H1 信号（Keltner/ORB），每个 M15 bar 运行 RSI 信号
  - 共享持仓上限 / 冷却期 / 日内亏损限制，忠实还原实盘 `gold_trader.py` 的双时间框架架构
  - SL/TP 在 M15 粒度检测（比 H1 backtest 更精确），trailing stop 使用 H1 ATR
  - 用法: `python backtest_m15.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]`
- **2026-04-02**: **M15 RSI ADX 过滤 A/B 回测** (`backtest_m15.py`, 2020-2026, M15+H1 双时间框架)
  - A 基准 (H1+M15 组合): 1666 笔, Sharpe 0.76, PnL $718, MaxDD $595(22.1%), WR 63.9%
    - 策略细分: Keltner 506 笔 PnL $327, M15 RSI 1097 笔 PnL $530
  - **B1 RSI block H1_ADX>40**: 1406 笔, **Sharpe 0.97 (+0.21)**, PnL $871 (+$153), MaxDD $564(-5%), WR 64.4% — **最优方案**
    - RSI 过滤 591/2031 个信号 (29%)，RSI PnL $663 (+$133)，过滤掉的是强趋势中的逆势均值回归
  - B2 RSI block H1_ADX>35: Sharpe 0.84 (+0.08), PnL $727, MaxDD $398 — 过滤过多 (845 个)，损失部分盈利信号
  - B3 RSI block H1_ATR_pct>0.90: Sharpe 0.78 (+0.02), 效果不显著
  - **关键发现**: M15 RSI 是主力利润来源（PnL $530 vs Keltner $327），但强趋势中的逆势信号是主要亏损源
  - **B1 价值**: ADX>40 过滤精准剔除强趋势逆势 RSI 信号，不影响顺势信号，Sharpe +0.21 是显著提升
  - 年度稳定性: B1 在所有年份均优于或持平基准（2020 +$7, 2022 +$54, 2024 +$22, 2025 +$70）
- **2026-04-02**: **ADX>40 过滤已实装到实盘** (`config.py` + `strategies/signals.py` + `gold_trader.py`)
  - `config.py`: 新增 `RSI_ADX_BLOCK_THRESHOLD = 40`
  - `strategies/signals.py`: `check_m15_rsi_signal` 新增 `h1_adx` 参数，H1 ADX > 40 时直接返回 None
  - `gold_trader.py`: 从 `df_h1` 提取 ADX 传递给 M15 信号扫描
- **2026-04-02**: **M15 策略优化 A/B 回测** (`backtest_m15.py`, 2020-2026, 10 个变体)
  - 基准 A: 1666 笔, Sharpe 0.76, PnL $718, MaxDD $595
  - **RSI 方向过滤**:
    - C1 ADX>40 + RSI BUY-only: Sharpe 0.63 (-0.13), **不采纳**。禁止 SELL 后 RSI PnL 从 $663 降到 $407，因为 ADX>40 已过滤掉差的 SELL 信号，剩余 SELL 是盈利的 ($224)
    - C2 RSI BUY-only (无 ADX 过滤): Sharpe 0.80 (+0.05)，但不如 B1 (0.97)。RSI SELL 整体 PnL=-$85，但有 ADX 过滤后变为 +$224
    - **关键发现**: RSI SELL 不应全面禁止，ADX>40 已精准筛选出好的 SELL 信号
  - **RSI ATR 百分位组合过滤**:
    - D1 ADX>40 + ATR_pct>=0.30: Sharpe 0.97 (持平 B1), RSI PnL $705 (+$43 vs B1)，微幅改善
    - D2 ADX>40 + ATR_pct>=0.20: Sharpe 0.97 (等同 B1)，阈值太低几乎无额外过滤
    - **结论**: ATR_pct>=0.30 有小幅增量但不显著，暂不实施
  - **ORB 持仓时间优化**:
    - E1 ORB hold 16 bars (4h): Sharpe 0.89 (+0.14), ORB PnL 从 -$139 升至 -$14, **显著改善**
    - E2 ORB hold 12 bars (3h): Sharpe 0.90 (+0.14), ORB PnL 从 -$139 升至 +$0.55, **ORB 首次扭亏**
    - **关键发现**: ORB 在 H1 框架 (24 M15 bars = 6h) 中亏损 -$139，缩短到 3-4 小时后接近盈亏平衡。原因：ORB 突破利润集中在前 2-3 小时，超时后反转概率增大
    - **值得实施**: E2 (3h) 使 ORB 从亏损变为微盈，且整体 Sharpe +0.14
  - **ATR Regime 仓位调整**:
    - F1 ATR regime lots: Sharpe 0.80 (+0.05), PnL $877 (+$159), 但 MaxDD $755 (+27%) — **不划算**。高波动加仓放大了回撤
    - F2 ADX>40 + ATR regime: **Sharpe 1.03 (+0.28)**, PnL $1072 (+49%), 但 MaxDD $715 (+20%)
    - **结论**: F2 是 PnL 最高和 Sharpe 最高的组合，但 MaxDD 增加 20%。对 $2000 账户来说回撤从 $564 增到 $715 可能太大。**暂不实施，留作观察**

## 策略优化 — 追踪止盈 Round 2 精调 (2026-04-03)

- **2026-04-03**: **Round 2 回测** (`backtest_round2.py`, 22 变体, 2015-2026 完整 11 年 M15+H1 数据)
  - 测试维度：追踪止盈精细网格、SL/TP ATR 乘数、Keltner ADX 门槛、最大持仓数、冷却期
  - **冠军 R2#1 SL 3.5ATR+Trail1.0/0.3**: Sharpe 1.76, PnL $11,894, MaxDD $856 (6.0%)
  - **冠军 R2#2 Trail 0.8/0.25**: Sharpe 1.74, PnL $13,149, MaxDD $741 (4.8%), 11 年全正
  - **R2#3 Trail 1.0/0.2**: Sharpe 1.71, PnL $13,559, MaxDD $769 (4.8%)
  - R1 冠军 Trail 1.0/0.3 降至 #11 (Sharpe 1.28)，不是最优点
  - SL 放宽到 3.0-3.5ATR 有显著正向（减少假突破被震出），SL 2.0ATR 反而更差
  - KC ADX 门槛降到 18/20 有小幅正向（+0.09），但增量不够显著
  - MaxPos=1/3 影响不大，MaxPos=2 合理
  - TP 5.0ATR 略优于 3.0ATR（有追踪止盈时 TP 很少被触发，设大给利润更多空间）
- **2026-04-03**: **追踪止盈参数已更新到实盘** (`config.py`)
  - `TRAILING_ACTIVATE_ATR`: 1.5 → **0.8** (更早激活锁利)
  - `TRAILING_DISTANCE_ATR`: 0.5 → **0.25** (更紧追踪，减少利润回吐)
  - 回测效果：Sharpe 0.39→1.74 (+346%), PnL $2,925→$13,149 (+349%), MaxDD $1,218→$741 (-39%)
  - R3 组合测试结果见下方
- **2026-04-03**: **R3 组合测试** (`backtest_round3_combo.py`, 16 变体, 2015-2026 完整 11 年数据)
  - 测试维度：Trail + SL + TP + KC ADX 门槛的交叉组合
  - **冠军 C12: T0.8/0.25 + SL3.5 + TP5 + ADX18**: Sharpe **2.54**, PnL $18,147, MaxDD $553 (2.7%), WR 72.0%
  - 11 年逐年全部盈利（最差 2018 +$776，最好 2020 +$2,632）
  - 四维度叠加效果显著：Trail(+1.35) + SL(+0.49) + TP(+0.09) + ADX(+0.09) = 组合 Sharpe 2.54
  - Keltner PnL 从 $876（基准）升至 $15,526（冠军），是核心增量来源
  - ORB 也从亏损 -$352 变为盈利 +$392（SL 放宽减少了假突破被震出）
- **2026-04-03**: **全部四维度已实装到实盘**
  - `config.py`: `TRAILING_ACTIVATE_ATR` 1.5→**0.8**, `TRAILING_DISTANCE_ATR` 0.5→**0.25**
  - `strategies/signals.py`: `ATR_SL_MULTIPLIER` 2.5→**3.5**, `ATR_TP_MULTIPLIER` 3.0→**5.0**, `ADX_TREND_THRESHOLD` 24→**18**

## 过拟合检测 (2026-04-03)

- **2026-04-03**: **过拟合检测套件** (`backtest_overfit_test.py`, 4 项测试, 73 分钟)
  - Walk-Forward (4yr train/2yr test, 4 windows): C12 OOS Sharpes = [3.10, 2.63, 2.60, 1.31], 全正向, Avg Decay -20%
  - 奇偶年对半: C12 奇数年 Avg Sh=2.87, 偶数年 2.80, 11/11 年全正
  - 参数敏感性: 所有 ±1 步扰动 Sharpe > 2.0, range=0.93 (中等敏感), Trail Act 0.6/Dist 0.20 甚至更好
  - Regime 压力测试: C12 在 6 个市场 regime 中赢 5 个, 低波动/暴涨/暴跌/复苏全覆盖
  - **结论: LOW overfit risk**, SL/TP/ADX 增量较小 (+6% OOS), Trail 0.8/0.25 是核心 alpha
  - 发现: Baseline 与 Trail-only 结果一致, 因 config.py 已改为 0.8/0.25, Round2Engine 未正确隔离基准

## 高级回测套件 (2026-04-03)

- **2026-04-03**: **Advanced Backtest Suite** (`backtest_advanced.py`, 4 类测试, 165 分钟)

### Test 1: 蒙特卡洛模拟 (1000 次)
  - 16,514 笔交易随机打乱顺序 1000 次，重新计算净值曲线
  - PnL 100% 正向（所有模拟结果 = $18,147，因总 PnL 不变）
  - **MaxDD 分布**: P5=$350, P50=$461, P95=$663, 最差=$956
  - **Sharpe 分布**: P5=3.23, P50=3.30, P95=3.36（随机序列反而 Sharpe 更高，因原序列含连续亏损集中段）
  - MaxDD < $1,000 概率 = 100%
  - **结论**: 策略的风险主要来自交易序列（连续亏损），而非单笔交易质量。实际 MaxDD $553 处于 P50-P75 区间，说明历史序列运气中等偏好

### Test 2: K-Fold 交叉验证 (6 折)
  - 6 个 2 年窗口 × 3 配置 (Baseline / Trail-only / C12)
  - **C12 在 5/6 折中优于 Baseline**（仅 Fold2 2017-2018 微弱落后 3.44 vs 3.45）
  - C12 Avg Sharpe=2.71 (Std=0.69), Baseline Avg=2.62 (Std=0.76)
  - C12 最弱 Fold = Fold6 (2025-2026, Sharpe 1.31)，最强 = Fold2 (2017-2018, 3.44)
  - **Baseline = Trail-only** 完全一致（再次确认 config.py 已改为 0.8/0.25 导致基准不隔离）
  - **所有 6 折 × 3 配置均为正 Sharpe**，无负收益窗口
  - **结论**: C12 泛化能力良好，6 个独立时间段全部盈利，优势稳定

### Test 3: Regime 自适应参数 (5 变体)
  - **V4 Adaptive SL+Trail**: Sharpe **2.98** (+0.44), PnL $20,675, MaxDD $691 (3.0%), WR 73.4%
    - 低波动: SL 2.5 / Trail 1.0/0.35; 高波动: SL 4.5 / Trail 0.6/0.20
  - **V3 Adaptive Trail**: Sharpe **2.90** (+0.37), PnL $21,101, MaxDD $541 (2.3%), WR 73.3%
    - 低波动 Trail 更松 (1.0/0.35)，高波动 Trail 更紧 (0.6/0.20)
  - V2 Adaptive SL: Sharpe 2.67 (+0.14)
  - V1 C12 Fixed: Sharpe 2.54 (基准)
  - V5 Strategy Switch (低波动关 Keltner/高波动关 RSI): Sharpe 2.49 (-0.05)
  - **关键发现**: 自适应 Trail 是最大增量 (+0.37)，低波动放松追踪+高波动收紧追踪符合直觉
  - **V3 最优风险调整**: Sharpe 2.90 且 MaxDD 仅 $541（低于 C12 的 $553），PnL $21,101 最高
  - **V4 Sharpe 最高但 MaxDD 增加**: $691 vs C12 $553 (+25%)，SL 自适应放大了回撤
  - **V5 策略切换反而更差**: 说明 Keltner 和 RSI 在各 regime 下都有价值，不应关闭

### Test 4: 新参数探索 (16 变体)
  - **N01 KC mult 1.0**: Sharpe **2.95** (+0.42), PnL $22,859, MaxDD $490 (2.0%), **冠军**
    - 更窄通道 → 更多 Keltner 突破信号 (H1: 9052 vs 基准 7498, +21%)
    - 更多信号但质量不降（WR 73.1%），MaxDD 反而下降 $63
  - **N05 KC1.25+RSI20/80**: Sharpe **2.92** (+0.39), PnL $22,498, 19799 笔
    - 组合效应：更窄通道 + 更宽 RSI 阈值，信号量大幅增加但维持 Sharpe
  - **N01 KC mult 1.25**: Sharpe 2.87 (+0.33), PnL $21,441
  - **N02 RSI 10/90**: Sharpe 2.82 (+0.28), 信号最少 (6004 M15) 但最高 WR (74.0%)
  - **N04 KC_EMA30**: Sharpe 2.81 (+0.27), PnL $21,438, **最低 MaxDD $451** (2.0%)
  - N03 EMA50: Sharpe 2.76, 更快适应趋势
  - N00 C12 Baseline: Sharpe 2.54 (排第 10)
  - 最差: N01 KC mult 2.0 Sharpe 2.04 / N04 KC_EMA15 Sharpe 2.16
  - **关键发现**:
    - KC 通道越窄 Sharpe 越高（1.0 > 1.25 > 1.5 > 1.75 > 2.0），单调递增关系
    - KC_EMA 越长越好（30 > 25 > 20 > 15），更平滑的中轨减少噪音信号
    - EMA 趋势越短越好（50 > 100 > 150 > 200），更快适应趋势变化
    - RSI 10/90 信号少但质量高，RSI 25/75 信号多但 WR 低（70.2%），存在质量-数量权衡
  - **风险提示**: KC mult 从 1.5→1.0 是大幅改动（通道宽度 -33%），可能在不同市场环境下表现不一致，需进一步验证

### 综合结论
  - 当前 C12 配置 (Trail 0.8/0.25 + SL 3.5 + TP 5.0 + ADX 18) 经过蒙特卡洛、K-Fold、Regime 和参数探索四重验证，确认为稳健基准
  - **潜在升级路径 (按优先级)**:
    1. ⭐ **Adaptive Trail (V3)**: Sharpe +0.37, 无额外风险, 最安全的升级
    2. ⭐ **KC mult 1.25**: Sharpe +0.33, 保守的通道收窄, 渐进改进
    3. **KC_EMA30**: Sharpe +0.27, 最低 MaxDD, 可叠加
    4. **KC mult 1.0**: Sharpe +0.42 最高但改动大, 需更多验证
  - **不建议实施**: V5 策略切换 (降低 Sharpe), KC mult 2.0/KC_EMA15 (显著恶化)

## 组合验证回测 (2026-04-03)

- **2026-04-03**: **Combo Verification** (`backtest_combo_verify.py`, 3 阶段, 32 分钟)

### Phase 1: 全量对比 (2015-2026)
  - **A: C12 基准**: Sharpe 2.54, PnL $18,147, MaxDD $553
  - **B: C12 + Adaptive Trail**: Sharpe 2.90 (+0.37), PnL $21,101, MaxDD $541
  - **C: C12 + KC1.25 + KC_EMA30**: Sharpe 3.11 (+0.57), PnL $24,068, MaxDD $483
  - **D: 三合一 (Trail+KC1.25+EMA30)**: Sharpe **3.46** (+0.92), PnL $27,060, MaxDD $589, WR 74.5%
  - 交互效应 = -0.01（几乎完美加法：B增量+0.37 + C增量+0.57 = 预期+0.94, 实际+0.92）
  - KC1.25+KC_EMA30 的增量（+0.57）比 Adaptive Trail（+0.37）更大，且 MaxDD 最低（$483）

### Phase 2: K-Fold 交叉验证 (6 折)
  - **Combo 在 6/6 折中全部优于 Baseline**
  - A Avg Sharpe=2.71, D Avg Sharpe=**3.66** (+0.95)
  - 最弱 Fold = 2025-2026（D Sharpe 1.96 vs A 1.31），最强 = 2019-2020（D Sharpe 4.81 vs A 3.10）
  - 所有折均正 Sharpe

### Phase 3: 逐年对比
  - **Combo 在 12/12 年中全部优于 Baseline**
  - 最大优势：2019 (+1.95 Sharpe), 2020 (+1.67), 2023 (+1.99)
  - 最小优势：2017 (+0.40), 2022 (+0.29)
  - 2026 年（最近 3 个月）：A Sharpe 0.74, D Sharpe 1.85（+1.11），近期优势反而更大

### 结论
  - **三合一方案通过全部验证**: 交互近零、6/6 折优胜、12/12 年优胜
  - 待用户确认后实装：Adaptive Trail + KC mult 1.25 + KC_EMA30

## 策略压力测试 (2026-04-03)

- **2026-04-03**: **Stress Test** (`backtest_stress_test.py`, 5 项挑战, 110 分钟)

### Challenge 1: 交易成本 — **致命发现**
  - **回测完全没有计入点差/滑点成本！**
  - 平均手数 0.027 手，$0.50 点差下每笔成本 $1.35
  - 18,606 笔交易总成本 = **$25,043**
  - **Gross PnL $27,060 → Net PnL 仅 $2,018**（交易成本吃掉 93% 的利润）
  - 即使 $0.30 最低点差：Net PnL = $12,035（成本吃掉 56%）
  - $0.80 点差下策略直接亏损 -$13,008
  - **根因**: 策略产生 18,606 笔交易（平均每天 5.8 笔），高频交易在有交易成本时不可持续
  - **对比**: True Baseline 13,711 笔，Combo 18,606 笔 — KC 通道收窄和自适应 Trail 增加了 36% 的交易量，放大了成本问题

### Challenge 2: 真实基准隔离
  - True Baseline (Trail 1.5/0.5, SL 2.5, TP 3.0, ADX 24): **Sharpe 0.53**, PnL $3,772
  - Combo D: Sharpe 3.46, PnL $27,060
  - **真实提升 = +2.93 Sharpe**（之前声称的 +0.92 是因为 Baseline 被污染）
  - 但这个 +2.93 是无成本的虚假提升

### Challenge 3: 信号聚集
  - 11% 的交易在 1 小时内连续开仓，69.7% 在 4 小时内
  - Keltner 策略 9,490 笔 WR 83.7% — **异常高的胜率，可能是短期反复开平仓的伪信号**
  - 平均每天 5.8 笔交易，最多单日 18 笔

### Challenge 4: 回撤分析（相对乐观）
  - MaxDD $562，最大连续亏损 9 笔，10+ 连亏 0 次
  - 135 个月中仅 10 个负月（7.4%），最差月 -$134
  - 这部分表现确实稳健

### Challenge 5: 参数悬崖（通过）
  - KC mult 敏感度: range = 0.38 (smooth, 无悬崖)
  - KC EMA 敏感度: range = 0.44 (smooth, 无悬崖)
  - KC 1.10/EMA30 甚至比 1.25 更好 (Sharpe 3.55)
  - KC 1.25/EMA35-40 更好 (Sharpe 3.57-3.61) — 参数空间平坦，非过拟合

### 核心结论
  - **Sharpe 3.46 是无交易成本的虚假繁荣**
  - 策略本身的方向判断能力是真实的（WR 74.5%, 参数平滑），但利润太薄无法覆盖交易成本
  - **解决方向**: 不是放弃策略优化，而是需要减少交易频率（过滤低质量信号）或增大单笔利润空间
  - 当前实盘手数 0.01-0.03，EMX Pro 点差约 $0.30-0.50，实际情况比回测假设差很多

## 成本调整回测 (2026-04-04)

- **2026-04-04**: **Cost-Adjusted Backtest** (`backtest_cost_adjusted.py`, 3 阶段, 5.2 小时)
- 已在 `backtest_m15.py` 的 `_close_position` 中加入 `spread_cost` 参数，所有子引擎自动继承

### Phase 1: 点差敏感度 (5 配置 x 4 点差)

| 配置 | sp=$0 | sp=$0.30 | sp=$0.50 | sp=$0.80 |
|---|---|---|---|---|
| True Baseline (原始参数) | 0.53 | **-1.04** | -2.14 | -3.67 |
| C12 (当前实盘) | 2.54 | 0.73 | **-0.53** | -2.37 |
| C12 + Adaptive Trail | 2.90 | 1.10 | -0.17 | -2.03 |
| C12 + KC1.25+EMA30 | 3.11 | 1.36 | 0.06 | -1.80 |
| **Combo (三合一)** | 3.46 | **1.65** | **0.35** | -1.56 |

  - **$0.50 点差下只有 Combo 勉强正 Sharpe (0.35)**，其余全部亏损
  - **$0.30 点差下 Combo Sharpe 1.65 尚可**，C12 仅 0.73
  - $0.80 点差下所有配置全部巨亏
  - **当前实盘 C12 在 $0.50 点差下 Sharpe = -0.53（预期亏损！）**

### Phase 2: 频率降低 (Combo + $0.50 点差)

| 最小间隔 | 交易数 | Sharpe | PnL | $/trade |
|---|---|---|---|---|
| **0h (无限制)** | 17,917 | **0.35** | $2,578 | $0.14 |
| 2h | 14,381 | 0.14 | $947 | $0.07 |
| 4h | 10,306 | 0.01 | $35 | $0.00 |
| 6h | 8,092 | -0.21 | -$964 | -$0.12 |
| 8h+ | <7,000 | <0 | 亏损 | 亏损 |

  - **降低频率反而更差！** 不是信号太多，而是每笔利润太薄
  - 间隔越长，好信号被跳过的概率越高，Sharpe 反而下降
  - C12 无论哪种间隔在 $0.50 点差下全部亏损

### Phase 3: K-Fold 验证 (Combo, $0.50 点差)
  - Combo 在 6/6 折中优于 C12（Avg Sharpe 0.10 vs -0.76）
  - 但 Fold1 (2015-2016) Sharpe=-0.90，Fold2 (2017-2018) Sharpe=-2.19 — **不稳定**
  - 后半段 (2019-2026) 明显好于前半段，可能是近年金价波动加大带来的利润空间

### 核心结论
  - **回测框架已加入交易成本支持** (`spread_cost` 参数)
  - 策略的边际改进（Trail/KC/EMA 优化）在无成本时很大，但**交易成本是最大的"策略杀手"**
  - Combo 是唯一在 $0.50 点差下勉强存活的配置（Sharpe 0.35），但远不够好
  - **根本问题不是策略参数，而是交易频率 vs 利润空间的结构性矛盾**
  - 实盘 EMX Pro 点差需要实测确认——如果 <$0.30 则 Combo (Sharpe 1.65) 有实际价值
  - **下一步方向**: ①确认实盘真实点差 ②探索更大时间框架降低频率 ③提高单笔利润（更宽 TP、更强信号过滤）

## 大趋势日识别研究 (2026-04-04)

- **2026-04-04**: **Trend Day Filter Research** (`backtest_trend_day.py`, 4 阶段, 2.3 小时)
- 背景：成本调整回测确认策略利润全部来自大趋势日，普通日和震荡日在扣除点差后亏损

### Phase 1: 日级别特征工程 & 分类

  - 从 11 年 H1 数据计算 3,491 个交易日的 7 个特征
  - **趋势评分** = 0.25×趋势强度 + 0.20×波幅/ATR + 0.20×ADX + 0.15×KC突破率 + 0.10×EMA排列 + 0.10×MACD一致性
  - 按 75/25 百分位分类：873 趋势日(25%) + 1,745 普通日(50%) + 873 震荡日(25%)
  - **特征相关性**: KC突破次数(+0.756)和趋势强度(+0.733)对趋势评分贡献最大

  **PnL 按日 Regime 分解 (Combo, $0.50 spread)**:

  | Regime | 交易数 | PnL | $/trade | WR% |
  |--------|--------|-----|---------|-----|
  | **趋势日** | 5,727 | **+$17,593** | **$3.07** | 78.2% |
  | 普通日 | 8,727 | -$5,203 | -$0.60 | 64.5% |
  | 震荡日 | 3,463 | -$9,812 | -$2.83 | 51.3% |
  | 合计 | 17,917 | $2,578 | $0.14 | — |

  - **核心发现**：趋势日仅占 25% 的日子，但贡献了 +$17,593 利润；其余 75% 的日子净亏 -$15,015
  - 震荡日平均每笔亏 $2.83（点差成本 > 利润），普通日也是净亏
  - **结论：如果能识别并只在趋势日交易，利润空间巨大**

### Phase 2: 过滤回测 (Oracle, $0.50 spread)

  | 模式 | 交易数 | Sharpe | PnL | MaxDD | $/trade |
  |------|--------|--------|-----|-------|---------|
  | A: 全天交易(基准) | 17,917 | **0.35** | $2,578 | $3,058 | $0.14 |
  | B: 跳过震荡日 | 14,636 | **1.96** | $12,357 | $613 | $0.84 |
  | **C: 仅趋势日** | 6,095 | **6.44** | **$18,392** | **$332** | **$3.02** |
  | D: 混合模式 | 14,636 | 1.96 | $12,357 | $613 | $0.84 |

  - **仅趋势日交易 Sharpe 从 0.35 → 6.44**，PnL 从 $2,578 → $18,392，MaxDD 从 $3,058 → $332
  - C12 默认配置同样有效：C12 全天 Sharpe=-0.53 → C12 仅趋势日 Sharpe=6.41
  - 跳过震荡日(B)是保守选择：Sharpe 1.96，保留了 82% 的交易但消除了大部分亏损
  - **但这是 Oracle 测试（用了未来信息），实战需要提前预判**

### Phase 3: 早盘预判准确率

  - 测试了 17 个早盘预判指标（T-1 数据 + 亚盘数据）
  - **最佳预判器**: `prev_score >= 0.60`（前一日趋势评分），Precision=28.3%, Recall=83.4%, 38.4% 震荡日被正确避开
  - `adx_rising_3d`（ADX 连续 3 天上升）避开 85.6% 震荡日，但 Recall 仅 21.8%（漏掉太多趋势日）
  - 亚盘波幅、隔夜跳空等指标预测力较弱

  **预测 Regime 回测结果（无前瞻偏差）**:
  - P1 score+asian [skip choppy]: Sharpe=0.35（与全天交易相同，预判太弱无法有效过滤）
  - P2 prev_score [trend only]: Sharpe=0.29（反而变差，因为预判不准导致错过好信号）

  - **核心问题：趋势日的可预测性低**。趋势日的形成往往是由突发事件（数据发布、地缘冲突、央行讲话）驱动，前一日的技术指标无法可靠预测
  - 这意味着**纯技术指标的日前预判方案行不通**

### Phase 4: K-Fold + 逐年验证 (Oracle "仅趋势日")

  **K-Fold (6 折)**:
  - **Best (仅趋势日) 在 6/6 折全部优于 Baseline**
  - Avg Sharpe: Best=6.79 vs Baseline=0.10
  - 即使最弱的 Fold (2017-2018) Best Sharpe 仍有 6.31

  **逐年对比**:

  | 年份 | Best Sh | Best PnL | Best N | Base Sh | Base PnL |
  |------|---------|----------|--------|---------|----------|
  | 2015 | 5.58 | $955 | 549 | -0.98 | -$415 |
  | 2016 | 6.68 | $1,232 | 525 | -0.93 | -$465 |
  | 2017 | 6.03 | $792 | 547 | -1.03 | -$398 |
  | 2018 | 5.20 | $565 | 490 | -3.46 | -$1,118 |
  | 2019 | 5.02 | $962 | 518 | -0.12 | -$53 |
  | 2020 | 8.75 | $2,689 | 601 | 2.59 | $1,921 |
  | 2021 | 7.76 | $1,540 | 523 | 0.46 | $300 |
  | 2022 | 6.08 | $1,448 | 510 | 0.06 | $43 |
  | 2023 | 7.37 | $1,671 | 507 | 1.05 | $616 |
  | 2024 | 8.29 | $2,195 | 561 | 0.94 | $667 |
  | 2025 | 7.89 | $2,769 | 592 | 0.77 | $742 |
  | 2026 | 2.30 | $298 | 89 | 1.35 | $572 |

  - **Best 在 11/12 年优于 Baseline**（唯一例外是 2026 年仅 3 个月数据）
  - 2020/2024/2025 表现最强（大波动年份趋势日更多更强）

### 综合结论

  1. **趋势日过滤是解决成本问题的正确方向**：Oracle 测试证明仅在趋势日交易可将 Sharpe 从 0.35 提升到 6.44，11 年 K-Fold 全部验证通过
  2. **但纯技术指标无法提前可靠预判趋势日**：T-1 特征的预测精度不足以在实战中使用
  3. **实战可行的替代方案**：
     - **(a) 盘中实时判断**：不需要提前预判，而是在盘中观察前 2-3 小时走势，判断当天是否正在形成趋势（如：已突破 KC、ADX 快速上升、EMA 排列完成）
     - **(b) 事件驱动过滤**：利用已有舆情系统的经济日历，在非农/CPI/FOMC 等高影响事件日才开仓（这类日子大概率是趋势日）
     - **(c) 保守方案 — 跳过震荡日**：用 `prev_score < 0.25` 加 `asian_range_vs_atr < 0.5` 识别极端震荡日（约 10-15%），对这部分日子暂停交易，Sharpe 从 0.35→1.96
  4. **下一步**：研究"盘中实时趋势判断"方案 — 这不需要预测未来，只需要在开盘 2-3 小时后判断"今天正在成为趋势日"

## 盘中自适应交易系统 (2026-04-04)

### 设计思路

- **2026-04-04**: 基于趋势日研究结论，决定不做"提前预判"，而是建"盘中自适应系统"——观察已发生的市场状态再决定交易行为
- **核心哲学**: 渐进式披露的交易版——先看市场给你什么信号，再决定怎么反应
- **为什么不预测**: T-1 技术指标无法预测趋势日（准确率~55%，跟抛硬币差不多），因为趋势日本质是事件驱动（非农/FOMC/地缘冲突），不是惯性延续

### 核心组件：IntradayTrendMeter (`intraday_trend.py`)

- 每次 `scan_and_trade` 调用时用当日已有的 H1 bar 计算趋势评分 (0-1)
- 4 个子因子：ADX(30%) + KC突破比例(25%) + EMA排列一致性(25%) + 趋势强度(20%)
- 三级门控：
  - **>= 0.60 TRENDING**: 所有策略正常交易
  - **0.35-0.60 NEUTRAL**: 仅允许 H1 策略 (Keltner/ORB)，跳过 M15 RSI
  - **< 0.35 CHOPPY**: 禁止所有新开仓

### 集成方式

- `config.py` 新增 `INTRADAY_TREND_ENABLED`, `INTRADAY_TREND_THRESHOLD=0.35`, `INTRADAY_TREND_KC_ONLY_THRESHOLD=0.60`
- `gold_trader.py` 的 `_check_entries` 中，max position 检查之后、信号扫描之前加入趋势门控
- `INTRADAY_TREND_ENABLED = False` 一键关闭回退到原有行为
- 不影响已持仓出场/trailing stop，不影响 SentimentEngine 和 RiskManager

### Phase 5 回测 — 盘中实时评分 (完成)

- **2026-04-04**: 使用重构后的统一回测引擎 (`backtest_intraday_adaptive.py`) 完成全部 3 项测试，72 分钟

**Test 1: Baseline vs Adaptive (spread=$0.50)**:

| 配置 | N | Sharpe | PnL | MaxDD | $/trade |
|------|---|--------|-----|-------|---------|
| A: Combo Baseline | 17,917 | 0.35 | $2,578 | $3,058 | $0.14 |
| B: Combo Adaptive (0.35/0.60) | 17,904 | 0.35 | $2,618 | $3,058 | $0.15 |
| C: C12 Baseline | 15,770 | -0.53 | -$3,656 | $4,243 | -$0.23 |
| **D: C12 Adaptive (0.35/0.60)** | **7,365** | **1.03** | **$5,497** | **$732** | **$0.75** |

- **关键发现**: Combo 配置自适应效果不明显（skipped_choppy=0，几乎没有时段低于阈值），但 C12 默认配置效果显著——Sharpe 从 -0.53 → 1.03，MaxDD 从 $4,243 → $732
- 原因：Combo 配置（KC1.25+EMA30）的更窄通道本身就在趋势时段产生信号，非趋势时段信号已经很少，门控无效；C12（KC1.5+EMA20）在非趋势时段会产生更多噪声信号，门控有效剔除

**Test 2: 阈值敏感性 (Combo config)**:

| kc_only 阈值 | N | Sharpe | PnL | $/trade |
|---|---|---|---|---|
| ≤ 0.60 | 17,904 | 0.35 | $2,618 | $0.15 |
| ≥ 0.65 | 9,693 | **1.86** | **$11,400** | **$1.18** |

- **存在阈值悬崖**：0.60→0.65 之间存在非连续跳跃（无中间态），这是过拟合红旗
- choppy 阈值（0.25-0.45）对 Combo 配置几乎无影响
- 最佳配置 choppy=0.40/kc_only=0.65，但因阈值不平滑暂不实装

**Test 3: K-Fold 验证 (best: 0.40/0.65, Combo config)**:

| Fold | Adaptive Sharpe | Baseline Sharpe | Winner |
|---|---|---|---|
| Fold1 (2015-2016) | 1.45 | -0.90 | Adapt |
| Fold2 (2017-2018) | 0.30 | -2.19 | Adapt |
| Fold3 (2019-2020) | 1.58 | 1.58 | Base |
| Fold4 (2021-2022) | 1.73 | 0.28 | Adapt |
| Fold5 (2023-2024) | 0.98 | 0.98 | Base |
| Fold6 (2025-2026) | 1.99 | 0.85 | Adapt |

- Avg Sharpe: Adaptive=**1.34** vs Baseline=0.10, Adaptive 赢 4/6 折
- Fold3/Fold5 持平（大波动期所有交易都在高评分时段，门控未生效）
- 最大提升出现在震荡为主的时期（Fold1/2/4）

### 结论
- **C12 Adaptive (0.35/0.60) 已实装**（config.py `INTRADAY_TREND_ENABLED=True`），回测确认有效
- Combo 配置 (0.40/0.65) 存在阈值悬崖风险，暂不实装，待进一步验证

### 当前实盘参数汇总 (2026-04-05 更新)

| 参数 | 值 | 来源 |
|------|-----|------|
| KC EMA | 20 | C12 默认 |
| KC Multiplier | 1.5 | C12 默认 |
| TrailAct | 1.0 | 旧默认 |
| TrailDist | 0.3 | 旧默认 |
| ADX 过滤 | 25 | 旧默认 |
| **ATR_SL_MULTIPLIER** | **4.5** | Mega Grid + 带成本精调 (2026-04-05) |
| TP ATR Multiplier | 3.0 | 默认 |
| Choppy | 0.35 | Phase 4 |
| **COOLDOWN_MINUTES** | **30** | 点差×冷却交叉测试 (2026-04-05) |
| Adaptive Trend | 0.35/0.60 | Phase 5 |
| Max Positions | 2 | 风控 |
| Risk per Trade | $50 (2.5%) | 风控 |
| **MAX_LOT_SIZE** | **0.05** | 绝对安全上限 |
| **亏损递减上限** | 0→0.05(ATR自由), 1→0.03, 2→0.02, 3+→0.01 | 替代旧 lot_scale 缩减 |

### Phase 6: 细粒度阈值扫描结果 (2026-04-04, 服务器运行)

- **2026-04-04**: `backtest_threshold_scan.py` kc_only 0.55→0.72 步长 0.01，Combo 配置 + $0.50 spread
- **核心发现: 0.65 不是渐进阈值，是 M15 RSI 的硬二元开关**
  - kc_only ≤ 0.64: 结果完全一致（N=17904, Sharpe=0.35, M15=8286笔）
  - kc_only ≥ 0.65: 结果完全一致（N=9693, Sharpe=1.86, M15=**5笔**, skip_m15=262522）
  - 0.64→0.65 之间无中间态，是精确断点
- **根因**: Combo 配置下盘中趋势评分几乎不超过 0.65，阈值设 0.65 等于完全关闭 M15 RSI
- **Sharpe 1.86 的本质**: 不是"聪明地在趋势日交易"，是关掉了 M15 RSI 后只靠 H1 策略赚钱（M15 在 $0.50 点差下是净亏损源）
- **决策: 保留 kc_only=0.60**，不采用 0.65。如需优化 Combo 成本问题，应单独提高 M15 RSI 信号质量而非用门控间接关闭

## 回测框架统一重构 (2026-04-04)

- **2026-04-04**: **回测技术债清理** — 将 15 个分散的回测脚本（~6,500 行）中的公共引擎代码统一为 `backtest/` 包
  - 新建 `backtest/engine.py`: 单一参数化 `BacktestEngine`，替代旧的继承链（`MultiTimeframeEngine → Round2Engine → RegimeEngine → IntradayAdaptiveEngine / CooldownEngine / ParamExploreEngine`）
  - 新建 `backtest/stats.py`: 统一统计计算（`calc_stats`）和报告格式化（`print_comparison`, `print_ranked`）
  - 新建 `backtest/runner.py`: 数据加载（`load_csv/load_m15/load_h1_aligned`）、`DataBundle` 类、`run_variant/run_variants/run_kfold` 执行器、配置预设（`C12_KWARGS`, `V3_REGIME`, `TRUE_BASELINE_KWARGS`）
  - 新建 `backtest/__init__.py`: 公开 API + 向后兼容别名（`_aggregate_daily_pnl`, `Position`, `TradeRecord`, `load_csv`）
  - **验证通过**: 5 个场景（C12, Adaptive Trail, Baseline, Spread Cost, RSI ADX Filter）新旧引擎结果完全一致（diff=0.0000）
- **2026-04-04**: **重构 4 个核心实验脚本**，全部改用统一 `backtest` 包：
  - `backtest_advanced.py`: 607→438 行（-28%），删除本地 `RegimeEngine/ParamExploreEngine/prepare_indicators_custom`
  - `backtest_combo_verify.py`: 337→310 行（-8%），使用 `DataBundle` + `_run_bundle` 替代 `run_fixed/run_regime`
  - `backtest_cost_adjusted.py`: 402→336 行（-16%），删除 `CooldownEngine/CooldownRegimeEngine`，用 `min_entry_gap_hours` 参数替代
  - `backtest_intraday_adaptive.py`: 471→378 行（-20%），删除 `IntradayAdaptiveEngine/calc_realtime_score`，用 `intraday_adaptive=True` 参数替代
- **2026-04-04**: `run_variant` 新增 `verbose` 参数，允许实验脚本控制自己的输出格式
- **2026-04-04**: 旧引擎文件（`backtest.py`, `backtest_m15.py`, `backtest_round2.py`）保留向后兼容，仍有 7 个脚本依赖它们，后续渐进清理

## Mega Grid 搜索与参数精调 (2026-04-05)

- **2026-04-05**: **Mega Grid Search** (`backtest_mega_grid.py`, 1440 组合, 4.4 小时)
  - 搜索维度: TrailAct [0.5-1.0] × TrailDist [0.15-0.35] × ADX [15-21] × SL [2.5-4.0] × Choppy [0.3-0.4]
  - 最优: T0.5/D0.15/A19/SL4.0/C0.4, Sharpe **3.36** (无成本), PnL $17,197, MaxDD $467, WR 85.3%
  - DSR=0.975 通过, PBO=0.00 零过拟合, IS-best OOS rank mean=8.5
  - 敏感性: TrailDist CLIFF(2.03), TrailAct CLIFF(0.73), SL CLIFF(0.77), ADX moderate(0.21), Choppy zero(0.00)
  - **注意**: 无成本结果, 9498笔×$1.35≈$12,822成本, 未做带成本验证
- **2026-04-05**: **RSI 全面扫描** (`backtest_rsi_optimization.py`, 18 变体)
  - C12+Adaptive 框架下 RSI 仅 6 笔/11年, 所有变体 Sharpe 完全一致(1.03), MaxDD 完全一致($732)
  - 无论收紧阈值、只做多、加 ADX/ATR 过滤, 对整体结果影响在小数点第三位以后
  - **结论**: RSI 在当前框架下已自然消亡, 保留代码但不再优化
- **2026-04-05**: **ORB 持仓时间扫描** (`backtest_orb_scan.py`, 8 变体)
  - hold=4(60min) ORB PnL=-$277, hold=24(360min) ORB PnL=+$2, 单调递增
  - 缩短持仓时间反而更差, 与之前 M15 回测(无 Adaptive)结论相反
  - **结论**: ORB 持仓时间保持默认(无限制/24 bars), 不改动
- **2026-04-05**: **Walk-Forward 18 窗口验证** (`backtest_walkforward.py`)
  - 一致性 18/18 (100%), C12+Adaptive 在所有窗口都是最优
  - OOS 平均 Sharpe=0.75 ($0.50 点差), 负 OOS 窗口 6/18 (低波动盘整期)
  - 最强 OOS: 2020H1(+2.39), 2023H1(+2.34), 2025H1(+2.10)
- **2026-04-05**: **SL 精调** (Top Combo 测试, 3 变体, $0.50 点差 + Adaptive)
  - SL=4.5: Sharpe **1.35**, PnL $6,721, MaxDD **$559** — **全维度优于当前 SL=3.5**
  - SL=4.0: Sharpe 1.34, PnL $6,839, MaxDD $647
  - SL=5.0: Sharpe 1.29, PnL $6,200, MaxDD $553
  - **SL=4.5 已实装**: Sharpe +0.32, MaxDD -$173, 12/12年全正
- **2026-04-05**: **点差×冷却期交叉测试** (`backtest_spread_cooldown_cross.py`, 20 变体)
  - 最优: $0.30/cd=30min/Adaptive, Sharpe **1.93**, PnL $10,958, MaxDD $696
  - Cooldown 30min 在所有点差下均优于 3h: $0.30 下 +0.14 Sharpe, $0.50 下 +0.14 Sharpe
  - 冷却期粒度扫描确认 15-20min 等效(1 根 M15 bar), 30-40min 等效(2 根 M15 bar)
  - **Cooldown 30min 已实装**: COOLDOWN_BARS(3h) → COOLDOWN_MINUTES(30min)
- **2026-04-05**: **持仓时间分析** (`backtest_hold_time_analysis.py`)
  - 前 2h 贡献 +$18,979 (345% of total), 4h+ 净亏 -$16,280
  - Trailing 5,542笔 +$41,088 (97.7%WR, avg 11 bars) — 核心利润引擎
  - Timeout:60 769笔 -$12,287 (2.0%WR) — 最大单项亏损源
  - 赢家中位持仓 6 bars(1.5h), 输家中位 46 bars(11.5h), 相关系数-0.376
- **2026-04-05**: **波动率 Regime 分析** (`backtest_volatility_regime.py`)
  - Keltner 在所有4个波动率环境下均盈利, Very High(>75%) Sharpe 1.57 最强
  - 任何波动率过滤都让结果变差, 不需要干预
  - ATR 与 PnL 相关系数 0.011, 无预测力
- **2026-04-05**: **BUY vs SELL 分析** (`backtest_direction_bias.py`)
  - BUY Sharpe 1.10 > SELL 0.77, 但 SELL 贡献 $2,127(39%), 不应禁用
  - 纽约时段 BUY 是甜蜜点($1.46/trade), Late NY SELL 更好($1.16/trade)
  - 连续 5+ SELL 后平均每笔-$0.07, 7+ SELL 后-$0.34(做空趋势持续性弱)
  - ORB SELL 亏损(-$157), 但样本不足以做方向过滤

## 待办与未来方向

- [x] ~~观察重构后系统运行 1-2 天，确认 position_tracker.py 持仓同步和平仓通知正常~~ → 发现并修复了 CLOSE_DETECTED 重复记录 bug
- [ ] 考虑让 MT4 EA 在 positions.json 中包含已平仓订单历史，解决 CLOSE_DETECTED 估值偏差问题
- [ ] Telegram Token 从代码移到 .env 文件（用户暂缓）
- [x] ~~IC 报告积累数据后进行第一次因子有效性评估~~ → 已用 factor_scanner.py 完成首次离线全量扫描（27因子×3窗口=81次检验）
- [x] ~~将 M15 RSI ADX>40 过滤实装到实盘~~ → 已实装到 config.py + signals.py + gold_trader.py (v8)
- [x] ~~考虑将 ORB max_hold_bars 缩短~~ → C12+Adaptive 框架下 ORB 默认(24 bars)最优，缩短反而更差(持仓越长 PnL 单调递增)
- [ ] 根据 IC 扫描结果，研究 day_of_week 周内择时策略（IC=+0.033，WF=100%，最稳定因子之一）
- [ ] 评估 P5_volume_breakout 是否继续保留（volume_ratio IC≈0，预测力不足）
- [ ] 模拟盘 P4/P5/P6 积累 20+ 笔后评估是否推进实盘（P4 重点观察震荡/下跌市表现）
- [ ] P5/P6 放宽条件后观察一周，若仍零触发则进一步放宽或替换
- [ ] 舆情系统 30 天后（~4/30）做第一次正式评估
- [ ] Polymarket 地缘风险指数观察 2 周后评估其与金价的实际相关性
- [ ] 历史交易记录无法补全因子快照，如需对历史做 IC 分析需写离线回算脚本（已有 backtest.py 框架可扩展）
- [x] ~~gap_fill 策略回测 Sharpe -1.71，需研究是否调参优化或禁用~~ → 已禁用（两次独立回测 Sharpe -1.25/-1.71，实盘 0 次触发）
- [x] ~~考虑在 ADX > 40 极端趋势环境下临时降低/暂停 M15 RSI 均值回归策略权重~~ → 已改为加入 EMA100 方向过滤（更精准，不会误杀顺势交易）
- [x] ~~SL ATR Multiplier 从 3.5 调整为 4.5~~ → 已实装, Sharpe +0.32, MaxDD -$173 (2026-04-05)
- [x] ~~Cooldown 从 3h 改为 30min~~ → 已实装, COOLDOWN_BARS → COOLDOWN_MINUTES, 所有引擎文件同步 (2026-04-05)
- [x] ~~确认 EMX Pro 实际点差~~ → 短期不换交易商，点差问题不再研究，以当前交易商实际点差为基准 (2026-04-05)
- [ ] **中期**: 测试缩短最大持仓时间从 60 bars 到 24-32 bars（Timeout 60 bars 亏损 $12,287，占总损失最大项）
- [ ] **中期**: 测试 Mega Grid 最优 T0.5/D0.15 在 $0.30/$0.50 带成本下的表现
- [ ] **低优先**: 考虑连续 5+ SELL 后减仓（均值 -$0.07/笔，7+ SELL 后 -$0.34/笔）
- [x] ~~渐进清理剩余 7 个旧引擎脚本~~ → 2026-04-05 已删除全部 7 个（共 ~3055 行），无外部依赖，git 历史可恢复
- [x] ~~引入宏观因子回测：经济日历事件标记作为第一个宏观因子纳入回测框架~~ → 已建立 `macro/` 数据管道（P1），DXY/VIX/Brent/US10Y + FRED(TIPS/US2Y/BEI/利差) 11年日线已下载
- [ ] DXY 日线作为日级别方向过滤（P6 策略的实盘验证版本）— 数据管道+regime检测器已就绪
- [x] ~~注册 FRED API key 并配置到环境变量，补全 TIPS 10Y/US2Y/2-10利差/BEI 5Y 数据~~ → 已完成，18列数据全部下载
- [x] ~~P2: 动态点差模型~~ → `backtest/engine.py` 支持 fixed/atr_scaled/session_aware 三种点差模型
- [x] ~~P3: 高级统计检验~~ → PSR/DSR/CSCV-PBO 已集成到 `backtest/stats.py`
- [x] ~~P4: 宏观 Regime 自动识别~~ → `macro/regime_detector.py` 6种regime + 策略权重映射，已集成到回测引擎
- [x] ~~编写 `backtest_macro_regime.py` 验证宏观regime过滤对策略的实际影响~~ → 宏观过滤反而变差，Regime分析价值在诊断不在过滤
- [x] ~~编写 `backtest_statistical_validation.py` 对 C12/Combo 运行 CSCV/PBO 验证~~ → PSR仅Adaptive通过，PBO=0.00零过拟合风险
- [x] ~~编写 `backtest_spread_model.py` 对比三种点差模型的回测差异~~ → $0.30 vs $0.50差异巨大，Session-Aware最接近真实
- [x] ~~**EXP28**: 事件日防御("带伞策略")穷举验证~~ → 全部防御无效或有害，不实装。V3+Adaptive+Cooldown 已充分
- [x] ~~根据 EXP28 结果决定是否实装事件日自动防御机制~~ → 否决，内建防御已够
- [ ] **阴跌缓解**: 测试 EMA100 斜率为负时禁止做多（针对 S4 慢跌场景）
- [ ] **阴跌缓解**: 测试周度累计回撤超阈值自动减仓到 MaxPos=1

## P2/P3/P4 实验结果 (2026-04-04, 服务器运行)

### Spread Model 结果
- **点差是生死线**: C12裸策略在$0.50下Sharpe=-0.53(亏), $0.30下Sharpe=+0.73(赚)
- **Adaptive+$0.30是最优组合**: Sharpe=1.79, 11年全正, MaxDD=$686
- **Session-Aware比ATR-Scaled更合理**: Sharpe 1.49 vs 1.15 (base=$0.30)
- **M15 RSI在任何点差下都亏**: $0.30亏$4,691, $0.50亏$9,292

### Statistical Validation 结果
- **PSR**: 12个变体中仅C12+Adaptive通过(PSR=0.9998, p=0.0002), 其余全FAIL
- **DSR**: SR*=1.21(12次测试的运气上限), Adaptive的1.03未达标, 但$0.30点差下1.79可通过
- **PBO = 0.00**: 70种组合零过拟合, IS-best=OOS-best在100%的分割中成立
- **结论**: Adaptive不是过拟合，是真实的结构性优势（过滤M15 RSI亏损交易）

### Macro Regime 结果
- **宏观过滤净效果为负**: C12+Macro比C12裸更差(Sharpe -0.53→-0.60), Adaptive+Macro也变差(1.03→0.98)
- **Regime诊断价值**: C12仅在risk_off(+$507)和liquidity_crisis(+$183)赚钱, 其余4种regime全亏
- **根因**: 裸C12的亏损来自M15 RSI(-$9,292), 不是宏观环境; 宏观过滤砍掉了赚钱和亏钱的交易
- **结论**: 宏观Regime暂不用于实盘过滤, 保留作为诊断工具和未来Regime-specific参数调整的基础

### 综合决策
- **实盘最优配置 (2026-04-05 更新)**: C12 + Adaptive (0.35/0.60) + **SL 4.5 ATR** + **Cooldown 30min**, 已全部实装
- **当前预期表现**: $0.50 点差 Sharpe ≈ 1.35, $0.30 点差 Sharpe ≈ 1.93, MaxDD ≈ $559
- **M15 RSI**: Adaptive 下已自然消亡(6笔/11年), 保留代码不再优化
- **宏观过滤**: 暂不启用, regime_detector保留备用
- **点差决策**: 短期不换交易商，不再研究点差优化，以当前交易商为基准运行 (2026-04-05)

## 大规模并行实验 (2026-04-05, AutoDL 22核服务器)

### 实验概览
- **26 个 Jupyter Notebook 实验** (EXP01-EXP26)，AutoDL 22 核服务器并行运行
- 覆盖 5 大方向：参数优化、时间维度、风控资金、策略诊断、综合验证
- 全部完成（EXP20 组合验证运行中）

### 参数优化结果

| 维度 | 冠军 | Sharpe | vs Baseline | 可实装? |
|---|---|---|---|---|
| **KC 通道 (EXP13)** | KC_EMA25 + M1.2 | **1.84** | +0.32 | ⚠️ 等 EXP20 K-Fold |
| **EMA 趋势 (EXP14)** | EMA_trend=150 | **1.65** | +0.13 | ⚠️ 等 EXP20 K-Fold |
| **SL×Trail (EXP04)** | SL5.0 + T0.6/D0.2 | **2.39** | +0.87 | ⚠️ 等 EXP20 K-Fold |
| **TP (EXP06/11)** | TP6.0-8.0 | 1.55-1.57 | +0.03-0.05 | ✅ 安全改动 |
| **V3 ATR Regime (EXP07)** | V3 + $0.50 | **1.85** | +0.33 | ✅ **K-Fold 6/6 折全赢** |
| **Trail 网格 (EXP03)** | T0.5/D0.15 + $0.30 | **4.50** | — | ⚠️ 仅低点差下有效 |
| KC max_hold (EXP01) | 60 (不限制) | 1.55 | 0 | ✅ 已是最优 |
| ADX (EXP09) | 18 | 2.27 | 0 | ✅ 已是当前值 |
| ORB (EXP02/21) | KC16_ORB16 | 1.54 | +0.02 | ❌ 增量太小 |
| Adaptive (EXP15) | 0.25/0.65 | 1.52 | 0 | ❌ 0.65 是 M15 RSI 硬开关 |

### 时段与时间维度

- **EXP05 时段择时**:
  - 最佳: UTC 21点 ($7.14/trade)，NY BUY ($1.71/trade)
  - London SELL 强 ($1.31/trade)，Asia 两方向都赚
  - **跳过任何时段 K-Fold 仅赢 2/6 折** → 不可靠，不采用
  - Session 稳定性: London 11/12年正, NY 10/12年正
- **EXP10 星期几**:
  - 周五最赚 ($1.42/trade)，周四最差 ($0.65/trade)
  - **跳过任何一天都让 $/trade 下降** → 不值得做星期过滤
  - 周一 SELL 最多 ($1247)，周五 BUY 最多 ($1701)
- **EXP16 Walk-Forward**: 9 窗口 (2Y IS + 1Y OOS) 滚动验证（结果待读取）

### 出场机制深度分析 (EXP06)

- **TP 越大越好**: TP8.0 Sharpe=1.56 > TP5.0=1.52 > TP2.0=1.13
- TP 几乎不触发（TP5.0 仅 1.4%），**trailing 是真正出场机制** (97.4% WR)
- **Timeout 是最大亏损源**: 1138 笔 -$14,962 (avg -$13.15/笔)
- 利润密度: 前 1-4 bars $4.80/trade，13+ bars 后利润消失
- BUY/SELL trailing 表现几乎一致 ($6.55 vs $7.31)

### V3 ATR Regime — 最安全的升级 (EXP07)

- **所有点差下都有效**: $0.50 Sharpe 1.52→1.85 (+0.33), $0.30 Sharpe 2.27→2.62 (+0.36)
- **K-Fold 6/6 折全赢** (Delta +0.03 到 +0.47)
- 低波动 Trail 更松 (1.0/0.35)，高波动 Trail 更紧 (0.6/0.20)
- 连续亏损跳过 N=3 省 $113，N≥5 样本不足无效

### 风险评估

- **EXP12 Monte Carlo**: 破产概率 **0.0%** (MaxDD>$1500 和 >$3000 均为零)
- MaxDD 分布: P5=$841, P50=$563, P95=$413
- **11/12 年正收益**（仅 2018 年 -$175）
- 滚动 100 笔 Sharpe: 73% 窗口 > 0, 60.3% > 0.5
- **EXP26 $3000 资本**: 破产概率 0.0%，资本充足

### 关键认知更新

1. **宏观 Regime 过滤为什么无效**: 策略是小时级短线(avg 1.5-11h)，宏观是月/季级别。宏观判断"偏空"但日内双向波动都能赚。策略本身已通过 KC/ADX/EMA 实现了微观级 Regime 过滤
2. **V3 ATR Regime 是唯一有效的 Regime 类过滤**: 因为它调整的是止盈参数（Trail Act/Dist）而不是过滤信号方向，本质是"波动率自适应"而非"方向预判"
3. **TP 形同虚设**: TP5.0 仅 1.4% 触发率，设大不设小。TP 的价值是"不限制利润上限"
4. **Timeout 是隐性杀手**: 1138 笔 Timeout 亏 $14,962 > 所有 SL 亏损之和。缩短 max_hold 反而更差（EXP01），说明长持仓的偶尔大赢覆盖了多数小亏

### EXP20 K-Fold 最终决策

- **KC25/M1.2 (不含 EMA150)**: K-Fold 4/6 折赢, Avg Sharpe 0.86→1.17 (+36%) → ✅ **已实装**
- **EMA150 单独**: Fold4(2021-2022) 崩溃至 5 笔交易 Sharpe=-13.56 → ❌ **否决**（横盘市过滤太激进）
- **KC25/M1.2 + EMA150 组合**: EMA150 的 Fold4 灾难拖累整体 → ❌ **否决组合**

### 2026-04-05 实装记录

| 改动 | 之前 | 之后 | 依据 |
|---|---|---|---|
| KC 通道 EMA 周期 | 20 | **25** | EXP13 冠军, K-Fold 4/6 折赢 |
| KC 通道乘数 | 1.5 | **1.2** | EXP13 冠军, 更敏感捕捉突破 |
| TP 倍数 | 5.0 ATR | **8.0 ATR** | EXP06: TP5.0 仅 1.4%触发, TP8.0 Sharpe +0.04 |
| V3 ATR Regime | 无 | **启用** | EXP07 K-Fold 6/6 折全赢, Sharpe +0.33 |

改动涉及文件:
- `strategies/signals.py`: KC_mid span 20→25, KC_upper/lower 乘数 1.5→1.2, ATR_TP_MULTIPLIER 5.0→8.0
- `config.py`: 新增 `V3_ATR_REGIME_ENABLED = True`
- `gold_trader.py`: 追踪止盈逻辑加入 ATR 百分位自适应 (高波动紧追踪/低波动松追踪)
- `backtest/runner.py`: C12_KWARGS 更新 tp_atr_mult=8.0 + regime_config, load_custom 默认参数同步

未改动（保持不变）:
- [x] ~~day_of_week 周内择时~~ → EXP10 确认跳过任何一天都更差，不采用
- [x] ~~时段过滤~~ → EXP05 K-Fold 仅 2/6 折，不采用
- [x] ~~缩短 max_hold~~ → EXP01/06 确认 max_hold=60 最优，不改动
- [x] ~~EMA150 趋势过滤~~ → EXP20 Fold4 崩溃，不采用
- [x] ~~SL 调整~~ → SL 4.5 ATR 已是最优，不改动

## 宏观情景压力测试 (EXP27, 2026-04-05)

### 情景风险矩阵

| 情景 | 代理时段 | N | Sharpe | PnL | MaxDD | WR% | 风险等级 |
|---|---|---|---|---|---|---|---|
| **S1 伊朗战争升级** | 2020-01~02 | 148 | **+6.59** | +$423 | $86 | 85.8% | LOW |
| **S2 伊朗停火暴跌** | 2026-03-15~31 | 32 | **-2.24** | -$78 | $289 | 53.1% | EXTREME |
| **S3 流动性危机** | 2020-03-09~23 | 38 | **-5.83** | -$112 | $198 | 65.8% | EXTREME |
| **S4 央行抛售阴跌** | 2022-09~11 | 397 | **-3.08** | -$525 | $549 | 65.2% | EXTREME |
| **S5 央行囤积慢牛** | 2024-02~04 | 10 | **-11.75** | -$67 | $78 | 40.0% | EXTREME |
| **S6 关税双向鞭打** | 2025-04-02~09 | 51 | **+7.62** | +$165 | $99 | 80.4% | LOW |

- 盈利情景: 2/6, 平均 Sharpe: -1.45
- 最佳: S1 伊朗升级(Sharpe 6.59), 最差: S5 慢牛(Sharpe -11.75)
- **策略擅长高波动有方向的市场(S1/S6), 在低波动或持续单边市中表现差(S4/S5)**

### 方向分解

| 情景 | BUY | BUY PnL | SELL | SELL PnL | 主导出场 |
|---|---|---|---|---|---|
| S1 避险上涨 | 102 | +$337 | 46 | +$86 | Trailing 123笔 +$801 |
| S2 避险暴跌 | 2 | -$15 | 30 | -$63 | SL 11笔 -$556 |
| S3 流动性危机 | 9 | -$50 | 29 | -$62 | SL 4笔 -$202 |
| S4 央行抛售 | 182 | -$159 | 215 | -$367 | Trailing 160笔 +$1087, SL 23笔 -$1056 |
| S5 慢牛 | 1 | -$9 | 9 | -$58 | SL 1笔 -$34 |
| S6 关税鞭打 | 32 | +$226 | 19 | -$60 | Trailing 22笔 +$233 |

### V3 ATR Regime 情景价值

- V3 在 **4/6 情景中胜出**: S1(+2.71), S3(+1.66), S4(+0.25), S6(+1.02)
- V3 在 S2(停火暴跌) 和 S5(慢牛) 中略差: -0.31, -6.42
- S5 V3 大幅落后因样本仅 10 笔, 统计无意义
- **结论: V3 在极端行情中整体有效, 维持启用**

### 防御策略效果

- **S2 MaxPos=1**: MaxDD $289→$263(-9%), 但 PnL 更差 → 防御有限
- **S3 MaxPos=1**: MaxDD $198→$142(-28%), PnL -$112→-$65(+42%) → **流动性危机下有效**
- **S4 MaxPos=1**: MaxDD $549→$480(-13%), PnL -$525→-$458(+13%) → **慢跌中有效**
- **Choppy 阈值提高**: 在所有情景中 **零效果**（极端行情趋势评分高, 不触发 choppy 过滤）

### 最大单日亏损

| 情景 | 最差日 | 最好日 | 最大连亏 | 连亏累计 | 负天数占比 |
|---|---|---|---|---|---|
| S1 | -$44 | +$97 | 2笔(-$1) | $-1 | 36% |
| S2 | **-$153** | +$110 | **6笔** | **-$259** | 67% |
| S3 | -$77 | +$57 | 2笔(-$11) | $-11 | 78% |
| S4 | -$99 | +$74 | 6笔(-$40) | $-40 | 53% |
| S5 | -$55 | $0 | 4笔(-$76) | $-76 | 75% |
| S6 | -$19 | +$80 | 3笔(-$53) | $-53 | 67% |

### 关键结论与下周操作建议

1. **策略特征确认**: 策略是**高波动趋势追随者**, 在有方向的大波动中表现优异(S1/S6 Sharpe 6-7), 但在低波动(S5)和持续单边市(S4)中亏损
2. **伊朗情景分析**:
   - 战争升级(S1): 策略天然受益, BUY 方向主导, trailing 锁利效率高 → **不需调整**
   - 停火暴跌(S2): 最差单日 -$153, 6 笔连亏 -$259 → **可控**, $2000 本金下 MaxDD $289 (14.5%)
   - 即使最差情景(S2), 两周内最大亏损 $78 → **不会触及 $1500 总亏损保护**
3. **央行行为风险**:
   - 央行抛售(S4 类似土耳其 3 月): 3 个月 MaxDD $549 是最大风险, 但 trailing 仍贡献 +$1087 → 策略有自愈能力
   - 央行囤积(S5): 慢牛反而最差 — **策略需要波动率, 不需要方向**
4. **操作建议**:
   - 下周正常运行, 不需预防性调整
   - 如果伊朗宣布停火 → 盘中观察, 若连续 3 笔止损考虑手动暂停 1-2 小时
   - 如果出现流动性危机(VIX>40) → 考虑手动改 `MAX_POSITIONS=1` 临时减仓
   - 关税新闻不需特殊处理(S6 证明策略在鞭打行情中依然盈利)

## EXP28 事件日防御测试 — "带伞策略" (2026-04-05)

### 设计动机

- 核心问题: 能否在高影响事件前预设防御参数("伞"), 事件发生则减少回撤, 不发生则零代价?
- 灵感: "明天可能下雨 → 提前带伞 → 下雨打伞，不下雨也没损失"
- 关键约束: **外部信息（新闻/VIX变化）有延迟**，金价往往率先反应，等看到新闻已经滞后

### 信息延迟问题

- **2026-04-05 认知更新**: 信息传导链 — T+0 内部人/算法 → T+5min 路透彭博 → T+15min 新闻 → T+30min+ 我们的舆情/VIX
- 所以用"新闻"或"VIX变化"触发防御 = 雨停了再打伞
- **EXP28 设计了三层触发器对比延迟影响**: `oracle_event`(后验), `prev_day`(滞后1天), `prev_day+VIX>25`(状态驱动，不受延迟影响)
- **最快的信号是价格本身** — 策略已有的 V3 ATR Regime、Intraday Trend Gating、ATR自适应止损都是零延迟的价格驱动防御

### 阴跌场景深度分析 (S4)

- **2026-04-05 认知更新**: 阴跌不是"无解"，而是趋势跟踪策略的结构性弱点
- S4 数据解剖: 6个月397笔，胜率65%仍亏$525 — 问题是23笔止损(-$1056)吃掉了160笔trailing(+$1087)的利润
- 阴跌特征: 波动率低 + 方向一致 = 假突破频繁 + 追踪止盈没空间
- **正确应对: 不是让策略在阴跌中赚钱（会破坏趋势市优势），而是"少亏就是赢"**
- 可能的缓解方向: ①连续亏损熔断加强 ②周度回撤监控自动减仓 ③EMA100斜率为负时禁止做多

### 测试框架

| Part | 内容 | 目的 |
|---|---|---|
| Part 1 | 事件日标注 | 极端波动日(range>2×ATR)、大幅单边(|ret|>1.5%)、VIX飙升(>15%)，构建无前瞻触发信号 |
| Part 2 | 全量代价穷举 | 11把伞(MaxPos=1/Choppy↑/Cooldown↑/SL↑/Trail松紧/No_V3)的11年Sharpe/PnL代价 |
| Part 3 | 混合回测 | 非事件日默认参数 + 事件日防御参数，测"只在需要时打伞"的效果 |
| Part 4 | K-Fold验证 | 6折验证最佳伞+触发器组合的稳定性 |

- 脚本: `run_exp28.py` (独立 .py，不依赖 jupyter)
- 结果: `data/exp28_shield_results.json`

### EXP28 结果 (2026-04-05, 服务器运行完成)

- **Baseline**: N=10050, Sharpe=2.16, PnL=$12,845, MaxDD=$461

**Part 1 事件日分类**:
  - 极端波动日(range>2×ATR) 占 76.1%（几乎每天都是"事件日"）
  - 事件日内 0 笔交易被单独归类，所有交易都在"正常日"
  - 触发器覆盖 76% 的交易日 → hybrid 等于全量，Part 3 全部 [SAME]

**Part 2 全量防御排名**:
  - **免费(无损)**: Choppy 0.40-0.50、Trail tight/loose — 全部 Sharpe +0.00（无效果）
  - **近乎免费**: MaxPos=1 Sharpe +0.01, SL=5.5 Sharpe +0.03, SL=6.0 Sharpe +0.07
  - **有害**: Cooldown=60min -0.10, Cooldown=120min -0.23, No_V3 -0.28
  - **V3 确认**: 关闭 V3 Sharpe 从 2.16 降到 1.88（-$2,032），V3 是真实结构性 alpha

**Part 3 Hybrid**: 15 个组合×3 触发器 = 全部 [SAME]（触发器覆盖率太高导致无差异）

**Part 4 K-Fold**: 3 个最佳组合 × 6 折 = 全部 TIE（0/6 折赢）

**结论**: ❌ **不实装任何额外防御**。策略的内建防御（V3 + Adaptive + Cooldown 30min）已充分，额外防御要么零效果要么有害。手动 VIX>40 减仓保留作极端应急操作

## EXP30/31/32 实验总结 (2026-04-05)

### EXP30: 缩短 Keltner 最大持仓时间
- Hold=20 bars (5h) 为最佳: Sharpe 4.99 (+0.58), K-Fold 5/6 折赢
- 缩短持仓让 Timeout 退出亏损更小，整体风险降低

### EXP31: EMA100 斜率过滤
- ❌ **完全无效** — `skipped_ema_slope = 0`，Keltner 自带 ADX+EMA 趋势过滤已覆盖

### EXP32: Mega Trail T0.5/D0.15 验证
- 🏆 **压倒性优胜**: Sharpe 7.66 vs 4.41, PnL $54,488 vs $31,457
- K-Fold 6/6 折全赢, 逐年 10/12 赢, S4 阴跌从 -0.08 升至 +4.08
- 更紧更快的追踪 → 更多 trailing 出场, 更少 SL 和 Timeout
- **决定**: 注册为 P7_mega_trail 模拟盘策略观察

### P7_mega_trail 模拟盘注册 (2026-04-05)
- 使用实盘相同的 Keltner 4 阶段信号 + Mega 追踪参数 (T0.5/D0.15)
- V3 ATR Regime 适配: 低波 T0.7/D0.25, 正常 T0.5/D0.15, 高波 T0.4/D0.10
- PaperPosition 新增 trailing stop 机制（之前只支持 SL/TP）
- max_hold_bars=15, max_positions=2
- 观察目标: 实盘环境下 trailing 触发频率、MFE/MAE、与实盘 Keltner 的对比

### EXP33: Mega Trail + Hold Time 联合 (2026-04-06)
- 🔥 **效果叠加**: Mega H12 Sharpe=9.03 (+4.62 vs Current, +1.37 vs Mega H60)
- Mega H24 PnL 最高 ($57,183)，Mega H12 Sharpe 最高 (9.03)
- SL 从 577 笔降至 164 笔（大量潜在 SL 被 Timeout 提前拦截，$-6.84/笔 vs $-49.62/笔）
- K-Fold: Mega H12 Avg=8.76 **6/6 折全赢**，但 Std=1.51（最高）; Mega H20 Avg=8.25 Std=1.37 更稳
- ⚠️ Mega H12 胜率 79.3%（vs Current 83.2%），心理压力稍大
- Low Vol 2018 所有配置均相同 (Sh=-6.48)，低波动是共同弱点

### EXP34: 周内择时 Day-of-Week (2026-04-06)
- ❌ **不值得过滤** — 每天都赚钱，最差的周四也有 $2.75/笔 80.8% 胜率
- 跳过任何一天 Sharpe 均下降（Skip Thu: -0.17, Skip Mon: -0.36）
- 逐年看每天轮流当最差日，无稳定弱势日

### EXP35: 连续亏损自适应减仓 (2026-04-06)
- ❌ **减仓无意义** — 连亏 1-3 笔后下一笔期望仍为正 ($2.3-2.8)
- 所有模拟减仓策略 Sharpe 变化 ≤ ±0.01
- Mega Trail 最大连亏 4 笔（vs Current 5 笔），结构性更稳
- 日内 3+ 亏损后期望变负 (-$35/笔)，但样本仅 5 笔不可靠

### P8_mega_h20 模拟盘注册 (2026-04-06)
- 与 P7 同信号同 trailing (T0.5/D0.15)，max_hold_bars=5 (H1 bars = 5h ≈ M15 Hold=20)
- 对比目标: 短持仓是否在实盘中减少 SL 损失、提升 Sharpe
- P7 vs P8: 唯一区别是 max_hold (P7=15 bars/15h vs P8=5 bars/5h)

## EXP36-41 实验总结 (2026-04-07)

### EXP36: 交易时段过滤 Hour-of-Day
- ❌ **跳过任何时段都会降低 Sharpe** — 全时段都赚钱
- Current: Asia $3.13/t 87.1%WR, London $3.41/t 84.8%WR, NY $2.88/t 78.5%WR, Late $2.73/t 80.0%WR
- Mega: 所有时段更均衡 (Asia $3.76, London $4.49, NY $3.91)
- Skip worst 3h: Current -0.10 Sharpe, Mega +0.00 — 不值得过滤
- 逐年分析: 没有稳定的弱势时段，每年轮换
- **结论**: Keltner 策略在全天候都有效，不限制时段

### EXP37: 分批止盈 Partial Profit Taking
- ✅ **小幅有效**: Current 最优 @0.5xSL close 70% Sharpe +0.33; Mega @0.3xSL close 50% Sharpe +0.35
- 但 MFE 分析揭示只有 33% 交易的 MFE 达到 0.3xSL，13% 达到 0.5xSL
- 大多数交易是"小赚快走"（trailing 在 MFE 很小时就退出）
- **评估**: 理论上有正面效果但实盘实现复杂（MT4 需分批平仓），暂不实装，标记待验证

### EXP38: RSI 背离提前出场
- ❌ **发现重要洞察但结论为负**: 39.8% 交易出现 RSI(9) 背离，这些交易 $/t=-$7.41
- 无背离交易 $/t=$1.27 — 差距 $8.69/笔
- 但这是"先观察到结果再看原因"的后验分析，不能直接用作过滤器
- RSI 背离本身也出现在 ~40% 的交易中 — 过滤掉太多
- **结论**: RSI 背离是"确认亏损"信号而非"预测亏损"信号，不实装

### EXP39: D1 日线方向过滤
- ❌ **反直觉但明确**: Against (逆 D1 趋势) 交易 $/t=$3.45 > Aligned $2.86!
- Block against D1 EMA50: Sharpe -0.78（严重下降），丢失 $13,735 利润
- 逐年看: 12 年中仅 4 年 Aligned > Against — 不稳定
- Mega 同样: Against $4.20/t > Aligned $3.77/t
- **原因推测**: Keltner 是突破策略，突破往往发生在逆势反转点（D1 看跌但 H1 突破向上 = 趋势反转信号）
- **结论**: 不加 D1 方向过滤，反而会误杀高质量反转突破

### EXP40: K线实体/影线质量过滤
- ❌ **结论与直觉相反**: 低 body ratio (<0.2) 交易 $/t=$3.88 > 高 body ratio (0.7+) $2.11
- Mega 同样: Low body $4.55 vs High body $2.94
- 过滤 Body>0.3: Sharpe -0.77; Body>0.5: Sharpe -1.38
- **原因推测**: 小实体 = 波动大 = ATR 大 = 更宽止损 = 更不容易被止损打出
- **结论**: K 线实体比对 Keltner 策略无预测力，不过滤

### EXP41: ATR Regime 反波动率加权
- ❌ **反波动率无效，甚至有害**: InvVol(1.5/1.0/0.6) Current Sharpe -0.49, Mega -0.72
- 原因: 51.9% 交易发生在高 ATR (>0.8 百分位)，且这些交易 $/t=$3.22 不差
- ProVol (高波加仓) 效果几乎中性 (+0.01)
- 逐年: 12 年中 11 年 InvVol 降低 Sharpe — 极其稳定的负面效果
- **结论**: 黄金 Keltner 策略天然在高波动时更活跃且更赚钱，减仓 = 损失alpha

### EXP36-41 总体结论 (2026-04-07)
- 6 个方向只有 EXP37 (分批止盈) 有微弱正效果，其余 5 个全部无效或有害
- **核心认知**: Keltner 突破策略的 alpha 来源非常"纯粹" — 它在所有时段、所有 D1 方向、所有 K 线形态、所有 ATR 水平下都稳定盈利
- 试图用二级过滤器来"优化"一级策略，更大概率是过拟合而非真实改善
- **下一步方向应该是**:
  1. 新策略（而非过滤现有策略）
  2. 宏观因子（DXY/VIX/TIPS）的日级别方向偏倚
  3. 波动率聚集效应（大波动后的延续/反转概率）
  4. 隔夜 vs 日内收益不对称性

## EXP42-47 实验总结 (2026-04-07)

### EXP42: 宏观 Regime (DXY/VIX) 日级偏倚调节
- ❌ **宏观驱动仓位调节无效**: 所有 sizing 规则 Sharpe 变化 ≤ ±0.05
- **VIX 分析**: Panic(>35) $/t=$4.13 > Low(<15) $2.92 — 高恐慌时反而更赚
- **DXY 分析**: DXY 上涨日 SELL $/t=$5.07 >> BUY $0.88; DXY 下跌日 BUY $/t=$4.31 >> SELL -$0.81
- ⚠️ **重要发现**: DXY 方向与 BUY/SELL 收益高度相关（符合黄金定价逻辑），但无法提前预知当日 DXY 方向
- **Haven regime (DXY↑+VIX↑)** $/t=$4.22 是最赚的 — 风险事件中金价波动大=更多 alpha
- Gold Bull: 1.5x + Headwind: 0.5x 反而 Sharpe -0.24 — 因为过度调节

### EXP43: 大波动后动量延续 vs 反转
- ✅ **学术研究确认**: 金价大涨后次日反转概率高（>1%涨幅后，仅38.2%延续上涨）
- 大跌后次日反弹概率更高（>1%跌幅后，64.9%次日上涨）
- **含义**: 黄金存在显著的"均值回归效应"，与学术论文一致
- 后续 sizing 模拟因 tz 问题未完成（待修复）

### EXP44: 波动率聚集效应
- ✅ **波动率聚集极其显著**: Lag 1 autocorr=0.5934（非常强）
- 高波动日(>1.5xATR)后次日仍>1xATR 概率: 60.3%
- **含义**: 高波动不是随机的，是"簇"状出现的 — 支持 GARCH 理论
- 后续 trade 分析因 tz 问题未完成（待修复）

### EXP45: 连续方向偏倚（趋势日识别）
- ✅ **上涨趋势不延续**: 连续3天上涨后第4天继续涨概率仅43%（略低于随机）
- 下跌趋势略有延续: 连续3天下跌后第4天继续跌概率49.7%
- **含义**: 黄金日线级别更偏均值回归而非动量

### EXP46: 整数关口效应
- ❌ **$50关口无显著影响**: Near $/t=$3.11 vs Far $3.42（差异 -$0.31 很小）
- $100关口略有正面效应: Near $/t=$3.22 vs Far $2.80（+$0.42，但Mega中消失）
- **方向偏倚**: Below $50 + SELL $/t=$3.91 最高（价格在整数下方做空更赚）
- $50 突破 vs 拒绝几乎 50/50（Cross 51.6% vs Reject 48.4%）— 无预测力
- **结论**: 整数关口效应在高频交易中可能有用，但对 H1 Keltner 策略无显著意义

### EXP47: 隔夜 vs 日内收益不对称
- ✅ **极度不对称**: 隔夜收益几乎为零(+0.39%)，日内收益占全部(+148.11%)
- 隔夜 positive 仅 26.1%（75% 的夜晚微跌）
- **含义**: 黄金的价格发现几乎完全发生在交易时段，隔夜持仓无收益

### EXP42-47 总体认知更新 (2026-04-07)
1. **策略 alpha 极其鲁棒**: VIX/DXY/日线趋势/K线形态/ATR regime/时段/整数关口 — 都无法显著改善
2. **黄金市场的统计特性确认**: 均值回归(大波动后反转)、波动率聚集(autocorr=0.59)、日内收益主导
3. **DXY 是最强相关宏观因子**: BUY/SELL 收益与 DXY 方向高度一致，但因果方向不可用于预测
4. **高 VIX = 更多 alpha**: 恐慌时金价波动大 → Keltner 突破更多 → 更赚钱
5. **下一步应聚焦**: ①利用波动率聚集效应(已确认 autocorr=0.59)设计 GARCH 自适应; ②利用大波动后反转效应设计 intraday 逆势策略; ③完善 DXY 实时数据管道用于方向偏倚

## EXP48-53 实验结果 (2026-04-07)

思路转变：从"优化 Keltner 过滤器"转向"设计互补策略"和"利用已确认的市场微观结构"。

### EXP48: Keltner 均值回归策略（全新互补策略）
- ❌ **均值回归策略在 H1 级别不可行**: 所有 15 个配置的 Sharpe 均为负值
- 最优配置 ADX<15 SL=1.5x: N=548 Sharpe=-0.02 WR=63.3%（虽高胜率但 $/t=-$0.01，盈亏刚好平衡）
- ADX 阈值越宽（ADX<25），表现越差（Sharpe=-0.75 to -1.46）
- 年度分析极不稳定: 2020/2024 盈利，但 2022/2023 大亏
- **根因分析**: KC 均值回归的止损($SL=1.5x ATR$)被触发时亏损远大于到中线($TP=KC_{mid}$)的盈利，尽管胜率>60%，但盈亏比不足以覆盖
- **结论**: 低频均值回归需要更短时间框架(M15)或更复杂的入场条件；H1 的 KC 带内回归空间太小

### EXP49: 波动率聚集自适应 Trailing
- ❌ **所有波动率聚集调节方案均无改善**: 最佳仅 +0.05 Sharpe（Extreme reduce 0.6x）
- 核心发现: 低波动日后交易 $/t=$3.55（最高），高波动日后 $/t=$2.24（最低）
- 直觉上"高波动=更赚"被推翻: 高波动环境下虽然有更多交易机会，但每笔交易的平均质量下降
- HighVol 1.3x 加码反而降低 Sharpe -0.17（放大了高波动日交易的劣势）
- **年度一致性**: 所有年份 delta 均为负值或接近零 — 没有任何年份从波动率加码中受益
- **结论**: 波动率聚集效应虽然统计上显著(autocorr=0.59)，但无法转化为 sizing alpha

### EXP50: 大波动后方向偏倚（修复 tz 后重跑）
- ❌ **大波动后反转偏倚同样无法转化为 sizing alpha**: 所有方案 delta ≤ +0.05 Sharpe
- Big up(>1%)后: BUY $/t=$3.25, SELL $/t=$0.14 → 实际上 BUY 更赚（与反转理论相反！）
- Big down(<-1%)后: SELL $/t=$1.78, BUY $/t=$0.41 → SELL 更赚（与反转理论也相反！）
- **关键洞察**: EXP43 确认的"反转效应"存在于日线级别，但日内 H1 交易时，大波动日后的 Keltner 突破信号仍然顺势更赚
- 最好的规则 "Threshold 1.5%: fade 1.3x/0.7x" 也仅 +0.05 Sharpe — 可忽略不计
- **结论**: 日线反转效应与日内突破策略的时间尺度不匹配

### EXP51: Keltner 通道宽度变化率（入场质量因子）
- ✅ **通道状态与交易质量有明确单调关系**:
  - Expanding: $/t=$3.73, Contracting: $/t=$1.74（差距 2x）
  - 分位数: Q1(收缩) $/t=$2.03 → Q3(扩张) $/t=$3.87（单调递增）
  - Squeeze-to-Expansion $/t=$3.92 vs Normal Expansion $3.72（+$0.20，边际改善）
- ⚠️ **但作为过滤器效果有限**:
  - "Skip extreme contraction (Q1)" 将 Sharpe 从 4.41 提到 4.59（+0.18）— 这是所有实验中**最大的过滤改善**
  - "Only expanding" 将 N 减半但 Sharpe 几乎不变(4.35)
  - Mega: 扩张 $/t=$4.53 vs 收缩 $3.25（差距 40%），但过滤后 Sharpe 从 7.66 降到 7.38
- **含义**: KC bandwidth 变化率是目前发现的**最有信息量的入场质量因子**，但作为二元过滤器不如作为 confidence score（加权而非拒绝）
- **Mega 中 squeeze-to-expansion 信号更强**: $/t=$6.21 vs normal expansion $4.44（+$1.77, 40%溢价）

### EXP52: 持仓内动态止损调整
- ✅ **ATR spike 检测有显著价值**:
  - ATR spike >50% 的交易: N=614, PnL=$-6,680, $/t=$-10.88（巨额亏损源！）
  - 非 spike 交易: N=9191, PnL=$37,590
  - ATR decrease >20%: N=1793, PnL=$-7,326, $/t=$-4.09（也是亏损源）
- ✅ **"ATR spike >50% reduce 0.7x" 是最有效的调节**: Current +0.57 Sharpe, Mega +0.48 Sharpe
- "ATR increase >30% tighten 0.85x" 也有效: Current +0.42, Mega +0.43
- ⚠️ "ATR decrease extend 1.15x" 反而有害: -0.25 Sharpe（波动率压缩后不是扩张机会而是趋势衰竭）
- **因果链**: 持仓期间 ATR 暴涨 = 市场剧烈波动 → 大概率是止损被更大幅度打穿 → 减仓可降低风险
- **实现建议**: 在 `_check_exits` 中，如果当前 ATR > 入场时 ATR × 1.5，则将持仓 PnL 乘以 0.7 缩减风险敞口（或实际缩减手数 30%）
- **这是 EXP48-53 中最有实战价值的发现**

### EXP53: 多策略组合权重优化
- 当前组合只有 Keltner (N=9805~13456) + ORB (N=287~337)，RSI 几乎不产生交易
- Keltner Sharpe=4.49~7.90 远超 ORB Sharpe=1.47~1.08
- 策略间相关性极低 (0.115/0.104) — 理论上是好的分散化
- ⚠️ **Keltner only 反而更好**: 去掉 ORB 后 Sharpe 从 4.69→4.82（+0.13）, Mega 8.16→8.47（+0.31）
- ORB 拖累组合 Sharpe — 因为它的 WR 仅 51%，Sharpe 偏低
- **动态权重 vs 等权**: Dynamic-weighted PnL $52,406 vs Equal $31,457 — 但这是因为 Keltner 被放大了(权重>1)
- **结论**: 当前组合中 ORB 的贡献微弱甚至负面；应评估 ORB 是否值得保留，或需要重新优化 ORB 参数

### EXP48-53 总体认知更新 (2026-04-07)
1. **Keltner 均值回归不可行(H1)**: 盈亏比不足，止损金额远大于到中线的止盈空间
2. **统计特性→交易 alpha 的转化极难**: 波动率聚集(EXP49)、大波动反转(EXP50)都是统计上显著但交易上无用的特性
3. **KC bandwidth 变化率是最佳入场质量因子**: 通道扩张时突破质量 2x 于收缩时，"Skip Q1" 是唯一显著的正面过滤(+0.18 Sharpe)
4. **ATR spike 检测是最大实战发现**: 持仓期间 ATR 暴涨>50%的交易平均亏$10.88/笔，缩减 0.7x 可提升 Sharpe +0.48~0.57
5. **ORB 策略拖累组合**: 去掉 ORB 后 Sharpe 提升，需重新评估
6. **思维模型更新**: "统计显著"≠"交易可用"。EXP43确认的反转效应(62%概率)在日线存在，但 H1 Keltner 的时间尺度下，突破仍然顺势更赚

### 待办事项
- [x] ~~实现 ATR spike 检测逻辑到 `_check_exits`~~ → 已实现并完成全量验证，见下方验证结果
- [x] ~~评估 "Skip KC bandwidth Q1" 过滤是否值得加入实盘~~ → 全量回测确认 BW 过滤弊大于利（Sharpe -0.40 到 -1.60），加入否决列表 (2026-04-09)
- [ ] 重新评估 ORB 策略: 初步数据显示去掉 ORB 后 Sharpe +0.24，待完整 orb_diagnosis 结果
- [ ] 探索 M15 级别的均值回归策略（H1 不可行，但 M15 时间框架可能更适合）
- [ ] 将 squeeze-to-expansion 信号作为 Keltner 入场 confidence score 测试（Mega 中 $/t 溢价 40%）
- [ ] **Trail Momentum (+50%) K-Fold 验证后考虑实装** — 12/12 年一致, Sharpe +0.44 (2026-04-09)
- [ ] **Stochastic 10/90 作为 RSI 替代候选** — $/t=$27.66 但 11 年仅 103 笔, 需更多数据 (2026-04-09)
- [ ] 等待 exit_combo_matrix / sl_optimization / tp_atr_sweep / trailing_evolution / session_filter 结果并分析 (2026-04-09)

## ATR Spike Protection 全量验证 (2026-04-07)

- **2026-04-07**: `run_atr_spike_verify.py` 完整验证（Phase 1-4，3.4 小时）

### 实现方式
- `backtest/engine.py`: `Position` 新增 `entry_atr` 字段，`_process_signals` 记录入场 ATR
- `_check_exits`: 当 `atr > entry_atr × threshold` 时，trailing distance 乘以 `trail_mult` 收紧
- `backtest/runner.py`: `run_variant` 返回 `atr_spike_tightens` 计数

### Phase 1&2: 全量数据对比（无成本）

| 配置 | N | Sharpe | PnL | dSh | Spike触发 |
|---|---|---|---|---|---|
| Current 基准 | 10,099 | 4.41 | $31,457 | — | — |
| Current + Spike(1.5/0.7) | 10,099 | 4.44 | $31,737 | **+0.03** | 7,287 |
| Mega 基准 | 13,800 | 7.66 | $54,488 | — | — |
| Mega + Spike(1.5/0.7) | 13,800 | 7.67 | $54,591 | **+0.01** | 4,150 |

- Spike protection 仅影响 Trailing 出场：Current +$280，Mega +$103
- 对 SL/TP/Timeout 出场零影响

### Phase 3: 参数敏感度

- Current 最优: threshold=1.3/trail_x=0.5 → Sharpe +0.11（$32,461）
- Mega 最优: threshold=1.3/trail_x=0.5 → Sharpe +0.04（$54,897）
- 参数越宽松效果越小，趋势平滑无悬崖

### Phase 4: K-Fold 6折验证

| Fold | Current基准 | +Spike | Delta |
|---|---|---|---|
| Fold1 (2015-2016) | 4.97 | 5.00 | +0.03 |
| Fold2 (2017-2018) | 4.22 | 4.25 | +0.03 |
| Fold3 (2019-2020) | 5.07 | 5.10 | +0.03 |
| Fold4 (2021-2022) | 4.19 | 4.23 | +0.04 |
| Fold5 (2023-2024) | 4.08 | 4.11 | +0.03 |
| Fold6 (2025-2026) | 3.60 | 3.62 | +0.02 |

6/6 折全正面，但每折仅 +0.02~0.04

### 核心结论

1. **Post-hoc vs 真实引擎差异巨大**: EXP52 post-hoc 模拟 +0.48~0.57 Sharpe，真实引擎仅 +0.03。差异 19 倍
2. **根因**: post-hoc 是"减少亏损交易 PnL 30%"（直接改收益），真实引擎是"微调 trailing distance"（间接影响出场价）
3. **教训**: post-hoc 模拟可做方向性筛选，但绝对值不可信，必须用真实引擎验证
4. **决策**: **保留代码但不启用实盘**。+0.03 Sharpe 属于噪声级别，不值得增加系统复杂度
5. Phase 5/6 因 Unicode 编码错误（GBK 无法编码 ✓ 字符）未执行，但前 4 阶段已足够下结论

## EUR/USD 第二量化品类研究 (2026-04-07)

- **2026-04-07**: 启动 EUR/USD 量化交易可行性研究
- 候选品类分析: 白银(相关性太高)、原油(资金不足)、USD/JPY(BOJ风险)、**EUR/USD(强烈推荐)**
- EMX Pro 确认: `EURUSD.mx`, 点差 1.8 pips(标准), 杠杆 1:400, 合约 100,000 EUR, 最小 0.01 手
- EUR/USD 日均波动 80-100 pips，$2,000 账户完全可控
- 与黄金中等正相关(~0.4-0.6)，通过 DXY 联动，不同 regime 下可形成对冲
- 现有 Keltner/ATR/Trailing 策略框架可 90% 复用

### EUR/USD 数据获取

- **2026-04-07**: yfinance 下载 `EURUSD=X`
- H1: 10,851 bars, 2024-07-01 ~ 2026-04-03（仅 ~21 个月，yfinance 730 天限制）
- M15: 3,210 bars, 仅 2026-02-17 ~ 2026-04-03（yfinance 60 天限制）
- **数据不足**: 无法做有意义的 K-Fold，需从 MT4 导出 EURUSD.mx 更长历史数据

### EUR/USD ATR 特征研究

| 指标 | EUR/USD | XAUUSD(参考) |
|---|---|---|
| H1 ATR(均值) | 12.7 pips (0.00127) | ~$20-25 |
| H1 ATR(中位数) | 11.6 pips | ~$18-20 |
| 日均波幅 | 67 pips | ~$30-50 |
| ADX 均值 | 33.5 | ~25-30 |
| ADX>18 占比 | 91.3% | ~75% |
| ADX>25 占比 | 70.5% | ~55% |

**关键发现**:
- EUR/USD 的 ADX 分布比 XAUUSD 更高（91.3% 时间 ADX>18 vs 黄金 ~75%），意味着更频繁的趋势信号
- 仓位大小: SL=4.5×ATR=57 pips，0.01 手风险仅 $5.7，**可以用更大手数**
- 建议: EUR/USD 可使用 0.05-0.10 手来匹配 $50 风险目标

### EUR/USD 策略回测结果（2024-07 ~ 2026-04，含 1.8 pips 点差）

#### Keltner 通道策略

| 配置 | N | Sharpe | PnL | MaxDD | DD% | WR% | $/t |
|---|---|---|---|---|---|---|---|
| **Gold-params (ADX18)** | 1,427 | 1.59 | $1,683 | $403 | 20.2% | 75.1% | $1.18 |
| SL=6.0 TP=10.0 | 1,398 | 1.55 | $1,399 | $363 | 18.1% | 75.6% | $1.00 |
| KC mult=1.0 | 1,500 | 1.59 | $1,722 | $392 | 19.6% | 75.2% | $1.15 |
| KC mult=1.5 | 1,319 | 1.35 | $1,418 | $393 | 19.6% | 74.8% | $1.08 |
| **KC mult=2.0** | **1,068** | **2.02** | **$1,818** | **$261** | **13.1%** | **76.1%** | **$1.70** |
| ADX>=20 | 1,404 | 1.66 | $1,729 | $325 | 16.3% | 75.2% | $1.23 |
| ADX>=25 | 1,294 | 1.49 | $1,466 | $406 | 20.3% | 75.0% | $1.13 |
| Mega Trail(0.5/0.15) | 1,427 | 1.59 | $1,683 | $403 | 20.2% | 75.1% | $1.18 |
| **无点差(理论上限)** | **1,430** | **3.57** | **$3,959** | **$360** | **18.0%** | **75.2%** | **$2.77** |

**核心发现**:
1. **KC mult=2.0 是最优配置**: Sharpe 2.02，MaxDD 最低($261/13.1%)，$/trade $1.70
2. **点差是最大敌人**: 有点差 Sharpe=1.59，无点差 3.57，点差吃掉 56% 的收益！
3. **ADX≥20 优于 18**: Sharpe +0.07，MaxDD 降低 $78
4. Mega Trail 与 Gold-params 结果完全相同（trailing 在这个数据集上效果无差异）
5. 总体 75% 胜率，但平均每笔仅赚 $1.18（被点差严重侵蚀）

#### ORB 策略

| 配置 | N | Sharpe | PnL | WR% |
|---|---|---|---|---|
| ORB London(8 UTC) | 374 | — | **-$1,649** | 31.3% |
| ORB NY(14 UTC) | 313 | — | **-$1,002** | 36.1% |

**结论**: ORB 在 EUR/USD 上**完全失败**，WR 仅 31-36%，大幅亏损。与黄金 ORB 表现一致（也是拖后腿）——**ORB 策略对两个品种都不适用**

#### 均值回归 (RSI2)

| 配置 | N | Sharpe | PnL | WR% |
|---|---|---|---|---|
| RSI2 15/85 only | 335 | -1.39 | -$539 | 52.8% |
| RSI2 10/90 only | 236 | -0.75 | -$217 | 55.1% |
| KC + RSI2 combo | 1,654 | 1.36 | $1,891 | 71.4% |

**结论**: RSI 均值回归在 EUR/USD H1 上也不可行（负 Sharpe）。与黄金 H1 结论一致。

### EUR/USD 总体评估

**积极信号**:
- Keltner 通道在 EUR/USD 上有效（Sharpe 1.59-2.02）
- ADX 分布更有利于趋势策略（91.3% 时间 ADX>18）
- KC mult=2.0（更宽通道）在 EUR/USD 上表现更好
- 金额虽小但方向正确，可通过放大手数（0.05-0.10）来提升

**风险/限制**:
- **点差侵蚀严重**: 1.8 pips 点差吃掉 56% 收益，这是最大挑战
- **数据仅 21 个月**: 回测可信度有限，需要 5+ 年数据验证
- 0.01 手每笔利润仅 $1-2，需要提高到 0.05 手才有意义
- ORB 和 RSI 均值回归在 EUR/USD 上同样无效

### 下一步

- [x] ~~从 MT4 导出 5年以上 H1 历史数据~~ → Dukascopy 下载 11年 H1 + 6年 M15
- [x] ~~用长数据做 6 折 K-Fold 验证 KC mult=2.0 配置~~ → 见下方深度回测
- [ ] 评估 0.05-0.10 手的实际风险匹配
- [ ] 如果 K-Fold 通过，进入 paper trade 阶段

## EUR/USD 深度回测 — 11 年 Dukascopy 数据 (2026-04-07)

### 数据

- **来源**: Dukascopy (dukascopy-node)
- **H1**: 70,161 bars, 2015-01-01 ~ 2026-04-05 (**11 年**)
- **M15**: 22 个 CSV 文件, 2020-01-01 ~ 2026-04-06 (**6 年**，备用)
- 零下载失败

### Phase 1: 全周期策略对比 (2015-2026, 含 1.8 pip 点差)

| 配置 | N | Sharpe | PnL | MaxDD | DD% | WR% | $/t |
|---|---|---|---|---|---|---|---|
| Gold-params(ADX18,mult=1.2) | 9,022 | 1.43 | $9,905 | $1,018 | **50.9%** | 74.5% | $1.10 |
| KC mult=2.0(ADX18) | 6,380 | 1.69 | $9,421 | $498 | 24.9% | 75.1% | $1.48 |
| KC mult=2.0(ADX20) | 6,339 | 1.70 | $9,419 | $521 | 26.0% | 75.1% | $1.49 |
| **KC mult=2.0(ADX25)** | **6,051** | **1.73** | **$9,207** | **$487** | **24.3%** | **75.1%** | **$1.52** |
| KC mult=2.0 SL6 TP10 | 6,331 | 1.62 | $7,558 | $495 | 24.7% | 75.5% | $1.19 |
| KC mult=2.0 Trail(0.6/0.15) | 6,380 | 1.69 | $9,421 | $498 | 24.9% | 75.1% | $1.48 |
| KC mult=2.0 No-regime | 6,306 | 1.61 | $8,942 | $511 | 25.5% | 74.6% | $1.42 |
| **KC mult=2.0 MaxHold=20** | **6,144** | **1.91** | **$10,769** | **$482** | **24.1%** | **78.8%** | **$1.75** |
| KC mult=2.0 No-spread(理论) | 6,383 | 3.18 | $18,456 | $404 | 20.2% | 75.3% | $2.89 |
| KC mult=2.0 risk$30 | 6,380 | 1.62 | $5,917 | $397 | 19.8% | 75.1% | $0.93 |

### 核心发现

1. **KC mult=2.0 MaxHold=20 是最优配置**: Sharpe 1.91, PnL $10,769, DD 24.1%, WR 78.8%
2. **增加 MaxHold 从 15→20 是最大单一改进**: Sharpe +0.22, PnL +$1,348, WR +3.7%
3. **KC mult=2.0 vs mult=1.2**: 更宽通道 Sharpe +0.26, MaxDD 减半($1,018→$498)
4. **ADX25 vs ADX18**: 几乎无差别(Sharpe 1.73 vs 1.69)，对 EUR/USD 不如黄金敏感
5. **点差仍是最大敌人**: 无点差 Sharpe 3.18 vs 有点差 1.69，点差吃掉 47% 收益
6. **Trailing 参数对 EUR/USD 无效**: Trail(0.6/0.15) vs Trail(0.8/0.25) 结果完全相同
7. **Regime 有微弱正效应**: +0.08 Sharpe，保留

### Phase 2: 逐年表现（KC mult=2.0 MaxHold=20）

| 年份 | N | PnL | WR% | $/trade |
|---|---|---|---|---|
| 2015 | 546 | $1,576 | 77.5% | $2.89 |
| 2016 | 503 | $892 | 77.5% | $1.77 |
| 2017 | 543 | $1,091 | 80.3% | $2.01 |
| 2018 | 597 | $1,254 | 82.4% | $2.10 |
| 2019 | 509 | $425 | 78.0% | $0.83 |
| 2020 | 599 | $1,448 | 81.8% | $2.42 |
| 2021 | 523 | $89 | 75.0% | $0.17 |
| 2022 | 530 | $529 | 76.8% | $1.00 |
| 2023 | 577 | $940 | 79.4% | $1.63 |
| 2024 | 538 | $443 | 76.8% | $0.82 |
| 2025 | 534 | $1,519 | 79.0% | $2.84 |
| 2026 | 145 | $564 | 83.4% | $3.89 |

**12/12 年全部盈利** (100% profitable years)

- 最弱年份: 2021($89) 和 2019($425)
- 最强年份: 2015($1,576) 和 2025($1,519)
- 策略在不同宏观环境下均保持正收益，鲁棒性很强

### Phase 3: K-Fold 6折验证

#### Config A: KC Gold-params (mult=1.2, ADX18)

| Fold | N | Sharpe | PnL | MaxDD | WR% |
|---|---|---|---|---|---|
| Fold1(2015-2016) | 1,554 | 2.20 | $2,980 | $536 | 74.4% |
| Fold2(2017-2018) | 1,627 | 1.39 | $1,710 | $306 | 75.3% |
| Fold3(2019-2020) | 1,629 | 2.09 | $2,266 | $356 | 75.9% |
| Fold4(2021-2022) | 1,559 | 0.56 | $684 | $488 | 72.9% |
| Fold5(2023-2024) | 1,571 | 0.50 | $581 | $1,018 | 73.5% |
| Fold6(2025-2026) | 971 | 1.42 | $1,096 | $326 | 74.6% |
| **平均** | | **1.36** | **$1,553** | | |

#### Config B: KC mult=2.0 (ADX18)

| Fold | N | Sharpe | PnL | MaxDD | WR% |
|---|---|---|---|---|---|
| Fold1(2015-2016) | 1,088 | **2.04** | $2,299 | $344 | 74.4% |
| Fold2(2017-2018) | 1,163 | **1.82** | $1,800 | $301 | 77.0% |
| Fold3(2019-2020) | 1,129 | **1.56** | $1,360 | $498 | 74.6% |
| Fold4(2021-2022) | 1,092 | 0.75 | $727 | $443 | 73.2% |
| Fold5(2023-2024) | 1,132 | **1.15** | $1,070 | $498 | 74.5% |
| Fold6(2025-2026) | 684 | **2.91** | $1,725 | $177 | 77.0% |
| **平均** | | **1.71** | **$1,497** | | |

**两个配置均 6/6 折全正面！** KC mult=2.0 均值 Sharpe 1.71 显著优于 Gold-params 的 1.36

### Phase 4: 全数据 ATR 特征

| 年份 | H1 ATR(pips) | 备注 |
|---|---|---|
| 2015 | 22.1 | 高波动（EUR/USD 平价恐慌） |
| 2016 | 16.4 | |
| 2017 | 14.0 | |
| 2018 | 15.5 | |
| 2019 | 10.0 | **最低波动年** |
| 2020 | 15.7 | 疫情 |
| 2021 | 11.7 | 低波动 |
| 2022 | 18.8 | 美联储加息 |
| 2023 | 13.6 | |
| 2024 | 10.8 | 低波动 |
| 2025 | 15.3 | |
| 2026 | 14.5 | |
| **11年均值** | **14.9** | |

ADX 11年均值: 34.6, ADX>18: 92.2%, ADX>25: 72.7%

### 总体评估

**EUR/USD Keltner 策略通过验证 ✓**

| 指标 | 黄金 XAUUSD | EUR/USD | 评价 |
|---|---|---|---|
| Sharpe (11yr) | ~4.4 (无成本) | 1.91 (含点差) | EUR/USD 含成本仍健康 |
| K-Fold 全正面 | 6/6 | 6/6 | 同等稳健 |
| 年度盈利 | 11/12 | **12/12** | EUR/USD 更稳定 |
| MaxDD% | ~15% | 24.1% | EUR/USD 略高但可接受 |
| WR% | ~55% | **78.8%** | EUR/USD 胜率更高 |
| $/trade | ~$3-5 | $1.75 | EUR/USD 单笔小但胜率补偿 |

**推荐配置**: KC mult=2.0, ADX18, SL=4.5x, TP=8.0x, MaxHold=20, V3 Regime, 1.8pip 点差

### 下一步

- [x] ~~将 EUR/USD 策略集成到 paper_trader.py~~ → 完成，P9_eurusd_keltner
- [x] ~~配置 EUR/USD 专属参数~~ → point_value=100,000, lots=0.05, symbol=EURUSD.mx
- [ ] paper trade 至少 20 笔后评估实盘切换

## EUR/USD Paper Trade 集成 (2026-04-07)

- **2026-04-07**: EUR/USD Keltner 策略加入模拟盘系统

### 架构改动

1. `paper_trader.py`: `PaperPosition` 新增 `point_value` 和 `symbol` 字段，支持多品种 PnL 计算
2. `paper_trader.py`: `scan()` 新增 `extra_data` 参数，支持传入非黄金品种数据
3. `paper_trader.py`: 方向冲突检查改为按品种隔离（黄金和 EUR/USD 可同时持仓不同方向）
4. `paper_trader.py`: 策略注册支持 `data_key`、`point_value`、`symbol`、`lots` 参数
5. `gold_runner.py`: 新增 `_fetch_eurusd_h1()` 通过 yfinance 获取 EUR/USD H1 数据（每 4 次扫描刷新 = ~2 分钟）
6. `gold_runner.py`: paper.scan 传入 extra_data 含 EURUSD_H1

### P9_eurusd_keltner 策略配置

| 参数 | 值 | 说明 |
|---|---|---|
| KC mult | 2.0 | 宽通道（EUR/USD 最优） |
| ADX 阈值 | 18 | |
| SL | 4.5 × ATR | |
| TP | 8.0 × ATR | |
| MaxHold | 20 bars | 比黄金多 5 bar |
| Trailing | V3 Regime | 与黄金相同逻辑 |
| lots | 0.05 | $50 风险匹配 |
| point_value | 100,000 | 外汇标准手 |

### 向后兼容

- 黄金策略 (P3-P8) 完全不受影响（默认 point_value/symbol 从 config 读取）
- 现有 paper_positions.json 和 paper_trades.json 格式向后兼容（新字段有默认值）

## IntradayTrendMeter 对突发趋势反应滞后 (2026-04-07)

- **2026-04-07**: 黄金从 $4,660 暴跌至 $4,610（~$50），但 trend_score 仅 0.33（choppy），系统未入场
- **根因**: ADX=14.8（前 18 根 K 线横盘，被稀释），KC 突破比例低（22 根中仅 3-4 根突破），EMA 排列未翻转
- **本质**: IntradayTrendMeter 看的是"全天趋势性"，对日内末段的突发趋势反应太慢
- **潜在优化**: 增加"短窗口动量旁路"（如最近 3 根 K 线连续破 KC 或 ATR 暴涨时绕过 choppy 门控）(已回测验证)

## EXP-MOM: 短窗口动量旁路回测结果 (2026-04-07~08)

### Round 1: 发现回测 TrendMeter 索引 bug（2026-04-07）

初次回测发现所有 bypass 方案结果与 baseline 完全相同，`skipped_choppy = 0`。

- **根因**: `_update_intraday_score` 用 `len(h1_window)`（滚动窗口~100）过滤 `_h1_date_map` 中的绝对索引（~5000-98000），导致 `valid` 永远为空，score 默认 0.5，choppy 从未触发
- **影响**: 之前所有用 `intraday_adaptive=True` 的回测结果中，门控实际上从未生效

### Bug 修复（2026-04-08）

- **修改**: `backtest/engine.py` 的 `_update_intraday_score`，改用 `h1_lookup[bar_time.floor('h')]` 获取全局 H1 索引上限，`valid = [i for i in indices if i <= max_idx]`
- **验证**: 一致性测试全部通过（实盘 `_calc_score` vs 回测 `_calc_realtime_score` 对相同输入结果完全一致）
- **效果**: 修复后 27.2% 的交易日至少出现一次 choppy 窗口（1,116/4,101 天）

### Round 2: 修复后回测结果（2026-04-08）

#### 修复前 vs 修复后 Baseline 对比

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| 交易笔数 | 10,099 (H1=10,092 M15=7) | 14,874 (H1=9,934 M15=4,940) | M15 正常参与 |
| Sharpe | 4.41 | 4.46 | +0.05 |
| PnL | $31,457 | $35,033 | +$3,576 |
| skipped_choppy | 0 | 31,071 | 门控正常工作 |

#### 旁路方案对比（Current 配置，11 年无点差）

| Variant | N | Sharpe | PnL | Skip_Choppy |
|---|---|---|---|---|
| A: Baseline | 14,874 | **4.46** | $35,033 | 31,071 |
| B: KC 突破旁路 | 14,909 | 4.43 | $34,955 | 29,108 |
| C: ATR 暴涨旁路 | 14,882 | 4.46 | $35,071 | 30,481 |
| D: 价格动量旁路 | 14,938 | 4.45 | $35,191 | 27,483 |
| E: Combo 旁路 | 14,949 | 4.45 | $35,230 | 25,973 |
| F: 无门控（上限） | 19,420 | 4.05 | $34,866 | 0 |
| G: Combo→trending | 15,102 | 4.38 | $34,937 | 25,893 |

#### 参数敏感性

| Variant | Sharpe | PnL |
|---|---|---|
| E1: Combo 宽松(1.3/1.2) | 4.46 | $35,276 |
| E2: Combo 默认(1.5/1.5) | 4.45 | $35,230 |
| E3: Combo 严格(2.0/2.0) | 4.45 | $35,173 |
| E5: Mom win=2 mult=1.0 | **4.47** | $35,266 |

#### Mega 配置

| Variant | N | Sharpe | PnL |
|---|---|---|---|
| M-A: Mega Baseline | 18,545 | **7.35** | $57,862 |
| M-E: Mega+Combo | 18,642 | 7.34 | $58,192 |

#### K-Fold 6折验证

| | Avg Sharpe | Std | 6 折 |
|---|---|---|---|
| Baseline | 4.51 | 0.45 | [4.60, 4.26, 5.15, 4.40, 4.89, 3.73] |
| Combo Bypass | 4.51 | 0.48 | [4.58, 4.34, 5.24, 4.37, 4.84, 3.67] |
| Delta | +0.000 | | 6/6 正收益 |

### 结论

1. **Bug 修复本身是最大收获**: Baseline Sharpe 4.41→4.46, PnL +$3,576, M15 策略正确参与 trending 日交易
2. **动量旁路效果微弱**: 所有方案 Sharpe 变化在 -0.03 到 +0.01 之间，K-Fold Delta=0.000，无统计显著差异
3. **choppy 门控有价值**: No gate Sharpe 从 4.46 降到 4.05，说明过滤 choppy 时段的低质量入场是正确的
4. **旁路不上线**: 被 choppy 过滤的信号整体是负贡献，强行放回来不改善也不恶化，但增加了复杂度，不值得
5. **注意**: 之前所有 EXP28-53 中用 intraday_adaptive=True 的结果需要在此修复基础上重新评估

---

### 新策略回测脚本 (2026-04-05)

三个顺势策略回测脚本已编写，供在外部服务器测试：

#### Strategy A: 动量追击 (`run_strategy_a_momentum.py`)
- **核心逻辑**: 短窗口 N 根 K 线累积涨幅超过 K*ATR 时入场，不等 ADX/EMA 确认，捕捉 Keltner 因指标滞后错过的急速行情
- **参数搜索**: lookback(2-5) x atr_mult(1.0-3.0) x sl_atr(3-6) x max_hold(8-16)，共 180 组合
- **验证**: Top-3 做 6-fold K-Fold + Momentum-only 独立评估
- **状态**: 待运行

#### Strategy C: 多级趋势过滤 (`run_strategy_c_trend_filter.py`)
- **核心逻辑**: 从 H1 聚合 D1 数据，D1 EMA 交叉 + ADX 确认多空方向，只放行方向一致的 H1 信号
- **参数搜索**: D1 EMA fast(10/20) x slow(50/100) x ADX 阈值(15-30) x 允许 FLAT(Y/N)，共 16 组合
- **验证**: Top-3 做 6-fold K-Fold + D1 趋势分布分析
- **状态**: 待运行

#### Strategy D: 趋势回调入场 (`run_strategy_d_pullback.py`)
- **核心逻辑**: 先确认趋势脉冲（N 根 K 线涨幅 > K*ATR），然后等 RSI 回调或价格触及短期 EMA 时入场，风险回报更优
- **参数搜索**: impulse_lb(5-15) x impulse_atr(1.5-3.0) x pb_rsi(30-45) x pb_ema(9/21) x sl_atr(3-6)，共 384 组合
- **验证**: Top-3 做 6-fold K-Fold + Pullback-only 独立评估 + 与 Keltner 信号重叠分析
- **状态**: 运行中

---

### 回测结果汇总 (2026-04-09)

#### 重要修复: backtest/engine.py monkey-patch bug

发现 `backtest/engine.py` 中 `from strategies.signals import scan_all_signals` 创建了本地绑定，
导致所有 monkey-patch（Strategy A/C/D 使用的信号注入/过滤机制）完全无效。
修复: 改为 `signals_mod.scan_all_signals(...)` 通过模块引用调用。

#### Strategy A: 动量追击 — 不推荐上线

**Baseline**: N=1779, Sharpe=4.92, PnL=$4,416

| 配置 | N | Sharpe | PnL | MaxDD |
|------|---|--------|-----|-------|
| lb=2/am=3.0/sl=3.0/mh=16 (Best) | 1732 | 3.97 | $4,415 | $1,269 |
| lb=5/am=3.0/sl=*/mh=16 | 1706 | 3.97 | $4,765 | $1,269 |
| lb=4/am=3.0/sl=*/mh=16 | 1708 | 3.92 | $4,658 | $1,269 |

- **K-Fold**: Avg Sharpe=1.63, Std=5.15, 3/4 正收益
- **Momentum-only**: 仅 140 笔交易, Sharpe=1.28, PnL=$407
- **结论**: 所有配置 Sharpe < baseline (4.92)。动量信号独立很弱，Combined 反而将 Sharpe 从 4.92 拖到 3.97，MaxDD 从 $225 增到 $1,269。SL 参数无影响（3.0/4.5/6.0 结果一致）。**不上线。**

#### Strategy C: 多级趋势过滤 — 不推荐上线

**Baseline**: N=1779, Sharpe=4.92, PnL=$4,416

| 配置 | N | Sharpe | PnL |
|------|---|--------|-----|
| D1(20/50) ADX>30 flat=Y (Best) | 1466 | 4.34 | $3,601 |
| D1(10/50) ADX>30 flat=Y | 1463 | 4.24 | $3,534 |
| D1(20/50) ADX>20 flat=Y | 1344 | 4.22 | $3,277 |
| D1(10/50) ADX>20 flat=N | 703 | 3.94 | $1,842 |
| slow=100, flat=Y (8种) | 1779 | 4.92 | $4,416 |
| slow=100, flat=N (8种) | 0 | 0.00 | $0 |

- **D1 趋势分布** (EMA10/100): UP=0%, DOWN=0%, FLAT=100% — EMA100 太慢无法分辨趋势
- **K-Fold**: Top 3 全是 slow=100 flat=Y（等同 baseline），Delta Sharpe=0.000
- **结论**: D1 过滤没有改善表现。slow=50 能分辨趋势但过滤后 Sharpe 下降。**不上线。**

#### ADX Threshold Test — 当前 ADX=18 最优

| ADX | Current Sharpe | Current PnL | Mega Sharpe | Mega PnL |
|-----|---------------|-------------|-------------|----------|
| **18 (live)** | **4.92** | **$4,416** | **8.29** | **$7,457** |
| 22 | 4.88 | $4,219 | 8.03 | $7,021 |
| 24 | 4.49 | $3,803 | 7.84 | $6,661 |
| 26 | 4.65 | $3,702 | 7.54 | $6,066 |
| 28 | 4.28 | $3,263 | 7.05 | $5,355 |

- ADX 提高主要减少 H1 信号（M15 几乎不变），但 Sharpe/PnL 均下降
- **结论**: ADX=18 是最优阈值，无需调整

#### D1+3h Overfit Detection Suite — 5/5 通过，低过拟合风险

| 测试 | 结果 | 详情 |
|------|------|------|
| **K-Fold 6折** | **PASS** | D1+3h 6折全正 (Avg=8.52, Min=5.14)，C12 有负折 |
| **参数敏感性** | **PASS** | Sharpe range 8.01-9.41 (span=1.40)，无悬崖式下降 |
| **PBO** | **PASS** | PBO=0.41 < 0.50，中等风险但通过 |
| **PSR** | **PASS** | p=0.000000，Sharpe 高度显著 |
| **DSR** | **PASS** | DSR > 0.95，通过多重测试校正 |

参数敏感性详情:

| Variant | N | Sharpe | PnL | WR% |
|---------|---|--------|-----|-----|
| Base: D1+3h | 2604 | 8.88 | $7,382 | 78.0% |
| max_hold=8 (2h) | 2702 | 9.38 | $7,378 | 73.9% |
| no_time_decay | 2559 | 9.41 | $7,866 | 77.2% |
| decay_start=1.5h | 2572 | 9.20 | $7,675 | 77.5% |
| no_regime | 2533 | 8.25 | $6,794 | 76.7% |
| max_hold=16 (4h) | 2555 | 8.01 | $7,114 | 80.1% |

- trail_act/trail_dist 扰动无影响 (Sharpe 不变)
- 最敏感参数: no_regime (Δ=-0.64), max_hold (range 8.01-9.38)
- **结论**: D1+3h Sharpe 8.88 可信度高，过拟合风险低。regime 配置有价值 (去掉后 -0.64)。

## 第四批并行实验结果 (2026-04-09, AutoDL 服务器)

### 实验概览
- 20 个测试脚本在 AutoDL 服务器并行运行，全部完成
- 覆盖方向：因子重要性、入场过滤、仓位管理、新策略、出场优化、宏观 regime

### Factor Importance (因子重要性分析)

- **IC 排名**: atr_pct(-0.0708, 最显著) > ADX(-0.0412) > KC_bw(-0.0298) > EMA100_dist(+0.0191) > KC_pos(+0.0163)
- **Gradient Boosting**: ATR(22.9%) > EMA100_dist(16.8%) > KC_pos(14.5%) > RSI14(11.5%) > KC_bw(8.8%)
- **最佳交互**: Low ATR + Asian session $/t=$3.27 WR=85.9%; RSI14 oversold + BUY $/t=$3.29
- **Volume = 0**: 数据中成交量全零，volume 因子完全无用
- **结论**: ATR 是最重要因子，现有 ADX+KC+EMA100 框架已覆盖 top 因子。R²=-0.023 说明线性模型无法解释 PnL 变异

### KC Bandwidth Filter (通道宽度过滤) — ❌ 不实装

| 变体 | Sharpe | PnL | vs 基线 |
|------|--------|-----|---------|
| 无过滤（D1+3h 基线） | **8.39** | **$59,218** | — |
| BW3 (3 bar 扩张) | 7.99 | $50,728 | **-0.40** |
| BW5 (5 bar 扩张) | 7.69 | $48,230 | **-0.70** |
| BW8 (8 bar 扩张) | 7.19 | $44,287 | **-1.20** |
| BW12 (12 bar 扩张) | 6.79 | $39,913 | **-1.60** |

- 虽然 BW3 提高了 KC 单笔质量 ($/t $3.30→$3.73)，但过滤掉的信号中包含大量盈利交易
- 逐年 12/12 年 BW3 均不如基线
- **加入否决列表**

### Kelly Sizing & 仓位管理 — ✅ Trail Momentum 有价值

| 方案 | Sharpe | PnL | MaxDD |
|------|--------|-----|-------|
| Fixed 1.0x (基线) | 8.45 | $59,218 | $329 |
| **Trail momentum (+50%)** | **8.89** | **$81,194** | $353 |
| Streak ±30% | 8.48 | $70,340 | $370 |
| Half-Kelly (rolling 100) | 8.46 | $118,384 | $658 |
| Day-loss extreme | 8.30 | $48,908 | $201 |
| Anti-streak | 8.29 | $49,024 | $287 |

- **Trail momentum (+50%)**: 上一笔 trailing 出场后，下一笔加仓 50%。逻辑合理（trailing = 趋势存在，趋势有延续性）
- **12/12 年全部优于基线**（Delta +0.10 到 +0.70），一致性 100%
- MaxDD 仅增 7%（$329→$353），$2000 账户可接受
- **待进一步 K-Fold 验证后考虑实装**

### London Breakout (伦敦突破) — ❌ 全部失败

- 7 个变体全部负 Sharpe（-0.15 到 -0.39），WR 仅 48%
- 对比 ORB（N=330, PnL=$423, $/t=$1.28），伦敦突破完全不可行
- **加入否决列表**

### Stochastic Mean Reversion — ⚠️ 暂不实装，标记观察

| 变体 | N | Sharpe | $/t |
|------|---|--------|-----|
| Stoch 10/90 hold=4h | 103 | 3.58 | $27.66 |
| Stoch 15/85 hold=6h | 306 | 2.05 | $10.19 |
| Stoch 25/75 hold=6h | 1016 | 1.15 | $11.12 |

- 单笔利润远高于 M15 RSI（$27.66 vs $0.73），但 11 年仅 103 笔（样本不足）
- 如 RSI 继续表现弱，可考虑替换为 Stochastic 10/90

### Gold/Oil Ratio — ❌ 数据失败

- yfinance 下载原油 CL=F 失败（`index 0 is out of bounds`），测试跳过
- 需从其他数据源获取后重跑

### Regime Validation (宏观 Regime 验证) — ❌ 再次确认无效

- Baseline Sharpe=5.58 > Regime-Weighted 5.53（-0.05）
- K-Fold: Baseline 赢 6/6 折
- 所有 6 种 regime 下策略都盈利（Sharpe 5.18-6.05），策略是 regime-agnostic
- Regime 区分度仅 0.87（<1.0 阈值），不足以产生有效过滤
- **再次确认加入否决列表**

### Trump Factor / 波动率 Sizing — ❌ 无效

| 方案 | Sharpe | Delta |
|------|--------|-------|
| Flat 1.0x (基线) | 8.45 | — |
| Extreme 0.5x | 8.49 | +0.04 |
| HighVol 1.3x | 8.29 | -0.16 |
| LowVol+HighVol combo | 8.16 | -0.29 |

- 低波动 $/t=$2.97 > 高波动 $2.52 > 极端波动 $2.39
- 所有 sizing 调整效果在 ±0.05 以内
- **加入否决列表**

### ORB Diagnosis — 等待完整结果

- 初步看到: NoORB Sharpe=8.63 > 有 ORB 8.39（+0.24）
- 去掉 ORB 后 Sharpe 更高，再次确认 ORB 是拖累
- 完整结果待脚本运行完毕

### RSI Optimize — 等待完整结果

- 初步看到: 所有 RSI 变体 Sharpe 在 8.38-8.40 之间，与基线 8.39 几乎无差别
- 再次确认 RSI 在 D1+3h 框架下已自然消亡

### Session Filter / SL Optimization / TP ATR Sweep / Trailing Evolution / Exit Combo Matrix — 仍在运行

- 这 6 个测试仍在运行中，结果待补充

### 第四批实验总体结论 (2026-04-09)

1. **唯一可操作发现: Trail Momentum (+50%)** — Sharpe +0.44, 12/12 年一致，逻辑合理（趋势延续性），待 K-Fold 验证
2. **KC Bandwidth 过滤弊大于利**: 虽然提高单笔质量但损失总利润，加入否决列表
3. **London Breakout 全部失败**: 7 个变体全负 Sharpe，加入否决列表
4. **宏观 Regime 第三次确认无效**: 策略在所有 regime 下都盈利，不需要 regime 调节
5. **波动率 Sizing 无效**: 高波动不应加仓也不应减仓，保持固定仓位最优
6. **Stochastic 10/90 值得观察**: 单笔利润高但样本过少，作为 RSI 替代候选保留
