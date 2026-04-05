"""
黄金量化交易主引擎
==================
编排层: 组合数据获取、风控、持仓追踪模块，执行交易流程

职责:
1. 调用 DataProvider 获取行情数据
2. 运行信号引擎检测入场/出场
3. 通过 MT4Bridge 执行交易
4. 调用 RiskManager 做风险管理
5. 调用 PositionTracker 记录交易日志
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

import config
from mt4_bridge import MT4Bridge
from data_provider import DataProvider
from risk_manager import RiskManager
from position_tracker import PositionTracker
from strategies.signals import (scan_all_signals, check_exit_signal,
                                get_orb_strategy, calc_auto_lot_size)
import notifier

# 舆情分析模块 (安全导入，失败不影响交易)
try:
    from sentiment import SentimentEngine
    SENTIMENT_AVAILABLE = True
except Exception as _import_err:
    SENTIMENT_AVAILABLE = False
    logging.getLogger(__name__).warning(f"舆情模块导入失败: {_import_err}")

# 跨资产宏观监控 (安全导入，失败不影响交易)
try:
    from sentiment.macro_monitor import MacroMonitor
    MACRO_MONITOR_AVAILABLE = True
except Exception as _import_err:
    MACRO_MONITOR_AVAILABLE = False
    logging.getLogger(__name__).warning(f"宏观监控模块导入失败: {_import_err}")

# 盘中趋势自适应 (安全导入)
try:
    from intraday_trend import IntradayTrendMeter
    TREND_METER_AVAILABLE = True
except Exception as _import_err:
    TREND_METER_AVAILABLE = False
    logging.getLogger(__name__).warning(f"盘中趋势模块导入失败: {_import_err}")

log = logging.getLogger(__name__)


class GoldTrader:
    """黄金量化交易器"""

    def __init__(self):
        self.bridge = MT4Bridge()
        self.data = DataProvider(self.bridge)
        self.risk = RiskManager(config.DATA_DIR)
        self.tracker = PositionTracker(self.bridge, config.DATA_DIR)

        # 初始化舆情引擎
        self.sentiment = None
        if SENTIMENT_AVAILABLE:
            try:
                self.sentiment = SentimentEngine(update_interval=300)
                log.info("🌐 舆情分析模块已加载")
            except Exception as e:
                log.warning(f"舆情模块初始化失败 (不影响交易): {e}")
        else:
            log.info("舆情模块未安装，纯技术面交易")

        # 初始化跨资产宏观监控 (观察模式，不影响交易)
        self.macro_monitor = None
        if MACRO_MONITOR_AVAILABLE:
            try:
                self.macro_monitor = MacroMonitor(cache_ttl_seconds=600)
                log.info("📊 跨资产宏观监控已加载 (观察模式)")
            except Exception as e:
                log.warning(f"宏观监控初始化失败 (不影响交易): {e}")

        # 盘中趋势自适应
        self.trend_meter = None
        if TREND_METER_AVAILABLE and config.INTRADAY_TREND_ENABLED:
            try:
                self.trend_meter = IntradayTrendMeter()
                log.info("📈 盘中趋势自适应已加载")
            except Exception as e:
                log.warning(f"盘中趋势模块初始化失败 (不影响交易): {e}")

    # ── 向后兼容属性 (供 gold_runner.py 使用) ──

    @property
    def total_pnl(self):
        return self.tracker.total_pnl

    @property
    def tracking(self):
        return self.tracker.tracking

    @property
    def daily_pnl(self):
        return self.risk.daily_pnl

    @property
    def daily_loss_count(self):
        return self.risk.daily_loss_count

    def get_strategy_positions(self) -> List[Dict]:
        return self.tracker.get_strategy_positions()

    def _sync_positions_tracking(self):
        self.tracker.sync_positions(risk_manager=self.risk)

    def get_hourly_data(self) -> Optional[pd.DataFrame]:
        return self.data.get_hourly_data()

    def get_m15_data(self) -> Optional[pd.DataFrame]:
        return self.data.get_m15_data()

    # ── 核心交易流程 ──

    def scan_and_trade(self) -> Dict:
        """完整扫描+交易流程"""
        now = datetime.now()
        log.info(f"\n{'='*60}")
        log.info(f"📊 黄金量化交易 — {now.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"{'='*60}")

        # 同步MT4持仓状态
        self.tracker.sync_positions(risk_manager=self.risk)

        # 总亏损检查
        if self.risk.check_total_loss_limit(self.tracker.total_pnl.get('total_pnl', 0)):
            return {"status": "stopped", "reason": "total_loss_limit"}

        # 日内亏损检查
        if self.risk.check_daily_loss_limit():
            log.warning(f"\n{'!'*60}")
            log.warning(f"🚨 日内亏损{self.risk.daily_loss_count}笔 (上限{config.DAILY_MAX_LOSSES}笔) | 日内PnL: ${self.risk.daily_pnl:.2f}")
            log.warning(f"🛑 系统停止交易，明日自动恢复")
            log.warning(f"{'!'*60}")
            notifier.notify_stop_review(self.risk.daily_pnl)
            return {"status": "STOP_REVIEW", "daily_pnl": self.risk.daily_pnl, "daily_losses": self.risk.daily_loss_count}

        # 获取舆情分析结果
        sentiment_ctx = self._get_sentiment_context()

        # 舆情状态落盘
        if sentiment_ctx:
            daily_state_file = config.DATA_DIR / "gold_daily_state.json"
            state = self.tracker._load_json(daily_state_file, {})
            s = sentiment_ctx["sentiment"]
            state["macro_sentiment"] = {
                "label": s["label"],
                "score": s["score"],
                "confidence": s["confidence"],
                "keyword_score": s.get("keyword_score", 0.0),
                "finbert_score": s.get("finbert_score"),
                "vader_score": s.get("vader_score", 0.0),
                "direction_bias": sentiment_ctx["trade_modifier"]["direction_bias"],
                "news_summary": sentiment_ctx.get("news_summary", ""),
            }
            if self.macro_monitor:
                try:
                    cross_assets = self.macro_monitor.get_cross_asset_snapshot()
                    if cross_assets:
                        state["macro_cross_assets"] = cross_assets
                except Exception as e:
                    log.debug(f"跨资产数据获取失败 (不影响交易): {e}")
            self.tracker._save_json(daily_state_file, state)
            self.tracker.append_sentiment_history(state)

        # 经济日历暂停检查
        if sentiment_ctx and not sentiment_ctx['trade_modifier']['allow_trading']:
            pause_reason = sentiment_ctx['calendar'].get('pause_reason', '未知原因')
            log.info(f"  🛑 舆情避险: {pause_reason}，暂停交易")
            log.info(f"{'='*60}")
            return {"status": "paused", "reason": pause_reason}

        # 获取多时间框架数据
        df_h1 = self.data.get_hourly_data()
        df_m15 = self.data.get_m15_data()

        if df_h1 is None and df_m15 is None:
            return {"status": "error", "reason": "no_data"}

        if df_h1 is not None:
            latest = df_h1.iloc[-1]
            close = float(latest['Close'])
            rsi14 = float(latest['RSI14']) if not pd.isna(latest['RSI14']) else 50
            macd_h = float(latest['MACD_hist']) if not pd.isna(latest['MACD_hist']) else 0
            log.info(f"  XAU/USD H1: ${close:.2f}  RSI(14): {rsi14:.1f}  MACD: {macd_h:+.2f}")

        if df_m15 is not None:
            m15_latest = df_m15.iloc[-1]
            m15_rsi = float(m15_latest['RSI2']) if not pd.isna(m15_latest['RSI2']) else 50
            log.info(f"  XAU/USD M15: RSI(2): {m15_rsi:.1f}")

        # Step 1: 检查现有持仓出场
        exits = []
        if df_h1 is not None:
            exits += self._check_exits(df_h1)
        if df_m15 is not None:
            exits += self._check_exits(df_m15)

        # Step 2: 检查新入场信号
        h1_adx = None
        if df_h1 is not None:
            _adx_raw = df_h1.iloc[-1].get('ADX', float('nan'))
            h1_adx = float(_adx_raw) if not pd.isna(_adx_raw) else None

        entries = []
        if df_h1 is not None:
            entries += self._check_entries(df_h1, 'H1', sentiment_ctx)
        if df_m15 is not None:
            entries += self._check_entries(df_m15, 'M15', sentiment_ctx, h1_adx=h1_adx)

        total = len(exits) + len(entries)
        log.info(f"\n{'='*60}")
        if total > 0:
            log.info(f"⚡ 执行了 {total} 笔操作")
        else:
            log.info(f"✅ 无操作")
        log.info(f"  总盈亏: ${self.tracker.total_pnl.get('total_pnl', 0):.2f} / 上限: -${config.MAX_TOTAL_LOSS}")
        log.info(f"  日内盈亏: ${self.risk.daily_pnl:.2f} | 亏损{self.risk.daily_loss_count}/{config.DAILY_MAX_LOSSES}笔")

        if df_h1 is not None:
            adx_val = float(df_h1.iloc[-1]['ADX']) if not pd.isna(df_h1.iloc[-1].get('ADX', float('nan'))) else 0
            atr_val = float(df_h1.iloc[-1]['ATR']) if not pd.isna(df_h1.iloc[-1].get('ATR', float('nan'))) else 0
            adx_status = '趋势✅' if adx_val >= 25 else '震荡⚠️'
            orb = get_orb_strategy()
            log.info(f"  ADX={adx_val:.1f} ({adx_status})  ATR=${atr_val:.2f}  止损=${atr_val*2.5:.2f}")
            log.info(f"  🇺🇸 ORB: {orb.get_status()}")

        if sentiment_ctx:
            s = sentiment_ctx['sentiment']
            m = sentiment_ctx['trade_modifier']
            label_cn = {'BULLISH': '看涨', 'BEARISH': '看跌', 'NEUTRAL': '中性'}
            log.info(f"  🌐 舆情: {label_cn.get(s['label'], s['label'])}({s['score']:.2f}) "
                     f"方向偏好: {m['direction_bias'] or '无'} 仓位系数: {m['lot_multiplier']:.1f}")

        if self.trend_meter:
            log.info(f"  📈 趋势: {self.trend_meter.status_line()}")

        if self.risk.cooldown_until:
            for strat, until in self.risk.cooldown_until.items():
                remaining = (until - datetime.now()).total_seconds() / 60
                if remaining > 0:
                    log.info(f"  ❄️ {strat} 冷却中 (还剩{remaining:.0f}分钟)")
        log.info(f"{'='*60}")

        self.tracker.append_equity_record(self.risk.daily_pnl, self.risk.daily_loss_count)

        return {"exits": exits, "entries": entries}

    def _check_exits(self, df: pd.DataFrame) -> List[Dict]:
        """检查出场信号"""
        positions = self.tracker.get_strategy_positions()
        if not positions:
            log.info(f"  📭 无策略持仓")
            return []

        now = datetime.now()
        exits = []

        log.info(f"\n  📋 持仓监控 ({len(positions)} 笔):")

        for pos in positions:
            ticket = pos.get('ticket', 0)
            lots = pos.get('lots', 0)
            open_price = pos.get('open_price', 0)
            current_price = pos.get('current_price', 0)
            profit = pos.get('profit', 0)

            track_key = str(ticket)
            track = self.tracker.tracking.get(track_key, {})
            strategy = track.get('strategy', 'unknown')
            entry_date_str = track.get('entry_date', now.isoformat())
            try:
                entry_date = datetime.fromisoformat(entry_date_str)
            except (ValueError, TypeError):
                entry_date = now
            hold_hours = (now - entry_date).total_seconds() / 3600
            hold_display = f"{hold_hours:.0f}h" if hold_hours < 48 else f"{hold_hours/24:.1f}天"

            pnl_pct = (current_price - open_price) / open_price * 100 if open_price > 0 else 0
            emoji = "🟢" if profit >= 0 else "🔴"
            log.info(f"    {emoji} #{ticket} {strategy}: {lots}手 @ {open_price:.2f} "
                     f"→ {current_price:.2f} ({pnl_pct:+.2f}%) ${profit:+.2f} {hold_display}")

            reason = None
            direction = track.get('direction', 'BUY')

            # 1. 策略出场信号
            if strategy in ('m15_rsi', 'm5_rsi') and hold_hours < 0.25:
                pass
            else:
                exit_sig = check_exit_signal(df, strategy, direction)
                if exit_sig:
                    reason = exit_sig

            # 2. Keltner Trailing Stop (V3 ATR Regime adaptive)
            if not reason and strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                atr = float(df.iloc[-1]['ATR']) if not pd.isna(df.iloc[-1].get('ATR', float('nan'))) else 0
                if atr > 0 and open_price > 0:
                    if direction == 'BUY':
                        float_profit = current_price - open_price
                    else:
                        float_profit = open_price - current_price

                    trail_act_mult = config.TRAILING_ACTIVATE_ATR
                    trail_dist_mult = config.TRAILING_DISTANCE_ATR
                    if config.V3_ATR_REGIME_ENABLED:
                        atr_series = df['ATR'].dropna()
                        if len(atr_series) >= 50:
                            atr_pct = (atr_series.iloc[-50:] < atr).mean()
                            if atr_pct > 0.70:
                                trail_act_mult = 0.6
                                trail_dist_mult = 0.20
                            elif atr_pct < 0.30:
                                trail_act_mult = 1.0
                                trail_dist_mult = 0.35
                    activate_threshold = atr * trail_act_mult
                    trail_distance = atr * trail_dist_mult

                    if float_profit >= activate_threshold:
                        extreme = track.get('extreme_price', current_price)
                        if direction == 'BUY':
                            extreme = max(extreme, current_price)
                            new_trail = round(extreme - trail_distance, 2)
                            old_trail = track.get('trailing_stop_price', 0)
                            trail_price = max(new_trail, old_trail)
                        else:
                            extreme = min(extreme, current_price) if extreme > 0 else current_price
                            new_trail = round(extreme + trail_distance, 2)
                            old_trail = track.get('trailing_stop_price', 0)
                            trail_price = min(new_trail, old_trail) if old_trail > 0 else new_trail

                        self.tracker.tracking[track_key]['trailing_stop_price'] = trail_price
                        self.tracker._save_tracking()

                        triggered = (direction == 'BUY' and current_price <= trail_price) or \
                                    (direction == 'SELL' and current_price >= trail_price)
                        if triggered:
                            reason = (f"📈 Trailing Stop: 浮盈${float_profit:.2f} "
                                      f"(激活阈值${activate_threshold:.2f}), "
                                      f"价格{current_price:.2f}触及追踪止盈{trail_price:.2f}")
                        else:
                            log.info(f"      📈 Trailing激活: 浮盈${float_profit:.2f} "
                                     f"追踪价{trail_price:.2f} (距离${trail_distance:.2f})")

            # 3. 时间止损
            max_hold = config.STRATEGIES.get(strategy, {}).get('max_hold_bars', 15)
            if not reason and hold_hours >= max_hold:
                reason = f"⏰ 时间止损: 持仓{hold_hours:.0f}小时 >= 上限{max_hold}小时"

            if reason:
                log.info(f"      → {reason}")
                success = self.bridge.close_order(ticket)

                trade = {
                    'action': 'CLOSE', 'ticket': ticket,
                    'strategy': strategy, 'lots': lots,
                    'open_price': open_price, 'close_price': current_price,
                    'profit': profit, 'pnl_pct': round(pnl_pct, 2),
                    'reason': reason, 'hold_hours': round(hold_hours, 1),
                    'time': now.isoformat(),
                }
                exits.append(trade)
                self.tracker.record_trade(trade)

                if success:
                    notifier.notify_close(ticket, strategy, profit, reason)
                    self.tracker.update_pnl(profit)
                    self.risk.update_daily_pnl(profit)
                    if profit < 0:
                        self.risk.add_cooldown(strategy, config.COOLDOWN_MINUTES)

                    if track_key in self.tracker.tracking:
                        del self.tracker.tracking[track_key]
                        self.tracker._save_tracking()
            else:
                log.info(f"      → 继续持有")

        return exits

    def _check_entries(self, df: pd.DataFrame, timeframe: str = 'H1',
                        sentiment_ctx: Optional[Dict] = None,
                        h1_adx: float = None) -> List[Dict]:
        """检查新入场信号"""
        current_positions = self.tracker.get_strategy_positions()
        if len(current_positions) >= config.MAX_POSITIONS:
            _missed_signals = scan_all_signals(df, timeframe, h1_adx=h1_adx)
            for _ms in _missed_signals:
                self.tracker.log_missed_signal(_ms, f"持仓已满({len(current_positions)}/{config.MAX_POSITIONS})")
            if _missed_signals:
                log.info(f"\n  📊 已持有 {len(current_positions)}/{config.MAX_POSITIONS} 笔，不再开仓 (错失{len(_missed_signals)}个信号)")
            else:
                log.info(f"\n  📊 已持有 {len(current_positions)}/{config.MAX_POSITIONS} 笔，不再开仓")
            return []

        # 盘中趋势门控
        if self.trend_meter and df is not None:
            h1_df = self.data.get_hourly_data()
            if h1_df is not None:
                self.trend_meter.update(h1_df)
            if not self.trend_meter.should_allow_entry(timeframe):
                regime = self.trend_meter.get_regime()
                score = self.trend_meter.get_score()
                log.info(f"\n  📈 {self.trend_meter.status_line()} — skip {timeframe} entries")
                return []

        current_dir = self._get_current_direction()

        sentiment_bias = None
        lot_multiplier = 1.0

        slots = config.MAX_POSITIONS - len(current_positions)
        log.info(f"\n  🔍 信号扫描 (可开 {slots} 笔):")

        signals = scan_all_signals(df, timeframe, h1_adx=h1_adx)

        if not signals:
            latest = df.iloc[-1]
            rsi2 = float(latest['RSI2']) if not pd.isna(latest['RSI2']) else 100
            if rsi2 < 20:
                log.info(f"    👀 RSI(2)={rsi2:.1f} 接近触发 (阈值<5)")
            else:
                log.info(f"    → 无信号")
            return []

        entries = []
        for sig in signals[:slots]:
            strategy = sig['strategy']
            reason = sig['reason']
            close = sig['close']

            # 冷却期检查
            if self.risk.is_in_cooldown(strategy):
                remaining = (self.risk.cooldown_until[strategy] - datetime.now()).total_seconds() / 60
                log.info(f"    ❄️ {reason} — 但{strategy}在冷却期中 (还剩{remaining:.0f}分钟)")
                self.tracker.log_missed_signal(sig, f"冷却期(还剩{remaining:.0f}分钟)")
                continue

            # 趋势耗尽熔断
            if strategy == 'keltner':
                rsi14 = float(df.iloc[-1]['RSI14']) if not pd.isna(df.iloc[-1].get('RSI14', float('nan'))) else 50
                direction = sig['signal']
                if direction == 'SELL' and rsi14 < config.KELTNER_EXHAUSTION_RSI_LOW:
                    log.info(f"    🛑 {reason} — 趋势耗尽熔断: RSI14={rsi14:.1f} < {config.KELTNER_EXHAUSTION_RSI_LOW} (超卖, 拒绝追空)")
                    self.tracker.log_missed_signal(sig, f"趋势耗尽(RSI14={rsi14:.1f}<{config.KELTNER_EXHAUSTION_RSI_LOW}, 超卖拒绝追空)")
                    continue
                if direction == 'BUY' and rsi14 > config.KELTNER_EXHAUSTION_RSI_HIGH:
                    log.info(f"    🛑 {reason} — 趋势耗尽熔断: RSI14={rsi14:.1f} > {config.KELTNER_EXHAUSTION_RSI_HIGH} (超买, 拒绝追多)")
                    self.tracker.log_missed_signal(sig, f"趋势耗尽(RSI14={rsi14:.1f}>{config.KELTNER_EXHAUSTION_RSI_HIGH}, 超买拒绝追多)")
                    continue

            # 同策略重复持仓检测
            active_strategies = set()
            for tk, track in self.tracker.tracking.items():
                active_strategies.add(track.get('strategy', ''))
            if strategy in active_strategies:
                direction = sig['signal']
                if strategy == 'keltner' and self._can_add_position(strategy, direction, df):
                    log.info(f"    🔥 {reason} — ADX强趋势+浮盈充足, 允许加仓")
                else:
                    log.info(f"    🚫 {reason} — 但{strategy}已有持仓，不重复开仓")
                    self.tracker.log_missed_signal(sig, f"同策略已持仓({strategy})")
                    continue

            # 方向冲突检测
            direction = sig['signal']
            if current_dir and direction != current_dir:
                log.info(f"    ⛔ {reason} — 但当前持仓方向为{current_dir}，跳过反向{direction}信号")
                self.tracker.log_missed_signal(sig, f"方向冲突(持仓{current_dir}, 信号{direction})")
                continue

            log.info(f"    🚀 {reason}")

            sl_pips = sig.get('sl', config.STOP_LOSS_PIPS)
            tp_pips = sig.get('tp', 0)

            if tp_pips <= 0:
                tp_pips = sl_pips * 2
                log.info(f"    🛡️ 安全止盈: ${tp_pips:.1f} (2×止损, MT4硬保护)")

            base_lots = calc_auto_lot_size(0, sl_pips)
            loss_scale = self.risk.get_lot_scale()
            actual_lots = round(base_lots * lot_multiplier * loss_scale, 2)
            actual_lots = max(config.MIN_LOT_SIZE, min(config.MAX_LOT_SIZE, actual_lots))
            if actual_lots != config.LOT_SIZE:
                log.info(f"    📊 自动调仓: 止损${sl_pips:.1f} → {actual_lots}手 (风险${sl_pips*actual_lots*config.POINT_VALUE_PER_LOT:.0f})")
            if loss_scale < 1.0:
                log.info(f"    📉 日内亏损自适应: {self.risk.daily_loss_count}笔亏损 → 仓位×{loss_scale:.1f}")
            if lot_multiplier != 1.0:
                log.info(f"    🌐 舆情仓位调整: ×{lot_multiplier:.1f}")

            if direction == 'BUY':
                success = self.bridge.buy(
                    lots=actual_lots,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    comment=f"GOLD_{strategy[:4]}",
                )
            else:
                success = self.bridge.sell(
                    lots=actual_lots,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    comment=f"GOLD_{strategy[:4]}",
                )

            if success:
                factors = self._snapshot_factors(df, strategy)
                trade = {
                    'action': 'OPEN', 'strategy': strategy,
                    'direction': direction,
                    'lots': actual_lots, 'price': close,
                    'sl_pips': sl_pips, 'tp_pips': tp_pips,
                    'reason': reason,
                    'sentiment_score': sentiment_ctx['sentiment']['score'] if sentiment_ctx else None,
                    'lot_multiplier': lot_multiplier,
                    'factors': factors,
                    'time': datetime.now().isoformat(),
                }
                entries.append(trade)
                self.tracker.record_trade(trade)

                temp_key = f"pending_{strategy}_{datetime.now().strftime('%H%M%S')}"
                self.tracker.tracking[temp_key] = {
                    'strategy': strategy,
                    'direction': direction,
                    'entry_price': close,
                    'entry_date': datetime.now().isoformat(),
                    'lots': actual_lots,
                    'sl': sl_pips,
                    'pending': True,
                    'trailing_stop_price': 0,
                    'extreme_price': close,
                    'is_pyramid': strategy in active_strategies,
                }
                self.tracker._save_tracking()

                log.info(f"    ✅ 已下单: {actual_lots}手 止损${sl_pips:.1f} 止盈${tp_pips:.1f}")
                notifier.notify_open(strategy, direction, actual_lots, close, sl_pips, reason)

        return entries

    def check_exits_only(self) -> Dict:
        """仅检查出场 (盘中监控, H1+M5双时间框架)"""
        self.tracker.sync_positions(risk_manager=self.risk)

        exits = []
        df_h1 = self.data.get_hourly_data()
        if df_h1 is not None:
            exits += self._check_exits(df_h1)
        df_m15 = self.data.get_m15_data()
        if df_m15 is not None:
            exits += self._check_exits(df_m15)
        return {"exits": exits}

    # ── 辅助方法 ──

    @staticmethod
    def _snapshot_factors(df: pd.DataFrame, strategy: str) -> Dict:
        """采集当前K线的因子快照，供 IC 分析使用"""
        row = df.iloc[-1]
        factors = {}
        for col in ('RSI14', 'RSI2', 'ADX', 'ATR', 'MACD_hist',
                     'EMA9', 'EMA21', 'EMA100', 'KC_upper', 'KC_lower',
                     'Volume', 'Vol_MA20'):
            val = row.get(col)
            if val is not None and not pd.isna(val):
                factors[col] = round(float(val), 4)

        close = float(row['Close'])
        atr = factors.get('ATR', 0)
        if atr > 0:
            kc_upper = factors.get('KC_upper', close)
            factors['kc_breakout_strength'] = round((close - kc_upper) / atr, 4)

        vol_ma = factors.get('Vol_MA20', 0)
        vol = factors.get('Volume', 0)
        if vol_ma > 0:
            factors['volume_ratio'] = round(vol / vol_ma, 4)

        atr_series = df['ATR'].dropna()
        if len(atr_series) >= 50:
            factors['atr_percentile'] = round((atr_series.iloc[-50:] < atr).mean(), 4)

        return factors

    def _get_sentiment_context(self) -> Optional[Dict]:
        """获取舆情分析结果 (失败返回None，不影响交易)"""
        if not self.sentiment:
            return None
        try:
            ctx = self.sentiment.get_trading_context()
            return ctx
        except Exception as e:
            log.warning(f"舆情分析异常 (不影响交易): {e}")
            return None

    def _get_current_direction(self) -> Optional[str]:
        """获取当前持仓方向 (BUY/SELL/None)"""
        positions = self.tracker.get_strategy_positions()
        if not positions:
            return None
        pos_type = positions[0].get('type', 0)
        return 'SELL' if pos_type == 1 else 'BUY'

    def _can_add_position(self, strategy: str, signal_direction: str, df: pd.DataFrame) -> bool:
        """检查是否允许同策略顺势加仓 (仅限Keltner强趋势)"""
        if not config.ADD_POSITION_ENABLED:
            return False

        adx = float(df.iloc[-1]['ADX']) if not pd.isna(df.iloc[-1].get('ADX', float('nan'))) else 0
        atr = float(df.iloc[-1]['ATR']) if not pd.isna(df.iloc[-1].get('ATR', float('nan'))) else 0
        if adx < config.ADD_POSITION_MIN_ADX or atr <= 0:
            return False

        same_strategy_count = 0
        existing_direction = None
        min_profit = float('inf')
        latest_entry_time = None
        latest_entry_price = None

        for tk, track in self.tracker.tracking.items():
            if track.get('strategy') == strategy:
                same_strategy_count += 1
                existing_direction = track.get('direction')
                profit = track.get('last_profit', 0)
                min_profit = min(min_profit, profit)
                try:
                    entry_t = datetime.fromisoformat(track.get('entry_date', ''))
                    if latest_entry_time is None or entry_t > latest_entry_time:
                        latest_entry_time = entry_t
                        latest_entry_price = track.get('entry_price', 0)
                except (ValueError, TypeError):
                    pass

        if same_strategy_count >= config.KELTNER_MAX_SAME_STRATEGY:
            return False
        if existing_direction and signal_direction != existing_direction:
            return False

        if latest_entry_time:
            elapsed_min = (datetime.now() - latest_entry_time).total_seconds() / 60
            if elapsed_min < config.ADD_POSITION_MIN_HOLD_MINUTES:
                log.info(f"      ⏳ 加仓冷却中: 上一笔入场仅{elapsed_min:.0f}分钟前 "
                         f"(需≥{config.ADD_POSITION_MIN_HOLD_MINUTES}分钟)")
                return False

        if latest_entry_price and latest_entry_price > 0:
            current_price = float(df.iloc[-1]['Close'])
            min_distance = atr * config.ADD_POSITION_MIN_DISTANCE_ATR
            if signal_direction == 'SELL':
                distance = latest_entry_price - current_price
            else:
                distance = current_price - latest_entry_price
            if distance < min_distance:
                log.info(f"      📏 加仓空间不足: 价格间距${distance:.2f} "
                         f"< 阈值${min_distance:.2f} (0.5×ATR)")
                return False

        profit_threshold = atr * config.ADD_POSITION_MIN_PROFIT_ATR
        lots = 0.01
        for tk, track in self.tracker.tracking.items():
            if track.get('strategy') == strategy:
                lots = track.get('lots', 0.01)
                break
        float_profit_dollars = min_profit
        float_profit_points = float_profit_dollars / (lots * config.POINT_VALUE_PER_LOT) if lots > 0 else 0

        if float_profit_points < profit_threshold:
            return False

        total_positions = len(self.tracker.get_strategy_positions())
        if total_positions >= config.MAX_POSITIONS:
            return False

        log.info(f"      ✅ 加仓条件满足: ADX={adx:.1f}≥{config.ADD_POSITION_MIN_ADX}, "
                 f"浮盈${min_profit:.2f}(≥{profit_threshold:.1f}点×手数), "
                 f"同策略{same_strategy_count}/{config.KELTNER_MAX_SAME_STRATEGY}")
        return True
