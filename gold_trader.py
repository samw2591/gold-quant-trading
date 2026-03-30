"""
黄金量化交易主引擎
==================
管理持仓、执行交易、风控保护

职责:
1. 从MT4获取行情数据
2. 运行信号引擎检测入场/出场
3. 通过MT4桥接执行交易
4. 风险管理 (总亏损保护、仓位控制)
5. 记录交易日志
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import config
from mt4_bridge import MT4Bridge
from strategies.signals import (prepare_indicators, scan_all_signals, check_exit_signal,
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

# 策略默认止损止盈 (美元)
STRATEGY_PARAMS = {
    'keltner': {'sl': 20, 'tp': 35, 'max_bars': 15},
    'macd': {'sl': 20, 'tp': 50, 'max_bars': 20},
    'm15_rsi': {'sl': 15, 'tp': 0, 'max_bars': 12},  # RSI出场，不用固定止盈
}

log = logging.getLogger(__name__)


class GoldTrader:
    """黄金量化交易器"""

    def __init__(self):
        self.bridge = MT4Bridge()
        self.tracking_file = config.DATA_DIR / "gold_position_tracking.json"
        self.log_file = config.DATA_DIR / "gold_trade_log.json"
        self.pnl_file = config.DATA_DIR / "gold_total_pnl.json"
        self.tracking = self._load_json(self.tracking_file, {})
        self.trade_log = self._load_json(self.log_file, [])
        self.total_pnl = self._load_json(self.pnl_file, {"total_pnl": 0, "trade_count": 0})
        
        # 错失信号记录
        self.missed_signals_file = config.DATA_DIR / "gold_missed_signals.json"
        self.missed_signals = self._load_json(self.missed_signals_file, [])
        
        # 冷却期跟踪: {strategy: 上次亏损时间}
        self.cooldown_until = {}
        # 日内亏损跟踪 (从文件恢复，防止重启丢失)
        self.daily_state_file = config.DATA_DIR / "gold_daily_state.json"
        self._load_daily_state()
        
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

    def _load_json(self, path, default):
        if path.exists():
            with open(path) as f:
                try:
                    return json.load(f)
                except:
                    return default
        return default

    def _save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _save_tracking(self):
        self._save_json(self.tracking_file, self.tracking)

    def _save_trade_log(self):
        self._save_json(self.log_file, self.trade_log)

    def _log_missed_signal(self, sig: Dict, filter_reason: str):
        """记录被过滤的信号，用于复盘分析机会成本"""
        record = {
            'time': datetime.now().isoformat(),
            'strategy': sig.get('strategy', ''),
            'direction': sig.get('signal', ''),
            'price': sig.get('close', 0),
            'reason': sig.get('reason', ''),
            'sl': sig.get('sl', 0),
            'tp': sig.get('tp', 0),
            'filter_reason': filter_reason,
        }
        self.missed_signals.append(record)
        # 只保留最近500条
        if len(self.missed_signals) > 500:
            self.missed_signals = self.missed_signals[-500:]
        self._save_json(self.missed_signals_file, self.missed_signals)

    def _save_pnl(self):
        self._save_json(self.pnl_file, self.total_pnl)

    # ── 数据获取 ──

    def _read_mt4_bars(self, filename: str) -> Optional[pd.DataFrame]:
        """从MT4桥接文件读K线数据 (优先数据源)"""
        filepath = config.BRIDGE_DIR / filename
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r') as f:
                data = json.loads(f.read())
            
            bars = data.get('bars', [])
            if len(bars) < 55:
                return None
            
            rows = []
            for b in bars:
                rows.append({
                    'Open': b['o'], 'High': b['h'], 'Low': b['l'],
                    'Close': b['c'], 'Volume': b.get('v', 0),
                })
            
            df = pd.DataFrame(rows)
            # 解析时间戳作为索引
            times = [pd.Timestamp(b['t'].replace('.', '-')) for b in bars]
            df.index = pd.DatetimeIndex(times)
            df.index.name = 'Datetime'
            
            return prepare_indicators(df)
        except Exception as e:
            log.debug(f"MT4本地数据读取失败 ({filename}): {e}")
            return None

    def _get_yfinance_data(self, interval='1h', period='60d') -> Optional[pd.DataFrame]:
        """从yfinance获取数据 (fallback备用)"""
        try:
            df = yf.download('GC=F', period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close'])
            if len(df) < 55:
                return None
            return prepare_indicators(df)
        except Exception as e:
            log.debug(f"yfinance {interval}数据获取失败: {e}")
            return None

    def get_hourly_data(self) -> Optional[pd.DataFrame]:
        """获取H1数据: MT4本地 > yfinance"""
        df = self._read_mt4_bars('bars_h1.json')
        if df is not None:
            log.debug("H1数据来源: MT4本地")
            return df
        log.debug("H1数据来源: yfinance (fallback)")
        return self._get_yfinance_data('1h', '60d')
    
    def get_m15_data(self) -> Optional[pd.DataFrame]:
        """获取M15数据: MT4本地 > yfinance"""
        df = self._read_mt4_bars('bars_m15.json')
        if df is not None:
            log.debug("M15数据来源: MT4本地")
            return df
        log.debug("M15数据来源: yfinance (fallback)")
        return self._get_yfinance_data('15m', '30d')

    # ── 风控检查 ──

    def check_total_loss_limit(self) -> bool:
        """检查是否超过总亏损上限"""
        total = self.total_pnl.get('total_pnl', 0)
        if total <= -config.MAX_TOTAL_LOSS:
            log.warning(f"🛑 总亏损已达 ${total:.2f}，超过上限 ${config.MAX_TOTAL_LOSS}，停止交易")
            return True
        return False

    def get_strategy_positions(self) -> List[Dict]:
        """获取本策略的持仓 (通过magic number过滤)"""
        all_pos = self.bridge.get_positions()
        return [p for p in all_pos if p.get('magic') == config.MAGIC_NUMBER]

    # ── 核心交易逻辑 ──

    def _sync_positions_tracking(self):
        """同步MT4持仓与tracking记录
        1. 更新现有持仓的实时盈亏
        2. 已平仓单(止损/手动)自动检测并记录盈亏
        3. 新开仓单自动补录tracking
        """
        mt4_positions = self.get_strategy_positions()
        mt4_tickets = {str(p['ticket']): p for p in mt4_positions}
        
        # 1. 更新现有持仓的实时盈亏 + 极值价格 (存入tracking，平仓/trailing stop时用)
        for pos in mt4_positions:
            tk = str(pos['ticket'])
            if tk in self.tracking:
                current = pos.get('current_price', 0)
                self.tracking[tk]['last_profit'] = pos.get('profit', 0)
                self.tracking[tk]['last_price'] = current
                direction = self.tracking[tk].get('direction', 'BUY')
                old_extreme = self.tracking[tk].get('extreme_price', current)
                if direction == 'BUY':
                    self.tracking[tk]['extreme_price'] = max(old_extreme, current)
                else:
                    self.tracking[tk]['extreme_price'] = min(old_extreme, current) if old_extreme > 0 else current
        self._save_tracking()
        
        # 2. 检测已消失的持仓 (被MT4止损/手动平仓)
        tracked_tickets = list(self.tracking.keys())
        for ticket_key in tracked_tickets:
            # 跳过pending条目，它们不是真实ticket
            if ticket_key.startswith('pending_'):
                continue
            if ticket_key not in mt4_tickets:
                track = self.tracking[ticket_key]
                strategy = track.get('strategy', 'unknown')
                direction = track.get('direction', 'BUY')
                entry_price = track.get('entry_price', 0)
                last_profit = track.get('last_profit', 0)
                lots = track.get('lots', config.LOT_SIZE)
                
                log.info(f"  ⚠️ 检测到 #{ticket_key} ({strategy}) 已被MT4平仓")
                log.info(f"     估算盈亏: ${last_profit:+.2f} (开仓价: {entry_price})")
                notifier.notify_close(int(ticket_key), strategy, last_profit, 'MT4自动平仓(止损或手动)')
                
                # 更新总盈亏
                self.total_pnl['total_pnl'] = round(
                    self.total_pnl.get('total_pnl', 0) + last_profit, 2
                )
                self.total_pnl['trade_count'] = self.total_pnl.get('trade_count', 0) + 1
                self._save_pnl()
                
                # 更新日内盈亏
                self._update_daily_pnl(last_profit)
                
                # 亏损时设置冷却期
                if last_profit < 0:
                    from datetime import timedelta
                    cooldown_hours = config.COOLDOWN_BARS  # 3小时
                    self.cooldown_until[strategy] = datetime.now() + timedelta(hours=cooldown_hours)
                    log.info(f"     ❄️ {strategy} 进入冷却期，{cooldown_hours}小时后才可开仓")
                
                # 记录到交易日志
                trade = {
                    'action': 'CLOSE_DETECTED', 'ticket': int(ticket_key),
                    'strategy': strategy, 'direction': direction,
                    'entry_price': entry_price, 'lots': lots,
                    'profit': last_profit,
                    'reason': '🚨 MT4自动平仓 (止损或手动)',
                    'time': datetime.now().isoformat(),
                }
                self.trade_log.append(trade)
                self._save_trade_log()
                
                del self.tracking[ticket_key]
                self._save_tracking()
        
        # 2. 清理pending条目: 真实ticket已到位时，删除临时的pending记录
        pending_keys = [k for k in self.tracking if k.startswith('pending_')]
        for pk in pending_keys:
            p_strategy = self.tracking[pk].get('strategy', '')
            # 检查MT4是否已有该策略的真实持仓
            matched = False
            for pos in mt4_positions:
                tk = str(pos['ticket'])
                if tk in self.tracking and self.tracking[tk].get('strategy') == p_strategy:
                    matched = True
                    break
                # 或者新单还没被记录，通过comment匹配
                comment = pos.get('comment', '').lower()
                if (p_strategy == 'keltner' and 'kelt' in comment) or \
                   (p_strategy == 'orb' and 'orb' in comment) or \
                   (p_strategy == 'm15_rsi' and ('rsi' in comment or 'm15' in comment)):
                    matched = True
                    break
            if matched:
                del self.tracking[pk]
                self._save_tracking()
                log.debug(f"  清理pending: {pk} (已匹配真实持仓)")
            else:
                # 超过10分钟的pending记录，认为开仓失败，清理
                entry_str = self.tracking[pk].get('entry_date', '')
                try:
                    entry_time = datetime.fromisoformat(entry_str)
                    if (datetime.now() - entry_time).total_seconds() > 600:
                        del self.tracking[pk]
                        self._save_tracking()
                        log.warning(f"  清理过期pending: {pk} (超过10分钟未匹配)")
                except:
                    pass
        
        # 3. 检测未tracking的新仓位 (开仓后ticket未同步)
        for pos in mt4_positions:
            ticket_key = str(pos['ticket'])
            if ticket_key not in self.tracking:
                # 从comment推断策略 (comment格式: GOLD_kelt / GOLD_orb / GOLD_m15_)
                comment = pos.get('comment', '')
                strategy = 'unknown'
                cl = comment.lower()
                if 'kelt' in cl:
                    strategy = 'keltner'
                elif 'orb' in cl:
                    strategy = 'orb'
                elif 'macd' in cl:
                    strategy = 'macd'
                elif 'rsi' in cl or 'm15' in cl:
                    strategy = 'm15_rsi'
                
                if strategy == 'unknown':
                    log.warning(f"  ⚠️ 无法识别策略: ticket={ticket_key} comment='{comment}'")
                
                direction = 'SELL' if pos.get('type', 0) == 1 else 'BUY'
                
                self.tracking[ticket_key] = {
                    'strategy': strategy,
                    'direction': direction,
                    'entry_price': pos.get('open_price', 0),
                    'entry_date': pos.get('open_time', datetime.now().isoformat()),
                    'lots': pos.get('lots', 0),
                    'sl': pos.get('sl', 0),
                    'trailing_stop_price': 0,
                    'extreme_price': pos.get('open_price', 0),
                    'is_pyramid': False,
                }
                self._save_tracking()
                log.info(f"  📝 同步新仓位: #{ticket_key} {strategy} {direction} @ {pos.get('open_price', 0)}")

    def scan_and_trade(self) -> Dict:
        """完整扫描+交易流程"""
        now = datetime.now()
        log.info(f"\n{'='*60}")
        log.info(f"📊 黄金量化交易 — {now.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"{'='*60}")

        # 同步MT4持仓状态
        self._sync_positions_tracking()

        # 总亏损检查
        if self.check_total_loss_limit():
            return {"status": "stopped", "reason": "total_loss_limit"}

        # 日内亏损检查
        if self._check_daily_loss_limit():
            log.warning(f"\n{'!'*60}")
            log.warning(f"🚨 日内亏损{self.daily_loss_count}笔 (上限{config.DAILY_MAX_LOSSES}笔) | 日内PnL: ${self.daily_pnl:.2f}")
            log.warning(f"🛑 系统停止交易，明日自动恢复")
            log.warning(f"{'!'*60}")
            notifier.notify_stop_review(self.daily_pnl)
            return {"status": "STOP_REVIEW", "daily_pnl": self.daily_pnl, "daily_losses": self.daily_loss_count}

        # 获取舆情分析结果
        sentiment_ctx = self._get_sentiment_context()

        # 舆情状态落盘（merge 写入，不覆盖技术面字段）
        if sentiment_ctx:
            state = self._load_json(self.daily_state_file, {})
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
            # 跨资产宏观数据落盘 (观察模式)
            if self.macro_monitor:
                try:
                    cross_assets = self.macro_monitor.get_cross_asset_snapshot()
                    if cross_assets:
                        state["macro_cross_assets"] = cross_assets
                except Exception as e:
                    log.debug(f"跨资产数据获取失败 (不影响交易): {e}")
            self._save_json(self.daily_state_file, state)

        # 经济日历暂停检查
        if sentiment_ctx and not sentiment_ctx['trade_modifier']['allow_trading']:
            pause_reason = sentiment_ctx['calendar'].get('pause_reason', '未知原因')
            log.info(f"  🛑 舆情避险: {pause_reason}，暂停交易")
            log.info(f"{'='*60}")
            return {"status": "paused", "reason": pause_reason}

        # 获取多时间框架数据
        df_h1 = self.get_hourly_data()
        df_m15 = self.get_m15_data()
        
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

        # Step 1: 检查现有持仓出场 (H1和M5都检查)
        exits = []
        if df_h1 is not None:
            exits += self._check_exits(df_h1)
        if df_m15 is not None:
            exits += self._check_exits(df_m15)

        # Step 2: 检查新入场信号 (H1 + M5)
        entries = []
        if df_h1 is not None:
            entries += self._check_entries(df_h1, 'H1', sentiment_ctx)
        if df_m15 is not None:
            entries += self._check_entries(df_m15, 'M15', sentiment_ctx)

        total = len(exits) + len(entries)
        log.info(f"\n{'='*60}")
        if total > 0:
            log.info(f"⚡ 执行了 {total} 笔操作")
        else:
            log.info(f"✅ 无操作")
        log.info(f"  总盈亏: ${self.total_pnl.get('total_pnl', 0):.2f} / 上限: -${config.MAX_TOTAL_LOSS}")
        log.info(f"  日内盈亏: ${self.daily_pnl:.2f} | 亏损{self.daily_loss_count}/{config.DAILY_MAX_LOSSES}笔")
        
        # 打印ADX和舆情
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
        
        # 打印冷却期状态
        if self.cooldown_until:
            for strat, until in self.cooldown_until.items():
                remaining = (until - datetime.now()).total_seconds() / 60
                if remaining > 0:
                    log.info(f"  ❄️ {strat} 冷却中 (还剩{remaining:.0f}分钟)")
        log.info(f"{'='*60}")

        return {"exits": exits, "entries": entries}

    def _check_exits(self, df: pd.DataFrame) -> List[Dict]:
        """检查出场信号"""
        positions = self.get_strategy_positions()
        if not positions:
            log.info(f"  📭 无策略持仓")
            return []

        now = datetime.now()
        exits = []

        log.info(f"\n  📋 持仓监控 ({len(positions)} 笔):")

        for pos in positions:
            ticket = pos.get('ticket', 0)
            symbol = pos.get('symbol', '')
            lots = pos.get('lots', 0)
            open_price = pos.get('open_price', 0)
            current_price = pos.get('current_price', 0)
            profit = pos.get('profit', 0)
            comment = pos.get('comment', '')

            # 从tracking获取策略信息
            track_key = str(ticket)
            track = self.tracking.get(track_key, {})
            strategy = track.get('strategy', 'unknown')
            entry_date_str = track.get('entry_date', now.isoformat())
            try:
                entry_date = datetime.fromisoformat(entry_date_str)
            except:
                entry_date = now
            hold_hours = (now - entry_date).total_seconds() / 3600
            hold_display = f"{hold_hours:.0f}h" if hold_hours < 48 else f"{hold_hours/24:.1f}天"

            pnl_pct = (current_price - open_price) / open_price * 100 if open_price > 0 else 0
            emoji = "🟢" if profit >= 0 else "🔴"
            log.info(f"    {emoji} #{ticket} {strategy}: {lots}手 @ {open_price:.2f} "
                     f"→ {current_price:.2f} ({pnl_pct:+.2f}%) ${profit:+.2f} {hold_display}")

            # 出场判断
            reason = None
            direction = track.get('direction', 'BUY')

            # 1. 策略出场信号
            exit_sig = check_exit_signal(df, strategy, direction)
            if exit_sig:
                reason = exit_sig

            # 2. Keltner Trailing Stop (追踪止盈)
            if not reason and strategy == 'keltner' and config.TRAILING_STOP_ENABLED:
                atr = float(df.iloc[-1]['ATR']) if not pd.isna(df.iloc[-1].get('ATR', float('nan'))) else 0
                if atr > 0 and open_price > 0:
                    if direction == 'BUY':
                        float_profit = current_price - open_price
                    else:
                        float_profit = open_price - current_price

                    activate_threshold = atr * config.TRAILING_ACTIVATE_ATR
                    trail_distance = atr * config.TRAILING_DISTANCE_ATR

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

                        self.tracking[track_key]['trailing_stop_price'] = trail_price
                        self._save_tracking()

                        triggered = (direction == 'BUY' and current_price <= trail_price) or \
                                    (direction == 'SELL' and current_price >= trail_price)
                        if triggered:
                            reason = (f"📈 Trailing Stop: 浮盈${float_profit:.2f} "
                                      f"(激活阈值${activate_threshold:.2f}), "
                                      f"价格{current_price:.2f}触及追踪止盈{trail_price:.2f}")
                        else:
                            log.info(f"      📈 Trailing激活: 浮盈${float_profit:.2f} "
                                     f"追踪价{trail_price:.2f} (距离${trail_distance:.2f})")

            # 3. 时间止损 (max_hold_bars = H1 K线数 = 小时数)
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
                self.trade_log.append(trade)
                self._save_trade_log()

                # 更新总盈亏 + 发送平仓通知
                if success:
                    notifier.notify_close(ticket, strategy, profit, reason)

                    self.total_pnl['total_pnl'] = round(
                        self.total_pnl.get('total_pnl', 0) + profit, 2
                    )
                    self.total_pnl['trade_count'] = self.total_pnl.get('trade_count', 0) + 1
                    self._save_pnl()

                    if track_key in self.tracking:
                        del self.tracking[track_key]
                        self._save_tracking()
            else:
                log.info(f"      → 继续持有")

        return exits

    def _load_daily_state(self):
        """从文件加载日内状态，防止重启丢失"""
        state = self._load_json(self.daily_state_file, {})
        saved_date = state.get('date', '')
        today = str(datetime.now().date())
        
        if saved_date == today:
            # 同一天，恢复状态
            self.daily_pnl = state.get('pnl', 0.0)
            self.daily_loss_count = state.get('loss_count', 0)
            self.daily_date = datetime.now().date()
            log.info(f"  📊 恢复日内状态: PnL=${self.daily_pnl:.2f}, 亏损{self.daily_loss_count}/{config.DAILY_MAX_LOSSES}笔")
        else:
            # 新的一天，重置
            self.daily_pnl = 0.0
            self.daily_loss_count = 0
            self.daily_date = datetime.now().date()

    def _save_daily_state(self):
        """保存日内状态到文件"""
        state = {
            'date': str(self.daily_date),
            'pnl': self.daily_pnl,
            'loss_count': self.daily_loss_count,
        }
        self._save_json(self.daily_state_file, state)

    def _update_daily_pnl(self, profit: float):
        """更新日内盈亏跟踪"""
        today = datetime.now().date()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_loss_count = 0
            self.daily_date = today
        self.daily_pnl = round(self.daily_pnl + profit, 2)
        if profit < 0:
            self.daily_loss_count += 1
            log.info(f"     📊 日内亏损第{self.daily_loss_count}笔 (上限{config.DAILY_MAX_LOSSES}笔)")
        self._save_daily_state()

    def _check_daily_loss_limit(self) -> bool:
        """检查是否超过日内最大亏损笔数"""
        today = datetime.now().date()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_loss_count = 0
            self.daily_date = today
            self._save_daily_state()
        # 笔数限制 (主要控制)
        if self.daily_loss_count >= config.DAILY_MAX_LOSSES:
            return True
        # 金额限制 (极端保护)
        if self.daily_pnl <= -config.DAILY_MAX_LOSS:
            return True
        return False

    def _is_in_cooldown(self, strategy: str) -> bool:
        """检查策略是否在冷却期"""
        if strategy in self.cooldown_until:
            if datetime.now() < self.cooldown_until[strategy]:
                remaining = (self.cooldown_until[strategy] - datetime.now()).total_seconds() / 60
                return True
            else:
                del self.cooldown_until[strategy]  # 冷却期已过
        return False

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
        positions = self.get_strategy_positions()
        if not positions:
            return None
        # 以第一笔持仓的方向为准 (0=BUY, 1=SELL)
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

        for tk, track in self.tracking.items():
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

        # 时间冷却: 距上一笔同策略入场至少 N 分钟
        if latest_entry_time:
            elapsed_min = (datetime.now() - latest_entry_time).total_seconds() / 60
            if elapsed_min < config.ADD_POSITION_MIN_HOLD_MINUTES:
                log.info(f"      ⏳ 加仓冷却中: 上一笔入场仅{elapsed_min:.0f}分钟前 "
                         f"(需≥{config.ADD_POSITION_MIN_HOLD_MINUTES}分钟)")
                return False

        # 空间验证: 价格距上一笔入场价至少 0.5×ATR
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
        for tk, track in self.tracking.items():
            if track.get('strategy') == strategy:
                lots = track.get('lots', 0.01)
                break
        float_profit_dollars = min_profit
        float_profit_points = float_profit_dollars / (lots * config.POINT_VALUE_PER_LOT) if lots > 0 else 0

        if float_profit_points < profit_threshold:
            return False

        total_positions = len(self.get_strategy_positions())
        if total_positions >= config.MAX_POSITIONS:
            return False

        log.info(f"      ✅ 加仓条件满足: ADX={adx:.1f}≥{config.ADD_POSITION_MIN_ADX}, "
                 f"浮盈${min_profit:.2f}(≥{profit_threshold:.1f}点×手数), "
                 f"同策略{same_strategy_count}/{config.KELTNER_MAX_SAME_STRATEGY}")
        return True

    def _check_entries(self, df: pd.DataFrame, timeframe: str = 'H1',
                        sentiment_ctx: Optional[Dict] = None) -> List[Dict]:
        """检查新入场信号"""
        # 持仓数检查
        current_positions = self.get_strategy_positions()
        if len(current_positions) >= config.MAX_POSITIONS:
            # 仍然扫描信号，记录错失的机会
            _missed_signals = scan_all_signals(df, timeframe)
            for _ms in _missed_signals:
                self._log_missed_signal(_ms, f"持仓已满({len(current_positions)}/{config.MAX_POSITIONS})")
            if _missed_signals:
                log.info(f"\n  📊 已持有 {len(current_positions)}/{config.MAX_POSITIONS} 笔，不再开仓 (错失{len(_missed_signals)}个信号)")
            else:
                log.info(f"\n  📊 已持有 {len(current_positions)}/{config.MAX_POSITIONS} 笔，不再开仓")
            return []

        # 获取当前持仓方向，防止开反向仓
        current_dir = self._get_current_direction()
        
        # 舆情参数 (方向过滤已关闭，仅保留经济日历暂停 + 日志显示)
        # 原因: 舆情准确率未验证，如果<55%只会增加噪音
        sentiment_bias = None   # 已关闭: 不再用舆情过滤方向
        lot_multiplier = 1.0    # 已关闭: 不再用舆情调整仓位

        slots = config.MAX_POSITIONS - len(current_positions)
        log.info(f"\n  🔍 信号扫描 (可开 {slots} 笔):")

        # 扫描对应时间框架的策略信号
        signals = scan_all_signals(df, timeframe)

        if not signals:
            # 打印接近触发的信号
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

            # 冷却期检查: 止损后等待冷却期结束
            if self._is_in_cooldown(strategy):
                remaining = (self.cooldown_until[strategy] - datetime.now()).total_seconds() / 60
                log.info(f"    ❄️ {reason} — 但{strategy}在冷却期中 (还剩{remaining:.0f}分钟)")
                self._log_missed_signal(sig, f"冷却期(还剩{remaining:.0f}分钟)")
                continue

            # 趋势耗尽熔断: RSI14极端时拦截Keltner追涨/追跌
            if strategy == 'keltner':
                rsi14 = float(df.iloc[-1]['RSI14']) if not pd.isna(df.iloc[-1].get('RSI14', float('nan'))) else 50
                direction = sig['signal']
                if direction == 'SELL' and rsi14 < config.KELTNER_EXHAUSTION_RSI_LOW:
                    log.info(f"    🛑 {reason} — 趋势耗尽熔断: RSI14={rsi14:.1f} < {config.KELTNER_EXHAUSTION_RSI_LOW} (超卖, 拒绝追空)")
                    self._log_missed_signal(sig, f"趋势耗尽(RSI14={rsi14:.1f}<{config.KELTNER_EXHAUSTION_RSI_LOW}, 超卖拒绝追空)")
                    continue
                if direction == 'BUY' and rsi14 > config.KELTNER_EXHAUSTION_RSI_HIGH:
                    log.info(f"    🛑 {reason} — 趋势耗尽熔断: RSI14={rsi14:.1f} > {config.KELTNER_EXHAUSTION_RSI_HIGH} (超买, 拒绝追多)")
                    self._log_missed_signal(sig, f"趋势耗尽(RSI14={rsi14:.1f}>{config.KELTNER_EXHAUSTION_RSI_HIGH}, 超买拒绝追多)")
                    continue

            # 同策略重复持仓检测: 同一策略已有持仓时不再开新仓
            # 例外: Keltner 在 ADX>=28 且已有持仓浮盈充足时允许顺势加仓
            active_strategies = set()
            for tk, track in self.tracking.items():
                active_strategies.add(track.get('strategy', ''))
            if strategy in active_strategies:
                direction = sig['signal']
                if strategy == 'keltner' and self._can_add_position(strategy, direction, df):
                    log.info(f"    🔥 {reason} — ADX强趋势+浮盈充足, 允许加仓")
                else:
                    log.info(f"    🚫 {reason} — 但{strategy}已有持仓，不重复开仓")
                    self._log_missed_signal(sig, f"同策略已持仓({strategy})")
                    continue

            # 方向冲突检测: 已有持仓时不开反向仓
            direction = sig['signal']  # 'BUY' 或 'SELL'
            if current_dir and direction != current_dir:
                log.info(f"    ⛔ {reason} — 但当前持仓方向为{current_dir}，跳过反向{direction}信号")
                self._log_missed_signal(sig, f"方向冲突(持仓{current_dir}, 信号{direction})")
                continue

            # 舆情方向过滤: 已关闭 (等待准确率验证)
            # if sentiment_bias and direction != sentiment_bias:
            #     log.info(f"    🌐 {reason} — 但舆情偏好{sentiment_bias}，跳过反向{direction}信号")
            #     continue

            log.info(f"    🚀 {reason}")

            # 止损/止盈距离 (优先用信号自带的ATR止损止盈)
            sl_pips = sig.get('sl', config.STOP_LOSS_PIPS)
            tp_pips = sig.get('tp', 0)
            
            # M15 RSI策略tp=0(用RSI信号出场)，但MT4仍需要硬止盈保护
            # 设置安全止盈 = 2×止损距离 (RR 1:2)
            if tp_pips <= 0:
                tp_pips = sl_pips * 2
                log.info(f"    🛡️ 安全止盈: ${tp_pips:.1f} (2×止损, MT4硬保护)")
            
            # ATR自动调仓: 根据止损距离计算手数，保持每笔风险$100
            base_lots = calc_auto_lot_size(0, sl_pips)
            actual_lots = round(base_lots * lot_multiplier, 2)
            actual_lots = max(config.MIN_LOT_SIZE, min(config.MAX_LOT_SIZE, actual_lots))
            if actual_lots != config.LOT_SIZE:
                log.info(f"    📊 自动调仓: 止损${sl_pips:.1f} → {actual_lots}手 (风险${sl_pips*actual_lots*config.POINT_VALUE_PER_LOT:.0f})")
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
                trade = {
                    'action': 'OPEN', 'strategy': strategy,
                    'direction': direction,
                    'lots': actual_lots, 'price': close,
                    'sl_pips': sl_pips, 'tp_pips': tp_pips,
                    'reason': reason,
                    'sentiment_score': sentiment_ctx['sentiment']['score'] if sentiment_ctx else None,
                    'lot_multiplier': lot_multiplier,
                    'time': datetime.now().isoformat(),
                }
                entries.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()
                
                # 立即写入tracking，防止同次扫描或下次扫描时重复开仓
                # 用临时key，下次_sync_positions_tracking时会用真实ticket替换
                temp_key = f"pending_{strategy}_{datetime.now().strftime('%H%M%S')}"
                self.tracking[temp_key] = {
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
                self._save_tracking()

                log.info(f"    ✅ 已下单: {actual_lots}手 止损${sl_pips:.1f} 止盈${tp_pips:.1f}")
                notifier.notify_open(strategy, direction, actual_lots, close, sl_pips, reason)

        return entries

    def check_exits_only(self) -> Dict:
        """仅检查出场 (盘中监控, H1+M5双时间框架)"""
        # 先同步持仓状态，检测止损/手动平仓
        self._sync_positions_tracking()
        
        exits = []
        df_h1 = self.get_hourly_data()
        if df_h1 is not None:
            exits += self._check_exits(df_h1)
        df_m15 = self.get_m15_data()
        if df_m15 is not None:
            exits += self._check_exits(df_m15)
        return {"exits": exits}
