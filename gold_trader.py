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
                                get_keltner_state_machine, get_orb_strategy, calc_auto_lot_size)

# 舆情分析模块 (安全导入，失败不影响交易)
try:
    from sentiment import SentimentEngine
    SENTIMENT_AVAILABLE = True
except Exception as _import_err:
    SENTIMENT_AVAILABLE = False
    logging.getLogger(__name__).warning(f"舆情模块导入失败: {_import_err}")

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
        
        # 冷却期跟踪: {strategy: 上次亏损时间}
        self.cooldown_until = {}
        # 日内亏损跟踪
        self.daily_pnl = 0.0
        self.daily_date = datetime.now().date()
        
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
        
        # 1. 更新现有持仓的实时盈亏 (存入tracking，平仓时用)
        for pos in mt4_positions:
            tk = str(pos['ticket'])
            if tk in self.tracking:
                self.tracking[tk]['last_profit'] = pos.get('profit', 0)
                self.tracking[tk]['last_price'] = pos.get('current_price', 0)
        self._save_tracking()
        
        # 2. 检测已消失的持仓 (被MT4止损/手动平仓)
        tracked_tickets = list(self.tracking.keys())
        for ticket_key in tracked_tickets:
            if ticket_key not in mt4_tickets:
                track = self.tracking[ticket_key]
                strategy = track.get('strategy', 'unknown')
                direction = track.get('direction', 'BUY')
                entry_price = track.get('entry_price', 0)
                last_profit = track.get('last_profit', 0)
                lots = track.get('lots', config.LOT_SIZE)
                
                log.info(f"  ⚠️ 检测到 #{ticket_key} ({strategy}) 已被MT4平仓")
                log.info(f"     估算盈亏: ${last_profit:+.2f} (开仓价: {entry_price})")
                
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
        
        # 2. 检测未tracking的新仓位 (开仓后ticket未同步)
        for pos in mt4_positions:
            ticket_key = str(pos['ticket'])
            if ticket_key not in self.tracking:
                # 从comment推断策略
                comment = pos.get('comment', '')
                strategy = 'unknown'
                if 'kelt' in comment.lower():
                    strategy = 'keltner'
                elif 'macd' in comment.lower():
                    strategy = 'macd'
                elif 'rsi' in comment.lower() or 'm15_r' in comment.lower():
                    strategy = 'm15_rsi'
                
                direction = 'SELL' if pos.get('type', 0) == 1 else 'BUY'
                
                self.tracking[ticket_key] = {
                    'strategy': strategy,
                    'direction': direction,
                    'entry_price': pos.get('open_price', 0),
                    'entry_date': pos.get('open_time', datetime.now().isoformat()),
                    'lots': pos.get('lots', 0),
                    'sl': pos.get('sl', 0),
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
            log.warning(f"🛑 日内亏损已达 ${self.daily_pnl:.2f}，超过限制 ${config.DAILY_MAX_LOSS}，暂停今日交易")
            return {"status": "daily_limit", "daily_pnl": self.daily_pnl}

        # 获取舆情分析结果
        sentiment_ctx = self._get_sentiment_context()

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
        log.info(f"  日内盈亏: ${self.daily_pnl:.2f} / 限制: -${config.DAILY_MAX_LOSS}")
        
        # 打印ADX和舆情
        if df_h1 is not None:
            adx_val = float(df_h1.iloc[-1]['ADX']) if not pd.isna(df_h1.iloc[-1].get('ADX', float('nan'))) else 0
            atr_val = float(df_h1.iloc[-1]['ATR']) if not pd.isna(df_h1.iloc[-1].get('ATR', float('nan'))) else 0
            adx_status = '趋势✅' if adx_val >= 25 else '震荡⚠️'
            sm = get_keltner_state_machine()
            orb = get_orb_strategy()
            log.info(f"  ADX={adx_val:.1f} ({adx_status})  ATR=${atr_val:.2f}  止损=${atr_val*2.5:.2f}")
            log.info(f"  🎰 Keltner状态机: {sm.get_status()}")
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
            hold_days = (now - entry_date).days

            pnl_pct = (current_price - open_price) / open_price * 100 if open_price > 0 else 0
            emoji = "🟢" if profit >= 0 else "🔴"
            log.info(f"    {emoji} #{ticket} {strategy}: {lots}手 @ {open_price:.2f} "
                     f"→ {current_price:.2f} ({pnl_pct:+.2f}%) ${profit:+.2f} {hold_days}天")

            # 出场判断
            reason = None
            direction = track.get('direction', 'BUY')

            # 1. 策略出场信号
            exit_sig = check_exit_signal(df, strategy, direction)
            if exit_sig:
                reason = exit_sig

            # 2. 时间止损
            max_hold = config.STRATEGIES.get(strategy, {}).get('max_hold_bars', 15)
            if not reason and hold_days >= max_hold:
                reason = f"⏰ 时间止损: {hold_days}天 >= {max_hold}天"

            if reason:
                log.info(f"      → {reason}")
                success = self.bridge.close_order(ticket)

                trade = {
                    'action': 'CLOSE', 'ticket': ticket,
                    'strategy': strategy, 'lots': lots,
                    'open_price': open_price, 'close_price': current_price,
                    'profit': profit, 'pnl_pct': round(pnl_pct, 2),
                    'reason': reason, 'hold_days': hold_days,
                    'time': now.isoformat(),
                }
                exits.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()

                # 更新总盈亏
                if success:
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

    def _update_daily_pnl(self, profit: float):
        """更新日内盈亏跟踪"""
        today = datetime.now().date()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_date = today
        self.daily_pnl = round(self.daily_pnl + profit, 2)

    def _check_daily_loss_limit(self) -> bool:
        """检查是否超过日内最大亏损"""
        today = datetime.now().date()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_date = today
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

    def _check_entries(self, df: pd.DataFrame, timeframe: str = 'H1',
                        sentiment_ctx: Optional[Dict] = None) -> List[Dict]:
        """检查新入场信号"""
        # 持仓数检查
        current_positions = self.get_strategy_positions()
        if len(current_positions) >= config.MAX_POSITIONS:
            log.info(f"\n  📊 已持有 {len(current_positions)}/{config.MAX_POSITIONS} 笔，不再开仓")
            return []

        # 获取当前持仓方向，防止开反向仓
        current_dir = self._get_current_direction()
        
        # 舆情修正参数
        sentiment_bias = None
        lot_multiplier = 1.0
        if sentiment_ctx:
            modifier = sentiment_ctx.get('trade_modifier', {})
            sentiment_bias = modifier.get('direction_bias')  # 'BUY'/'SELL'/None
            lot_multiplier = modifier.get('lot_multiplier', 1.0)

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
                continue

            # 方向冲突检测: 已有持仓时不开反向仓
            direction = sig['signal']  # 'BUY' 或 'SELL'
            if current_dir and direction != current_dir:
                log.info(f"    ⛔ {reason} — 但当前持仓方向为{current_dir}，跳过反向{direction}信号")
                continue

            # 舆情方向过滤: 如果舆情有明确偏好且与信号方向相反，跳过
            if sentiment_bias and direction != sentiment_bias:
                log.info(f"    🌐 {reason} — 但舆情偏好{sentiment_bias}，跳过反向{direction}信号")
                continue

            log.info(f"    🚀 {reason}")

            # 止损距离 (优先用信号自带的ATR止损)
            sl_pips = sig.get('sl', config.STOP_LOSS_PIPS)
            
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
                    comment=f"GOLD_{strategy[:4]}",
                )
            else:
                success = self.bridge.sell(
                    lots=actual_lots,
                    sl_pips=sl_pips,
                    comment=f"GOLD_{strategy[:4]}",
                )

            if success:
                trade = {
                    'action': 'OPEN', 'strategy': strategy,
                    'direction': direction,
                    'lots': actual_lots, 'price': close,
                    'sl_pips': sl_pips, 'reason': reason,
                    'sentiment_score': sentiment_ctx['sentiment']['score'] if sentiment_ctx else None,
                    'lot_multiplier': lot_multiplier,
                    'time': datetime.now().isoformat(),
                }
                entries.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()

                log.info(f"    ✅ 已下单: {actual_lots}手 止损{sl_pips}点")

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
