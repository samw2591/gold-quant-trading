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
from strategies.signals import prepare_indicators, scan_all_signals, check_exit_signal

# 策略默认止损止盈 (美元)
STRATEGY_PARAMS = {
    'keltner': {'sl': 20, 'tp': 35, 'max_bars': 15},
    'macd': {'sl': 20, 'tp': 50, 'max_bars': 20},
    'm5_rsi': {'sl': 15, 'tp': 0, 'max_bars': 12},  # RSI出场，不用固定止盈
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

    def get_data(self, interval='1h', period='60d') -> Optional[pd.DataFrame]:
        """获取黄金数据 (COMEX黄金期货GC=F)"""
        try:
            df = yf.download('GC=F', period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close'])
            if len(df) < 55:
                log.warning(f"黄金{interval}数据不足55根K线")
                return None
            return prepare_indicators(df)
        except Exception as e:
            log.error(f"获取{interval}数据失败: {e}")
            return None

    def get_hourly_data(self) -> Optional[pd.DataFrame]:
        return self.get_data('1h', '60d')
    
    def get_m5_data(self) -> Optional[pd.DataFrame]:
        return self.get_data('5m', '5d')

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

    def scan_and_trade(self) -> Dict:
        """完整扫描+交易流程"""
        now = datetime.now()
        log.info(f"\n{'='*60}")
        log.info(f"📊 黄金量化交易 — {now.strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"{'='*60}")

        # 总亏损检查
        if self.check_total_loss_limit():
            return {"status": "stopped", "reason": "total_loss_limit"}

        # 获取多时间框架数据
        df_h1 = self.get_hourly_data()
        df_m5 = self.get_m5_data()
        
        if df_h1 is None and df_m5 is None:
            return {"status": "error", "reason": "no_data"}

        if df_h1 is not None:
            latest = df_h1.iloc[-1]
            close = float(latest['Close'])
            rsi14 = float(latest['RSI14']) if not pd.isna(latest['RSI14']) else 50
            macd_h = float(latest['MACD_hist']) if not pd.isna(latest['MACD_hist']) else 0
            log.info(f"  XAU/USD H1: ${close:.2f}  RSI(14): {rsi14:.1f}  MACD: {macd_h:+.2f}")
        
        if df_m5 is not None:
            m5_latest = df_m5.iloc[-1]
            m5_rsi = float(m5_latest['RSI2']) if not pd.isna(m5_latest['RSI2']) else 50
            log.info(f"  XAU/USD M5: RSI(2): {m5_rsi:.1f}")

        # Step 1: 检查现有持仓出场 (H1和M5都检查)
        exits = []
        if df_h1 is not None:
            exits += self._check_exits(df_h1)
        if df_m5 is not None:
            exits += self._check_exits(df_m5)

        # Step 2: 检查新入场信号 (H1 + M5)
        entries = []
        if df_h1 is not None:
            entries += self._check_entries(df_h1, 'H1')
        if df_m5 is not None:
            entries += self._check_entries(df_m5, 'M5')

        total = len(exits) + len(entries)
        log.info(f"\n{'='*60}")
        if total > 0:
            log.info(f"⚡ 执行了 {total} 笔操作")
        else:
            log.info(f"✅ 无操作")
        log.info(f"  总盈亏: ${self.total_pnl.get('total_pnl', 0):.2f} / 上限: -${config.MAX_TOTAL_LOSS}")
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

            # 1. 策略出场信号
            exit_sig = check_exit_signal(df, strategy, direction)
            if exit_sig:
                reason = exit_sig

            # 2. 时间止损
            max_hold = config.STRATEGIES.get(strategy, {}).get('max_hold_bars', 15)
            if not reason and hold_days >= max_hold:
                reason = f"⏰ 时间止损: {hold_days}天 >= {max_hold}天"

            # 硬止损由MT4的SL单自动处理
            # 策略出场信号需要知道方向
            direction = track.get('direction', 'BUY')

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

    def _check_entries(self, df: pd.DataFrame, timeframe: str = 'H1') -> List[Dict]:
        """检查新入场信号"""
        # 持仓数检查
        current_positions = self.get_strategy_positions()
        if len(current_positions) >= config.MAX_POSITIONS:
            log.info(f"\n  📊 已持有 {len(current_positions)}/{config.MAX_POSITIONS} 笔，不再开仓")
            return []

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

            log.info(f"    🚀 {reason}")

            # 计算止损价
            sl_pips = config.STRATEGIES.get(strategy, {}).get('stop_loss', config.STOP_LOSS_PIPS)

            # 执行交易 (做多或做空)
            direction = sig['signal']  # 'BUY' 或 'SELL'
            sl_pips = sig.get('sl', config.STOP_LOSS_PIPS)
            
            if direction == 'BUY':
                success = self.bridge.buy(
                    lots=config.LOT_SIZE,
                    sl_pips=sl_pips,
                    comment=f"GOLD_{strategy[:4]}",
                )
            else:
                success = self.bridge.sell(
                    lots=config.LOT_SIZE,
                    sl_pips=sl_pips,
                    comment=f"GOLD_{strategy[:4]}",
                )

            if success:
                trade = {
                    'action': 'OPEN', 'strategy': strategy,
                    'direction': direction,
                    'lots': config.LOT_SIZE, 'price': close,
                    'sl_pips': sl_pips, 'reason': reason,
                    'time': datetime.now().isoformat(),
                }
                entries.append(trade)
                self.trade_log.append(trade)
                self._save_trade_log()

                # 记录到tracking (用ticket作为key，但这里还不知道ticket)
                # EA执行后会更新positions文件，下次扫描时同步
                log.info(f"    ✅ 已下单: {config.LOT_SIZE}手 止损{sl_pips}点")

        return entries

    def check_exits_only(self) -> Dict:
        """仅检查出场 (盘中监控, H1+M5双时间框架)"""
        exits = []
        df_h1 = self.get_hourly_data()
        if df_h1 is not None:
            exits += self._check_exits(df_h1)
        df_m5 = self.get_m5_data()
        if df_m5 is not None:
            exits += self._check_exits(df_m5)
        return {"exits": exits}
