"""
持仓追踪与交易记录模块
======================
管理持仓同步、交易日志、盈亏记录、净值曲线
"""
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import config
from mt4_bridge import MT4Bridge
import notifier

log = logging.getLogger(__name__)


class PositionTracker:
    """持仓追踪与交易记录"""

    def __init__(self, bridge: MT4Bridge, data_dir: Path):
        self.bridge = bridge
        self.tracking_file = data_dir / "gold_position_tracking.json"
        self.log_file = data_dir / "gold_trade_log.json"
        self.pnl_file = data_dir / "gold_total_pnl.json"
        self.missed_signals_file = data_dir / "gold_missed_signals.json"

        self.tracking = self._load_json(self.tracking_file, {})
        self.trade_log = self._load_json(self.log_file, [])
        self.total_pnl = self._load_json(self.pnl_file, {"total_pnl": 0, "trade_count": 0})
        self.missed_signals = self._load_json(self.missed_signals_file, [])

    # ── 持仓查询 ──

    def get_strategy_positions(self) -> List[Dict]:
        """获取本策略的持仓 (通过magic number过滤)"""
        all_pos = self.bridge.get_positions()
        return [p for p in all_pos if p.get('magic') == config.MAGIC_NUMBER]

    # ── 持仓同步 ──

    def sync_positions(self, risk_manager=None):
        """同步MT4持仓与tracking记录

        Args:
            risk_manager: RiskManager 实例，用于更新日内盈亏和冷却期
        """
        mt4_positions = self.get_strategy_positions()
        mt4_tickets = {str(p['ticket']): p for p in mt4_positions}

        # 1. 更新现有持仓的实时盈亏 + 极值价格
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
            if ticket_key.startswith('pending_'):
                continue
            if ticket_key not in mt4_tickets:
                if self._ticket_already_closed(int(ticket_key)):
                    log.info(f"  ℹ️ #{ticket_key} 已有平仓记录, 仅清理tracking")
                    del self.tracking[ticket_key]
                    self._save_tracking()
                    continue

                track = self.tracking[ticket_key]
                strategy = track.get('strategy', 'unknown')
                direction = track.get('direction', 'BUY')
                entry_price = track.get('entry_price', 0)
                last_profit = track.get('last_profit', 0)
                lots = track.get('lots', config.LOT_SIZE)

                log.info(f"  ⚠️ 检测到 #{ticket_key} ({strategy}) 已被MT4平仓")
                log.info(f"     估算盈亏: ${last_profit:+.2f} (开仓价: {entry_price})")
                notifier.notify_close(int(ticket_key), strategy, last_profit, 'MT4自动平仓(止损或手动)')

                self.total_pnl['total_pnl'] = round(
                    self.total_pnl.get('total_pnl', 0) + last_profit, 2
                )
                self.total_pnl['trade_count'] = self.total_pnl.get('trade_count', 0) + 1
                self._save_pnl()

                if risk_manager:
                    risk_manager.update_daily_pnl(last_profit)
                    if last_profit < 0:
                        risk_manager.add_cooldown(strategy, config.COOLDOWN_MINUTES)

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

        # 3. 清理pending条目
        pending_keys = [k for k in self.tracking if k.startswith('pending_')]
        for pk in pending_keys:
            p_strategy = self.tracking[pk].get('strategy', '')
            matched = False
            for pos in mt4_positions:
                tk = str(pos['ticket'])
                if tk in self.tracking and self.tracking[tk].get('strategy') == p_strategy:
                    matched = True
                    break
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
                entry_str = self.tracking[pk].get('entry_date', '')
                try:
                    entry_time = datetime.fromisoformat(entry_str)
                    if (datetime.now() - entry_time).total_seconds() > 600:
                        del self.tracking[pk]
                        self._save_tracking()
                        log.warning(f"  清理过期pending: {pk} (超过10分钟未匹配)")
                except (ValueError, TypeError):
                    pass

        # 4. 检测未tracking的新仓位 (排除已有平仓记录的ticket)
        for pos in mt4_positions:
            ticket_key = str(pos['ticket'])
            if ticket_key not in self.tracking and not self._ticket_already_closed(pos['ticket']):
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

    def _ticket_already_closed(self, ticket: int) -> bool:
        """检查某 ticket 是否已有 CLOSE/CLOSE_DETECTED 记录，防止重复计算 PnL"""
        for entry in reversed(self.trade_log):
            if entry.get('ticket') == ticket and entry.get('action') in ('CLOSE', 'CLOSE_DETECTED'):
                return True
        return False

    # ── 交易记录 ──

    def record_trade(self, trade: Dict):
        """追加交易记录"""
        self.trade_log.append(trade)
        self._save_trade_log()

    def update_pnl(self, profit: float):
        """更新总盈亏"""
        self.total_pnl['total_pnl'] = round(
            self.total_pnl.get('total_pnl', 0) + profit, 2
        )
        self.total_pnl['trade_count'] = self.total_pnl.get('trade_count', 0) + 1
        self._save_pnl()

    def log_missed_signal(self, sig: Dict, filter_reason: str):
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
        if len(self.missed_signals) > 500:
            self.missed_signals = self.missed_signals[-500:]
        self._save_json(self.missed_signals_file, self.missed_signals)

    # ── 净值与舆情历史 ──

    def append_equity_record(self, daily_pnl: float, daily_loss_count: int):
        """追加今日净值曲线 + 分策略绩效"""
        equity_file = config.DATA_DIR / "equity_curve.json"
        history = self._load_json(equity_file, [])
        today = str(datetime.now().date())

        total_pnl = self.total_pnl.get("total_pnl", 0)
        equity = round(config.CAPITAL + total_pnl, 2)

        strat_stats: Dict[str, Dict] = {}
        daily_wins = 0
        daily_trade_count = 0
        for entry in self.trade_log:
            if entry.get("action") not in ("CLOSE", "CLOSE_DETECTED"):
                continue
            t = entry.get("time", "")
            if not t.startswith(today):
                continue
            daily_trade_count += 1
            profit = entry.get("profit", 0)
            if profit > 0:
                daily_wins += 1
            strat = entry.get("strategy", "unknown")
            if strat not in strat_stats:
                strat_stats[strat] = {"trades": 0, "pnl": 0.0}
            strat_stats[strat]["trades"] += 1
            strat_stats[strat]["pnl"] = round(strat_stats[strat]["pnl"] + profit, 2)

        record = {
            "date": today,
            "equity": equity,
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
            "daily_trades": daily_trade_count,
            "daily_wins": daily_wins,
            "daily_losses": daily_loss_count,
            "strategies": strat_stats,
        }

        replaced = False
        for i, r in enumerate(history):
            if r.get("date") == today:
                history[i] = record
                replaced = True
                break
        if not replaced:
            history.append(record)

        self._save_json(equity_file, history)

    def append_sentiment_history(self, daily_state: dict):
        """追加今日舆情快照到历史文件"""
        history_file = config.DATA_DIR / "sentiment_history.json"
        history = self._load_json(history_file, [])
        today = daily_state.get("date", str(datetime.now().date()))
        ms = daily_state.get("macro_sentiment", {})
        ca = daily_state.get("macro_cross_assets", {})

        record = {
            "date": today,
            "label": ms.get("label"),
            "score": ms.get("score"),
            "confidence": ms.get("confidence"),
            "keyword_score": ms.get("keyword_score"),
            "finbert_score": ms.get("finbert_score"),
            "vader_score": ms.get("vader_score"),
            "direction_bias": ms.get("direction_bias"),
            "brent_oil_price": ca.get("brent_oil_price"),
            "brent_oil_change_pct": ca.get("brent_oil_change_pct"),
            "us10y_yield_price": ca.get("us10y_yield_price"),
            "us10y_yield_change_pct": ca.get("us10y_yield_change_pct"),
        }

        replaced = False
        for i, r in enumerate(history):
            if r.get("date") == today:
                history[i] = record
                replaced = True
                break
        if not replaced:
            history.append(record)

        self._save_json(history_file, history)

    # ── 文件操作 ──

    def _save_tracking(self):
        self._save_json(self.tracking_file, self.tracking)

    def _save_trade_log(self):
        self._save_json(self.log_file, self.trade_log)

    def _save_pnl(self):
        self._save_json(self.pnl_file, self.total_pnl)

    def _load_json(self, path, default):
        if path.exists():
            with open(path) as f:
                try:
                    return json.load(f)
                except (json.JSONDecodeError, ValueError):
                    return default
        return default

    def _save_json(self, path, data):
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(Path(path).parent), suffix='.tmp')
        try:
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, str(path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
