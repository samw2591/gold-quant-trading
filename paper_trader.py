"""
模拟盘交易系统 (Paper Trading)
================================
与实盘并行运行，测试新策略，不实际下单

功能:
1. 读取MT4实时行情数据（和实盘共享数据源）
2. 运行独立的策略逻辑
3. 虚拟下单/平仓，记录到独立的JSON文件
4. 每笔交易记录完整信息，供周报分析

使用方式:
- 在 paper_strategies/ 目录下添加新策略
- 在 PAPER_STRATEGIES 中注册
- gold_runner.py 主循环中调用 paper_trader.scan()
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import config

log = logging.getLogger(__name__)


class PaperPosition:
    """模拟持仓"""
    def __init__(self, strategy: str, direction: str, entry_price: float,
                 sl: float, tp: float, lots: float, reason: str):
        self.strategy = strategy
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.lots = lots
        self.reason = reason
        self.entry_time = datetime.now().isoformat()
        self.bars_held = 0
        self.max_favorable = 0.0  # MFE
        self.max_adverse = 0.0    # MAE

    def update(self, high: float, low: float, close: float) -> Optional[Dict]:
        """
        更新持仓状态，检查是否触发出场
        返回: 平仓信息dict 或 None(继续持有)
        """
        self.bars_held += 1

        if self.direction == 'BUY':
            pnl = close - self.entry_price
            self.max_favorable = max(self.max_favorable, high - self.entry_price)
            self.max_adverse = max(self.max_adverse, self.entry_price - low)

            # 止损
            if low <= self.entry_price - self.sl:
                return self._close(-self.sl, 'sl')
            # 止盈
            if self.tp > 0 and high >= self.entry_price + self.tp:
                return self._close(self.tp, 'tp')
        else:
            pnl = self.entry_price - close
            self.max_favorable = max(self.max_favorable, self.entry_price - low)
            self.max_adverse = max(self.max_adverse, high - self.entry_price)

            if high >= self.entry_price + self.sl:
                return self._close(-self.sl, 'sl')
            if self.tp > 0 and low <= self.entry_price - self.tp:
                return self._close(self.tp, 'tp')

        return None

    def _close(self, pnl: float, exit_reason: str) -> Dict:
        return {
            'strategy': self.strategy,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.entry_price + pnl if self.direction == 'BUY' else self.entry_price - pnl,
            'sl': self.sl,
            'tp': self.tp,
            'lots': self.lots,
            'pnl': round(pnl * self.lots * config.POINT_VALUE_PER_LOT, 2),
            'pnl_points': round(pnl, 2),
            'exit_reason': exit_reason,
            'reason': self.reason,
            'entry_time': self.entry_time,
            'exit_time': datetime.now().isoformat(),
            'bars_held': self.bars_held,
            'mfe': round(self.max_favorable, 2),
            'mae': round(self.max_adverse, 2),
        }

    def force_close(self, close: float) -> Dict:
        """超时强制平仓"""
        if self.direction == 'BUY':
            pnl = close - self.entry_price
        else:
            pnl = self.entry_price - close
        return {
            'strategy': self.strategy,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': close,
            'sl': self.sl, 'tp': self.tp, 'lots': self.lots,
            'pnl': round(pnl * self.lots * config.POINT_VALUE_PER_LOT, 2),
            'pnl_points': round(pnl, 2),
            'exit_reason': 'timeout',
            'reason': self.reason,
            'entry_time': self.entry_time,
            'exit_time': datetime.now().isoformat(),
            'bars_held': self.bars_held,
            'mfe': round(self.max_favorable, 2),
            'mae': round(self.max_adverse, 2),
        }


class PaperTrader:
    """模拟盘交易器"""

    def __init__(self):
        self.data_dir = config.DATA_DIR / "paper"
        self.data_dir.mkdir(exist_ok=True)

        self.trades_file = self.data_dir / "paper_trades.json"
        self.positions_file = self.data_dir / "paper_positions.json"
        self.state_file = self.data_dir / "paper_state.json"

        # 加载历史
        self.trades = self._load_json(self.trades_file, [])
        self.state = self._load_json(self.state_file, {
            'total_pnl': 0, 'trade_count': 0, 'wins': 0, 'losses': 0
        })

        # 活跃持仓 (内存中)
        self.positions: List[PaperPosition] = []

        # 已注册的策略
        self.strategies: Dict[str, dict] = {}

        log.info(f"📝 模拟盘启动 | 历史交易{self.state['trade_count']}笔 | 累计PnL ${self.state['total_pnl']:.2f}")

    def _load_json(self, path, default):
        try:
            if path.exists():
                with open(path) as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
        except:
            pass
        return default

    def _save_json(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def register_strategy(self, name: str, config_dict: dict):
        """
        注册一个模拟策略

        config_dict 示例:
        {
            'signal_func': my_signal_function,  # 接收df返回信号dict或None
            'exit_func': my_exit_function,      # 可选，自定义出场逻辑
            'max_hold_bars': 15,
            'max_positions': 1,                 # 该策略最大同时持仓
            'enabled': True,
        }
        """
        self.strategies[name] = config_dict
        log.info(f"📝 模拟盘注册策略: {name}")

    def scan(self, df_h1: Optional[pd.DataFrame] = None,
             df_m15: Optional[pd.DataFrame] = None):
        """
        主扫描入口，在gold_runner主循环中调用

        Args:
            df_h1: H1数据 (和实盘共享)
            df_m15: M15数据 (和实盘共享)
        """
        if not self.strategies:
            return

        # 更新现有持仓
        self._update_positions(df_h1, df_m15)

        # 扫描新信号
        self._scan_signals(df_h1, df_m15)

    def _update_positions(self, df_h1, df_m15):
        """更新持仓，检查出场"""
        to_close = []

        for pos in self.positions:
            # 选择对应时间框架的数据
            df = df_h1  # 默认用H1
            if df is None or len(df) < 5:
                continue

            latest = df.iloc[-1]
            high = float(latest['High'])
            low = float(latest['Low'])
            close = float(latest['Close'])

            # 检查SL/TP
            result = pos.update(high, low, close)
            if result:
                to_close.append((pos, result))
                continue

            # 检查超时
            strat_config = self.strategies.get(pos.strategy, {})
            max_bars = strat_config.get('max_hold_bars', 15)
            if pos.bars_held >= max_bars:
                result = pos.force_close(close)
                to_close.append((pos, result))
                continue

            # 自定义出场逻辑
            exit_func = strat_config.get('exit_func')
            if exit_func:
                exit_signal = exit_func(df, pos.direction)
                if exit_signal:
                    if pos.direction == 'BUY':
                        pnl = close - pos.entry_price
                    else:
                        pnl = pos.entry_price - close
                    result = {
                        **pos._close(pnl, 'signal_exit'),
                        'pnl': round(pnl * pos.lots * config.POINT_VALUE_PER_LOT, 2),
                        'pnl_points': round(pnl, 2),
                        'exit_reason': f'signal: {exit_signal}',
                    }
                    to_close.append((pos, result))

        # 处理平仓
        for pos, result in to_close:
            self._record_close(result)
            self.positions.remove(pos)

    def _scan_signals(self, df_h1, df_m15):
        """扫描各策略信号"""
        for name, strat_config in self.strategies.items():
            if not strat_config.get('enabled', True):
                continue

            # 检查该策略持仓上限
            max_pos = strat_config.get('max_positions', 1)
            active = sum(1 for p in self.positions if p.strategy == name)
            if active >= max_pos:
                continue

            # 调用策略信号函数
            signal_func = strat_config.get('signal_func')
            if not signal_func:
                continue

            # 传入对应时间框架
            df = df_h1 if strat_config.get('timeframe', 'H1') == 'H1' else df_m15
            if df is None or len(df) < 50:
                continue

            sig = signal_func(df)
            if sig is None:
                continue

            # 方向冲突检查
            active_dirs = set(p.direction for p in self.positions)
            if active_dirs and sig['signal'] not in active_dirs:
                continue

            # 虚拟开仓
            entry_price = float(df.iloc[-1]['Close'])
            sl = sig.get('sl', 20)
            tp = sig.get('tp', 0)
            lots = 0.01  # 模拟盘固定小手数

            pos = PaperPosition(
                strategy=name,
                direction=sig['signal'],
                entry_price=entry_price,
                sl=sl, tp=tp, lots=lots,
                reason=sig.get('reason', name),
            )
            self.positions.append(pos)

            log.info(f"  📝 [模拟] {sig['signal']} {name} @ {entry_price:.2f} "
                     f"SL={sl:.1f} TP={tp:.1f} | {sig.get('reason', '')}")

    def _record_close(self, result: Dict):
        """记录平仓"""
        result['mode'] = 'paper'
        self.trades.append(result)

        # 更新统计
        self.state['trade_count'] += 1
        self.state['total_pnl'] = round(self.state['total_pnl'] + result['pnl'], 2)
        if result['pnl'] > 0:
            self.state['wins'] += 1
        else:
            self.state['losses'] += 1

        # 保存
        self._save_json(self.trades_file, self.trades)
        self._save_json(self.state_file, self.state)

        emoji = '✅' if result['pnl'] > 0 else '❌'
        log.info(f"  📝 [模拟平仓] {emoji} {result['strategy']} {result['direction']} "
                 f"PnL=${result['pnl']:+.2f} ({result['exit_reason']}) "
                 f"持仓{result['bars_held']}根 MFE=${result['mfe']:.1f}")

    def get_summary(self) -> str:
        """获取模拟盘摘要"""
        s = self.state
        total = s['trade_count']
        if total == 0:
            return "📝 模拟盘: 暂无交易"

        winrate = 100 * s['wins'] / total if total > 0 else 0
        return (f"📝 模拟盘: {total}笔交易 | "
                f"PnL ${s['total_pnl']:+.2f} | "
                f"胜率 {winrate:.0f}%")


# ═══════════════════════════════════════════════════════════════
# 模拟策略注册区 (在这里添加新策略)
# ═══════════════════════════════════════════════════════════════

def setup_paper_strategies(paper: PaperTrader):
    """
    注册模拟盘策略。每周根据周报建议在这里添加/修改/启停策略。

    添加新策略示例:
    ─────────────────────────────
    def my_new_signal(df):
        '''返回信号dict或None'''
        close = float(df.iloc[-1]['Close'])
        rsi = float(df.iloc[-1]['RSI14'])  # 用RSI14替代RSI2
        if rsi < 30:
            return {'signal': 'BUY', 'sl': 25, 'tp': 50,
                    'reason': f'RSI14做多: RSI={rsi:.1f}<30'}
        if rsi > 70:
            return {'signal': 'SELL', 'sl': 25, 'tp': 50,
                    'reason': f'RSI14做空: RSI={rsi:.1f}>70'}
        return None

    paper.register_strategy('rsi14_test', {
        'signal_func': my_new_signal,
        'timeframe': 'H1',
        'max_hold_bars': 10,
        'max_positions': 1,
        'enabled': True,
    })
    ─────────────────────────────

    当前无活跃模拟策略，等待周报建议后添加。
    """
    pass  # 暂无策略，等周报建议
