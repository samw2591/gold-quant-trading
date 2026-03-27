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
        self.scaled_in = False    # 是否已加过仓
        self.is_scale_in = False  # 是否为加仓单
        self.original_sl = sl     # 原始止损(加仓前), 用于计算1R

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

        # 活跃持仓 (从文件恢复，防止重启丢失)
        self.positions: List[PaperPosition] = self._load_positions()

        # 已注册的策略
        self.strategies: Dict[str, dict] = {}
        
        # 上次信号时间跟踪，防止同一根K线重复触发
        self._last_signal_bar: Dict[str, str] = {}  # {strategy: bar_time_str}

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

    def _load_positions(self) -> List[PaperPosition]:
        """从文件恢复持仓"""
        data = self._load_json(self.positions_file, [])
        positions = []
        for p in data:
            pos = PaperPosition(
                strategy=p.get('strategy', ''),
                direction=p.get('direction', 'BUY'),
                entry_price=p.get('entry_price', 0),
                sl=p.get('sl', 20),
                tp=p.get('tp', 0),
                lots=p.get('lots', 0.01),
                reason=p.get('reason', ''),
            )
            pos.entry_time = p.get('entry_time', '')
            pos.bars_held = p.get('bars_held', 0)
            pos.max_favorable = p.get('max_favorable', 0)
            pos.max_adverse = p.get('max_adverse', 0)
            pos.scaled_in = p.get('scaled_in', False)
            pos.is_scale_in = p.get('is_scale_in', False)
            pos.original_sl = p.get('original_sl', p.get('sl', 20))
            positions.append(pos)
        if positions:
            log.info(f"  📝 恢复{len(positions)}笔模拟持仓")
        return positions

    def _save_positions(self):
        """保存当前持仓到文件"""
        data = []
        for pos in self.positions:
            data.append({
                'strategy': pos.strategy,
                'direction': pos.direction,
                'entry_price': pos.entry_price,
                'sl': pos.sl, 'tp': pos.tp,
                'lots': pos.lots,
                'reason': pos.reason,
                'entry_time': pos.entry_time,
                'bars_held': pos.bars_held,
                'max_favorable': pos.max_favorable,
                'max_adverse': pos.max_adverse,
                'scaled_in': pos.scaled_in,
                'is_scale_in': pos.is_scale_in,
                'original_sl': pos.original_sl,
            })
        self._save_json(self.positions_file, data)

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

        # ── 加仓检测 (在平仓之前) ──
        new_scale_positions = []
        for pos in self.positions:
            if pos in [p for p, _ in to_close]:
                continue  # 即将平仓的不做加仓检测

            strat_config = self.strategies.get(pos.strategy, {})
            scale_in_cfg = strat_config.get('scale_in', {})
            if not scale_in_cfg.get('enabled', False):
                continue
            if pos.is_scale_in or pos.scaled_in:
                continue

            df = df_h1 if strat_config.get('timeframe', 'H1') == 'H1' else df_m15
            if df is None or len(df) < 105:
                continue

            latest = df.iloc[-1]
            close = float(latest['Close'])
            adx = float(latest['ADX']) if not pd.isna(latest.get('ADX', float('nan'))) else 0

            # 计算浮盈(点数)
            if pos.direction == 'BUY':
                unrealized = close - pos.entry_price
            else:
                unrealized = pos.entry_price - close

            trigger_r = scale_in_cfg.get('trigger_r', 1.0)
            adx_min = scale_in_cfg.get('adx_min', 24)

            # 条件: 浮盈 >= 1R 且 ADX 仍在趋势区间
            if unrealized >= pos.original_sl * trigger_r and adx >= adx_min:
                # 检查已有加仓层数
                layers = sum(1 for p in self.positions if p.is_scale_in and p.strategy == pos.strategy)
                max_layers = scale_in_cfg.get('max_layers', 1)
                if layers >= max_layers:
                    continue

                # 首仓止损移至保本
                if scale_in_cfg.get('move_sl_to_breakeven', True):
                    pos.sl = 0  # breakeven = entry_price ± 0
                    log.info(f"  📝 [模拟加仓] 首仓止损移至保本 (原SL=${pos.original_sl:.1f})")

                pos.scaled_in = True

                # 创建加仓单
                from strategies.signals import _calc_atr_stop, _calc_atr_tp
                new_sl = _calc_atr_stop(df)
                new_tp = _calc_atr_tp(df)

                scale_pos = PaperPosition(
                    strategy=pos.strategy,
                    direction=pos.direction,
                    entry_price=close,
                    sl=new_sl,
                    tp=new_tp,
                    lots=0.01,
                    reason=f"Keltner加仓: 首仓浮盈{unrealized:.1f}>={pos.original_sl*trigger_r:.1f}(1R), ADX={adx:.1f}",
                )
                scale_pos.is_scale_in = True
                scale_pos.original_sl = new_sl
                new_scale_positions.append(scale_pos)

                log.info(f"  📝 [模拟加仓] {pos.direction} {pos.strategy} @ {close:.2f} "
                         f"SL={new_sl:.1f} TP={new_tp:.1f} | 浮盈${unrealized:.1f} ADX={adx:.1f}")

        if new_scale_positions:
            self.positions.extend(new_scale_positions)
            self._save_positions()

        # 处理平仓
        for pos, result in to_close:
            self._record_close(result, pos)
            self.positions.remove(pos)

        if to_close:
            self._save_positions()

    def _scan_signals(self, df_h1, df_m15):
        """扫描各策略信号"""
        for name, strat_config in self.strategies.items():
            if not strat_config.get('enabled', True):
                continue

            # 检查该策略持仓上限
            # 对于有 scale_in 配置的策略, _scan_signals 只负责开首仓(最多1笔)
            # 加仓由 _update_positions 中的 scale-in 逻辑处理
            scale_in_cfg = strat_config.get('scale_in', {})
            if scale_in_cfg.get('enabled', False):
                # 只计算首仓数量(非加仓单), 最多允许1笔首仓
                first_pos_count = sum(1 for p in self.positions 
                                     if p.strategy == name and not p.is_scale_in)
                if first_pos_count >= 1:
                    continue
            else:
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

            # 防止同一根K线重复触发
            bar_time_str = str(df.index[-1])
            if self._last_signal_bar.get(name) == bar_time_str:
                continue

            # 方向冲突检查
            active_dirs = set(p.direction for p in self.positions)
            if active_dirs and sig['signal'] not in active_dirs:
                continue

            # 记录触发时间
            self._last_signal_bar[name] = bar_time_str

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
            self._save_positions()

            log.info(f"  📝 [模拟] {sig['signal']} {name} @ {entry_price:.2f} "
                     f"SL={sl:.1f} TP={tp:.1f} | {sig.get('reason', '')}")

    def _record_close(self, result: Dict, pos: Optional['PaperPosition'] = None):
        """记录平仓"""
        result['mode'] = 'paper'
        if pos is not None:
            result['is_scale_in'] = pos.is_scale_in
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

    """
    # ── 策略P1: Stochastic极端 + EMA100趋势过滤 ──
    # 来源: Algomatic Trading研究, 回测Sharpe 0.85, 胜率46.3%, 年均109笔
    # 原理: Stochastic在超买/超卖区金叉/死叉 + 顺大趋势方向
    def stoch_extreme_signal(df):
        if len(df) < 105:
            return None
        row = df.iloc[-1]; prev = df.iloc[-2]
        
        # 计算Stochastic
        low14 = df['Low'].iloc[-14:].min()
        high14 = df['High'].iloc[-14:].max()
        stk = 100 * (float(row['Close']) - low14) / (high14 - low14) if high14 != low14 else 50
        
        low14_p = df['Low'].iloc[-15:-1].min()
        high14_p = df['High'].iloc[-15:-1].max()
        pstk = 100 * (float(prev['Close']) - low14_p) / (high14_p - low14_p) if high14_p != low14_p else 50
        
        # 简化: 用K线方向代替Stoch_D交叉
        ema100 = float(row['EMA100'])
        close = float(row['Close'])
        atr = float(row['ATR'])
        
        if any(pd.isna(v) for v in [ema100, atr]):
            return None
        
        sl = max(10, min(40, round(atr * 2.0, 2)))
        tp = round(atr * 3.0, 2)
        
        # 做多: Stoch从超卖区回升 + 顺势
        if stk > 20 and pstk <= 20 and close > ema100:
            return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                    'reason': f'P1 Stoch做多: K={stk:.0f}从超卖回升, 价>{ema100:.0f}'}
        # 做空: Stoch从超买区回落 + 顺势
        if stk < 80 and pstk >= 80 and close < ema100:
            return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                    'reason': f'P1 Stoch做空: K={stk:.0f}从超买回落, 价<{ema100:.0f}'}
        return None

    paper.register_strategy('P1_stoch_extreme', {
        'signal_func': stoch_extreme_signal,
        'timeframe': 'H1',
        'max_hold_bars': 12,
        'max_positions': 1,
        'enabled': True,
    })

    # ── 策略P2: London-NY重叠时段动量 ──
    # 自研策略, 回测Sharpe 0.66, 胜率49.3%
    # 原理: 伦敦和NY重叠时段(UTC 13-16)流动性最高, 顺EMA+MACD方向做动量
    def london_ny_signal(df):
        if len(df) < 105:
            return None
        bar_time = df.index[-1]
        hour = bar_time.hour
        if hour < 13 or hour > 16:
            return None
        
        row = df.iloc[-1]
        close = float(row['Close'])
        ema9 = float(row['EMA9'])
        ema21 = float(row['EMA21'])
        adx = float(row['ADX'])
        atr = float(row['ATR'])
        macd_h = float(row['MACD_hist'])
        
        if any(pd.isna(v) for v in [ema9, ema21, adx, atr, macd_h]):
            return None
        if adx < 20:
            return None
        
        sl = max(10, min(35, round(atr * 1.5, 2)))
        tp = round(atr * 3.0, 2)
        
        if ema9 > ema21 and macd_h > 0 and close > ema21:
            return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                    'reason': f'P2 LN动量做多: EMA9>{ema21:.0f}, MACD>0, ADX={adx:.0f}'}
        if ema9 < ema21 and macd_h < 0 and close < ema21:
            return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                    'reason': f'P2 LN动量做空: EMA9<{ema21:.0f}, MACD<0, ADX={adx:.0f}'}
        return None

    paper.register_strategy('P2_london_ny', {
        'signal_func': london_ny_signal,
        'timeframe': 'H1',
        'max_hold_bars': 8,
        'max_positions': 1,
        'enabled': True,
    })

    # ── 策略P3: Keltner分层加仓 ──
    # 测试目的: 验证在ADX 26-29趋势段，浮盈后加仓是否能提升收益
    # 规则: 首仓浮盈>=1R + ADX>=24 → 首仓移保本 + 加仓1层
    from strategies.signals import check_keltner_signal, ADX_TREND_THRESHOLD

    paper.register_strategy('P3_keltner_scalein', {
        'signal_func': check_keltner_signal,
        'timeframe': 'H1',
        'max_hold_bars': 15,
        'max_positions': 2,   # 首仓+加仓 = 最多2层
        'enabled': True,
        'scale_in': {
            'enabled': True,
            'trigger_r': 1.0,                  # 浮盈>=1R触发
            'adx_min': ADX_TREND_THRESHOLD,    # ADX>=24
            'max_layers': 1,                   # 最多加仓1次
            'move_sl_to_breakeven': True,      # 加仓时首仓移保本
        },
    })
