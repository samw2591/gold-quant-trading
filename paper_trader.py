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
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import config
from strategies.exit_logic import check_time_decay_tp

log = logging.getLogger(__name__)


class PaperPosition:
    """模拟持仓"""
    def __init__(self, strategy: str, direction: str, entry_price: float,
                 sl: float, tp: float, lots: float, reason: str,
                 factors: Optional[Dict] = None,
                 trailing_activate: float = 0, trailing_distance: float = 0,
                 point_value: float = 0, symbol: str = ""):
        self.strategy = strategy
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.lots = lots
        self.reason = reason
        self.factors = factors or {}
        self.entry_time = datetime.now().isoformat()
        self.bars_held = 0
        self.max_favorable = 0.0  # MFE
        self.max_adverse = 0.0    # MAE
        self.trailing_activate = trailing_activate
        self.trailing_distance = trailing_distance
        self.trailing_active = False
        self.trailing_stop_price = 0.0
        self.point_value = point_value or config.POINT_VALUE_PER_LOT
        self.symbol = symbol or config.SYMBOL

    def update(self, high: float, low: float, close: float) -> Optional[Dict]:
        """
        更新持仓状态，检查是否触发出场
        返回: 平仓信息dict 或 None(继续持有)
        """
        self.bars_held += 1

        if self.direction == 'BUY':
            self.max_favorable = max(self.max_favorable, high - self.entry_price)
            self.max_adverse = max(self.max_adverse, self.entry_price - low)

            if low <= self.entry_price - self.sl:
                return self._close(-self.sl, 'sl')
            if self.tp > 0 and high >= self.entry_price + self.tp:
                return self._close(self.tp, 'tp')

            # Trailing stop
            if self.trailing_activate > 0 and self.trailing_distance > 0:
                if not self.trailing_active and self.max_favorable >= self.trailing_activate:
                    self.trailing_active = True
                    self.trailing_stop_price = high - self.trailing_distance
                if self.trailing_active:
                    self.trailing_stop_price = max(self.trailing_stop_price, high - self.trailing_distance)
                    if low <= self.trailing_stop_price:
                        pnl = self.trailing_stop_price - self.entry_price
                        return self._close(pnl, 'trailing')
        else:
            self.max_favorable = max(self.max_favorable, self.entry_price - low)
            self.max_adverse = max(self.max_adverse, high - self.entry_price)

            if high >= self.entry_price + self.sl:
                return self._close(-self.sl, 'sl')
            if self.tp > 0 and low <= self.entry_price - self.tp:
                return self._close(self.tp, 'tp')

            # Trailing stop
            if self.trailing_activate > 0 and self.trailing_distance > 0:
                if not self.trailing_active and self.max_favorable >= self.trailing_activate:
                    self.trailing_active = True
                    self.trailing_stop_price = low + self.trailing_distance
                if self.trailing_active:
                    self.trailing_stop_price = min(self.trailing_stop_price, low + self.trailing_distance)
                    if high >= self.trailing_stop_price:
                        pnl = self.entry_price - self.trailing_stop_price
                        return self._close(pnl, 'trailing')

        return None

    def _close(self, pnl: float, exit_reason: str) -> Dict:
        return {
            'strategy': self.strategy,
            'direction': self.direction,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'exit_price': self.entry_price + pnl if self.direction == 'BUY' else self.entry_price - pnl,
            'sl': self.sl,
            'tp': self.tp,
            'lots': self.lots,
            'pnl': round(pnl * self.lots * self.point_value, 2),
            'pnl_points': round(pnl, 2),
            'exit_reason': exit_reason,
            'reason': self.reason,
            'factors': self.factors,
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
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'exit_price': close,
            'sl': self.sl, 'tp': self.tp, 'lots': self.lots,
            'pnl': round(pnl * self.lots * self.point_value, 2),
            'pnl_points': round(pnl, 2),
            'exit_reason': 'timeout',
            'reason': self.reason,
            'factors': self.factors,
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
        except (json.JSONDecodeError, ValueError, OSError):
            pass
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
                factors=p.get('factors', {}),
                trailing_activate=p.get('trailing_activate', 0),
                trailing_distance=p.get('trailing_distance', 0),
                point_value=p.get('point_value', config.POINT_VALUE_PER_LOT),
                symbol=p.get('symbol', config.SYMBOL),
            )
            pos.entry_time = p.get('entry_time', '')
            pos.bars_held = p.get('bars_held', 0)
            pos.max_favorable = p.get('max_favorable', 0)
            pos.max_adverse = p.get('max_adverse', 0)
            pos.trailing_active = p.get('trailing_active', False)
            pos.trailing_stop_price = p.get('trailing_stop_price', 0.0)
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
                'symbol': pos.symbol,
                'entry_price': pos.entry_price,
                'sl': pos.sl, 'tp': pos.tp,
                'lots': pos.lots,
                'reason': pos.reason,
                'factors': pos.factors,
                'entry_time': pos.entry_time,
                'bars_held': pos.bars_held,
                'max_favorable': pos.max_favorable,
                'max_adverse': pos.max_adverse,
                'trailing_activate': pos.trailing_activate,
                'trailing_distance': pos.trailing_distance,
                'trailing_active': pos.trailing_active,
                'trailing_stop_price': pos.trailing_stop_price,
                'point_value': pos.point_value,
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
             df_m15: Optional[pd.DataFrame] = None,
             extra_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        主扫描入口，在gold_runner主循环中调用

        Args:
            df_h1: H1数据 (黄金, 和实盘共享)
            df_m15: M15数据 (黄金, 和实盘共享)
            extra_data: 其他品种数据 {"EURUSD_H1": df, ...}
        """
        if not self.strategies:
            return

        self._extra_data = extra_data or {}

        # 更新现有持仓
        self._update_positions(df_h1, df_m15)

        # 扫描新信号
        self._scan_signals(df_h1, df_m15)

    def _get_df_for_strategy(self, strategy_name, df_h1, df_m15):
        """根据策略配置返回对应的 DataFrame"""
        strat_config = self.strategies.get(strategy_name, {})
        data_key = strat_config.get('data_key')
        if data_key and data_key in self._extra_data:
            return self._extra_data[data_key]
        tf = strat_config.get('timeframe', 'H1')
        return df_h1 if tf == 'H1' else df_m15

    def _update_positions(self, df_h1, df_m15):
        """更新持仓，检查出场"""
        to_close = []

        for pos in self.positions:
            df = self._get_df_for_strategy(pos.strategy, df_h1, df_m15)
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

            # 时间衰减止盈（与实盘共享逻辑）
            strat_config = self.strategies.get(pos.strategy, {})
            if strat_config.get('time_decay_tp', False):
                atr_val = float(latest.get('ATR', 0)) if not pd.isna(latest.get('ATR', 0)) else 0
                td_reason = check_time_decay_tp(
                    direction=pos.direction,
                    current_price=close,
                    open_price=pos.entry_price,
                    hold_hours=pos.bars_held,
                    atr=atr_val,
                    trailing_active=pos.trailing_active,
                )
                if td_reason:
                    if pos.direction == 'BUY':
                        pnl = close - pos.entry_price
                    else:
                        pnl = pos.entry_price - close
                    result = {
                        **pos._close(pnl, 'time_decay_tp'),
                        'pnl': round(pnl * pos.lots * pos.point_value, 2),
                        'pnl_points': round(pnl, 2),
                    }
                    to_close.append((pos, result))
                    continue

            # 检查超时
            max_bars = strat_config.get('max_hold_bars', 15)
            if pos.bars_held >= max_bars:
                result = pos.force_close(close)
                to_close.append((pos, result))
                continue

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
                        'pnl': round(pnl * pos.lots * pos.point_value, 2),
                        'pnl_points': round(pnl, 2),
                        'exit_reason': f'signal: {exit_signal}',
                    }
                    to_close.append((pos, result))

        # 处理平仓
        for pos, result in to_close:
            self._record_close(result)
            self.positions.remove(pos)
        
        if to_close:
            self._save_positions()

    def _scan_signals(self, df_h1, df_m15):
        """扫描各策略信号"""
        for name, strat_config in self.strategies.items():
            if not strat_config.get('enabled', True):
                continue

            max_pos = strat_config.get('max_positions', 1)
            active = sum(1 for p in self.positions if p.strategy == name)
            if active >= max_pos:
                continue

            signal_func = strat_config.get('signal_func')
            if not signal_func:
                continue

            df = self._get_df_for_strategy(name, df_h1, df_m15)
            if df is None or len(df) < 50:
                continue

            sig = signal_func(df)
            if sig is None:
                continue

            bar_time_str = str(df.index[-1])
            if self._last_signal_bar.get(name) == bar_time_str:
                continue

            # 方向冲突检查（仅限同品种）
            strat_symbol = strat_config.get('symbol', config.SYMBOL)
            active_dirs = set(p.direction for p in self.positions if p.symbol == strat_symbol)
            if active_dirs and sig['signal'] not in active_dirs:
                continue

            self._last_signal_bar[name] = bar_time_str

            entry_price = float(df.iloc[-1]['Close'])
            sl = sig.get('sl', 20)
            tp = sig.get('tp', 0)
            lots = strat_config.get('lots', 0.01)
            pv = strat_config.get('point_value', config.POINT_VALUE_PER_LOT)

            factors = self._snapshot_factors(df, name)

            pos = PaperPosition(
                strategy=name,
                direction=sig['signal'],
                entry_price=entry_price,
                sl=sl, tp=tp, lots=lots,
                reason=sig.get('reason', name),
                factors=factors,
                trailing_activate=sig.get('trailing_activate', 0),
                trailing_distance=sig.get('trailing_distance', 0),
                point_value=pv,
                symbol=strat_symbol,
            )
            self.positions.append(pos)
            self._save_positions()

            sym_tag = f"[{strat_symbol}] " if strat_symbol != config.SYMBOL else ""
            log.info(f"  📝 [模拟] {sym_tag}{sig['signal']} {name} @ {entry_price:.5f} "
                     f"SL={sl:.5f} TP={tp:.5f} | {sig.get('reason', '')}")

    @staticmethod
    def _snapshot_factors(df: pd.DataFrame, strategy: str) -> Dict:
        """采集当前K线的因子快照，供 IC 分析使用"""
        row = df.iloc[-1]
        factors: Dict = {}
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
            factors['atr_percentile'] = round(float((atr_series.iloc[-50:] < atr).mean()), 4)

        return factors

    def _record_close(self, result: Dict):
        """记录平仓"""
        result['mode'] = 'paper'
        self.trades.append(result)

        # 更新总统计
        self.state['trade_count'] += 1
        self.state['total_pnl'] = round(self.state['total_pnl'] + result['pnl'], 2)
        if result['pnl'] > 0:
            self.state['wins'] += 1
        else:
            self.state['losses'] += 1

        # 更新分策略统计
        strat = result.get('strategy', '?')
        by_strat = self.state.setdefault('by_strategy', {})
        s = by_strat.setdefault(strat, {'n': 0, 'pnl': 0.0, 'wins': 0, 'losses': 0})
        s['n'] += 1
        s['pnl'] = round(s['pnl'] + result['pnl'], 2)
        if result['pnl'] > 0:
            s['wins'] += 1
        else:
            s['losses'] += 1

        # 保存
        self._save_json(self.trades_file, self.trades)
        self._save_json(self.state_file, self.state)

        emoji = '✅' if result['pnl'] > 0 else '❌'
        log.info(f"  📝 [模拟平仓] {emoji} {result['strategy']} {result['direction']} "
                 f"PnL=${result['pnl']:+.2f} ({result['exit_reason']}) "
                 f"持仓{result['bars_held']}根 MFE=${result['mfe']:.1f}")

    def get_summary(self) -> str:
        """获取模拟盘摘要（分策略统计）"""
        s = self.state
        total = s['trade_count']
        if total == 0:
            return "📝 模拟盘: 暂无交易"

        by_strat: Dict[str, Dict] = {}
        for t in self.trades:
            name = t.get('strategy', '?')
            if name not in by_strat:
                by_strat[name] = {'n': 0, 'pnl': 0.0, 'wins': 0}
            by_strat[name]['n'] += 1
            by_strat[name]['pnl'] += t.get('pnl', 0)
            if t.get('pnl', 0) > 0:
                by_strat[name]['wins'] += 1

        lines = [f"📝 模拟盘汇总: {total}笔 PnL ${s['total_pnl']:+.2f}"]
        for name, st in sorted(by_strat.items()):
            wr = 100 * st['wins'] / st['n'] if st['n'] > 0 else 0
            lines.append(f"  {name}: {st['n']}笔 ${st['pnl']:+.2f} 胜率{wr:.0f}%")
        return "\n".join(lines)


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
    # ── 策略P3: 周五持仓过周末 (模拟盘) ──
    # 回测: Sharpe 0.96 (ADX>24), 胜率44.2%, 但平均盈亏+$2.6/笔
    # 原理: 周五收盘前顺趋势方向开仓，赌周末事件不会反转趋势
    def friday_hold_signal(df):
        if len(df) < 105:
            return None
        bar_time = df.index[-1]
        # 只在周五UTC 19-21点触发 (SGT周六凌晨3-5点, 接近收盘)
        if bar_time.weekday() != 4 or bar_time.hour < 19:
            return None
        
        row = df.iloc[-1]
        close = float(row['Close'])
        ema100 = float(row['EMA100'])
        adx = float(row['ADX'])
        atr = float(row['ATR'])
        
        if any(pd.isna(v) for v in [ema100, adx, atr]):
            return None
        if adx < 24:
            return None
        
        sl = max(15, min(50, round(atr * 2.0, 2)))
        tp = round(atr * 3.0, 2)
        
        if close > ema100:
            return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                    'reason': f'P3 周末持仓做多: 趋势向上(>{ema100:.0f}), ADX={adx:.0f}'}
        else:
            return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                    'reason': f'P3 周末持仓做空: 趋势向下(<{ema100:.0f}), ADX={adx:.0f}'}

    paper.register_strategy('P3_friday_hold', {
        'signal_func': friday_hold_signal,
        'timeframe': 'H1',
        'max_hold_bars': 48,  # 持仓到周一 (~48小时)
        'max_positions': 1,
        'enabled': True,
    })

    # ── 策略P4: ATR Regime 自适应 ──
    # 根据ATR百分位区分波动率环境，低波用RSI均值回归，正常波动用动量，高波跳过
    def atr_regime_signal(df):
        if len(df) < 105:
            return None
        row = df.iloc[-1]
        close = float(row['Close'])
        atr = float(row['ATR'])
        rsi14 = float(row['RSI14'])
        ema9 = float(row['EMA9'])
        ema21 = float(row['EMA21'])
        ema100 = float(row['EMA100'])
        macd_h = float(row['MACD_hist'])

        if any(pd.isna(v) for v in [atr, rsi14, ema9, ema21, ema100, macd_h]):
            return None

        atr_series = df['ATR'].dropna()
        if len(atr_series) < 50:
            return None
        atr_pct = (atr_series.iloc[-50:] < atr).mean()

        if atr_pct > 0.70:
            return None

        if atr_pct < 0.30:
            sl = max(8, min(25, round(atr * 2.0, 2)))
            tp = round(atr * 2.0, 2)
            if rsi14 < 30 and close > ema100:
                return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                        'reason': f'P4 低波RSI做多: RSI={rsi14:.0f}<30, ATR%={atr_pct:.0%}'}
            if rsi14 > 70 and close < ema100:
                return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                        'reason': f'P4 低波RSI做空: RSI={rsi14:.0f}>70, ATR%={atr_pct:.0%}'}
        else:
            sl = max(10, min(35, round(atr * 2.0, 2)))
            tp = round(atr * 2.5, 2)
            if ema9 > ema21 and macd_h > 0 and close > ema100:
                return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                        'reason': f'P4 正常动量做多: EMA9>21, MACD>0, ATR%={atr_pct:.0%}'}
            if ema9 < ema21 and macd_h < 0 and close < ema100:
                return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                        'reason': f'P4 正常动量做空: EMA9<21, MACD<0, ATR%={atr_pct:.0%}'}
        return None

    paper.register_strategy('P4_atr_regime', {
        'signal_func': atr_regime_signal,
        'timeframe': 'H1',
        'max_hold_bars': 12,
        'max_positions': 1,
        'enabled': True,
    })

    # ── 策略P5: Volume确认突破 ──
    # Keltner通道突破 + 成交量放大确认，过滤低量假突破
    def volume_breakout_signal(df):
        if len(df) < 105:
            return None
        row = df.iloc[-1]
        close = float(row['Close'])
        kc_upper = float(row['KC_upper'])
        kc_lower = float(row['KC_lower'])
        adx = float(row['ADX'])
        atr = float(row['ATR'])
        ema100 = float(row['EMA100'])
        volume = float(row['Volume'])
        vol_ma = float(row['Vol_MA20'])

        if any(pd.isna(v) for v in [kc_upper, kc_lower, adx, atr, ema100, vol_ma]):
            return None
        if adx < 22:
            return None
        if vol_ma <= 0:
            return None
        if volume < vol_ma * 1.2:
            return None

        sl = max(10, min(45, round(atr * 2.5, 2)))
        tp = round(atr * 3.0, 2)
        vol_ratio = volume / vol_ma

        if close > kc_upper and close > ema100:
            return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                    'reason': f'P5 放量突破做多: 破KC上轨, Vol={vol_ratio:.1f}x均量, ADX={adx:.0f}'}
        if close < kc_lower and close < ema100:
            return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                    'reason': f'P5 放量突破做空: 破KC下轨, Vol={vol_ratio:.1f}x均量, ADX={adx:.0f}'}
        return None

    paper.register_strategy('P5_volume_breakout', {
        'signal_func': volume_breakout_signal,
        'timeframe': 'H1',
        'max_hold_bars': 15,
        'max_positions': 1,
        'enabled': True,
    })

    # ── 策略P6: DXY过滤动量 ──
    # 伦敦-纽约时段动量 + DXY负相关过滤 + 降低TP到2xATR
    _dxy_cache = {'data': None, 'ts': 0}

    def _get_dxy_change():
        """获取DXY日涨跌幅，10分钟缓存"""
        import time
        now = time.monotonic()
        if _dxy_cache['data'] is not None and (now - _dxy_cache['ts']) < 600:
            return _dxy_cache['data']
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            end = datetime.now()
            start = end - timedelta(days=5)
            data = yf.download('DX-Y.NYB', start=start.strftime('%Y-%m-%d'),
                               end=end.strftime('%Y-%m-%d'), progress=False)
            if data.empty or len(data) < 2:
                return None
            if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                data.columns = data.columns.droplevel(1)
            latest = float(data['Close'].iloc[-1])
            prev = float(data['Close'].iloc[-2])
            change_pct = (latest - prev) / prev * 100
            _dxy_cache['data'] = round(change_pct, 3)
            _dxy_cache['ts'] = now
            return _dxy_cache['data']
        except Exception:
            return None

    def dxy_filtered_signal(df):
        if len(df) < 105:
            return None
        bar_time = df.index[-1]
        hour = bar_time.hour if hasattr(bar_time, 'hour') else -1
        if hour < 12 or hour > 20:
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
        tp = round(atr * 2.0, 2)

        dxy_chg = _get_dxy_change()

        if ema9 > ema21 and macd_h > 0 and close > ema21:
            if dxy_chg is not None and dxy_chg > 0.05:
                return None
            dxy_info = f', DXY={dxy_chg:+.2f}%' if dxy_chg is not None else ''
            return {'signal': 'BUY', 'sl': sl, 'tp': tp,
                    'reason': f'P6 DXY做多: EMA9>{ema21:.0f}, MACD>0{dxy_info}'}
        if ema9 < ema21 and macd_h < 0 and close < ema21:
            if dxy_chg is not None and dxy_chg < -0.05:
                return None
            dxy_info = f', DXY={dxy_chg:+.2f}%' if dxy_chg is not None else ''
            return {'signal': 'SELL', 'sl': sl, 'tp': tp,
                    'reason': f'P6 DXY做空: EMA9<{ema21:.0f}, MACD<0{dxy_info}'}
        return None

    paper.register_strategy('P6_dxy_filtered', {
        'signal_func': dxy_filtered_signal,
        'timeframe': 'H1',
        'max_hold_bars': 8,
        'max_positions': 1,
        'enabled': True,
    })

    # ── 策略P7: Mega Trail (T0.5/D0.15) ──
    # EXP32 冠军: 与实盘相同的 Keltner 信号，但追踪止盈更紧更快
    # 回测 11 年: Sharpe 7.66 vs 当前 4.41, K-Fold 6/6 折全赢, 逐年 10/12 赢
    # 核心差异: Trail Activate 0.8→0.5 ATR, Trail Distance 0.25→0.15 ATR
    from strategies.signals import check_keltner_signal

    def mega_trail_signal(df):
        if len(df) < 105:
            return None
        row = df.iloc[-1]
        atr = float(row.get('ATR', 0))
        if pd.isna(atr) or atr <= 0:
            return None

        sig = check_keltner_signal(df)
        if not sig:
            return None

        direction = sig.get('signal', '')
        if direction not in ('BUY', 'SELL'):
            return None

        sl = round(atr * 4.5, 2)
        tp = round(atr * 8.0, 2)

        # V3 ATR Regime: adjust trailing by volatility
        atr_series = df['ATR'].dropna()
        atr_pct = 0.5
        if len(atr_series) >= 50:
            atr_pct = float((atr_series.iloc[-50:] < atr).mean())

        if atr_pct < 0.30:
            trail_act = round(atr * 0.7, 2)
            trail_dist = round(atr * 0.25, 2)
        elif atr_pct > 0.70:
            trail_act = round(atr * 0.4, 2)
            trail_dist = round(atr * 0.10, 2)
        else:
            trail_act = round(atr * 0.5, 2)
            trail_dist = round(atr * 0.15, 2)

        return {
            'signal': direction,
            'sl': sl, 'tp': tp,
            'trailing_activate': trail_act,
            'trailing_distance': trail_dist,
            'reason': f"P7 Mega: {sig.get('reason', direction)} T={trail_act:.1f}/D={trail_dist:.1f} ATR%={atr_pct:.0%}",
        }

    paper.register_strategy('P7_mega_trail', {
        'signal_func': mega_trail_signal,
        'timeframe': 'H1',
        'max_hold_bars': 15,
        'max_positions': 2,
        'enabled': True,
        'time_decay_tp': True,
    })

    # ── 策略P8: Mega Trail + 短持仓 (H20 = 5h) ──
    # EXP33 冠军组合: Mega T0.5/D0.15 + Hold=20 bars
    # K-Fold 6/6 折全赢, Avg Sharpe 8.25 vs P7的7.47
    # 与 P7 同信号同 trailing，仅 max_hold 从 15 bars 缩短到 5 bars
    # 对比目标: 短持仓是否在实盘中也能减少 SL 损失
    paper.register_strategy('P8_mega_h20', {
        'signal_func': mega_trail_signal,
        'timeframe': 'H1',
        'max_hold_bars': 5,
        'max_positions': 2,
        'enabled': True,
        'time_decay_tp': True,
    })

    # ══════════════════════════════════════════════════════════════
    # EUR/USD 策略 — 第二量化品类
    # 数据通过 extra_data["EURUSD_H1"] 传入
    # PnL: 1 标准手 1 pip = $10, 所以 point_value = 100,000
    #       (price_diff * lots * 100000 = pnl in USD)
    # ══════════════════════════════════════════════════════════════

    EURUSD_POINT_VALUE = 100_000  # 1 standard lot = 100,000 units
    EURUSD_SYMBOL = "EURUSD.mx"

    def _calc_eurusd_indicators(df):
        """Calculate indicators for EUR/USD with KC mult=2.0 (optimized params)."""
        if 'KC_mid' in df.columns:
            return df
        df = df.copy()
        df['EMA100'] = df['Close'].ewm(span=100).mean()
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA21'] = df['Close'].ewm(span=21).mean()
        df['SMA50'] = df['Close'].rolling(50).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        df['KC_mid'] = df['Close'].ewm(span=25).mean()
        df['KC_upper'] = df['KC_mid'] + 2.0 * df['ATR']
        df['KC_lower'] = df['KC_mid'] - 2.0 * df['ATR']

        # RSI14
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss
        df['RSI14'] = 100 - (100 / (1 + rs))

        # ADX
        high, low, close = df['High'], df['Low'], df['Close']
        plus_dm = high.diff()
        minus_dm = low.diff().apply(lambda x: -x)
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr_14 = tr.ewm(span=14, min_periods=14).mean()
        plus_di = 100 * plus_dm.ewm(span=14, min_periods=14).mean() / atr_14
        minus_di = 100 * minus_dm.ewm(span=14, min_periods=14).mean() / atr_14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['ADX'] = dx.ewm(span=14, min_periods=14).mean()

        df['atr_percentile'] = df['ATR'].rolling(500, min_periods=50).rank(pct=True).fillna(0.5)
        return df

    # ── 策略 P9: EUR/USD Keltner KC2.0 ──
    # 11年回测: Sharpe 1.91, 12/12年盈利, K-Fold 6/6折正面
    # 最优参数: KC mult=2.0, ADX18, SL=4.5xATR, TP=8.0xATR, MaxHold=20
    def eurusd_keltner_signal(df):
        df = _calc_eurusd_indicators(df)
        if len(df) < 150:
            return None
        row = df.iloc[-1]
        close = float(row['Close'])
        atr = float(row.get('ATR', 0))
        adx = float(row.get('ADX', 0))
        kc_upper = float(row.get('KC_upper', 0))
        kc_lower = float(row.get('KC_lower', 0))
        ema100 = float(row.get('EMA100', 0))

        if any(pd.isna(v) for v in [atr, adx, kc_upper, kc_lower, ema100]):
            return None
        if atr <= 0 or adx < 18 or kc_upper <= 0:
            return None

        sl = round(atr * 4.5, 6)
        tp = round(atr * 8.0, 6)

        # V3 ATR Regime trailing
        atr_pct = float(row.get('atr_percentile', 0.5))
        if pd.isna(atr_pct):
            atr_pct = 0.5
        if atr_pct < 0.30:
            trail_act = round(atr * 1.0, 6)
            trail_dist = round(atr * 0.35, 6)
        elif atr_pct > 0.70:
            trail_act = round(atr * 0.6, 6)
            trail_dist = round(atr * 0.20, 6)
        else:
            trail_act = round(atr * 0.8, 6)
            trail_dist = round(atr * 0.25, 6)

        signal = None
        if close > kc_upper and close > ema100:
            signal = 'BUY'
        elif close < kc_lower and close < ema100:
            signal = 'SELL'

        if not signal:
            return None

        atr_pips = atr * 10000
        return {
            'signal': signal,
            'sl': sl,
            'tp': tp,
            'trailing_activate': trail_act,
            'trailing_distance': trail_dist,
            'reason': f'P9 EURUSD KC2.0: {signal} ADX={adx:.0f} ATR={atr_pips:.1f}pip',
        }

    paper.register_strategy('P9_eurusd_keltner', {
        'signal_func': eurusd_keltner_signal,
        'data_key': 'EURUSD_H1',
        'timeframe': 'H1',
        'max_hold_bars': 20,
        'max_positions': 1,
        'enabled': True,
        'point_value': EURUSD_POINT_VALUE,
        'symbol': EURUSD_SYMBOL,
        'lots': 0.05,
    })
