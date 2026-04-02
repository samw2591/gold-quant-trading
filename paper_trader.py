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

log = logging.getLogger(__name__)


class PaperPosition:
    """模拟持仓"""
    def __init__(self, strategy: str, direction: str, entry_price: float,
                 sl: float, tp: float, lots: float, reason: str,
                 factors: Optional[Dict] = None):
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
            'entry_price': self.entry_price,
            'exit_price': close,
            'sl': self.sl, 'tp': self.tp, 'lots': self.lots,
            'pnl': round(pnl * self.lots * config.POINT_VALUE_PER_LOT, 2),
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
            )
            pos.entry_time = p.get('entry_time', '')
            pos.bars_held = p.get('bars_held', 0)
            pos.max_favorable = p.get('max_favorable', 0)
            pos.max_adverse = p.get('max_adverse', 0)
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
                'factors': pos.factors,
                'entry_time': pos.entry_time,
                'bars_held': pos.bars_held,
                'max_favorable': pos.max_favorable,
                'max_adverse': pos.max_adverse,
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

            factors = self._snapshot_factors(df, name)

            pos = PaperPosition(
                strategy=name,
                direction=sig['signal'],
                entry_price=entry_price,
                sl=sl, tp=tp, lots=lots,
                reason=sig.get('reason', name),
                factors=factors,
            )
            self.positions.append(pos)
            self._save_positions()

            log.info(f"  📝 [模拟] {sig['signal']} {name} @ {entry_price:.2f} "
                     f"SL={sl:.1f} TP={tp:.1f} | {sig.get('reason', '')}")

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
        if volume < vol_ma * 1.5:
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
