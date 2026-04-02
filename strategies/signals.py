"""
黄金多时间框架量化交易信号引擎 v3
====================================
v3 核心升级 (基于GitHub backtrader-pullback-window-xauusd项目研究):
1. 4阶段状态机入场 — SCANNING→ARMED→WINDOW_OPEN→ENTRY
   不再"突破就入场"，改为等待回撤确认后再入场
2. EMA100趋势过滤 — 只在大趋势方向上交易
3. ADX趋势强度过滤 — ADX>25才允许趋势策略开仓
4. ATR自适应止损 — 2.5×ATR (基于backtrader项目实测参数)
5. 做空条件放宽 — 用ADX+EMA100过滤

策略组合:
1. H1 Keltner通道突破 (主力) + 状态机 + ADX + EMA100
2. H1 MACD+SMA50趋势 (补充) + ADX + EMA100
3. M15 RSI均值回归 (低风险补充)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

import config

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 技术指标计算
# ═══════════════════════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ADX (平均趋向指标)"""
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=1/period, min_periods=period).mean()


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA100'] = df['Close'].ewm(span=100).mean()   # v3新增: 趋势过滤
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['KC_mid'] = df['Close'].ewm(span=20).mean()
    df['KC_upper'] = df['KC_mid'] + 1.5 * df['ATR']
    df['KC_lower'] = df['KC_mid'] - 1.5 * df['ATR']
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['RSI14'] = calc_rsi(df['Close'], 14)
    df['ADX'] = calc_adx(df, 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    return df


# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════
ADX_TREND_THRESHOLD = 24    # 回测最优: Sharpe 0.60 (25→0.52, 23→0.45)
ATR_SL_MULTIPLIER = 2.5     # v3: 改为2.5×ATR (backtrader项目实测值)
ATR_SL_MIN = 10
ATR_SL_MAX = 50
ATR_TP_MULTIPLIER = 3.0     # v5: 止盈3.0×ATR (6.5×太远, MFE P90才$17, 几乎永远不触发)


def _calc_atr_stop(df: pd.DataFrame) -> float:
    atr = float(df.iloc[-1]['ATR'])
    if pd.isna(atr) or atr <= 0:
        return 20
    sl = round(atr * ATR_SL_MULTIPLIER, 2)
    return max(ATR_SL_MIN, min(ATR_SL_MAX, sl))


def _calc_atr_tp(df: pd.DataFrame) -> float:
    atr = float(df.iloc[-1]['ATR'])
    if pd.isna(atr) or atr <= 0:
        return 50
    return round(atr * ATR_TP_MULTIPLIER, 2)


# ═══════════════════════════════════════════════════════════════
# 4阶段状态机 (核心v3升级)
# ═══════════════════════════════════════════════════════════════

class KeltnerStateMachine:
    """
    Keltner突破的4阶段状态机入场系统
    基于 backtrader-pullback-window-xauusd 项目

    Phase 1 SCANNING:  检测到Keltner突破信号
    Phase 2 ARMED:     等待1-2根回撤K线确认
    Phase 3 WINDOW:    在回撤K线高低点设置突破窗口
    Phase 4 ENTRY:     价格突破窗口 → 真正入场

    如果窗口超时或反向突破 → 重置
    """

    # 状态常量
    SCANNING = "SCANNING"
    ARMED = "ARMED"
    WINDOW = "WINDOW"

    def __init__(self):
        self.state = self.SCANNING
        self.direction = None          # 'BUY' / 'SELL'
        self.pullback_count = 0
        self.window_top = None
        self.window_bottom = None
        self.window_bars_left = 0
        self.armed_bar_count = 0       # ARMED状态持续的bar数
        self.last_signal_reason = ""
        self._reset_count = 0

    def reset(self, reason: str = ""):
        """重置到扫描状态"""
        if self.state != self.SCANNING:
            log.debug(f"  [状态机] 重置: {self.state}→SCANNING ({reason})")
        self.state = self.SCANNING
        self.direction = None
        self.pullback_count = 0
        self.window_top = self.window_bottom = None
        self.window_bars_left = 0
        self.armed_bar_count = 0

    def update(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        每根新K线调用一次，返回入场信号或None

        Args:
            df: 包含指标的DataFrame (最新K线在最后)
        Returns:
            入场信号dict 或 None
        """
        if len(df) < 105:  # 需要EMA100有值
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(latest['Close'])
        open_ = float(latest['Open']) if 'Open' in latest else close
        high = float(latest['High'])
        low = float(latest['Low'])
        kc_upper = float(latest['KC_upper'])
        kc_lower = float(latest['KC_lower'])
        ema100 = float(latest['EMA100'])
        adx = float(latest['ADX'])
        atr = float(latest['ATR'])

        if any(pd.isna(v) for v in [kc_upper, kc_lower, ema100, adx, atr]):
            return None

        # ── Phase 1: SCANNING ──
        if self.state == self.SCANNING:
            return self._phase_scanning(close, kc_upper, kc_lower, ema100, adx)

        # ── Phase 2: ARMED (等待回撤) ──
        if self.state == self.ARMED:
            return self._phase_armed(close, open_, high, low, atr, df)

        # ── Phase 3: WINDOW (等待突破确认) ──
        if self.state == self.WINDOW:
            return self._phase_window(high, low, close, df)

        return None

    def _phase_scanning(self, close, kc_upper, kc_lower, ema100, adx):
        """Phase 1: 扫描信号"""
        # ADX过滤
        if adx < ADX_TREND_THRESHOLD:
            return None

        # 做多: 价格突破上轨 + 价格>EMA100 (大趋势过滤)
        if close > kc_upper and close > ema100:
            self.state = self.ARMED
            self.direction = 'BUY'
            self.pullback_count = 0
            self.armed_bar_count = 0
            self.last_signal_reason = f"Keltner做多: 价格{close:.2f} > 上轨{kc_upper:.2f} (ADX={adx:.1f})"
            log.info(f"  [状态机] SCANNING→ARMED(BUY): {self.last_signal_reason}")
            return None  # 不立即入场，等回撤

        # 做空: 价格跌破下轨 + 价格<EMA100
        if close < kc_lower and close < ema100:
            self.state = self.ARMED
            self.direction = 'SELL'
            self.pullback_count = 0
            self.armed_bar_count = 0
            self.last_signal_reason = f"Keltner做空: 价格{close:.2f} < 下轨{kc_lower:.2f} (ADX={adx:.1f})"
            log.info(f"  [状态机] SCANNING→ARMED(SELL): {self.last_signal_reason}")
            return None

        return None

    def _phase_armed(self, close, open_, high, low, atr, df):
        """Phase 2: 等待回撤K线"""
        self.armed_bar_count += 1

        # 超时保护: 最多等5根K线 (5小时H1 / 75分钟M15)
        if self.armed_bar_count > 5:
            self.reset("ARMED超时(5根K线)")
            return None

        # 判断是否为回撤K线
        is_pullback = False
        if self.direction == 'BUY':
            is_pullback = close < open_  # 红色K线 = 回撤
        elif self.direction == 'SELL':
            is_pullback = close > open_  # 绿色K线 = 回撤

        if is_pullback:
            self.pullback_count += 1
            log.info(f"  [状态机] 回撤K线 #{self.pullback_count} (H={high:.2f} L={low:.2f})")

            # 1根回撤即可进入窗口 (H1时间框架不能等太久)
            if self.pullback_count >= 1:
                # 计算突破窗口
                candle_range = high - low
                offset = candle_range * 0.5

                self.window_top = high + offset
                self.window_bottom = low - offset
                self.window_bars_left = 3  # 窗口持续3根K线

                self.state = self.WINDOW
                log.info(f"  [状态机] ARMED→WINDOW: 窗口[{self.window_bottom:.2f}, {self.window_top:.2f}] 持续{self.window_bars_left}根K线")
                return None
        else:
            # 非回撤K线：如果是顺向强势K线，可能是有效突破，不重置
            # 但如果方向完全反转，重置
            if self.direction == 'BUY' and close < df.iloc[-3]['Low'] if len(df) >= 3 else False:
                self.reset("BUY信号失效(价格大幅回落)")
            elif self.direction == 'SELL' and close > df.iloc[-3]['High'] if len(df) >= 3 else False:
                self.reset("SELL信号失效(价格大幅反弹)")

        return None

    def _phase_window(self, high, low, close, df):
        """Phase 3: 等待突破确认"""
        self.window_bars_left -= 1

        sl = _calc_atr_stop(df)
        tp = _calc_atr_tp(df)

        if self.direction == 'BUY':
            # 价格突破窗口上沿 → 入场
            if high >= self.window_top:
                signal = {
                    'strategy': 'keltner',
                    'signal': 'BUY',
                    'reason': f"✅ Keltner做多确认: 回撤后突破{self.window_top:.2f} ({self.last_signal_reason})",
                    'close': close,
                    'sl': sl,
                    'tp': tp,
                }
                log.info(f"  [状态机] WINDOW→ENTRY(BUY): 突破确认!")
                self.reset("入场完成")
                return signal

            # 价格跌破窗口下沿 → 失效
            if low <= self.window_bottom:
                log.info(f"  [状态机] WINDOW失效: 价格{low:.2f} < 下沿{self.window_bottom:.2f}")
                self.reset("窗口失效(反向突破)")
                return None

        elif self.direction == 'SELL':
            if low <= self.window_bottom:
                signal = {
                    'strategy': 'keltner',
                    'signal': 'SELL',
                    'reason': f"✅ Keltner做空确认: 回撤后突破{self.window_bottom:.2f} ({self.last_signal_reason})",
                    'close': close,
                    'sl': sl,
                    'tp': tp,
                }
                log.info(f"  [状态机] WINDOW→ENTRY(SELL): 突破确认!")
                self.reset("入场完成")
                return signal

            if high >= self.window_top:
                log.info(f"  [状态机] WINDOW失效: 价格{high:.2f} > 上沿{self.window_top:.2f}")
                self.reset("窗口失效(反向突破)")
                return None

        # 窗口超时
        if self.window_bars_left <= 0:
            log.info(f"  [状态机] WINDOW超时，重置")
            self.reset("窗口超时")

        return None

    def get_status(self) -> str:
        """返回当前状态描述（用于日志）"""
        if self.state == self.SCANNING:
            return "扫描中"
        elif self.state == self.ARMED:
            return f"等回撤({self.direction}, {self.pullback_count}根)"
        elif self.state == self.WINDOW:
            return f"窗口({self.direction}, 剩{self.window_bars_left}根, [{self.window_bottom:.0f}-{self.window_top:.0f}])"
        return self.state


# ═══════════════════════════════════════════════════════════════
# 全局状态机实例 (在模块级别保持状态)
# ═══════════════════════════════════════════════════════════════
_keltner_sm = KeltnerStateMachine()


def get_keltner_state_machine() -> KeltnerStateMachine:
    """获取全局状态机实例"""
    return _keltner_sm


# ═══════════════════════════════════════════════════════════════
# 信号检测函数
# ═══════════════════════════════════════════════════════════════

def check_keltner_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    Keltner通道突破信号 v4 — 简单版 + ADX过滤 + EMA100趋势过滤
    
    回测验证: 11年Sharpe 0.52, 特朗普2期Sharpe 1.47
    状态机版本回测后Sharpe仅0.52降到0.23，确认简单版更优
    """
    if len(df) < 105:
        return None
    
    latest = df.iloc[-1]
    close = float(latest['Close'])
    kc_upper = float(latest['KC_upper'])
    kc_lower = float(latest['KC_lower'])
    ema100 = float(latest['EMA100'])
    adx = float(latest['ADX'])
    
    if any(pd.isna(v) for v in [kc_upper, kc_lower, ema100, adx]):
        return None
    
    # ADX过滤: 趋势不够强不开仓
    if adx < ADX_TREND_THRESHOLD:
        return None
    
    sl = _calc_atr_stop(df)
    tp = _calc_atr_tp(df)
    
    # 做多: 突破上轨 + 价格>EMA100
    if close > kc_upper and close > ema100:
        return {
            'strategy': 'keltner',
            'signal': 'BUY',
            'reason': f"Keltner做多: 价格{close:.2f} > 上轨{kc_upper:.2f} (ADX={adx:.1f})",
            'close': close,
            'sl': sl,
            'tp': tp,
        }
    
    # 做空: 跌破下轨 + 价格<EMA100
    if close < kc_lower and close < ema100:
        return {
            'strategy': 'keltner',
            'signal': 'SELL',
            'reason': f"Keltner做空: 价格{close:.2f} < 下轨{kc_lower:.2f} (ADX={adx:.1f})",
            'close': close,
            'sl': sl,
            'tp': tp,
        }
    
    return None


def check_macd_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    MACD趋势信号 v3 — 增加EMA100趋势过滤
    """
    if len(df) < 105:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest['Close'])
    macd_hist = float(latest['MACD_hist'])
    macd_hist_prev = float(prev['MACD_hist'])
    ema100 = float(latest['EMA100'])
    adx = float(latest['ADX'])

    if any(pd.isna(v) for v in [macd_hist, macd_hist_prev, ema100, adx]):
        return None

    if adx < ADX_TREND_THRESHOLD:
        return None

    sl = _calc_atr_stop(df)
    tp = _calc_atr_tp(df)

    # 做多: MACD转正 + 价格>EMA100
    if macd_hist > 0 and macd_hist_prev <= 0 and close > ema100:
        return {
            'strategy': 'macd',
            'signal': 'BUY',
            'reason': f"MACD做多: 柱状图转正, 价格{close:.2f} > EMA100 (ADX={adx:.1f})",
            'close': close,
            'sl': sl,
            'tp': tp,
        }

    # 做空: MACD转负 + 价格<EMA100
    if macd_hist < 0 and macd_hist_prev >= 0 and close < ema100:
        return {
            'strategy': 'macd',
            'signal': 'SELL',
            'reason': f"MACD做空: 柱状图转负, 价格{close:.2f} < EMA100 (ADX={adx:.1f})",
            'close': close,
            'sl': sl,
            'tp': tp,
        }

    return None


def check_exit_signal(df: pd.DataFrame, strategy: str, direction: str) -> Optional[str]:
    """检查出场信号"""
    if len(df) < 5:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(latest['Close'])

    if strategy == 'keltner':
        # v5优化: 去掉KC中轨出场 (回测验证: 中轨出场94.7%是亏损平仓，
        # 去掉后Sharpe 0.74→0.80, 特朗普2期 1.98→2.10, 6个时段5个提升)
        # Keltner现在只靠 TP(6.5×ATR) / SL(2.5×ATR) / 超时(15根K线) 出场
        pass

    elif strategy == 'macd':
        macd_hist = float(latest['MACD_hist'])
        macd_hist_prev = float(prev['MACD_hist'])
        if not pd.isna(macd_hist):
            if direction == 'BUY' and macd_hist < 0 and macd_hist_prev >= 0:
                return "MACD多头出场: 柱状图转负"
            elif direction == 'SELL' and macd_hist > 0 and macd_hist_prev <= 0:
                return "MACD空头出场: 柱状图转正"

    elif strategy in ('m5_rsi', 'm15_rsi'):
        rsi2 = float(latest['RSI2'])
        if not pd.isna(rsi2):
            if direction == 'BUY' and rsi2 > 55:
                return f"M15 RSI多头出场: RSI(2)={rsi2:.1f} > 55"
            elif direction == 'SELL' and rsi2 < 45:
                return f"M15 RSI空头出场: RSI(2)={rsi2:.1f} < 45"

    return None


def check_m15_rsi_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    M15 RSI均值回归信号 + EMA100趋势方向过滤

    v7: 加入EMA100过滤，防止在强趋势中逆势开仓
    回测: Sharpe 1.02→1.28(+25%), MaxDD $662→$524(-21%), 胜率 56.4%→58.6%
    """
    if len(df) < 105:
        return None
    latest = df.iloc[-1]
    close = float(latest['Close'])
    rsi2 = float(latest['RSI2'])
    sma50 = float(latest['SMA50'])
    ema100 = float(latest['EMA100'])
    if pd.isna(rsi2) or pd.isna(sma50) or pd.isna(ema100):
        return None

    sl = _calc_atr_stop(df) if not pd.isna(df.iloc[-1]['ATR']) else 15

    if rsi2 < 15 and close > sma50 and close > ema100:
        return {
            'strategy': 'm15_rsi', 'signal': 'BUY',
            'reason': f"M15 RSI做多: RSI(2)={rsi2:.1f} < 15, 超卖反弹 (EMA100={ema100:.0f})",
            'close': close, 'sl': sl, 'tp': 0,
        }
    if rsi2 > 85 and close < sma50 and close < ema100:
        return {
            'strategy': 'm15_rsi', 'signal': 'SELL',
            'reason': f"M15 RSI做空: RSI(2)={rsi2:.1f} > 85, 超买回落 (EMA100={ema100:.0f})",
            'close': close, 'sl': sl, 'tp': 0,
        }
    return None


# ═══════════════════════════════════════════════════════════════
# NY开盘区间突破 (ORB) 策略
# ═══════════════════════════════════════════════════════════════



class ORBStrategy:
    """
    NY开盘区间突破策略 (Opening Range Breakout)

    原理:
    - 纽约开盘后前15分钟的高低点形成当日区间
    - 价格突破区间上沿→做多
    - 价格跌破区间下沿→做空
    - 止损=区间宽度, 止盈=2.2×区间宽度
    - 窗口有效期2小时 (过时不入场)
    - 每日只交易一次

    胜率61%, RR 1:2.2 (根据历史研究)
    """

    def __init__(self):
        self.range_high = None
        self.range_low = None
        self.range_set_date = None     # 区间设定日期
        self.traded_today = False      # 今日是否已交易
        self.window_open = False
        self.window_expiry = None

    def reset_daily(self):
        """每日重置"""
        self.range_high = None
        self.range_low = None
        self.range_set_date = None
        self.traded_today = False
        self.window_open = False
        self.window_expiry = None

    def update(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        用H1数据检测ORB信号

        逻辑:
        1. 识别NY开盘K线 (UTC 14:xx) → 设定区间
        2. 后续的K线检查是否突破
        """
        if not config.ORB_ENABLED:
            return None
        if len(df) < 10:
            return None

        latest = df.iloc[-1]
        close = float(latest['Close'])
        high = float(latest['High'])
        low = float(latest['Low'])

        # 获取当前K线的UTC小时
        bar_time = df.index[-1]
        if hasattr(bar_time, 'hour'):
            bar_hour = bar_time.hour
        else:
            bar_hour = -1

        today = bar_time.date() if hasattr(bar_time, 'date') else None

        # 新的一天重置
        if today and self.range_set_date and today != self.range_set_date:
            self.reset_daily()

        # Step 1: 识别NY开盘K线 → 设定区间
        # 回溯最近3根K线，防止因扫描间隔错过开盘K线
        if self.range_high is None:
            for lookback in range(min(3, len(df))):
                idx = -(lookback + 1)
                check_bar = df.iloc[idx]
                check_time = df.index[idx]
                check_hour = check_time.hour if hasattr(check_time, 'hour') else -1
                check_date = check_time.date() if hasattr(check_time, 'date') else None
                
                if check_hour == config.ORB_NY_OPEN_HOUR_UTC:
                    self.range_high = float(check_bar['High'])
                    self.range_low = float(check_bar['Low'])
                    self.range_set_date = check_date
                    self.window_open = True
                    # 窗口有效期减去已过去的K线数
                    self.window_expiry = max(1, config.ORB_EXPIRY_MINUTES // 60 - lookback)

                    range_width = self.range_high - self.range_low
                    if lookback > 0:
                        log.info(f"  [🇺🇸 ORB] 回溯{lookback}根K线找到NY开盘")
                    log.info(f"  [🇺🇸 ORB] NY开盘区间设定: [{self.range_low:.2f} - {self.range_high:.2f}] "
                             f"宽度=${range_width:.2f} 窗口{self.window_expiry}根K线")
                    
                    # 如果是回溯找到的，当前K线已经是后续K线，直接检查突破
                    if lookback > 0:
                        break  # 跳出循环，下面Step 2会检查突破
                    return None  # 开盘K线本身不交易

        # Step 2: 检查突破
        if self.window_open and self.range_high is not None and not self.traded_today:
            self.window_expiry -= 1

            # 窗口超时
            if self.window_expiry <= 0:
                log.info(f"  [🇺🇸 ORB] 窗口超时，今日不再触发")
                self.window_open = False
                return None

            range_width = self.range_high - self.range_low
            atr = float(df.iloc[-1]['ATR']) if not pd.isna(df.iloc[-1]['ATR']) else 0

            if atr <= 0:
                log.warning(f"  [🇺🇸 ORB] ATR数据无效({atr}), 无法评估风险, 跳过")
                return None

            min_range = max(8, 0.5 * atr)
            if range_width < min_range:
                log.info(f"  [🇺🇸 ORB] 区间${range_width:.2f} < 最小${min_range:.2f} (0.5×ATR), 跳过")
                return None
            max_range = min(60, 3.0 * atr)
            if range_width > max_range:
                log.info(f"  [🇺🇸 ORB] 区间${range_width:.2f} > 上限${max_range:.2f} (3.0×ATR), 跳过")
                self.window_open = False
                return None
            # v6: Range/ATR过滤 — 区间>2.0×ATR说明开盘已剧烈波动，假突破概率高
            # 回测: Sharpe 1.52→2.45, 胜率 48.2%→51.6%, Trump2.0期 Sharpe 2.50→3.13
            if range_width > 2.0 * atr:
                log.info(f"  [🇺🇸 ORB] Range/ATR={range_width/atr:.1f}>2.0，假突破风险高，跳过")
                return None

            sl = round(range_width * config.ORB_SL_MULTIPLIER, 2)
            min_sl = round(atr * config.ORB_SL_MIN_ATR_MULTIPLIER, 2) if atr > 0 else ATR_SL_MIN
            sl = max(sl, min_sl)
            tp = round(range_width * config.ORB_TP_MULTIPLIER, 2)

            # 突破上沿 → 做多
            if high > self.range_high:
                self.traded_today = True
                self.window_open = False
                return {
                    'strategy': 'orb',
                    'signal': 'BUY',
                    'reason': f"🇺🇸 ORB做多: 价格{high:.2f} 突破开盘区间上沿{self.range_high:.2f} (区间${range_width:.1f})",
                    'close': close,
                    'sl': sl,
                    'tp': tp,
                }

            # 跌破下沿 → 做空
            if low < self.range_low:
                self.traded_today = True
                self.window_open = False
                return {
                    'strategy': 'orb',
                    'signal': 'SELL',
                    'reason': f"🇺🇸 ORB做空: 价格{low:.2f} 跌破开盘区间下沿{self.range_low:.2f} (区间${range_width:.1f})",
                    'close': close,
                    'sl': sl,
                    'tp': tp,
                }

        return None

    def get_status(self) -> str:
        if self.range_high is None:
            return "等待NY开盘"
        if self.traded_today:
            return "今日已交易"
        if self.window_open:
            return f"窗口开启 [{self.range_low:.0f}-{self.range_high:.0f}] 剩{self.window_expiry}根K线"
        return "窗口已关闭"


# 全局ORB实例
_orb_strategy = ORBStrategy()

def get_orb_strategy() -> ORBStrategy:
    return _orb_strategy

def check_orb_signal(df: pd.DataFrame) -> Optional[Dict]:
    """检查ORB信号"""
    return _orb_strategy.update(df)


# ═══════════════════════════════════════════════════════════════
# ATR自动调仓
# ═══════════════════════════════════════════════════════════════

def calc_auto_lot_size(atr: float, sl_distance: float) -> float:
    """
    根据ATR/止损距离自动计算手数，保持每笔风险金额恒定

    公式: lots = RISK_PER_TRADE / (sl_distance × POINT_VALUE_PER_LOT)

    例如:
    - RISK_PER_TRADE=$100, sl=$37.5, POINT_VALUE=100
    - lots = 100 / (37.5 × 100) = 0.027 → 0.03手
    - 实际风险 = 37.5 × 0.03 × 100 = $112.5

    - RISK_PER_TRADE=$100, sl=$77.5 (高ATR), POINT_VALUE=100
    - lots = 100 / (77.5 × 100) = 0.013 → 0.01手
    - 实际风险 = 77.5 × 0.01 × 100 = $77.5
    """
    if not config.AUTO_LOT_SIZING:
        return config.LOT_SIZE

    if sl_distance <= 0:
        return config.LOT_SIZE

    lots = config.RISK_PER_TRADE / (sl_distance * config.POINT_VALUE_PER_LOT)
    # 四舍五入到小数点后两位
    lots = round(lots, 2)
    # 限制范围
    lots = max(config.MIN_LOT_SIZE, min(config.MAX_LOT_SIZE, lots))
    return lots


# ═══════════════════════════════════════════════════════════════
# 周一跳空回补策略 (Monday Gap Fill)
# ═══════════════════════════════════════════════════════════════


# 周五收盘价存储 (全局状态)
_friday_close_price = None
_gap_traded_today = False


def update_friday_close(df: pd.DataFrame):
    """周五最后一根K线时记录收盘价"""
    global _friday_close_price
    if len(df) < 5:
        return
    bar_time = df.index[-1]
    if hasattr(bar_time, 'weekday') and bar_time.weekday() == 4:
        _friday_close_price = float(df.iloc[-1]['Close'])


def check_monday_gap_fill(df: pd.DataFrame) -> Optional[Dict]:
    """
    周一跳空回补策略
    
    回测: Sharpe 0.92, 胜率79.3%, 11年426笔
    原理: 77%的周末跳空会在周一当天回补
    - 跳高开盘 → 做空 (等回补)
    - 跳低开盘 → 做多 (等回补)
    - SL = 跳空幅度×1.0, TP = 跳空幅度 (回补完整跳空)
    - 最小跳空$2, 每周最多1笔
    """
    global _friday_close_price, _gap_traded_today
    
    if not config.STRATEGIES.get('gap_fill', {}).get('enabled', False):
        return None
    if _friday_close_price is None:
        return None
    if len(df) < 5:
        return None
    
    bar_time = df.index[-1]
    if not hasattr(bar_time, 'weekday'):
        return None
    
    # 只在周一前3根K线 (UTC 22-01, 即开盘后前3小时)
    if bar_time.weekday() != 0:
        _gap_traded_today = False  # 重置
        return None
    if bar_time.hour > 3:  # 只在开盘后前几小时
        return None
    if _gap_traded_today:
        return None
    
    current_price = float(df.iloc[-1]['Close'])
    gap = current_price - _friday_close_price
    
    # 最小跳空$2
    if abs(gap) < 2:
        return None
    
    sl = round(abs(gap) * 1.0, 2)  # SL = 跳空幅度
    tp = round(abs(gap), 2)         # TP = 回补完整跳空
    sl = max(3, min(30, sl))        # 限制范围
    
    _gap_traded_today = True
    
    if gap > 0:
        # 跳高 → 做空等回补
        return {
            'strategy': 'gap_fill',
            'signal': 'SELL',
            'reason': f'📅 周一跳空回补: 跳高${gap:.1f}(周五收{_friday_close_price:.0f}), 做空等回补',
            'close': current_price,
            'sl': sl, 'tp': tp,
        }
    else:
        # 跳低 → 做多等回补
        return {
            'strategy': 'gap_fill',
            'signal': 'BUY',
            'reason': f'📅 周一跳空回补: 跳低${abs(gap):.1f}(周五收{_friday_close_price:.0f}), 做多等回补',
            'close': current_price,
            'sl': sl, 'tp': tp,
        }


# ═══════════════════════════════════════════════════════════════
# 信号扫描入口
# ═══════════════════════════════════════════════════════════════

def scan_all_signals(df: pd.DataFrame, timeframe: str = 'H1') -> List[Dict]:
    """扫描所有已启用策略的信号"""
    signals = []
    if timeframe == 'H1':
        sig = check_keltner_signal(df)
        if sig:
            signals.append(sig)
        if config.STRATEGIES.get('macd', {}).get('enabled', True):
            sig = check_macd_signal(df)
            if sig:
                signals.append(sig)
        # ORB策略 (也用H1数据)
        if config.ORB_ENABLED:
            sig = check_orb_signal(df)
            if sig:
                signals.append(sig)
        # 周一跳空回补
        update_friday_close(df)  # 每次都更新周五收盘价
        sig = check_monday_gap_fill(df)
        if sig:
            signals.append(sig)
    elif timeframe in ('M5', 'M15'):
        sig = check_m15_rsi_signal(df)
        if sig:
            signals.append(sig)
    return signals
