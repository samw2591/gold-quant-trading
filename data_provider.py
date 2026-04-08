"""
行情数据提供模块
================
从 MT4 本地桥接文件或 yfinance 获取 K 线数据

数据源优先级: MT4本地 > yfinance (fallback)
"""
import json
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

import config
from mt4_bridge import MT4Bridge
from strategies.signals import prepare_indicators

log = logging.getLogger(__name__)


class DataProvider:
    """行情数据获取"""

    def __init__(self, bridge: MT4Bridge):
        self.bridge = bridge

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

    def _read_mt4_bars(self, filename: str) -> Optional[pd.DataFrame]:
        """从黄金桥接目录读K线数据 (优先数据源)"""
        df = self._read_mt4_bars_from(filename, config.BRIDGE_DIR)
        if df is not None:
            return prepare_indicators(df)
        return None

    def get_eurusd_h1(self) -> Optional[pd.DataFrame]:
        """获取 EUR/USD H1 数据: MT4 桥接 > yfinance fallback"""
        df = self._read_mt4_bars_from('bars_h1.json', config.BRIDGE_DIR_EURUSD)
        if df is not None:
            log.debug("EURUSD H1数据来源: MT4本地")
            return df
        log.debug("EURUSD H1数据来源: yfinance (fallback)")
        return self._get_yfinance_data('1h', '30d', ticker='EURUSD=X', apply_indicators=False)

    def _read_mt4_bars_from(self, filename: str, bridge_dir) -> Optional[pd.DataFrame]:
        """从指定桥接目录读K线数据"""
        filepath = bridge_dir / filename
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
            times = [pd.Timestamp(b['t'].replace('.', '-')) for b in bars]
            df.index = pd.DatetimeIndex(times)
            df.index.name = 'Datetime'
            return df
        except Exception as e:
            log.debug(f"MT4本地数据读取失败 ({bridge_dir.name}/{filename}): {e}")
            return None

    def _get_yfinance_data(self, interval='1h', period='60d',
                           ticker='GC=F', apply_indicators=True) -> Optional[pd.DataFrame]:
        """从yfinance获取数据 (fallback备用)"""
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close'])
            if len(df) < 55:
                return None
            if apply_indicators:
                return prepare_indicators(df)
            return df
        except Exception as e:
            log.debug(f"yfinance {ticker} {interval}数据获取失败: {e}")
            return None
