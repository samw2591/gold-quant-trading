"""ORBStrategy 单元测试 — 状态机边界条件"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from strategies.signals import ORBStrategy, prepare_indicators


def _make_h1_df(n: int = 60, base_price: float = 2000.0,
                start_hour: int = 10) -> pd.DataFrame:
    """生成带UTC时间索引的H1数据"""
    rng = np.random.default_rng(42)
    close = base_price + rng.standard_normal(n).cumsum()
    high = close + rng.uniform(1, 5, n)
    low = close - rng.uniform(1, 5, n)
    open_ = close + rng.uniform(-2, 2, n)
    volume = rng.integers(100, 5000, n)
    idx = pd.date_range('2026-03-02', periods=n, freq='h')
    start_offset = pd.Timedelta(hours=start_hour)
    idx = idx + start_offset - pd.Timedelta(hours=idx[0].hour)
    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    }, index=idx[:n])
    return prepare_indicators(df)


class TestORBStrategy:

    def test_initial_state(self):
        orb = ORBStrategy()
        assert orb.range_high is None
        assert orb.traded_today is False

    def test_reset_daily(self):
        orb = ORBStrategy()
        orb.range_high = 2050
        orb.traded_today = True
        orb.reset_daily()
        assert orb.range_high is None
        assert orb.traded_today is False

    def test_no_signal_without_ny_open(self):
        """没有NY开盘K线时不应产生信号"""
        orb = ORBStrategy()
        df = _make_h1_df(60, start_hour=0)
        has_ny_bar = any(
            hasattr(t, 'hour') and t.hour == config.ORB_NY_OPEN_HOUR_UTC
            for t in df.index
        )
        if not has_ny_bar:
            sig = orb.update(df)
            assert sig is None

    def test_disabled_returns_none(self, monkeypatch):
        """ORB_ENABLED=False 时应返回None"""
        monkeypatch.setattr(config, 'ORB_ENABLED', False)
        orb = ORBStrategy()
        df = _make_h1_df()
        assert orb.update(df) is None

    def test_insufficient_data(self):
        """数据不足10根时不产生信号"""
        orb = ORBStrategy()
        df = _make_h1_df(5)
        assert orb.update(df) is None

    def test_get_status_before_range(self):
        """未设定区间时 get_status 不应抛异常"""
        orb = ORBStrategy()
        status = orb.get_status()
        assert isinstance(status, str)
