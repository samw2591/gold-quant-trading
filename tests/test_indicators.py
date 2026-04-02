"""prepare_indicators 单元测试 — 确保输出列完整"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from strategies.signals import prepare_indicators

EXPECTED_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA50', 'EMA100', 'EMA9', 'EMA12', 'EMA21', 'EMA26',
    'ATR', 'KC_mid', 'KC_upper', 'KC_lower',
    'MACD', 'MACD_signal', 'MACD_hist',
    'RSI2', 'RSI14', 'ADX',
]


def _make_ohlcv(n: int = 120) -> pd.DataFrame:
    """生成模拟 OHLCV 数据"""
    rng = np.random.default_rng(42)
    close = 2000 + rng.standard_normal(n).cumsum()
    high = close + rng.uniform(0.5, 3, n)
    low = close - rng.uniform(0.5, 3, n)
    open_ = close + rng.uniform(-1, 1, n)
    volume = rng.integers(100, 5000, n)
    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    }, index=pd.date_range('2026-01-01', periods=n, freq='h'))
    return df


class TestPrepareIndicators:

    def test_all_columns_present(self):
        """输出应包含所有期望的技术指标列"""
        df = prepare_indicators(_make_ohlcv())
        for col in EXPECTED_COLUMNS:
            assert col in df.columns, f"缺少列: {col}"

    def test_no_nan_in_last_row(self):
        """最后一行的关键指标不应为NaN (数据足够多时)"""
        df = prepare_indicators(_make_ohlcv(200))
        last = df.iloc[-1]
        for col in ['SMA50', 'EMA100', 'ATR', 'MACD', 'RSI14', 'ADX']:
            assert not pd.isna(last[col]), f"最后一行 {col} 为 NaN"

    def test_output_length_unchanged(self):
        """输出行数应与输入相同"""
        raw = _make_ohlcv(100)
        df = prepare_indicators(raw)
        assert len(df) == len(raw)

    def test_does_not_mutate_input(self):
        """不应修改输入DataFrame"""
        raw = _make_ohlcv(100)
        original_cols = list(raw.columns)
        prepare_indicators(raw)
        assert list(raw.columns) == original_cols
