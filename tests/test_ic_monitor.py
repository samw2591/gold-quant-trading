"""IC Monitor 核心逻辑测试"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ic_monitor import ICMonitor, _ic_quality_label, _rolling_rank_ic, MIN_TRADES_FOR_IC


class TestICQualityLabel:
    def test_excellent(self):
        assert _ic_quality_label(0.15, 0.6) == '优秀'

    def test_good(self):
        assert _ic_quality_label(0.06, 0.35) == '良好'

    def test_average(self):
        assert _ic_quality_label(0.04, 0.1) == '一般'

    def test_invalid(self):
        assert _ic_quality_label(0.01, 0.01) == '无效'


class TestOverallIC:
    def _make_df(self, n=50, correlation=0.5):
        np.random.seed(42)
        factor = np.random.randn(n)
        noise = np.random.randn(n) * (1 - abs(correlation))
        pnl = factor * correlation + noise
        return pd.DataFrame({'pnl': pnl, 'f_RSI14': factor})

    def test_calc_overall_ic_returns_dict(self):
        df = self._make_df(n=60, correlation=0.6)
        result = ICMonitor._calc_overall_ic(df, ['f_RSI14'])
        assert 'RSI14' in result
        assert 'ic' in result['RSI14']
        assert 'ic_ir' in result['RSI14']

    def test_positive_correlation_gives_positive_ic(self):
        df = self._make_df(n=80, correlation=0.7)
        result = ICMonitor._calc_overall_ic(df, ['f_RSI14'])
        assert result['RSI14']['ic'] > 0

    def test_insufficient_data_skipped(self):
        df = self._make_df(n=5)
        result = ICMonitor._calc_overall_ic(df, ['f_RSI14'])
        assert len(result) == 0


class TestDecayDetection:
    def test_stable_factor(self):
        np.random.seed(42)
        n = 60
        factor = np.random.randn(n)
        pnl = factor * 0.5 + np.random.randn(n) * 0.3
        df = pd.DataFrame({'pnl': pnl, 'f_ATR': factor})
        result = ICMonitor._detect_decay(df, ['f_ATR'])
        assert 'ATR' in result
        assert result['ATR']['status'] in ('stable', 'strengthening')


class TestFormatTelegram:
    def test_format_insufficient_data(self):
        monitor = ICMonitor()
        report = {
            'date': '2026-04-01',
            'live': {'status': 'insufficient_data', 'trade_count': 5, 'min_required': 20},
            'paper': {'status': 'no_factors', 'trade_count': 10},
        }
        msg = monitor.format_telegram_summary(report)
        assert '实盘' in msg
        assert '模拟盘' in msg
        assert '数据不足' in msg


class TestSnapshotFactors:
    def test_gold_trader_snapshot(self):
        from gold_trader import GoldTrader
        from strategies.signals import prepare_indicators

        np.random.seed(42)
        n = 80
        df = pd.DataFrame({
            'Open': np.random.uniform(2000, 2100, n),
            'High': np.random.uniform(2100, 2200, n),
            'Low': np.random.uniform(1900, 2000, n),
            'Close': np.random.uniform(2000, 2100, n),
            'Volume': np.random.randint(100, 10000, n),
        }, index=pd.date_range('2026-01-01', periods=n, freq='h'))

        df = prepare_indicators(df)
        factors = GoldTrader._snapshot_factors(df, 'keltner')

        assert isinstance(factors, dict)
        assert 'RSI14' in factors
        assert 'ATR' in factors
        assert all(isinstance(v, float) for v in factors.values())

    def test_paper_trader_snapshot(self):
        from paper_trader import PaperTrader
        from strategies.signals import prepare_indicators

        np.random.seed(42)
        n = 80
        df = pd.DataFrame({
            'Open': np.random.uniform(2000, 2100, n),
            'High': np.random.uniform(2100, 2200, n),
            'Low': np.random.uniform(1900, 2000, n),
            'Close': np.random.uniform(2000, 2100, n),
            'Volume': np.random.randint(100, 10000, n),
        }, index=pd.date_range('2026-01-01', periods=n, freq='h'))

        df = prepare_indicators(df)
        factors = PaperTrader._snapshot_factors(df, 'P4_atr_regime')

        assert isinstance(factors, dict)
        assert 'RSI14' in factors
        assert 'ATR' in factors
