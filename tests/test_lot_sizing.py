"""calc_auto_lot_size 单元测试 — 资金安全相关"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from strategies.signals import calc_auto_lot_size


class TestAutoLotSizing:
    """测试ATR自动调仓各种边界"""

    def test_normal_sl(self):
        """正常止损距离应返回合理手数"""
        lots = calc_auto_lot_size(0, 37.5)
        assert config.MIN_LOT_SIZE <= lots <= config.MAX_LOT_SIZE

    def test_zero_sl_returns_default(self):
        """止损为0时应返回默认手数"""
        lots = calc_auto_lot_size(0, 0)
        assert lots == config.LOT_SIZE

    def test_negative_sl_returns_default(self):
        """负止损值应返回默认手数"""
        lots = calc_auto_lot_size(0, -10)
        assert lots == config.LOT_SIZE

    def test_very_large_sl(self):
        """大止损距离应返回最小手数"""
        lots = calc_auto_lot_size(0, 10000)
        assert lots == config.MIN_LOT_SIZE

    def test_very_small_sl(self):
        """极小止损距离应返回最大手数"""
        lots = calc_auto_lot_size(0, 0.01)
        assert lots == config.MAX_LOT_SIZE

    def test_result_is_rounded(self):
        """结果应四舍五入到两位小数"""
        lots = calc_auto_lot_size(0, 33.33)
        assert lots == round(lots, 2)

    def test_auto_lot_disabled(self, monkeypatch):
        """AUTO_LOT_SIZING=False 时应返回固定手数"""
        monkeypatch.setattr(config, 'AUTO_LOT_SIZING', False)
        lots = calc_auto_lot_size(0, 37.5)
        assert lots == config.LOT_SIZE

    def test_min_lot_floor(self):
        """结果不应低于 MIN_LOT_SIZE"""
        lots = calc_auto_lot_size(0, 999999)
        assert lots >= config.MIN_LOT_SIZE

    def test_max_lot_cap(self):
        """结果不应超过 MAX_LOT_SIZE"""
        lots = calc_auto_lot_size(0, 0.001)
        assert lots <= config.MAX_LOT_SIZE
