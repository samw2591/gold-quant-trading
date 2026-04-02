"""RiskManager 单元测试 — 日内亏损限制、冷却期"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import tempfile
from datetime import datetime

import config
from risk_manager import RiskManager


def _make_risk_manager(tmp_path: Path) -> RiskManager:
    """在临时目录创建 RiskManager"""
    return RiskManager(tmp_path)


class TestTotalLossLimit:

    def test_within_limit(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        assert rm.check_total_loss_limit(-100) is False

    def test_at_limit(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        assert rm.check_total_loss_limit(-config.MAX_TOTAL_LOSS) is True

    def test_beyond_limit(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        assert rm.check_total_loss_limit(-config.MAX_TOTAL_LOSS - 1) is True

    def test_positive_pnl(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        assert rm.check_total_loss_limit(500) is False


class TestDailyLossLimit:

    def test_under_limit(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        rm.update_daily_pnl(-10)
        assert rm.check_daily_loss_limit() is False

    def test_at_loss_count_limit(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        for _ in range(config.DAILY_MAX_LOSSES):
            rm.update_daily_pnl(-10)
        assert rm.check_daily_loss_limit() is True

    def test_winning_trades_dont_count(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        for _ in range(10):
            rm.update_daily_pnl(50)
        assert rm.check_daily_loss_limit() is False

    def test_state_persists(self, tmp_path):
        rm1 = _make_risk_manager(tmp_path)
        rm1.update_daily_pnl(-20)
        rm1.update_daily_pnl(-15)

        rm2 = _make_risk_manager(tmp_path)
        assert rm2.daily_loss_count == 2
        assert rm2.daily_pnl == -35.0


class TestCooldown:

    def test_no_cooldown_by_default(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        assert rm.is_in_cooldown('keltner') is False

    def test_cooldown_active(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        rm.add_cooldown('keltner', 3)
        assert rm.is_in_cooldown('keltner') is True

    def test_cooldown_expired(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        rm.add_cooldown('keltner', 0)
        assert rm.is_in_cooldown('keltner') is False

    def test_cooldown_per_strategy(self, tmp_path):
        rm = _make_risk_manager(tmp_path)
        rm.add_cooldown('keltner', 3)
        assert rm.is_in_cooldown('orb') is False
