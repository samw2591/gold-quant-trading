"""
风控管理模块
============
总亏损保护、日内亏损限制、策略冷却期
"""
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import config

log = logging.getLogger(__name__)


class RiskManager:
    """交易风险管理"""

    def __init__(self, data_dir: Path):
        self.daily_state_file = data_dir / "gold_daily_state.json"
        self.cooldown_until: Dict[str, datetime] = {}
        self._load_daily_state()

    # ── 总亏损 ──

    def check_total_loss_limit(self, total_pnl: float) -> bool:
        """检查是否超过总亏损上限"""
        if total_pnl <= -config.MAX_TOTAL_LOSS:
            log.warning(f"🛑 总亏损已达 ${total_pnl:.2f}，超过上限 ${config.MAX_TOTAL_LOSS}，停止交易")
            return True
        return False

    # ── 日内亏损 ──

    def check_daily_loss_limit(self) -> bool:
        """检查是否超过日内最大亏损笔数"""
        today = datetime.now().date()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_loss_count = 0
            self.daily_date = today
            self._save_daily_state()
        if self.daily_loss_count >= config.DAILY_MAX_LOSSES:
            return True
        if self.daily_pnl <= -config.DAILY_MAX_LOSS:
            return True
        return False

    def update_daily_pnl(self, profit: float):
        """更新日内盈亏跟踪"""
        today = datetime.now().date()
        if today != self.daily_date:
            self.daily_pnl = 0.0
            self.daily_loss_count = 0
            self.daily_date = today
        self.daily_pnl = round(self.daily_pnl + profit, 2)
        if profit < 0:
            self.daily_loss_count += 1
            log.info(f"     📊 日内亏损第{self.daily_loss_count}笔 (上限{config.DAILY_MAX_LOSSES}笔)")
        self._save_daily_state()

    # ── 日内自适应仓位缩减 ──

    LOSS_LOT_SCALE = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1}

    def get_lot_scale(self) -> float:
        """根据日内亏损笔数返回仓位缩减系数 (回测验证: Sharpe +6.3%, MaxDD -14.4%)"""
        return self.LOSS_LOT_SCALE.get(self.daily_loss_count, 0.1)

    # ── 冷却期 ──

    def is_in_cooldown(self, strategy: str) -> bool:
        """检查策略是否在冷却期"""
        if strategy in self.cooldown_until:
            if datetime.now() < self.cooldown_until[strategy]:
                return True
            else:
                del self.cooldown_until[strategy]
        return False

    def add_cooldown(self, strategy: str, hours: int):
        """为策略设置冷却期"""
        self.cooldown_until[strategy] = datetime.now() + timedelta(hours=hours)
        log.info(f"     ❄️ {strategy} 进入冷却期，{hours}小时后才可开仓")

    # ── 日内状态持久化 ──

    def _load_daily_state(self):
        """从文件加载日内状态，防止重启丢失"""
        state = self._load_json(self.daily_state_file, {})
        saved_date = state.get('date', '')
        today = str(datetime.now().date())

        if saved_date == today:
            self.daily_pnl = state.get('pnl', 0.0)
            self.daily_loss_count = state.get('loss_count', 0)
            self.daily_date = datetime.now().date()
            log.info(f"  📊 恢复日内状态: PnL=${self.daily_pnl:.2f}, 亏损{self.daily_loss_count}/{config.DAILY_MAX_LOSSES}笔")
        else:
            self.daily_pnl = 0.0
            self.daily_loss_count = 0
            self.daily_date = datetime.now().date()

    def _save_daily_state(self):
        """保存日内状态到文件"""
        state = {
            'date': str(self.daily_date),
            'pnl': self.daily_pnl,
            'loss_count': self.daily_loss_count,
        }
        self._save_json(self.daily_state_file, state)

    def _load_json(self, path, default):
        if path.exists():
            with open(path) as f:
                try:
                    return json.load(f)
                except (json.JSONDecodeError, ValueError):
                    return default
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
