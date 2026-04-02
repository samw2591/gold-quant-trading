"""
MT4 文件桥接模块
================
通过文件系统与 MT4 EA 通信:
- Python 写指令文件 → EA 读取并执行
- EA 写状态文件 → Python 读取结果

通信目录: MT4数据文件夹/MQL4/Files/DWX/
"""
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import config

log = logging.getLogger(__name__)


class MT4Bridge:
    """MT4 文件桥接"""

    def __init__(self):
        self.bridge_dir = config.BRIDGE_DIR
        self.orders_file = self.bridge_dir / "orders.json"
        self.positions_file = self.bridge_dir / "positions.json"
        self.account_file = self.bridge_dir / "account.json"
        self.commands_file = self.bridge_dir / "commands.json"
        self.response_file = self.bridge_dir / "response.json"

        # 确保目录存在
        self.bridge_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(self, filepath: Path) -> Optional[Dict]:
        """读取JSON文件"""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"读取 {filepath.name} 失败: {e}")
        return None

    def _write_json(self, filepath: Path, data: Dict):
        """写入JSON文件 (紧凑格式, 无空格, 确保EA能正确解析)"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, separators=(',', ':'), default=str)
        except IOError as e:
            log.error(f"写入 {filepath.name} 失败: {e}")

    def get_account(self) -> Optional[Dict]:
        """获取账户信息"""
        return self._read_json(self.account_file)

    def get_positions(self) -> List[Dict]:
        """获取当前持仓"""
        data = self._read_json(self.positions_file)
        if data and 'positions' in data:
            return data['positions']
        return []

    def get_open_orders(self) -> List[Dict]:
        """获取挂单"""
        data = self._read_json(self.orders_file)
        if data and 'orders' in data:
            return data['orders']
        return []

    def send_order(self, symbol: str, order_type: str, lots: float,
                   price: float = 0, sl: float = 0, tp: float = 0,
                   comment: str = "", magic: int = 0) -> bool:
        """
        发送交易指令

        Args:
            symbol: 交易品种 (XAUUSD)
            order_type: 订单类型 (BUY, SELL, BUYLIMIT, SELLLIMIT, BUYSTOP, SELLSTOP)
            lots: 手数
            price: 价格 (市价单传0)
            sl: 止损价
            tp: 止盈价 (0=不设)
            comment: 订单备注
            magic: 魔术号
        """
        command = {
            "action": "OPEN",
            "symbol": symbol,
            "type": order_type,
            "lots": lots,
            "price": price,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "magic": magic or config.MAGIC_NUMBER,
            "slippage": config.SLIPPAGE,
            "timestamp": datetime.now().isoformat(),
        }

        self._clear_response()  # 清除旧响应
        self._write_json(self.commands_file, command)
        log.info(f"📤 发送指令: {order_type} {symbol} {lots}手 SL={sl} TP={tp}")

        # 等待EA响应
        return self._wait_response(timeout=10)

    def close_order(self, ticket: int) -> bool:
        """平仓指定订单"""
        command = {
            "action": "CLOSE",
            "ticket": ticket,
            "timestamp": datetime.now().isoformat(),
        }

        self._clear_response()  # 清除旧响应
        self._write_json(self.commands_file, command)
        log.info(f"📤 发送平仓指令: ticket={ticket}")

        return self._wait_response(timeout=10)

    def modify_order(self, ticket: int, sl: float = 0, tp: float = 0) -> bool:
        """修改订单止损止盈"""
        command = {
            "action": "MODIFY",
            "ticket": ticket,
            "sl": sl,
            "tp": tp,
            "timestamp": datetime.now().isoformat(),
        }

        self._clear_response()  # 清除旧响应
        self._write_json(self.commands_file, command)
        return self._wait_response(timeout=10)

    def _clear_response(self):
        """清除旧的响应文件（在发送指令前调用）"""
        try:
            if self.response_file.exists():
                self.response_file.unlink()
        except OSError:
            pass

    def _wait_response(self, timeout: int = 10) -> bool:
        """等待EA执行结果"""
        start = time.time()
        while time.time() - start < timeout:
            resp = self._read_json(self.response_file)
            if resp and 'success' in resp:
                success = resp.get('success', False)
                message = resp.get('message', '')
                if success:
                    log.info(f"✅ 执行成功: {message}")
                else:
                    log.error(f"❌ 执行失败: {message}")
                # 清除响应文件
                try:
                    self.response_file.unlink()
                except OSError:
                    pass
                return success
            time.sleep(0.5)

        log.warning("⏰ 等待EA响应超时 (10秒)")
        return False

    def is_connected(self) -> bool:
        """检查EA是否在线 (通过心跳文件)"""
        heartbeat = self.bridge_dir / "heartbeat.json"
        data = self._read_json(heartbeat)
        if data:
            last_beat = data.get('timestamp', '')
            try:
                # MT4 TimeToString格式: "2026.03.28 01:05:00"
                beat_time = datetime.strptime(last_beat, "%Y.%m.%d %H:%M:%S")
                if (datetime.now() - beat_time).total_seconds() < 30:
                    return True
            except ValueError:
                pass
        return False

    def buy(self, lots: float = None, sl_pips: float = None, tp_pips: float = 0, comment: str = "") -> bool:
        """市价买入 + 自动止损止盈"""
        lots = lots or config.LOT_SIZE
        sl_pips = sl_pips or config.STOP_LOSS_PIPS

        # 获取当前价格来计算止损止盈
        account = self.get_account()
        if account and 'bid' in account:
            current_price = account['bid']
            sl_price = round(current_price - sl_pips, 2)
            tp_price = round(current_price + tp_pips, 2) if tp_pips > 0 else 0
        else:
            sl_price = 0
            tp_price = 0

        return self.send_order(
            symbol=config.SYMBOL,
            order_type="BUY",
            lots=lots,
            sl=sl_price,
            tp=tp_price,
            comment=comment,
            magic=config.MAGIC_NUMBER,
        )

    def sell(self, lots: float = None, sl_pips: float = None, tp_pips: float = 0, comment: str = "") -> bool:
        """市价卖出 + 自动止损止盈"""
        lots = lots or config.LOT_SIZE
        sl_pips = sl_pips or config.STOP_LOSS_PIPS

        account = self.get_account()
        if account and 'ask' in account:
            current_price = account['ask']
            sl_price = round(current_price + sl_pips, 2)
            tp_price = round(current_price - tp_pips, 2) if tp_pips > 0 else 0
        else:
            sl_price = 0
            tp_price = 0

        return self.send_order(
            symbol=config.SYMBOL,
            order_type="SELL",
            lots=lots,
            sl=sl_price,
            tp=tp_price,
            comment=comment,
            magic=config.MAGIC_NUMBER,
        )
