# 🥇 黄金 XAU/USD 量化交易系统

基于 21 年历史数据回测的黄金量化交易策略，通过 MT4 文件桥接自动执行。

## 策略 (基于COMEX黄金期货GC=F真实价格回测 2005-2026)

| 策略 | Sharpe | 胜率 | 单笔均收 | 最大回撤 | 年均交易 |
|---|---|---|---|---|---|
| 布林带均值回归 | **2.21** | 75.0% | +$9.6 | -8.9% | 2笔 |
| 窄幅突破 | **1.27** | 43.2% | +$13.6 | -9.9% | 5笔 |
| ATR收缩突破 | **1.19** | 43.0% | +$21.8 | -20.0% | 4笔 |

## 风控参数

- 本金: $3,000
- 手数: 0.01手 (1盎司, $1/点)
- 单笔止损: 50点 = $50 (1.7%本金)
- 最大同时持仓: 2笔
- 总亏损上限: $1,500 (达到后自动停止)

## 安装步骤

### 1. Python 环境
```bash
pip install numpy pandas yfinance
```

### 2. 下载代码
```bash
git clone https://github.com/linhuang1313/gold-quant-trading.git
cd gold-quant-trading
```

### 3. MT4 设置
1. 打开 MT4，菜单 `文件` → `打开数据文件夹`
2. 将 `mt4_ea/GoldBridge_EA.mq4` 复制到 `MQL4/Experts/` 目录
3. 在 MT4 中编译 EA (Navigator → Expert Advisors → 右键 Compile)
4. 打开 XAUUSD 图表，将 EA 拖到图表上
5. 确保开启 `自动交易` 和 `允许DLL导入`

### 4. 配置
编辑 `config.py`:
```python
# 修改为你的MT4数据文件夹路径
METATRADER_DIR_PATH = r"C:\Users\你的用户名\AppData\Roaming\MetaQuotes\Terminal\你的ID"
```

在 MT4 中查找路径: `文件` → `打开数据文件夹`

### 5. 运行
```bash
python gold_runner.py
```

## 架构

```
Python (gold_runner.py)          MT4 (GoldBridge_EA.mq4)
      │                                │
      ├─ 每60秒扫描信号                  ├─ 每500ms检查指令
      ├─ 写 commands.json ──────────→  ├─ 读取并执行交易
      ├─ 读 positions.json ←────────── ├─ 写入持仓状态
      ├─ 读 account.json  ←────────── ├─ 写入账户信息
      └─ 读 heartbeat.json ←───────── └─ 每5秒心跳
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `gold_runner.py` | 主程序，24/5循环运行 |
| `gold_trader.py` | 交易引擎，管理信号和持仓 |
| `mt4_bridge.py` | MT4文件桥接模块 |
| `strategies/signals.py` | 3种策略信号引擎 |
| `config.py` | 所有参数配置 |
| `mt4_ea/GoldBridge_EA.mq4` | MT4 EA (桥接服务端) |

## 注意事项

- 程序必须和 MT4 在同一台电脑上运行
- MT4 必须保持打开且 EA 已启用
- 首次运行建议用模拟账户测试
- 日志文件在 `logs/gold_runner.log`
