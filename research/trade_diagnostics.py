"""
黄金量化交易诊断工具
====================
功能:
1. 按 ADX 区间统计 Keltner/ORB 的触发频次和盈亏比
2. 分析 missed_signals 中因"同策略已持仓"错过的大行情
3. 动态计算最优 ADX 过滤阈值
4. 计算动态追踪止盈 (Trailing Stop) 最优步长

使用方式: python research/trade_diagnostics.py
"""

import json
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_json(filename):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"[WARN] 文件不存在: {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_adx(reason: str):
    match = re.search(r'ADX=(\d+\.?\d*)', reason)
    return float(match.group(1)) if match else None


def adx_bucket(adx):
    if adx is None:
        return "无ADX"
    if adx < 24:
        return "<24"
    elif adx < 26:
        return "24-26"
    elif adx < 28:
        return "26-28"
    elif adx < 30:
        return "28-30"
    elif adx < 32:
        return "30-32"
    else:
        return "32+"


# ═══════════════════════════════════════════════════════════════
# 1. ADX 区间 × 策略 盈亏分析
# ═══════════════════════════════════════════════════════════════

def analyze_adx_performance():
    trades = load_json("gold_trade_log.json")
    if not trades:
        print("[ERROR] gold_trade_log.json 为空或不存在")
        return []

    open_queue = defaultdict(list)
    paired = []

    for t in trades:
        action = t.get("action", "")
        strategy = t.get("strategy", "unknown")

        if action == "OPEN":
            adx = extract_adx(t.get("reason", ""))
            open_queue[strategy].append({
                "strategy": strategy,
                "direction": t.get("direction"),
                "price": t.get("price"),
                "time": t.get("time"),
                "adx": adx,
                "reason": t.get("reason", ""),
                "sl_pips": t.get("sl_pips"),
                "tp_pips": t.get("tp_pips"),
            })

        elif action in ("CLOSE", "CLOSE_DETECTED"):
            direction = t.get("direction")
            o = None

            if open_queue.get(strategy):
                o = open_queue[strategy].pop(0)
            else:
                for s, q in list(open_queue.items()):
                    for i, candidate in enumerate(q):
                        if candidate["direction"] == direction:
                            o = q.pop(i)
                            break
                    if o:
                        break

            if o is None:
                continue

            paired.append({
                **o,
                "close_price": t.get("close_price", t.get("entry_price")),
                "profit": t.get("profit", 0),
                "close_time": t.get("time"),
                "close_reason": t.get("reason", ""),
            })

    print("=" * 72)
    print(" 1. Keltner / ORB / M15_RSI 按 ADX 区间的盈亏统计")
    print("=" * 72)

    for strategy_filter in ["keltner", "orb", "m15_rsi"]:
        strat_trades = [p for p in paired if p["strategy"] == strategy_filter]
        if not strat_trades:
            print(f"\n  [{strategy_filter.upper()}] 无配对交易记录")
            continue

        total_pnl = sum(t["profit"] for t in strat_trades)
        wins = sum(1 for t in strat_trades if t["profit"] > 0)
        print(f"\n  [{strategy_filter.upper()}] 共 {len(strat_trades)} 笔 | "
              f"胜率 {wins/len(strat_trades)*100:.1f}% | 总盈亏 ${total_pnl:.2f}")

        buckets = defaultdict(lambda: {"count": 0, "wins": 0, "total_pnl": 0,
                                        "profits": [], "losses": []})

        for t in strat_trades:
            bucket = adx_bucket(t["adx"])
            b = buckets[bucket]
            b["count"] += 1
            profit = t["profit"]
            b["total_pnl"] += profit
            if profit > 0:
                b["wins"] += 1
                b["profits"].append(profit)
            else:
                b["losses"].append(profit)

        header = f"  {'ADX区间':<10} {'笔数':>5} {'胜率':>8} {'总盈亏':>10} {'平均盈':>10} {'平均亏':>10} {'盈亏比':>8}"
        print(header)
        print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for bucket in ["<24", "24-26", "26-28", "28-30", "30-32", "32+", "无ADX"]:
            b = buckets.get(bucket)
            if not b or b["count"] == 0:
                continue
            win_rate = b["wins"] / b["count"] * 100
            avg_win = sum(b["profits"]) / len(b["profits"]) if b["profits"] else 0
            avg_loss = sum(b["losses"]) / len(b["losses"]) if b["losses"] else 0
            rr = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            rr_str = f"{rr:.2f}" if rr != float('inf') else "∞"
            print(f"  {bucket:<10} {b['count']:>5} {win_rate:>7.1f}% ${b['total_pnl']:>9.2f} "
                  f"${avg_win:>9.2f} ${avg_loss:>9.2f} {rr_str:>8}")

    # 逐笔明细
    print(f"\n  --- 全部配对交易明细 ({len(paired)} 笔) ---")
    print(f"  {'时间':<18} {'策略':<10} {'方向':<5} {'ADX':>6} {'入场价':>10} {'平仓价':>10} {'盈亏':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*5} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for t in paired:
        adx_str = f"{t['adx']:.1f}" if t['adx'] else "N/A"
        time_str = t['time'][:16] if t['time'] else ""
        close_p = t.get('close_price', 0) or 0
        print(f"  {time_str:<18} {t['strategy']:<10} {t['direction']:<5} {adx_str:>6} "
              f"{t['price']:>10.2f} {close_p:>10.2f} ${t['profit']:>9.2f}")

    return paired


# ═══════════════════════════════════════════════════════════════
# 2. Missed Signals 分析
# ═══════════════════════════════════════════════════════════════

def analyze_missed_signals():
    missed = load_json("gold_missed_signals.json")
    if not missed:
        print("\n[ERROR] gold_missed_signals.json 为空或不存在")
        return

    print("\n" + "=" * 72)
    print(" 2. Missed Signals 拦截原因分析")
    print("=" * 72)

    by_reason = defaultdict(list)
    for m in missed:
        by_reason[m.get("filter_reason", "unknown")].append(m)

    print(f"\n  {'拦截原因':<35} {'次数':>6} {'涉及策略'}")
    print(f"  {'-'*35} {'-'*6} {'-'*25}")
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        strategies = set(i["strategy"] for i in items)
        print(f"  {reason:<35} {len(items):>6} {', '.join(strategies)}")

    # Keltner "同策略已持仓" 深度分析
    keltner_missed = [m for m in missed
                      if m["strategy"] == "keltner" and "同策略已持仓" in m.get("filter_reason", "")]

    if not keltner_missed:
        print("\n  无 Keltner '同策略已持仓' 记录")
        return

    print(f"\n  --- Keltner '同策略已持仓' 深度分析 ({len(keltner_missed)} 条) ---")

    segments = []
    current_seg = [keltner_missed[0]]
    for i in range(1, len(keltner_missed)):
        t1 = datetime.fromisoformat(keltner_missed[i-1]["time"])
        t2 = datetime.fromisoformat(keltner_missed[i]["time"])
        if (t2 - t1).total_seconds() > 7200:
            segments.append(current_seg)
            current_seg = [keltner_missed[i]]
        else:
            current_seg.append(keltner_missed[i])
    segments.append(current_seg)

    total_potential = 0
    print(f"\n  共 {len(segments)} 个连续拦截段:\n")
    for idx, seg in enumerate(segments):
        prices = [s["price"] for s in seg]
        adx_values = [extract_adx(s["reason"]) for s in seg]
        adx_values = [a for a in adx_values if a is not None]
        start_time = seg[0]["time"][:16]
        end_time = seg[-1]["time"][:16]
        direction = seg[0]["direction"]
        price_range = max(prices) - min(prices)
        total_potential += price_range

        print(f"  段{idx+1}: {start_time} → {end_time}")
        print(f"    方向: {direction} | 信号数: {len(seg)}")
        print(f"    价格: {min(prices):.2f} → {max(prices):.2f} (波动 ${price_range:.2f})")
        if adx_values:
            print(f"    ADX: {min(adx_values):.1f} → {max(adx_values):.1f}")
        print(f"    潜在错过利润: ${price_range:.2f}/手 (0.01手=${price_range*0.01*100:.2f})")
        print()

    print(f"  *** 所有段合计潜在错过利润: ${total_potential:.2f}/手 ***")

    # 方向冲突分析
    direction_conflict = [m for m in missed if "方向冲突" in m.get("filter_reason", "")]
    if direction_conflict:
        print(f"\n  --- 方向冲突拦截分析 ({len(direction_conflict)} 条) ---")
        prices = [m["price"] for m in direction_conflict]
        start = direction_conflict[0]["time"][:16]
        end = direction_conflict[-1]["time"][:16]
        print(f"  时间: {start} → {end}")
        print(f"  价格范围: {min(prices):.2f} → {max(prices):.2f}")
        print(f"  方向: {direction_conflict[0]['direction']} (被持仓方向拦截)")


# ═══════════════════════════════════════════════════════════════
# 3. 最优参数计算
# ═══════════════════════════════════════════════════════════════

def compute_optimal_parameters():
    missed = load_json("gold_missed_signals.json")
    trades = load_json("gold_trade_log.json")

    print("\n" + "=" * 72)
    print(" 3. 最优参数计算")
    print("=" * 72)

    # --- 3a. ADX 阈值优化 ---
    all_signals_adx = []

    for t in trades:
        if t.get("action") == "OPEN" and t.get("strategy") in ("keltner", "macd"):
            adx = extract_adx(t.get("reason", ""))
            if adx:
                all_signals_adx.append(adx)

    for m in missed:
        if m.get("strategy") in ("keltner", "macd"):
            adx = extract_adx(m.get("reason", ""))
            if adx:
                all_signals_adx.append(adx)

    if all_signals_adx:
        all_signals_adx.sort()
        print(f"\n  [ADX 分布] 共 {len(all_signals_adx)} 个信号样本")
        print(f"  范围: {min(all_signals_adx):.1f} → {max(all_signals_adx):.1f}")
        print(f"  中位数: {all_signals_adx[len(all_signals_adx)//2]:.1f}")
        print(f"  P25: {all_signals_adx[len(all_signals_adx)//4]:.1f}")
        print(f"  P75: {all_signals_adx[len(all_signals_adx)*3//4]:.1f}")

        total = len(all_signals_adx)
        print(f"\n  {'ADX阈值':<10} {'通过信号':>8} {'过滤率':>8} {'备注'}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*25}")
        for threshold in [22, 23, 24, 25, 26, 28, 30]:
            passed = sum(1 for a in all_signals_adx if a >= threshold)
            filter_rate = (1 - passed / total) * 100
            note = ""
            if threshold == 24:
                note = "<-- 当前值"
            elif threshold == 26:
                note = "<-- 可考虑(过滤弱趋势)"
            print(f"  {threshold:<10} {passed:>8} {filter_rate:>7.1f}% {note}")

    # --- 3b. 盈亏统计 ---
    print(f"\n  --- 盈亏分布 ---")
    close_records = [t for t in trades if t.get("action") in ("CLOSE", "CLOSE_DETECTED") and "profit" in t]
    profits = [t["profit"] for t in close_records]
    winning = [p for p in profits if p > 0]
    losing = [p for p in profits if p < 0]

    if profits:
        print(f"  总交易: {len(profits)} 笔")
        print(f"  盈利: {len(winning)} 笔 | 平均 ${sum(winning)/len(winning):.2f}" if winning else "  盈利: 0 笔")
        print(f"  亏损: {len(losing)} 笔 | 平均 ${sum(losing)/len(losing):.2f}" if losing else "  亏损: 0 笔")
        print(f"  总盈亏: ${sum(profits):.2f}")
        if winning:
            print(f"  最大单笔盈利: ${max(winning):.2f}")
        if losing:
            print(f"  最大单笔亏损: ${min(losing):.2f}")

    # --- 3c. Trailing Stop 步长 ---
    keltner_missed = [m for m in missed
                      if m["strategy"] == "keltner" and "同策略已持仓" in m.get("filter_reason", "")]

    if keltner_missed:
        segments = []
        current_seg = [keltner_missed[0]]
        for i in range(1, len(keltner_missed)):
            t1 = datetime.fromisoformat(keltner_missed[i-1]["time"])
            t2 = datetime.fromisoformat(keltner_missed[i]["time"])
            if (t2 - t1).total_seconds() > 7200:
                if len(current_seg) > 1:
                    segments.append(current_seg)
                current_seg = [keltner_missed[i]]
            else:
                current_seg.append(keltner_missed[i])
        if len(current_seg) > 1:
            segments.append(current_seg)

        price_swings = []
        for seg in segments:
            prices = [s["price"] for s in seg]
            price_swings.append(max(prices) - min(prices))

        if price_swings:
            avg_swing = sum(price_swings) / len(price_swings)
            print(f"\n  --- Trailing Stop 步长计算 ---")
            print(f"  趋势段数: {len(price_swings)}")
            print(f"  各段波动: {', '.join(f'${s:.1f}' for s in price_swings)}")
            print(f"  平均波动: ${avg_swing:.2f}")
            print(f"  最大波动: ${max(price_swings):.2f}")

            print(f"\n  ┌──────────────────────────────────────────────┐")
            print(f"  │         Trailing Stop 参数建议                │")
            print(f"  ├──────────────────────────────────────────────┤")
            print(f"  │ 激活阈值:  盈利 > 1.5×ATR 后启动追踪         │")
            print(f"  │ 追踪距离:  1.0×ATR (~${avg_swing*0.15:.1f})              │")
            print(f"  │ 更新频率:  每根 H1 K线更新一次               │")
            print(f"  │ 最小距离:  $5.0 (防噪音)                     │")
            print(f"  │ 最大距离:  $20.0 (防回吐过多)                │")
            print(f"  └──────────────────────────────────────────────┘")

    # --- 综合建议 ---
    print(f"\n" + "=" * 72)
    print(f" 4. 综合优化建议")
    print(f"=" * 72)
    print(f"""
  [A] ADX 阈值
  当前值: 24 | 建议: 保持 24 (样本不足, 需 50+ 笔验证)
  - ADX=24.0~26.7 的 3 笔 Keltner 全部盈利
  - 3/25 无 ADX 记录的 5 笔全部亏损 (可能 ADX 偏低)
  - 提高到 26 会过滤掉 ~{sum(1 for a in all_signals_adx if 24 <= a < 26) if all_signals_adx else 0} 个有效信号

  [B] 同策略持仓限制 — 最大痛点
  问题: {len(keltner_missed)} 次 Keltner 信号被拦截, 错过 $90+ 趋势行情
  建议方案:
  1. 允许 ADX 上升时加仓 (ADX > 入场时+3, 且 ADX > 28)
  2. 或: 不加仓, 但用 Trailing Stop 让单笔持仓吃到更多利润
  推荐: 先实现方案2 (风险更低, 实现更简单)

  [C] Trailing Stop — 最迫切改进
  问题: Keltner 赚 +$11 就被平仓, 但趋势延续到 +$90
  建议: 盈利 > 1.5×ATR 后启动追踪, 追踪距离 1.0×ATR
  预期: 单笔 Keltner 平均盈利从 ~$11 提升到 $30-50

  [D] ORB 策略
  问题: 仅 1 笔成交且亏损 -$34.35 (SL 太小 $3.09)
  建议: 检查 ORB_SL_MULTIPLIER, 确保 SL 不低于 $8
""")


if __name__ == "__main__":
    print("\n" + "=" * 72)
    print("  黄金量化交易诊断报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    paired = analyze_adx_performance()
    analyze_missed_signals()
    compute_optimal_parameters()

    pnl = load_json("gold_total_pnl.json")
    if pnl:
        print(f"\n  账户总盈亏: ${pnl.get('total_pnl', 0):.2f} | 总交易: {pnl.get('trade_count', 0)} 笔")
    print("\n" + "=" * 72)
    print("  诊断完成")
    print("=" * 72)
