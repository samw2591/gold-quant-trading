"""
因子深度研究
============
在 factor_scanner.py 的 IC 扫描基础上，对有价值的因子做深入分析：
1. day_of_week 周内择时：各星期几的收益分布
2. ADX 分箱分析：非线性条件过滤效果
3. 因子交叉组合：RSI2 × ATR_percentile 等
4. volume_ratio 诊断：为什么 IC ≈ 0

用法:
    python factor_deep_dive.py <csv_path>
"""
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from factor_scanner import load_data, compute_extended_factors, _fast_rank_corr


def build_matrix(df: pd.DataFrame, horizons=(1, 4, 8), warmup=150):
    """构建因子-收益矩阵（复用 factor_scanner 逻辑）。"""
    result = df.copy()
    if 'is_flat' in result.columns:
        result = result[~result['is_flat']]

    close = result['Close']
    for h in horizons:
        future = close.shift(-h)
        result[f'ret_{h}'] = (future - close) / close

    result = result.iloc[warmup:]
    max_h = max(horizons)
    result = result.iloc[:-max_h]
    return result


# ═══════════════════════════════════════════════════════════════
# 研究1: day_of_week 周内择时
# ═══════════════════════════════════════════════════════════════

def study_day_of_week(m: pd.DataFrame):
    print()
    print("=" * 70)
    print("  研究1: day_of_week 周内择时 (IC=+0.033, WF=100%)")
    print("=" * 70)

    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}

    for h in [1, 4, 8]:
        ret_col = f'ret_{h}'
        if ret_col not in m.columns:
            continue

        print(f"\n  --- 未来 {h} bar 收益 ---")
        print(f"  {'星期':<6} {'均值(bp)':>10} {'中位数(bp)':>10} {'标准差(bp)':>10} "
              f"{'正收益%':>8} {'样本':>8} {'t-stat':>8} {'p-value':>8}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        overall_mean = m[ret_col].mean()

        for dow in range(5):
            subset = m[m.index.dayofweek == dow][ret_col].dropna()
            if len(subset) < 100:
                continue

            mean_bp = subset.mean() * 10000
            median_bp = subset.median() * 10000
            std_bp = subset.std() * 10000
            pos_pct = (subset > 0).mean() * 100
            n = len(subset)

            # t-test: 该星期收益是否显著不等于总体均值
            t_stat, p_val = stats.ttest_1samp(subset, overall_mean)

            name = day_names.get(dow, str(dow))
            sig = '*' if p_val < 0.05 else ' '
            print(f"  {name:<6} {mean_bp:>+10.2f} {median_bp:>+10.2f} {std_bp:>10.2f} "
                  f"{pos_pct:>7.1f}% {n:>8} {t_stat:>+8.2f} {p_val:>8.4f}{sig}")

        # 最优/最差星期
        day_means = {}
        for dow in range(5):
            subset = m[m.index.dayofweek == dow][ret_col].dropna()
            if len(subset) > 0:
                day_means[dow] = subset.mean()

        if day_means:
            best = max(day_means, key=day_means.get)
            worst = min(day_means, key=day_means.get)
            print(f"\n  最优: {day_names[best]} ({day_means[best]*10000:+.2f}bp)  "
                  f"最差: {day_names[worst]} ({day_means[worst]*10000:+.2f}bp)")

    # 分年度稳定性检查
    print(f"\n  --- 分年度 day_of_week 效应稳定性 (ret_4) ---")
    print(f"  {'年份':<6}", end='')
    for dow in range(5):
        print(f"  {day_names[dow]:>8}", end='')
    print(f"  {'最优':>6}  {'最差':>6}")

    years = sorted(m.index.year.unique())
    consistency_best = []
    consistency_worst = []

    for year in years:
        year_data = m[m.index.year == year]
        if len(year_data) < 200:
            continue
        print(f"  {year:<6}", end='')
        year_day_means = {}
        for dow in range(5):
            subset = year_data[year_data.index.dayofweek == dow]['ret_4'].dropna()
            if len(subset) > 0:
                mean_bp = subset.mean() * 10000
                year_day_means[dow] = mean_bp
                print(f"  {mean_bp:>+8.2f}", end='')
            else:
                print(f"  {'N/A':>8}", end='')

        if year_day_means:
            best_d = max(year_day_means, key=year_day_means.get)
            worst_d = min(year_day_means, key=year_day_means.get)
            print(f"  {day_names[best_d]:>6}  {day_names[worst_d]:>6}")
            consistency_best.append(best_d)
            consistency_worst.append(worst_d)
        else:
            print()

    # 统计哪些星期反复出现为最优/最差
    if consistency_best:
        from collections import Counter
        best_counts = Counter(consistency_best)
        worst_counts = Counter(consistency_worst)
        print(f"\n  最优日出现频率: {', '.join(f'{day_names[d]}={c}次' for d, c in best_counts.most_common())}")
        print(f"  最差日出现频率: {', '.join(f'{day_names[d]}={c}次' for d, c in worst_counts.most_common())}")


# ═══════════════════════════════════════════════════════════════
# 研究2: ADX 分箱分析（非线性条件过滤）
# ═══════════════════════════════════════════════════════════════

def study_adx_bins(m: pd.DataFrame):
    print()
    print("=" * 70)
    print("  研究2: ADX 分箱分析 (整体 IC~0, 但可能有非线性效果)")
    print("=" * 70)

    adx = m['ADX']
    bins = [0, 15, 20, 25, 30, 40, 100]
    labels = ['<15', '15-20', '20-25', '25-30', '30-40', '>40']
    m_copy = m.copy()
    m_copy['ADX_bin'] = pd.cut(adx, bins=bins, labels=labels, right=False)

    for h in [1, 4, 8]:
        ret_col = f'ret_{h}'
        if ret_col not in m.columns:
            continue

        print(f"\n  --- ret_{h} 在不同 ADX 区间的表现 ---")
        print(f"  {'ADX区间':<10} {'均值(bp)':>10} {'标准差(bp)':>10} {'正收益%':>8} "
              f"{'样本':>8} {'|Sharpe|':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

        for label in labels:
            subset = m_copy[m_copy['ADX_bin'] == label][ret_col].dropna()
            if len(subset) < 50:
                continue
            mean_bp = subset.mean() * 10000
            std_bp = subset.std() * 10000
            pos_pct = (subset > 0).mean() * 100
            sharpe = abs(mean_bp / std_bp) if std_bp > 0 else 0
            print(f"  {label:<10} {mean_bp:>+10.2f} {std_bp:>10.2f} {pos_pct:>7.1f}% "
                  f"{len(subset):>8} {sharpe:>8.3f}")

    # ADX 对 Keltner 突破方向收益的条件分析
    print(f"\n  --- ADX 对 Keltner 突破信号质量的影响 ---")
    kc_upper = m['KC_upper']
    kc_lower = m['KC_lower']
    close = m['Close']

    bullish_breakout = close > kc_upper
    bearish_breakout = close < kc_lower

    for h in [1, 4]:
        ret_col = f'ret_{h}'
        print(f"\n  ADX 条件 × Keltner 突破 → ret_{h}:")
        print(f"  {'条件':<30} {'均值(bp)':>10} {'胜率%':>8} {'样本':>8}")
        print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8}")

        for adx_thresh in [20, 24, 28, 32]:
            adx_high = adx >= adx_thresh

            bull = m[bullish_breakout & adx_high][ret_col].dropna()
            bear = m[bearish_breakout & adx_high][ret_col].dropna()

            if len(bull) > 20:
                print(f"  BUY  (ADX>={adx_thresh}, close>KC_upper)  "
                      f"{bull.mean()*10000:>+10.2f} {(bull>0).mean()*100:>7.1f}% {len(bull):>8}")
            if len(bear) > 20:
                print(f"  SELL (ADX>={adx_thresh}, close<KC_lower)  "
                      f"{(-bear).mean()*10000:>+10.2f} {(bear<0).mean()*100:>7.1f}% {len(bear):>8}")


# ═══════════════════════════════════════════════════════════════
# 研究3: 因子交叉组合
# ═══════════════════════════════════════════════════════════════

def study_factor_combinations(m: pd.DataFrame):
    print()
    print("=" * 70)
    print("  研究3: 因子交叉组合效应")
    print("=" * 70)

    ret_col = 'ret_1'

    combos = [
        ("RSI2<15 (超卖)", m['RSI2'] < 15, "BUY"),
        ("RSI2<15 + ATR_pct>70", (m['RSI2'] < 15) & (m['ATR_percentile'] > 0.7), "BUY"),
        ("RSI2<15 + ATR_pct<30", (m['RSI2'] < 15) & (m['ATR_percentile'] < 0.3), "BUY"),
        ("RSI2<15 + close>EMA100", (m['RSI2'] < 15) & (m['Close'] > m['EMA100']), "BUY"),
        ("RSI2<15 + close<EMA100", (m['RSI2'] < 15) & (m['Close'] < m['EMA100']), "BUY"),
        ("RSI2>85 (超买)", m['RSI2'] > 85, "SELL"),
        ("RSI2>85 + ATR_pct>70", (m['RSI2'] > 85) & (m['ATR_percentile'] > 0.7), "SELL"),
        ("RSI2>85 + ATR_pct<30", (m['RSI2'] > 85) & (m['ATR_percentile'] < 0.3), "SELL"),
        ("RSI2>85 + close<EMA100", (m['RSI2'] > 85) & (m['Close'] < m['EMA100']), "SELL"),
        ("RSI2>85 + close>EMA100", (m['RSI2'] > 85) & (m['Close'] > m['EMA100']), "SELL"),
    ]

    for h in [1, 4]:
        ret_col = f'ret_{h}'
        print(f"\n  --- 因子组合 → ret_{h} ---")
        print(f"  {'条件':<35} {'方向':<5} {'均值(bp)':>10} {'胜率%':>8} {'样本':>8} {'vs基准':>10}")
        print(f"  {'-'*35} {'-'*5} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

        baseline_mean = m[ret_col].mean()

        for name, mask, direction in combos:
            subset = m[mask][ret_col].dropna()
            if len(subset) < 20:
                continue

            if direction == "SELL":
                effective_ret = -subset
            else:
                effective_ret = subset

            mean_bp = effective_ret.mean() * 10000
            win_rate = (effective_ret > 0).mean() * 100
            vs_base = mean_bp - baseline_mean * 10000

            print(f"  {name:<35} {direction:<5} {mean_bp:>+10.2f} {win_rate:>7.1f}% "
                  f"{len(subset):>8} {vs_base:>+10.2f}")

    # 动量 + 波动率组合
    print(f"\n  --- 动量 × 波动率组合 → ret_4 ---")
    ret_col = 'ret_4'
    mom = m['momentum_5']
    atr_pct = m['ATR_percentile']

    mom_bins = [(-np.inf, -0.01), (-0.01, 0), (0, 0.01), (0.01, np.inf)]
    atr_bins = [(0, 0.3), (0.3, 0.7), (0.7, 1.0)]

    print(f"  {'':>20}", end='')
    for al, ah in atr_bins:
        print(f"  ATR_pct {al:.0%}-{ah:.0%}", end='')
    print()

    for ml, mh in mom_bins:
        label = f"mom5 [{ml:+.0%},{mh:+.0%})"
        if ml == -np.inf:
            label = "mom5 < -1%"
        elif mh == np.inf:
            label = "mom5 > +1%"
        print(f"  {label:>20}", end='')

        for al, ah in atr_bins:
            mask = (mom >= ml) & (mom < mh) & (atr_pct >= al) & (atr_pct < ah)
            subset = m[mask][ret_col].dropna()
            if len(subset) >= 20:
                mean_bp = subset.mean() * 10000
                print(f"  {mean_bp:>+7.1f}bp({len(subset):>4})", end='')
            else:
                print(f"  {'N/A':>15}", end='')
        print()


# ═══════════════════════════════════════════════════════════════
# 研究4: volume_ratio 诊断
# ═══════════════════════════════════════════════════════════════

def study_volume_ratio(m: pd.DataFrame):
    print()
    print("=" * 70)
    print("  研究4: volume_ratio 诊断 (整体 IC~0)")
    print("=" * 70)

    vol_ratio = m['volume_ratio']

    # 基本统计
    print(f"\n  volume_ratio 基本统计:")
    print(f"    均值: {vol_ratio.mean():.3f}")
    print(f"    中位数: {vol_ratio.median():.3f}")
    print(f"    P10: {vol_ratio.quantile(0.1):.3f}")
    print(f"    P90: {vol_ratio.quantile(0.9):.3f}")
    print(f"    非零占比: {(vol_ratio > 0).mean()*100:.1f}%")
    print(f"    零值占比: {(vol_ratio == 0).mean()*100:.1f}%")

    # 分箱分析
    bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 100]
    labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '1.5-2.0', '>2.0']
    m_copy = m.copy()
    m_copy['vol_bin'] = pd.cut(vol_ratio, bins=bins, labels=labels, right=False)

    for h in [1, 4]:
        ret_col = f'ret_{h}'
        print(f"\n  volume_ratio 分箱 → ret_{h}:")
        print(f"  {'区间':<10} {'均值(bp)':>10} {'|均值|(bp)':>12} {'波动(bp)':>10} {'样本':>8}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

        for label in labels:
            subset = m_copy[m_copy['vol_bin'] == label][ret_col].dropna()
            if len(subset) < 20:
                continue
            mean_bp = subset.mean() * 10000
            abs_mean_bp = subset.abs().mean() * 10000
            std_bp = subset.std() * 10000
            print(f"  {label:<10} {mean_bp:>+10.2f} {abs_mean_bp:>12.2f} {std_bp:>10.2f} {len(subset):>8}")

    # 放量 + Keltner 突破组合
    print(f"\n  volume_ratio × Keltner 突破 → ret_4:")
    kc_upper = m['KC_upper']
    kc_lower = m['KC_lower']
    close = m['Close']

    for vol_thresh in [1.0, 1.2, 1.5, 2.0]:
        high_vol = vol_ratio >= vol_thresh

        bull_hv = m[(close > kc_upper) & high_vol]['ret_4'].dropna()
        bull_lv = m[(close > kc_upper) & ~high_vol]['ret_4'].dropna()

        bear_hv = m[(close < kc_lower) & high_vol]['ret_4'].dropna()
        bear_lv = m[(close < kc_lower) & ~high_vol]['ret_4'].dropna()

        print(f"\n  vol_ratio >= {vol_thresh}:")
        if len(bull_hv) > 10 and len(bull_lv) > 10:
            print(f"    BUY+放量:  {bull_hv.mean()*10000:>+8.2f}bp 胜率{(bull_hv>0).mean()*100:.0f}% (n={len(bull_hv)})")
            print(f"    BUY+缩量:  {bull_lv.mean()*10000:>+8.2f}bp 胜率{(bull_lv>0).mean()*100:.0f}% (n={len(bull_lv)})")
        if len(bear_hv) > 10 and len(bear_lv) > 10:
            print(f"    SELL+放量: {(-bear_hv).mean()*10000:>+8.2f}bp 胜率{(bear_hv<0).mean()*100:.0f}% (n={len(bear_hv)})")
            print(f"    SELL+缩量: {(-bear_lv).mean()*10000:>+8.2f}bp 胜率{(bear_lv<0).mean()*100:.0f}% (n={len(bear_lv)})")


# ═══════════════════════════════════════════════════════════════
# 研究5: 时段分析（伦敦/纽约/亚洲）
# ═══════════════════════════════════════════════════════════════

def study_session_analysis(m: pd.DataFrame):
    print()
    print("=" * 70)
    print("  研究5: 交易时段分析 (is_london_ny_overlap IC=0.25 的真实含义)")
    print("=" * 70)

    hour = m.index.hour

    sessions = {
        'Asia (00-08 UTC)':    (hour >= 0) & (hour < 8),
        'London (08-13 UTC)':  (hour >= 8) & (hour < 13),
        'LN-NY overlap (13-17)': (hour >= 13) & (hour < 17),
        'NY only (17-21 UTC)': (hour >= 17) & (hour < 21),
        'Late (21-24 UTC)':    (hour >= 21),
    }

    for h in [1, 4]:
        ret_col = f'ret_{h}'
        print(f"\n  --- 各时段 ret_{h} ---")
        print(f"  {'时段':<25} {'均值(bp)':>10} {'|均值|(bp)':>12} {'波动(bp)':>10} "
              f"{'正收益%':>8} {'样本':>8}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

        for name, mask in sessions.items():
            subset = m[mask][ret_col].dropna()
            if len(subset) < 100:
                continue
            mean_bp = subset.mean() * 10000
            abs_mean_bp = subset.abs().mean() * 10000
            std_bp = subset.std() * 10000
            pos_pct = (subset > 0).mean() * 100
            print(f"  {name:<25} {mean_bp:>+10.2f} {abs_mean_bp:>12.2f} {std_bp:>10.2f} "
                  f"{pos_pct:>7.1f}% {len(subset):>8}")

    # 逐小时分析
    print(f"\n  --- 逐小时 ret_1 均值 (bp) ---")
    for h_val in range(24):
        subset = m[hour == h_val]['ret_1'].dropna()
        if len(subset) < 50:
            continue
        mean_bp = subset.mean() * 10000
        bar = '+' * int(abs(mean_bp) * 10) if mean_bp > 0 else '-' * int(abs(mean_bp) * 10)
        print(f"  {h_val:02d}:00 UTC  {mean_bp:>+7.2f}bp  {bar}")


# ═══════════════════════════════════════════════════════════════
# 研究6: 策略信号条件下的因子增量
# ═══════════════════════════════════════════════════════════════

def study_strategy_factor_filter(m: pd.DataFrame):
    print()
    print("=" * 70)
    print("  研究6: 在 Keltner 突破信号上叠加因子过滤的增量效果")
    print("=" * 70)

    close = m['Close']
    kc_upper = m['KC_upper']
    kc_lower = m['KC_lower']
    ema100 = m['EMA100']
    adx = m['ADX']

    base_buy = (close > kc_upper) & (close > ema100) & (adx >= 24)
    base_sell = (close < kc_lower) & (close < ema100) & (adx >= 24)

    filters = [
        ("基准 (Keltner v4)", base_buy, base_sell),
        ("+ ATR_pct > 50%", base_buy & (m['ATR_percentile'] > 0.5), base_sell & (m['ATR_percentile'] > 0.5)),
        ("+ ATR_pct > 70%", base_buy & (m['ATR_percentile'] > 0.7), base_sell & (m['ATR_percentile'] > 0.7)),
        ("+ momentum_5 > 0 (BUY)", base_buy & (m['momentum_5'] > 0), base_sell & (m['momentum_5'] < 0)),
        ("+ body_ratio > 0.5", base_buy & (m['body_range_ratio'] > 0.5), base_sell & (m['body_range_ratio'] > 0.5)),
        ("+ MACD_hist_chg 顺向", base_buy & (m['MACD_hist_change'] > 0), base_sell & (m['MACD_hist_change'] < 0)),
        ("+ 非周五", base_buy & (m.index.dayofweek != 4), base_sell & (m.index.dayofweek != 4)),
        ("+ LN-NY时段(13-17)", base_buy & (m.index.hour >= 13) & (m.index.hour < 17),
                                base_sell & (m.index.hour >= 13) & (m.index.hour < 17)),
    ]

    for h in [1, 4]:
        ret_col = f'ret_{h}'
        print(f"\n  --- Keltner + 过滤器 → ret_{h} ---")
        print(f"  {'过滤条件':<30} {'BUY均值bp':>10} {'BUY胜率':>8} {'BUY样本':>8} "
              f"{'SELL均值bp':>10} {'SELL胜率':>8} {'SELL样本':>8}")
        print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

        for name, buy_mask, sell_mask in filters:
            buy_ret = m[buy_mask][ret_col].dropna()
            sell_ret = m[sell_mask][ret_col].dropna()

            buy_bp = buy_ret.mean() * 10000 if len(buy_ret) > 0 else 0
            buy_wr = (buy_ret > 0).mean() * 100 if len(buy_ret) > 0 else 0
            sell_bp = (-sell_ret).mean() * 10000 if len(sell_ret) > 0 else 0
            sell_wr = (sell_ret < 0).mean() * 100 if len(sell_ret) > 0 else 0

            print(f"  {name:<30} {buy_bp:>+10.2f} {buy_wr:>7.1f}% {len(buy_ret):>8} "
                  f"{sell_bp:>+10.2f} {sell_wr:>7.1f}% {len(sell_ret):>8}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='因子深度研究')
    parser.add_argument('csv_path', help='历史数据 CSV')
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    args = parser.parse_args()

    print("加载数据...")
    df = load_data(args.csv_path, args.start, args.end)
    print(f"K 线: {len(df)}")

    print("计算指标...")
    df = compute_extended_factors(df)

    print("构建分析矩阵...")
    m = build_matrix(df)
    print(f"有效样本: {len(m)}")

    study_day_of_week(m)
    study_adx_bins(m)
    study_factor_combinations(m)
    study_volume_ratio(m)
    study_session_analysis(m)
    study_strategy_factor_filter(m)

    print("\n\n" + "=" * 70)
    print("  深度研究完成。")
    print("=" * 70)


if __name__ == '__main__':
    main()
