"""
因子 IC 离线扫描器
==================
在历史数据上系统性评估所有候选因子的预测力。

核心流程:
1. 加载 CSV → prepare_indicators() 计算技术指标
2. 逐 bar 采集因子快照 + 未来 N-bar 收益
3. 对每个因子计算 Rank IC / IC_IR
4. Walk-Forward 分段验证 (防过拟合)
5. Bootstrap 置换检验 (统计显著性)
6. 输出因子排名表 + 详细报告

用法:
    python factor_scanner.py <csv_path> [--horizons 1 4 8] [--wf-splits 5]
"""
import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from strategies.signals import prepare_indicators


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_data(path: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    }, inplace=True)
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['is_flat'] = (
        (df['Open'] == df['High']) &
        (df['High'] == df['Low']) &
        (df['Low'] == df['Close'])
    )
    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]
    return df


# ═══════════════════════════════════════════════════════════════
# 扩展因子计算
# ═══════════════════════════════════════════════════════════════

def compute_extended_factors(df: pd.DataFrame) -> pd.DataFrame:
    """在 prepare_indicators() 基础上添加候选因子列。"""
    df = prepare_indicators(df)

    close = df['Close']
    high = df['High']
    low = df['Low']
    atr = df['ATR']

    # ── 波动率因子 ──
    atr_roll = atr.rolling(100)
    df['ATR_percentile'] = atr.rolling(100).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0 if len(x) == 100 else np.nan,
        raw=False,
    )
    df['ATR_ratio'] = atr / atr.rolling(50).mean()

    # Bollinger 带宽 (波动率的另一个视角)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_width'] = (2 * bb_std) / bb_mid

    # ── 动量因子 ──
    df['momentum_5'] = close / close.shift(5) - 1
    df['momentum_10'] = close / close.shift(10) - 1
    df['momentum_20'] = close / close.shift(20) - 1
    df['ROC_5'] = close.pct_change(5)

    # ── K 线形态因子 ──
    body = (close - df['Open']).abs()
    full_range = high - low
    df['body_range_ratio'] = body / full_range.replace(0, np.nan)
    upper_shadow = high - pd.concat([close, df['Open']], axis=1).max(axis=1)
    lower_shadow = pd.concat([close, df['Open']], axis=1).min(axis=1) - low
    df['upper_shadow_ratio'] = upper_shadow / full_range.replace(0, np.nan)
    df['lower_shadow_ratio'] = lower_shadow / full_range.replace(0, np.nan)

    # ── 通道因子 ──
    kc_width = df['KC_upper'] - df['KC_lower']
    df['KC_position'] = (close - df['KC_lower']) / kc_width.replace(0, np.nan)
    df['KC_breakout_strength'] = (close - df['KC_upper']) / atr.replace(0, np.nan)

    # ── 成交量因子 ──
    vol = df['Volume']
    vol_ma20 = df['Vol_MA20']
    df['volume_ratio'] = vol / vol_ma20.replace(0, np.nan)

    # ── 均线距离因子 ──
    df['close_ema100_dist'] = (close - df['EMA100']) / atr.replace(0, np.nan)
    df['ema9_ema21_cross'] = (df['EMA9'] - df['EMA21']) / atr.replace(0, np.nan)

    # ── 时间因子 ──
    if hasattr(df.index, 'hour'):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 17)).astype(float)
    if hasattr(df.index, 'dayofweek'):
        df['day_of_week'] = df.index.dayofweek.astype(float)

    # ── MACD 衍生 ──
    df['MACD_hist_change'] = df['MACD_hist'] - df['MACD_hist'].shift(1)

    return df


# ═══════════════════════════════════════════════════════════════
# 因子定义注册表（因子名 → 经济逻辑）
# ═══════════════════════════════════════════════════════════════

FACTOR_REGISTRY = {
    # 已有因子 (prepare_indicators)
    'RSI2':               '超短期超买超卖，均值回归',
    'RSI14':              '标准超买超卖',
    'ATR':                '绝对波动率',
    'ADX':                '趋势强度',
    'MACD_hist':          'MACD 柱状图动量',
    'EMA9':               '短期均线（用距离）',
    'EMA21':              '中期均线（用距离）',

    # 扩展因子
    'ATR_percentile':     '波动率聚类：高波动后趋势延续概率高',
    'ATR_ratio':          '波动率相对历史的扩张/收缩',
    'BB_width':           'Bollinger带宽，波动率的标准化度量',
    'momentum_5':         '5期动量效应',
    'momentum_10':        '10期动量效应',
    'momentum_20':        '20期动量效应',
    'ROC_5':              '5期变化率',
    'body_range_ratio':   'K线实体占比→买卖力量确认',
    'upper_shadow_ratio': '上影线占比→卖压',
    'lower_shadow_ratio': '下影线占比→买盘支撑',
    'KC_position':        '价格在Keltner通道中的位置',
    'KC_breakout_strength': '突破KC上轨的ATR标准化强度',
    'volume_ratio':       '成交量/MA20，放量确认',
    'close_ema100_dist':  '价格偏离EMA100的ATR标准化距离',
    'ema9_ema21_cross':   'EMA9-EMA21交叉强度（ATR标准化）',
    'hour_sin':           '交易时段周期（正弦编码）',
    'hour_cos':           '交易时段周期（余弦编码）',
    'is_london_ny_overlap': '伦敦/纽约重叠时段（最高流动性）',
    'day_of_week':        '星期效应',
    'MACD_hist_change':   'MACD柱状图变化速度',
}


# ═══════════════════════════════════════════════════════════════
# IC 计算核心
# ═══════════════════════════════════════════════════════════════

def build_factor_return_matrix(
    df: pd.DataFrame,
    horizons: List[int],
    min_warmup: int = 150,
) -> pd.DataFrame:
    """构建 (因子值, 未来收益) 对齐矩阵。"""

    factor_cols = [c for c in FACTOR_REGISTRY if c in df.columns]
    result = df[factor_cols].copy()

    # 过滤休市填充 bar
    if 'is_flat' in df.columns:
        result = result[~df['is_flat']]

    # 计算未来收益
    close = df.loc[result.index, 'Close']
    for h in horizons:
        future_close = close.shift(-h)
        result[f'ret_{h}'] = (future_close - close) / close

    # 去掉前 warmup 行（指标未稳定）和尾部（无未来收益）
    result = result.iloc[min_warmup:]
    max_h = max(horizons)
    result = result.iloc[:-max_h] if max_h > 0 else result

    result.dropna(how='all', subset=factor_cols, inplace=True)

    return result


def _fast_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation via numpy (avoids scipy overhead per call)."""
    n = len(a)
    if n < 3:
        return np.nan
    rank_a = stats.rankdata(a)
    rank_b = stats.rankdata(b)
    d = rank_a - rank_b
    return 1 - 6 * np.sum(d * d) / (n * (n * n - 1))


def calc_rank_ic(
    matrix: pd.DataFrame,
    factor_col: str,
    return_col: str,
) -> Tuple[float, float, float, int]:
    """计算单因子-单收益窗口的 Rank IC。

    Returns: (ic, ic_mean_rolling, ic_ir, n_samples)
    """
    valid = matrix[[factor_col, return_col]].dropna()
    n = len(valid)
    if n < 30:
        return np.nan, np.nan, np.nan, n

    factor_vals = valid[factor_col].values
    return_vals = valid[return_col].values
    ic = _fast_rank_corr(factor_vals, return_vals)

    # 滚动 IC — 采样而非逐 bar (66K 行全跑太慢)
    window = min(200, n // 3)
    if window < 30:
        return ic, ic, 0.0, n

    step = max(1, window // 4)
    rolling_ics = []
    for i in range(window, n, step):
        f_chunk = factor_vals[i - window:i]
        r_chunk = return_vals[i - window:i]
        r = _fast_rank_corr(f_chunk, r_chunk)
        if not np.isnan(r):
            rolling_ics.append(r)

    if not rolling_ics:
        return ic, ic, 0.0, n

    ic_mean = float(np.mean(rolling_ics))
    ic_std = float(np.std(rolling_ics))
    ic_ir = ic_mean / ic_std if ic_std > 1e-6 else 0.0

    return ic, ic_mean, ic_ir, n


# ═══════════════════════════════════════════════════════════════
# Walk-Forward 验证
# ═══════════════════════════════════════════════════════════════

def walk_forward_ic(
    matrix: pd.DataFrame,
    factor_col: str,
    return_col: str,
    n_splits: int = 5,
) -> Dict:
    """将数据分为 n_splits 段，计算每段的 IC，检查一致性。"""
    valid = matrix[[factor_col, return_col]].dropna()
    n = len(valid)
    if n < n_splits * 30:
        return {'status': 'insufficient_data', 'n': n}

    segment_size = n // n_splits
    ics = []
    for i in range(n_splits):
        start = i * segment_size
        end = start + segment_size if i < n_splits - 1 else n
        seg = valid.iloc[start:end]
        if len(seg) < 20:
            continue
        r = _fast_rank_corr(seg[factor_col].values, seg[return_col].values)
        ics.append(r if not np.isnan(r) else 0.0)

    if not ics:
        return {'status': 'no_valid_segments'}

    signs = [1 if x > 0 else -1 for x in ics]
    sign_consistency = abs(sum(signs)) / len(signs)

    return {
        'status': 'ok',
        'segment_ics': [round(x, 4) for x in ics],
        'mean_ic': round(float(np.mean(ics)), 4),
        'std_ic': round(float(np.std(ics)), 4),
        'sign_consistency': round(sign_consistency, 2),
        'all_same_sign': all(x > 0 for x in ics) or all(x < 0 for x in ics),
    }


# ═══════════════════════════════════════════════════════════════
# Bootstrap 置换检验
# ═══════════════════════════════════════════════════════════════

def bootstrap_significance(
    matrix: pd.DataFrame,
    factor_col: str,
    return_col: str,
    n_permutations: int = 500,
) -> Dict:
    """通过打乱收益序列评估 IC 的统计显著性（向量化加速版）。"""
    valid = matrix[[factor_col, return_col]].dropna()
    n = len(valid)
    if n < 30:
        return {'p_value': 1.0, 'significant': False}

    factor_vals = valid[factor_col].values
    return_vals = valid[return_col].values
    actual_ic = _fast_rank_corr(factor_vals, return_vals)

    # 向量化：预先生成所有置换索引，批量计算
    rng = np.random.default_rng(42)
    factor_ranks = stats.rankdata(factor_vals)
    null_ics = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled_ranks = stats.rankdata(rng.permutation(return_vals))
        d = factor_ranks - shuffled_ranks
        null_ics[i] = 1 - 6 * np.sum(d * d) / (n * (n * n - 1))

    p_value = float(np.mean(np.abs(null_ics) >= abs(actual_ic)))

    return {
        'actual_ic': round(actual_ic, 4),
        'p_value': round(p_value, 4),
        'null_ic_mean': round(float(np.mean(null_ics)), 4),
        'null_ic_std': round(float(np.std(null_ics)), 4),
        'percentile': round(float(stats.percentileofscore(np.abs(null_ics), abs(actual_ic))), 1),
        'significant_005': p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════
# 衰减检测
# ═══════════════════════════════════════════════════════════════

def detect_decay(
    matrix: pd.DataFrame,
    factor_col: str,
    return_col: str,
) -> Dict:
    """对比前半 vs 后半 IC，检测因子衰减。"""
    valid = matrix[[factor_col, return_col]].dropna()
    n = len(valid)
    if n < 60:
        return {'status': 'insufficient_data'}

    mid = n // 2
    first = valid.iloc[:mid]
    second = valid.iloc[mid:]

    ic1 = _fast_rank_corr(first[factor_col].values, first[return_col].values)
    ic2 = _fast_rank_corr(second[factor_col].values, second[return_col].values)

    if np.isnan(ic1) or np.isnan(ic2):
        return {'status': 'nan_ic'}

    status = 'stable'
    if abs(ic1) > 0.03 and abs(ic2) < abs(ic1) * 0.5:
        status = 'decaying'
    elif abs(ic2) > abs(ic1) * 1.5 and abs(ic2) > 0.03:
        status = 'strengthening'

    return {
        'status': status,
        'ic_first_half': round(float(ic1), 4),
        'ic_second_half': round(float(ic2), 4),
        'change': round(float(ic2 - ic1), 4),
    }


# ═══════════════════════════════════════════════════════════════
# 主扫描流程
# ═══════════════════════════════════════════════════════════════

def ic_quality_label(ic_abs: float, ir_abs: float) -> str:
    if ic_abs >= 0.10 and ir_abs >= 0.5:
        return 'EXCELLENT'
    elif ic_abs >= 0.05 and ir_abs >= 0.3:
        return 'GOOD'
    elif ic_abs >= 0.03:
        return 'FAIR'
    return 'WEAK'


def scan_all_factors(
    df: pd.DataFrame,
    horizons: List[int] = None,
    wf_splits: int = 5,
    bootstrap_n: int = 500,
) -> pd.DataFrame:
    """全量因子扫描主流程。"""
    if horizons is None:
        horizons = [1, 4, 8]

    print("构建因子-收益矩阵...", end='', flush=True)
    matrix = build_factor_return_matrix(df, horizons)
    print(f" 完成 ({len(matrix)} 行)")

    factor_cols = [c for c in FACTOR_REGISTRY if c in matrix.columns]
    print(f"候选因子: {len(factor_cols)} 个")
    print(f"收益窗口: {horizons} bars")
    print(f"Walk-Forward 分段: {wf_splits}")
    print(f"Bootstrap 置换次数: {bootstrap_n}")
    print()

    # Bonferroni 修正阈值
    n_tests = len(factor_cols) * len(horizons)
    bonferroni_threshold = 0.05 / n_tests
    print(f"Bonferroni 修正: p < {bonferroni_threshold:.4f} (共 {n_tests} 次检验)")
    print("=" * 80)

    rows = []
    total = len(factor_cols) * len(horizons)
    done = 0

    for f_col in factor_cols:
        for h in horizons:
            ret_col = f'ret_{h}'
            done += 1
            print(f"\r  [{done}/{total}] {f_col} × ret_{h}...", end='', flush=True)

            ic, ic_mean, ic_ir, n_samples = calc_rank_ic(matrix, f_col, ret_col)
            if np.isnan(ic):
                continue

            wf = walk_forward_ic(matrix, f_col, ret_col, wf_splits)
            bs = bootstrap_significance(matrix, f_col, ret_col, bootstrap_n)
            decay = detect_decay(matrix, f_col, ret_col)

            quality = ic_quality_label(abs(ic), abs(ic_ir))
            wf_consistency = wf.get('sign_consistency', 0) if wf.get('status') == 'ok' else 0
            wf_all_same = wf.get('all_same_sign', False) if wf.get('status') == 'ok' else False

            rows.append({
                'factor': f_col,
                'horizon': h,
                'ic': round(ic, 4),
                'ic_mean': round(ic_mean, 4),
                'ic_ir': round(ic_ir, 4),
                'n_samples': n_samples,
                'quality': quality,
                'p_value': bs.get('p_value', 1.0),
                'significant_bonf': bs.get('p_value', 1.0) < bonferroni_threshold,
                'significant_005': bs.get('significant_005', False),
                'wf_consistency': wf_consistency,
                'wf_all_same_sign': wf_all_same,
                'wf_segment_ics': wf.get('segment_ics', []),
                'decay_status': decay.get('status', 'unknown'),
                'ic_first_half': decay.get('ic_first_half', np.nan),
                'ic_second_half': decay.get('ic_second_half', np.nan),
                'logic': FACTOR_REGISTRY.get(f_col, ''),
            })

    print("\r" + " " * 60 + "\r", end='')

    result = pd.DataFrame(rows)
    if not result.empty:
        result.sort_values('ic', key=lambda x: x.abs(), ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)

    return result


# ═══════════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════════

def print_report(result: pd.DataFrame, horizons: List[int]):
    if result.empty:
        print("无有效结果。")
        return

    n_tests = result.shape[0]
    print()
    print("=" * 90)
    print("  因子 IC 扫描报告  (Factor IC Scanner Report)")
    print("=" * 90)
    print(f"  总检验数: {n_tests}")
    print(f"  收益窗口: {horizons} bars")
    print()

    for h in horizons:
        subset = result[result['horizon'] == h].head(30)
        if subset.empty:
            continue

        print(f"  ── 未来 {h} bar 收益 ──")
        print(f"  {'因子':<24} {'IC':>7} {'IC_IR':>7} {'质量':<10} {'p值':>7} "
              f"{'Bonf':>5} {'WF一致':>7} {'衰减':>12} {'样本':>6}")
        print(f"  {'-'*24} {'-'*7} {'-'*7} {'-'*10} {'-'*7} "
              f"{'-'*5} {'-'*7} {'-'*12} {'-'*6}")

        for _, row in subset.iterrows():
            bonf = 'Y' if row['significant_bonf'] else ''
            wf_con = f"{row['wf_consistency']:.0%}" if row['wf_consistency'] > 0 else 'N/A'
            decay = row['decay_status']

            icon = {'EXCELLENT': '*', 'GOOD': '+', 'FAIR': '.', 'WEAK': ' '}.get(row['quality'], ' ')

            print(f"  {icon}{row['factor']:<23} {row['ic']:>+7.4f} {row['ic_ir']:>7.3f} "
                  f"{row['quality']:<10} {row['p_value']:>7.3f} {bonf:>5} "
                  f"{wf_con:>7} {decay:>12} {row['n_samples']:>6}")

        print()

    # 综合推荐
    print("=" * 90)
    print("  综合推荐 (通过 Bonferroni + Walk-Forward 一致性 >= 60%)")
    print("=" * 90)

    recommended = result[
        (result['significant_bonf']) &
        (result['wf_consistency'] >= 0.6) &
        (result['decay_status'] != 'decaying')
    ]

    if recommended.empty:
        recommended_relaxed = result[
            (result['significant_005']) &
            (result['wf_consistency'] >= 0.6) &
            (result['decay_status'] != 'decaying')
        ]
        if not recommended_relaxed.empty:
            print("  (无因子通过 Bonferroni 修正，以下为 p<0.05 的候选)")
            print()
            for _, row in recommended_relaxed.iterrows():
                print(f"  {row['factor']:<24} ret_{row['horizon']}  IC={row['ic']:+.4f}  "
                      f"IR={row['ic_ir']:.3f}  p={row['p_value']:.3f}  "
                      f"WF={row['wf_consistency']:.0%}  [{row['logic']}]")
        else:
            print("  无因子通过筛选。可能原因：")
            print("  1. 单因子对黄金收益的线性预测力确实有限")
            print("  2. 因子有效性可能是非线性的（需条件组合）")
            print("  3. 预测窗口不匹配（尝试调整 --horizons）")
    else:
        for _, row in recommended.iterrows():
            print(f"  {row['factor']:<24} ret_{row['horizon']}  IC={row['ic']:+.4f}  "
                  f"IR={row['ic_ir']:.3f}  p={row['p_value']:.3f}  "
                  f"WF={row['wf_consistency']:.0%}  [{row['logic']}]")

    # 衰减警告
    decaying = result[result['decay_status'] == 'decaying']
    if not decaying.empty:
        print()
        print("  [!] 衰减因子 (前半 IC 明显高于后半):")
        for _, row in decaying.drop_duplicates(subset='factor').iterrows():
            print(f"    {row['factor']}: IC {row['ic_first_half']:+.4f} → {row['ic_second_half']:+.4f}")

    # 增强因子
    strengthening = result[result['decay_status'] == 'strengthening']
    if not strengthening.empty:
        print()
        print("  [^] 增强因子 (后半 IC 明显高于前半):")
        for _, row in strengthening.drop_duplicates(subset='factor').iterrows():
            print(f"    {row['factor']}: IC {row['ic_first_half']:+.4f} → {row['ic_second_half']:+.4f}")

    print()
    print("=" * 90)


def save_csv_report(result: pd.DataFrame, output_path: str):
    cols = [c for c in result.columns if c != 'wf_segment_ics']
    result[cols].to_csv(output_path, index=False)
    print(f"详细报告已保存: {output_path}")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='因子 IC 离线扫描器')
    parser.add_argument('csv_path', help='历史数据 CSV 路径')
    parser.add_argument('--start', default=None, help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default=None, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 4, 8],
                        help='未来收益窗口 (bar 数), 默认 1 4 8')
    parser.add_argument('--wf-splits', type=int, default=5,
                        help='Walk-Forward 分段数, 默认 5')
    parser.add_argument('--bootstrap', type=int, default=500,
                        help='Bootstrap 置换次数, 默认 500')
    parser.add_argument('--output', default=None,
                        help='输出 CSV 路径 (默认: data/factor_ic_report.csv)')
    args = parser.parse_args()

    print("=" * 60)
    print("  因子 IC 离线扫描器 (Factor IC Scanner)")
    print("=" * 60)
    print()

    print("加载数据...")
    df = load_data(args.csv_path, args.start, args.end)
    n_flat = df['is_flat'].sum()
    print(f"总 K 线: {len(df)} (其中休市填充: {n_flat})")

    print("计算技术指标 + 扩展因子...")
    df = compute_extended_factors(df)

    result = scan_all_factors(df, args.horizons, args.wf_splits, args.bootstrap)
    print_report(result, args.horizons)

    output_path = args.output or 'data/factor_ic_report.csv'
    save_csv_report(result, output_path)


if __name__ == '__main__':
    main()
