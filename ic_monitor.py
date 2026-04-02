"""
因子 IC (Information Coefficient) 监控模块
==========================================
从交易日志中提取因子快照，计算各因子与收益的 Rank IC，
滚动检测因子有效性衰减，辅助策略迭代决策。

核心指标:
- IC: 因子截面值与未来收益的 Spearman 相关系数
- IC_IR: IC均值 / IC标准差，衡量因子稳定性 (>0.5 优秀)
- IC半衰期: IC从峰值衰减到50%的窗口长度
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

IC_REPORT_FILE = config.DATA_DIR / "ic_report.json"
MIN_TRADES_FOR_IC = 20


class ICMonitor:
    """因子 IC 计算与衰减检测"""

    def __init__(self):
        self.live_log_file = config.DATA_DIR / "gold_trade_log.json"
        self.paper_log_file = config.DATA_DIR / "paper" / "paper_trades.json"

    def generate_report(self) -> Dict:
        """生成完整 IC 报告 (实盘 + 模拟盘)"""
        report = {
            'date': datetime.now().isoformat(),
            'live': self._analyze_source(self.live_log_file, 'live'),
            'paper': self._analyze_source(self.paper_log_file, 'paper'),
        }
        self._save_report(report)
        return report

    def _analyze_source(self, log_file: Path, source: str) -> Dict:
        """分析单个交易日志来源"""
        trades = self._load_trades_with_factors(log_file, source)
        if len(trades) < MIN_TRADES_FOR_IC:
            return {
                'status': 'insufficient_data',
                'trade_count': len(trades),
                'min_required': MIN_TRADES_FOR_IC,
            }

        df = pd.DataFrame(trades)
        factor_cols = [c for c in df.columns if c.startswith('f_')]

        if not factor_cols:
            return {'status': 'no_factors', 'trade_count': len(trades)}

        overall_ic = self._calc_overall_ic(df, factor_cols)
        strategy_ic = self._calc_strategy_ic(df, factor_cols)
        decay = self._detect_decay(df, factor_cols)

        return {
            'status': 'ok',
            'trade_count': len(trades),
            'overall': overall_ic,
            'by_strategy': strategy_ic,
            'decay': decay,
        }

    def _load_trades_with_factors(self, log_file: Path, source: str) -> List[Dict]:
        """从日志加载含因子的平仓记录"""
        if not log_file.exists():
            return []

        try:
            with open(log_file) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []

        result = []
        for entry in raw:
            factors = entry.get('factors')
            if not factors:
                continue

            if source == 'live':
                pnl = entry.get('profit', 0)
                strategy = entry.get('strategy', 'unknown')
                action = entry.get('action', '')
                if action not in ('CLOSE', 'CLOSE_DETECTED'):
                    pnl = self._find_close_pnl(raw, entry, strategy)
                    if pnl is None:
                        continue
                trade_time = entry.get('time', '')
            else:
                pnl = entry.get('pnl', 0)
                strategy = entry.get('strategy', 'unknown')
                trade_time = entry.get('exit_time', entry.get('entry_time', ''))

            row = {'pnl': pnl, 'strategy': strategy, 'time': trade_time}
            for k, v in factors.items():
                row[f'f_{k}'] = v
            result.append(row)

        return result

    @staticmethod
    def _find_close_pnl(raw: List[Dict], open_entry: Dict, strategy: str) -> Optional[float]:
        """在实盘日志中，OPEN 记录的 factors 需要匹配对应的 CLOSE 获取 pnl"""
        open_time = open_entry.get('time', '')
        for entry in raw:
            if entry.get('action') not in ('CLOSE', 'CLOSE_DETECTED'):
                continue
            if entry.get('strategy') != strategy:
                continue
            close_time = entry.get('time', '')
            if close_time > open_time:
                return entry.get('profit', 0)
        return None

    @staticmethod
    def _calc_overall_ic(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, Dict]:
        """计算每个因子的总体 IC / IC_IR"""
        results = {}
        pnl = df['pnl']

        for col in factor_cols:
            vals = df[col].dropna()
            valid_idx = vals.index.intersection(pnl.index)
            if len(valid_idx) < MIN_TRADES_FOR_IC:
                continue

            ic = pnl.loc[valid_idx].corr(vals.loc[valid_idx], method='spearman')
            if pd.isna(ic):
                continue

            factor_name = col[2:]  # strip 'f_' prefix

            rolling_ic = _rolling_rank_ic(df, col, window=min(30, len(valid_idx)))
            ic_mean = float(rolling_ic.mean()) if len(rolling_ic) > 0 else 0.0
            ic_std = float(rolling_ic.std()) if len(rolling_ic) > 1 else 1.0
            ic_ir = round(ic_mean / ic_std, 4) if ic_std > 0.001 else 0.0

            results[factor_name] = {
                'ic': round(float(ic), 4),
                'ic_mean': round(ic_mean, 4),
                'ic_std': round(ic_std, 4),
                'ic_ir': ic_ir,
                'samples': len(valid_idx),
                'quality': _ic_quality_label(abs(ic), abs(ic_ir)),
            }

        return dict(sorted(results.items(), key=lambda x: abs(x[1]['ic']), reverse=True))

    @staticmethod
    def _calc_strategy_ic(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, Dict]:
        """按策略分组计算 IC"""
        results = {}
        for strat, grp in df.groupby('strategy'):
            if len(grp) < MIN_TRADES_FOR_IC:
                results[strat] = {'status': 'insufficient_data', 'trade_count': len(grp)}
                continue

            strat_ic = {}
            pnl = grp['pnl']
            for col in factor_cols:
                vals = grp[col].dropna()
                valid_idx = vals.index.intersection(pnl.index)
                if len(valid_idx) < 10:
                    continue
                ic = pnl.loc[valid_idx].corr(vals.loc[valid_idx], method='spearman')
                if pd.isna(ic):
                    continue
                strat_ic[col[2:]] = round(float(ic), 4)

            results[strat] = {'status': 'ok', 'trade_count': len(grp), 'ic': strat_ic}

        return results

    @staticmethod
    def _detect_decay(df: pd.DataFrame, factor_cols: List[str]) -> Dict[str, Dict]:
        """检测因子衰减: 对比前半段 vs 后半段 IC"""
        if len(df) < MIN_TRADES_FOR_IC * 2:
            return {}

        mid = len(df) // 2
        first_half = df.iloc[:mid]
        second_half = df.iloc[mid:]
        decay_info = {}

        for col in factor_cols:
            pnl_1 = first_half['pnl']
            vals_1 = first_half[col].dropna()
            idx_1 = vals_1.index.intersection(pnl_1.index)

            pnl_2 = second_half['pnl']
            vals_2 = second_half[col].dropna()
            idx_2 = vals_2.index.intersection(pnl_2.index)

            if len(idx_1) < 10 or len(idx_2) < 10:
                continue

            ic_1 = pnl_1.loc[idx_1].corr(vals_1.loc[idx_1], method='spearman')
            ic_2 = pnl_2.loc[idx_2].corr(vals_2.loc[idx_2], method='spearman')

            if pd.isna(ic_1) or pd.isna(ic_2):
                continue

            change = float(ic_2) - float(ic_1)
            factor_name = col[2:]
            status = 'stable'
            if abs(ic_1) > 0.05 and abs(ic_2) < abs(ic_1) * 0.5:
                status = 'decaying'
            elif abs(ic_2) > abs(ic_1) * 1.5 and abs(ic_2) > 0.05:
                status = 'strengthening'

            decay_info[factor_name] = {
                'ic_first_half': round(float(ic_1), 4),
                'ic_second_half': round(float(ic_2), 4),
                'change': round(change, 4),
                'status': status,
            }

        return decay_info

    def _save_report(self, report: Dict):
        """保存 IC 报告到文件"""
        try:
            with open(IC_REPORT_FILE, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except OSError as e:
            log.warning(f"IC报告保存失败: {e}")

    def format_telegram_summary(self, report: Dict) -> str:
        """将 IC 报告格式化为 Telegram 消息"""
        lines = ["📊 <b>因子 IC 监控报告</b>", ""]

        for source_name, source_label in [('live', '实盘'), ('paper', '模拟盘')]:
            data = report.get(source_name, {})
            status = data.get('status', 'unknown')
            lines.append(f"<b>{'─' * 3} {source_label} {'─' * 3}</b>")

            if status == 'insufficient_data':
                lines.append(f"  数据不足 ({data.get('trade_count', 0)}/{data.get('min_required', MIN_TRADES_FOR_IC)}笔)")
                lines.append("")
                continue
            if status == 'no_factors':
                lines.append(f"  {data.get('trade_count', 0)}笔交易，暂无因子数据")
                lines.append("")
                continue
            if status != 'ok':
                lines.append(f"  状态异常: {status}")
                lines.append("")
                continue

            overall = data.get('overall', {})
            if overall:
                top_factors = list(overall.items())[:5]
                for name, info in top_factors:
                    ic = info['ic']
                    ir = info['ic_ir']
                    quality = info['quality']
                    icon = _factor_icon(quality)
                    lines.append(f"  {icon} {name}: IC={ic:+.3f} IR={ir:.2f} [{quality}]")

            decay = data.get('decay', {})
            decaying = [k for k, v in decay.items() if v.get('status') == 'decaying']
            if decaying:
                lines.append(f"  ⚠️ 衰减因子: {', '.join(decaying)}")

            strengthening = [k for k, v in decay.items() if v.get('status') == 'strengthening']
            if strengthening:
                lines.append(f"  🔥 增强因子: {', '.join(strengthening)}")

            lines.append("")

        return '\n'.join(lines)


def _rolling_rank_ic(df: pd.DataFrame, factor_col: str, window: int = 30) -> pd.Series:
    """计算滚动 Rank IC"""
    ic_values = []
    for i in range(window, len(df)):
        subset = df.iloc[i - window:i]
        pnl = subset['pnl']
        vals = subset[factor_col].dropna()
        idx = vals.index.intersection(pnl.index)
        if len(idx) < 10:
            continue
        ic = pnl.loc[idx].corr(vals.loc[idx], method='spearman')
        if not pd.isna(ic):
            ic_values.append(ic)
    return pd.Series(ic_values)


def _ic_quality_label(ic_abs: float, ir_abs: float) -> str:
    """根据 |IC| 和 |IC_IR| 判定因子质量"""
    if ic_abs >= 0.1 and ir_abs >= 0.5:
        return '优秀'
    elif ic_abs >= 0.05 and ir_abs >= 0.3:
        return '良好'
    elif ic_abs >= 0.03:
        return '一般'
    else:
        return '无效'


def _factor_icon(quality: str) -> str:
    if quality == '优秀':
        return '🟢'
    elif quality == '良好':
        return '🟡'
    elif quality == '一般':
        return '🟠'
    return '🔴'
