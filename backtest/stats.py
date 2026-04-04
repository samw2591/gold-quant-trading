"""
Backtest Statistics & Reporting
================================
Unified statistics calculation and report formatting.
Replaces: backtest.print_report, backtest_m15.calc_stats/print_comparison
"""
from typing import Dict, List, Optional

import itertools
import random

import numpy as np
import pandas as pd
from scipy import stats

import config
from backtest.engine import TradeRecord


def aggregate_daily_pnl(trades: List[TradeRecord]) -> List[float]:
    """Aggregate PnL by exit date."""
    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0) + t.pnl
    return list(daily.values())


def calc_stats(trades: List[TradeRecord], equity_curve: List[float]) -> Dict:
    """Calculate comprehensive backtest statistics.

    Returns a dict with keys: n, total_pnl, sharpe, win_rate, max_dd, max_dd_pct,
    rr, avg_win, avg_loss, plus per-strategy and per-year breakdowns.
    """
    if not trades:
        return {
            'n': 0, 'total_pnl': 0, 'sharpe': 0, 'win_rate': 0,
            'max_dd': 0, 'max_dd_pct': 0, 'rr': 0,
            'avg_win': 0, 'avg_loss': 0, 'year_pnl': {},
            'rsi_n': 0, 'rsi_pnl': 0, 'rsi_wr': 0,
            'rsi_buy_n': 0, 'rsi_buy_pnl': 0,
            'rsi_sell_n': 0, 'rsi_sell_pnl': 0,
            'keltner_n': 0, 'keltner_pnl': 0, 'keltner_wr': 0,
            'orb_n': 0, 'orb_pnl': 0, 'orb_wr': 0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    daily_pnl = aggregate_daily_pnl(trades)
    sharpe = 0.0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)

    eq = np.array(equity_curve) if equity_curve else np.array([config.CAPITAL])
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())
    max_dd_pct = max_dd / peak[np.argmin(dd)] * 100 if peak[np.argmin(dd)] > 0 else 0

    # Per-strategy breakdown
    def _strat_stats(strat_name):
        st = [t for t in trades if t.strategy == strat_name]
        n = len(st)
        pnl = sum(t.pnl for t in st)
        wr = len([t for t in st if t.pnl > 0]) / n * 100 if n else 0
        return n, pnl, wr

    rsi_n, rsi_pnl, rsi_wr = _strat_stats('m15_rsi')
    keltner_n, keltner_pnl, keltner_wr = _strat_stats('keltner')
    orb_n, orb_pnl, orb_wr = _strat_stats('orb')

    rsi_buy = [t for t in trades if t.strategy == 'm15_rsi' and t.direction == 'BUY']
    rsi_sell = [t for t in trades if t.strategy == 'm15_rsi' and t.direction == 'SELL']

    year_pnl = {}
    for t in trades:
        y = pd.Timestamp(t.exit_time).year
        year_pnl[y] = year_pnl.get(y, 0) + t.pnl

    return {
        'n': len(pnls), 'total_pnl': total_pnl, 'win_rate': win_rate,
        'avg_win': avg_win, 'avg_loss': avg_loss, 'rr': rr,
        'sharpe': sharpe, 'max_dd': max_dd, 'max_dd_pct': max_dd_pct,
        'rsi_n': rsi_n, 'rsi_pnl': rsi_pnl, 'rsi_wr': rsi_wr,
        'rsi_buy_n': len(rsi_buy), 'rsi_buy_pnl': sum(t.pnl for t in rsi_buy),
        'rsi_sell_n': len(rsi_sell), 'rsi_sell_pnl': sum(t.pnl for t in rsi_sell),
        'keltner_n': keltner_n, 'keltner_pnl': keltner_pnl, 'keltner_wr': keltner_wr,
        'orb_n': orb_n, 'orb_pnl': orb_pnl, 'orb_wr': orb_wr,
        'year_pnl': year_pnl,
    }


# ═══════════════════════════════════════════════════════════════
# Report printing
# ═══════════════════════════════════════════════════════════════

def print_comparison(variants: List[Dict], title: str = "Backtest Comparison"):
    """Print a ranked comparison table of multiple variants."""
    print("\n")
    print("=" * 120)
    print(f"  {title}")
    print("=" * 120)

    header = (f"  {'Variant':<40} {'Trades':>6} {'Sharpe':>8} {'PnL':>10} "
              f"{'MaxDD':>10} {'DD%':>6} {'WinR%':>7} {'RR':>6}")
    print(header)
    print(f"  {'-'*40} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*6}")

    baseline = variants[0] if variants else {}
    base_sharpe = baseline.get('sharpe', 0)

    for v in variants:
        if v.get('n', 0) == 0:
            print(f"  {v.get('label', '?'):<40}   (no trades)")
            continue
        sharpe_diff = v['sharpe'] - base_sharpe
        marker = f"({sharpe_diff:+.2f})" if v != baseline else ""
        print(f"  {v.get('label', '?'):<40} {v['n']:>6} {v['sharpe']:>8.2f} ${v['total_pnl']:>9.2f} "
              f"${v['max_dd']:>9.2f} {v['max_dd_pct']:>5.1f}% {v['win_rate']:>6.1f}% "
              f"{v['rr']:>5.2f} {marker}")

    # Strategy breakdown
    print(f"\n  --- Strategy Breakdown ---")
    print(f"  {'Variant':<40} {'K_N':>5} {'K_PnL':>10} {'ORB_N':>5} {'ORB_PnL':>10} "
          f"{'RSI_N':>5} {'RSI_PnL':>10} {'RSI_WR':>6} {'Filt':>5}")
    print(f"  {'-'*40} {'-'*5} {'-'*10} {'-'*5} {'-'*10} "
          f"{'-'*5} {'-'*10} {'-'*6} {'-'*5}")
    for v in variants:
        if v.get('n', 0) == 0:
            continue
        print(f"  {v.get('label', '?'):<40} {v.get('keltner_n', 0):>5} ${v.get('keltner_pnl', 0):>9.2f} "
              f"{v.get('orb_n', 0):>5} ${v.get('orb_pnl', 0):>9.2f} "
              f"{v.get('rsi_n', 0):>5} ${v.get('rsi_pnl', 0):>9.2f} "
              f"{v.get('rsi_wr', 0):>5.1f}% {v.get('rsi_filtered', 0):>5}")

    # Year-by-year
    all_years = set()
    for v in variants:
        all_years.update(v.get('year_pnl', {}).keys())
    years = sorted(all_years)

    if years:
        print(f"\n  --- Year-by-Year PnL ($) ---")
        print(f"  {'Year':<6}", end='')
        for v in variants:
            print(f"  {v.get('label', '?')[:18]:>20}", end='')
        print()
        for y in years:
            print(f"  {y:<6}", end='')
            for v in variants:
                pnl = v.get('year_pnl', {}).get(y, 0)
                print(f"  ${pnl:>19.2f}", end='')
            print()

    print("\n" + "=" * 120)


def print_ranked(results: List[Dict], title: str = "Ranked Results"):
    """Print results sorted by Sharpe, with delta from baseline."""
    results_sorted = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)
    baseline_list = [r for r in results if 'Baseline' in r.get('label', '') or r == results[0]]
    base_sharpe = baseline_list[0].get('sharpe', 0) if baseline_list else 0

    print(f"\n  {'='*130}")
    print(f"  {title}")
    print(f"  {'='*130}")

    print(f"\n  {'Rank':<5} {'Variant':<35} {'N':>6} {'Sharpe':>8} {'dSh':>6} {'PnL':>10} "
          f"{'MaxDD':>10} {'DD%':>6} {'WR%':>6} {'RR':>5}")
    print(f"  {'-'*5} {'-'*35} {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*5}")

    for rank, v in enumerate(results_sorted, 1):
        if v.get('n', 0) == 0:
            print(f"  {rank:<5} {v.get('label', '?'):<35}   (no trades)")
            continue
        ds = v['sharpe'] - base_sharpe
        print(f"  {rank:<5} {v.get('label', '?'):<35} {v['n']:>6} {v['sharpe']:>8.2f} {ds:>+5.2f} "
              f"${v['total_pnl']:>9.0f} ${v['max_dd']:>9.0f} {v['max_dd_pct']:>5.1f}% "
              f"{v['win_rate']:>5.1f}% {v['rr']:>4.2f}")

    print(f"\n  {'='*130}")


# ═══════════════════════════════════════════════════════════════
# Advanced statistical tests (PSR / DSR / PBO)
# ═══════════════════════════════════════════════════════════════

_EULER_MASCHERONI = 0.5772156649015329


def _annualized_daily_sharpe(returns: List[float]) -> float:
    """Annualized Sharpe from daily PnL: (mean/std) * sqrt(252)."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std <= 0 or not np.isfinite(std):
        return 0.0
    mu = float(np.mean(arr))
    return float((mu / std) * np.sqrt(252.0))


def _sharpe_ratio_std(sr: float, n: int, skew: float, excess_kurt: float) -> float:
    """Asymptotic std of Sharpe estimator (Bailey & López de Prado, 2012)."""
    if n < 2:
        return float("nan")
    inner = 1.0 - skew * sr + ((excess_kurt - 1.0) / 4.0) * (sr ** 2)
    if inner < 0:
        inner = 0.0
    v = inner / float(n - 1)
    return float(np.sqrt(v)) if v >= 0 and np.isfinite(v) else float("nan")


def probabilistic_sharpe(returns: List[float], sharpe_benchmark: float = 0) -> Dict:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Tests if observed SR significantly exceeds sharpe_benchmark.
    Returns dict with: sharpe_obs, psr, p_value, n, skew, kurtosis, sr_std
    """
    default = {
        "sharpe_obs": 0.0,
        "psr": 0.0,
        "p_value": 1.0,
        "n": 0,
        "skew": 0.0,
        "kurtosis": 0.0,
        "sr_std": float("nan"),
    }
    if not returns:
        return default
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n < 10:
        return {**default, "n": n}
    std = float(np.std(arr, ddof=1))
    if std <= 0 or not np.isfinite(std):
        return {**default, "n": n}
    skew = float(stats.skew(arr, bias=False))
    excess_kurt = float(stats.kurtosis(arr, fisher=True, bias=False))
    sharpe_obs = _annualized_daily_sharpe(arr.tolist())
    # Use non-annualized SR in the std formula (Bailey & LdP use daily SR)
    daily_sr = float(np.mean(arr) / std)
    sr_std = _sharpe_ratio_std(daily_sr, n, skew, excess_kurt)
    if not np.isfinite(sr_std) or sr_std <= 0:
        return {
            "sharpe_obs": sharpe_obs,
            "psr": 0.0,
            "p_value": 1.0,
            "n": n,
            "skew": skew,
            "kurtosis": excess_kurt,
            "sr_std": sr_std,
        }
    sr_std_annualized = sr_std * np.sqrt(252)
    z = (sharpe_obs - sharpe_benchmark) / sr_std_annualized if sr_std_annualized > 0 else 0.0
    psr = float(stats.norm.cdf(z))
    p_value = float(1.0 - psr)
    return {
        "sharpe_obs": sharpe_obs,
        "psr": psr,
        "p_value": p_value,
        "n": n,
        "skew": skew,
        "kurtosis": excess_kurt,
        "sr_std": sr_std,
    }


def deflated_sharpe(
    returns: List[float],
    n_trials: int,
    all_sharpes_var: float = None,
) -> Dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts SR for multiple testing by computing E[max(SR)] under null.
    Returns dict with: sharpe_obs, sr_star (expected max under null),
                       dsr (deflated probability), n_trials, passed (bool)
    """
    default = {
        "sharpe_obs": 0.0,
        "sr_star": float("nan"),
        "dsr": 0.0,
        "n_trials": int(n_trials) if n_trials is not None else 0,
        "passed": False,
    }
    if not returns or n_trials is None or n_trials < 1:
        return default
    N = int(n_trials)
    if N < 1:
        return default
    psr_parts = probabilistic_sharpe(returns, sharpe_benchmark=0.0)
    n = psr_parts["n"]
    sharpe_obs = psr_parts["sharpe_obs"]
    skew = psr_parts["skew"]
    excess_kurt = psr_parts["kurtosis"]
    if n < 10:
        return {
            **default,
            "sharpe_obs": sharpe_obs,
            "n_trials": N,
        }
    if all_sharpes_var is not None and np.isfinite(all_sharpes_var) and all_sharpes_var >= 0:
        v_sr = float(all_sharpes_var)
    else:
        sr_std = psr_parts["sr_std"]
        if not np.isfinite(sr_std) or sr_std <= 0:
            return {
                "sharpe_obs": sharpe_obs,
                "sr_star": float("nan"),
                "dsr": 0.0,
                "n_trials": N,
                "passed": False,
            }
        v_sr = sr_std ** 2

    sqrt_v = float(np.sqrt(v_sr))
    term1 = stats.norm.ppf(1.0 - 1.0 / N)
    term2 = stats.norm.ppf(1.0 - 1.0 / (N * np.e))
    sr_star = sqrt_v * (
        (1.0 - _EULER_MASCHERONI) * term1 + _EULER_MASCHERONI * term2
    )
    if not np.isfinite(sr_star):
        return {
            "sharpe_obs": sharpe_obs,
            "sr_star": float("nan"),
            "dsr": 0.0,
            "n_trials": N,
            "passed": False,
        }
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    daily_std = float(np.std(arr, ddof=1))
    daily_sr = float(np.mean(arr) / daily_std) if daily_std > 0 else 0.0
    sr_std_dsr = _sharpe_ratio_std(daily_sr, n, skew, excess_kurt)
    if not np.isfinite(sr_std_dsr) or sr_std_dsr <= 0:
        dsr = 0.0
    else:
        z = (sharpe_obs - sr_star) / (sr_std_dsr * np.sqrt(252))
        dsr = float(stats.norm.cdf(z))
    passed = bool(dsr > 0.95)
    return {
        "sharpe_obs": sharpe_obs,
        "sr_star": float(sr_star),
        "dsr": dsr,
        "n_trials": N,
        "passed": passed,
    }


def compute_pbo(
    daily_pnls_by_variant: Dict[str, List[float]],
    n_partitions: int = 8,
) -> Dict:
    """Probability of Backtest Overfitting (Bailey et al., 2015).

    Splits time series into n_partitions blocks, creates C(S, S/2)
    train/test combinations. For each: finds IS-best variant, checks
    its OOS rank. PBO = proportion where IS-best ranks below median OOS.

    Args:
        daily_pnls_by_variant: Dict mapping variant label to list of daily PnL
        n_partitions: Number of time blocks (must be even, default 8)

    Returns dict with: pbo, n_combinations, is_best_oos_ranks,
                       logit_distribution, overfit_risk (LOW/MEDIUM/HIGH)
    """
    empty = {
        "pbo": 0.0,
        "n_combinations": 0,
        "is_best_oos_ranks": [],
        "logit_distribution": [],
        "overfit_risk": "LOW",
    }
    if not daily_pnls_by_variant:
        return empty
    if n_partitions < 2 or n_partitions % 2 != 0:
        return empty

    labels = list(daily_pnls_by_variant.keys())
    k_variants = len(labels)
    if k_variants < 2:
        return empty

    lengths = [len(daily_pnls_by_variant[lbl]) for lbl in labels]
    t = min(lengths)
    if t < n_partitions * 2:
        return empty

    series = {
        lbl: np.asarray(daily_pnls_by_variant[lbl][:t], dtype=float) for lbl in labels
    }

    s_blocks = n_partitions
    edges = np.linspace(0, t, s_blocks + 1, dtype=int)
    block_indices = [
        (int(edges[i]), int(edges[i + 1])) for i in range(s_blocks)
    ]

    def _sharpe_blocks(lbl: str, block_ids: tuple) -> float:
        parts = []
        for b in block_ids:
            a, bnd = block_indices[b]
            parts.append(series[lbl][a:bnd])
        if not parts:
            return float("-inf")
        sl = np.concatenate(parts)
        sl = sl[np.isfinite(sl)]
        if len(sl) < 10:
            return float("-inf")
        std = float(np.std(sl, ddof=1))
        if std <= 0:
            return float("-inf")
        return _annualized_daily_sharpe(sl.tolist())

    all_combos = list(itertools.combinations(range(s_blocks), s_blocks // 2))
    n_all = len(all_combos)
    max_combos = 100
    if n_all > max_combos:
        all_combos = random.sample(all_combos, max_combos)
        n_combinations = max_combos
    else:
        n_combinations = n_all

    is_best_oos_ranks: List[int] = []
    logit_distribution: List[float] = []
    overfit_count = 0

    for is_blocks in all_combos:
        is_set = set(is_blocks)
        oos_blocks = tuple(b for b in range(s_blocks) if b not in is_set)

        is_sharpes = {lbl: _sharpe_blocks(lbl, is_blocks) for lbl in labels}
        best_lbl = max(is_sharpes, key=is_sharpes.get)
        if not np.isfinite(is_sharpes[best_lbl]) or is_sharpes[best_lbl] == float(
            "-inf"
        ):
            continue

        oos_sharpes = {lbl: _sharpe_blocks(lbl, oos_blocks) for lbl in labels}
        sorted_lbls = sorted(
            labels,
            key=lambda x: oos_sharpes[x],
            reverse=True,
        )
        rank = sorted_lbls.index(best_lbl) + 1

        is_best_oos_ranks.append(rank)
        denom = max(k_variants - rank, 1e-12)
        logit_distribution.append(float(np.log(rank / denom)))

        if rank > k_variants / 2.0:
            overfit_count += 1

    n_used = len(is_best_oos_ranks)
    if n_used == 0:
        return {
            "pbo": 0.0,
            "n_combinations": n_combinations,
            "is_best_oos_ranks": [],
            "logit_distribution": [],
            "overfit_risk": "LOW",
        }

    pbo = overfit_count / float(n_used)
    if pbo < 0.20:
        risk = "LOW"
    elif pbo < 0.40:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return {
        "pbo": float(pbo),
        "n_combinations": n_combinations,
        "is_best_oos_ranks": is_best_oos_ranks,
        "logit_distribution": logit_distribution,
        "overfit_risk": risk,
    }
