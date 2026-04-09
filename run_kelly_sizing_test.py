#!/usr/bin/env python3
"""
Kelly Criterion & Dynamic Position Sizing Test
================================================
核心认知: "改善方向应聚焦风控和仓位管理而非新信号"
测试不同仓位管理方案对净值曲线和 Sharpe 的影响。

方案:
1. Fixed $50 risk (baseline)
2. Half-Kelly (基于滚动胜率+盈亏比)
3. ATR-percentile sizing (高波动加仓/低波动减仓)
4. 连胜递增 / 连败递减
5. 日内亏损后极小仓位 (已有，测极端版)
6. 追踪止盈触发后加仓下一笔
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "kelly_sizing_output.txt"


class TeeOutput:
    def __init__(self, fp):
        self.file = open(fp, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, d):
        self.stdout.write(d)
        self.file.write(d)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

print("=" * 80)
print("KELLY CRITERION & DYNAMIC POSITION SIZING TEST")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

MEGA_D1_3H = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "trailing_activate_atr": 0.5,
    "trailing_distance_atr": 0.15,
    "regime_config": {
        'low':    {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high':   {'trail_act': 0.4, 'trail_dist': 0.10},
    },
    "keltner_max_hold_m15": 12,
    "time_decay_tp": True,
    "time_decay_start_hour": 1.0,
    "time_decay_atr_start": 0.30,
    "time_decay_atr_step": 0.10,
}

print("\n--- Running baseline ---")
stats = run_variant(data, "D1+3h Baseline", **MEGA_D1_3H)
trades = stats['_trades']
print(f"  {len(trades)} trades")


def simulate_sizing(trades, scheme_func, label):
    """Re-weight PnL by a sizing function that sees prior trade history."""
    state = {'consec_wins': 0, 'consec_losses': 0, 'trail_last': False,
             'rolling_wins': 0, 'rolling_total': 0, 'day_losses': 0,
             'last_day': '', 'equity': 2000.0}
    pnls = []
    sizes = []

    for i, t in enumerate(trades):
        day = t.entry_time.strftime('%Y-%m-%d')
        if day != state['last_day']:
            state['day_losses'] = 0
            state['last_day'] = day

        # Calculate sizing multiplier
        mult = scheme_func(t, i, state)
        pnls.append(t.pnl * mult)
        sizes.append(mult)

        # Update state
        state['equity'] += t.pnl * mult
        if t.pnl > 0:
            state['consec_wins'] += 1
            state['consec_losses'] = 0
            state['rolling_wins'] += 1
        else:
            state['consec_losses'] += 1
            state['consec_wins'] = 0
            state['day_losses'] += 1
        state['rolling_total'] += 1
        state['trail_last'] = 'Trailing' in getattr(t, 'exit_reason', '')

    return pnls, sizes


def compute_sharpe_from_pnls(trades, pnls):
    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        daily[t.entry_time.strftime('%Y-%m-%d')] += p
    vals = list(daily.values())
    if len(vals) > 1 and np.std(vals) > 0:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0


def max_dd_from_pnls(pnls):
    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    return min(eq[j] - max(eq[:j+1]) for j in range(len(eq)))


# ── Sizing schemes ──

def fixed_1x(t, i, s):
    return 1.0

def half_kelly(t, i, s):
    """Half-Kelly: size = 0.5 * (win_rate - (1-win_rate)/avg_rr)"""
    if s['rolling_total'] < 50:
        return 1.0
    wr = s['rolling_wins'] / s['rolling_total']
    avg_win = np.mean([tr.pnl for tr in trades[max(0,i-100):i] if tr.pnl > 0]) if s['rolling_wins'] > 0 else 50
    avg_loss = abs(np.mean([tr.pnl for tr in trades[max(0,i-100):i] if tr.pnl < 0])) if s['rolling_total'] - s['rolling_wins'] > 0 else 50
    if avg_loss == 0:
        return 1.0
    rr = avg_win / avg_loss
    kelly = wr - (1 - wr) / rr
    half_k = max(0.3, min(2.0, 0.5 * kelly / 0.025))  # normalize to base risk 2.5%
    return half_k

def atr_pct_sizing(t, i, s):
    """High ATR percentile → larger size (trends = profit source)."""
    if not hasattr(t, 'entry_time'):
        return 1.0
    return 1.0  # placeholder, actual ATR pct from trade not available, use proxy

def streak_sizing(t, i, s):
    """Win streak → increase, loss streak → decrease."""
    if s['consec_wins'] >= 3:
        return 1.3
    elif s['consec_wins'] >= 2:
        return 1.15
    elif s['consec_losses'] >= 3:
        return 0.5
    elif s['consec_losses'] >= 2:
        return 0.7
    return 1.0

def anti_streak(t, i, s):
    """Opposite of streak: after losses increase (mean reversion)."""
    if s['consec_losses'] >= 3:
        return 1.3
    elif s['consec_losses'] >= 2:
        return 1.15
    elif s['consec_wins'] >= 3:
        return 0.7
    return 1.0

def day_loss_extreme(t, i, s):
    """Very aggressive day-loss reduction: 1st loss → 0.5x, 2nd → 0.2x, 3rd+ → skip."""
    if s['day_losses'] >= 3:
        return 0.1
    elif s['day_losses'] >= 2:
        return 0.2
    elif s['day_losses'] >= 1:
        return 0.5
    return 1.0

def trail_momentum(t, i, s):
    """If last trade was trailing stop win → increase next trade size."""
    if s['trail_last']:
        return 1.5
    return 1.0

def combo_kelly_dayguard(t, i, s):
    """Half-Kelly + day loss guard."""
    k = half_kelly(t, i, s)
    d = day_loss_extreme(t, i, s)
    return k * d

schemes = [
    ("Fixed 1.0x (baseline)", fixed_1x),
    ("Half-Kelly (rolling 100)", half_kelly),
    ("Streak +30%/−30%", streak_sizing),
    ("Anti-streak (loss→increase)", anti_streak),
    ("Day-loss extreme (0.5/0.2/0.1)", day_loss_extreme),
    ("Trail momentum (+50% after trail)", trail_momentum),
    ("Kelly + DayGuard combo", combo_kelly_dayguard),
]

results = []
for label, func in schemes:
    pnls, sizes = simulate_sizing(trades, func, label)
    sharpe = compute_sharpe_from_pnls(trades, pnls)
    total_pnl = sum(pnls)
    max_dd = max_dd_from_pnls(pnls)
    avg_size = np.mean(sizes)
    wins = sum(1 for p in pnls if p > 0)
    wr = 100.0 * wins / len(pnls)
    results.append({
        'label': label,
        'sharpe': sharpe,
        'pnl': total_pnl,
        'max_dd': max_dd,
        'avg_size': avg_size,
        'wr': wr,
        'n': len(pnls),
    })


print("\n\n" + "=" * 80)
print("SIZING SCHEME COMPARISON")
print("=" * 80)
hdr = f"{'Scheme':<38} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'AvgSize':>8}"
print(hdr)
print("-" * len(hdr))
base_sh = results[0]['sharpe']
for r in results:
    delta = r['sharpe'] - base_sh
    marker = " <<<" if delta > 0.1 else ""
    print(f"{r['label']:<38} {r['sharpe']:>7.2f} ${r['pnl']:>9,.0f} ${r['max_dd']:>7,.0f} {r['avg_size']:>8.2f}{marker}")


# ── Year-by-year for best ──
best = max(results[1:], key=lambda r: r['sharpe'])
print(f"\n\nYEAR-BY-YEAR: {best['label']} vs Baseline")
print("-" * 60)

def yearly_sharpe(trades, pnl_func):
    from collections import defaultdict
    pnls, _ = simulate_sizing(trades, pnl_func, "")
    years = defaultdict(lambda: {'pnls': [], 'trades': []})
    for t, p in zip(trades, pnls):
        yr = t.entry_time.year
        years[yr]['pnls'].append(p)
        years[yr]['trades'].append(t)
    out = {}
    for yr in sorted(years.keys()):
        p = years[yr]['pnls']
        sh = compute_sharpe_from_pnls(years[yr]['trades'], p)
        out[yr] = {'pnl': sum(p), 'sharpe': sh, 'n': len(p)}
    return out

base_yr = yearly_sharpe(trades, fixed_1x)
best_func = [f for l, f in schemes if l == best['label']][0]
best_yr = yearly_sharpe(trades, best_func)

print(f"{'Year':<6} {'Base_Sh':>8} {'Base_PnL':>10} | {'Best_Sh':>8} {'Best_PnL':>10} {'ΔSh':>7}")
print("-" * 58)
for yr in sorted(set(list(base_yr.keys()) + list(best_yr.keys()))):
    b = base_yr.get(yr, {'sharpe': 0, 'pnl': 0})
    d = best_yr.get(yr, {'sharpe': 0, 'pnl': 0})
    print(f"{yr:<6} {b['sharpe']:>8.2f} ${b['pnl']:>9,.0f} | {d['sharpe']:>8.2f} ${d['pnl']:>9,.0f} {d['sharpe']-b['sharpe']:>+7.2f}")

print(f"\n\nCONCLUSION")
print("=" * 80)
for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
    delta = r['sharpe'] - base_sh
    print(f"  {r['label']:<38} Sharpe={r['sharpe']:.2f} (Δ{delta:+.2f})")

total_elapsed = time.time() - t_total
print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
