#!/usr/bin/env python3
"""
EXP-NEW BATCH — 4 New Experiments (Non-overlapping with running tests)
=======================================================================
  EXP-A: Trail Momentum (+50%) K-Fold Validation
  EXP-B: ORB Complete Diagnosis + NoORB K-Fold
  EXP-C: Stochastic 10/90 Extremes with Spread Cost K-Fold
  EXP-D: Squeeze-to-Expansion Confidence Score (Keltner entry weighting)

Prerequisites: 12/12 year-consistent Trail Momentum data, NoORB Sharpe+0.24,
Stochastic $/t=$27.66 (N=103), Squeeze-to-expansion Mega $/t premium 40%.

Shared data load, serial execution.
"""
import sys, os, time, gc
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold
from backtest.stats import calc_stats

OUTPUT_FILE = "exp_new_batch_output.txt"


class TeeOutput:
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()


tee = TeeOutput(OUTPUT_FILE)
sys.stdout = tee

print("=" * 80)
print("EXP-NEW BATCH — 4 NON-OVERLAPPING EXPERIMENTS")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

CURRENT = {**C12_KWARGS, "intraday_adaptive": True}
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
# Shared baselines
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SHARED BASELINES")
print("=" * 80)

base_cur = run_variant(data, "Current-Baseline", **CURRENT)
base_mega = run_variant(data, "Mega-Baseline", **MEGA)

print(f"\n  Current: N={base_cur['n']:,} Sharpe={base_cur['sharpe']:.2f} PnL=${base_cur['total_pnl']:,.0f}")
print(f"  Mega:    N={base_mega['n']:,} Sharpe={base_mega['sharpe']:.2f} PnL=${base_mega['total_pnl']:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-A: Trail Momentum (+50%) K-Fold Validation
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-A: TRAIL MOMENTUM (+50% after trailing win) — K-FOLD")
print("  Prior: 12/12 year consistent, full-sample Sharpe +0.44")
print("  Method: post-hoc sizing simulation per fold")
print("=" * 80)


def simulate_trail_momentum(trades, multiplier=1.5):
    """Apply trail momentum sizing: +50% after trailing stop win."""
    pnls = []
    trail_last = False
    for t in trades:
        mult = multiplier if trail_last else 1.0
        pnls.append(t.pnl * mult)
        trail_last = (t.pnl > 0 and 'Trailing' in getattr(t, 'exit_reason', ''))
    return pnls


def sizing_stats(trades, pnls, label):
    """Compute stats from sized PnL list."""
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    wr = 100.0 * wins / n if n > 0 else 0
    ppt = total_pnl / n if n > 0 else 0

    daily = defaultdict(float)
    for t, p in zip(trades, pnls):
        daily[t.entry_time.strftime('%Y-%m-%d')] += p
    vals = list(daily.values())
    sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0

    eq = [0.0]
    for p in pnls:
        eq.append(eq[-1] + p)
    max_dd = min(eq[j] - max(eq[:j + 1]) for j in range(len(eq))) if len(eq) > 1 else 0

    return {'label': label, 'n': n, 'sharpe': sharpe, 'total_pnl': total_pnl,
            'wr': wr, 'ppt': ppt, 'max_dd': max_dd}


FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

MULTIPLIERS = [1.3, 1.5, 1.8, 2.0]

for config_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    print(f"\n--- Trail Momentum K-Fold on {config_label} ---")
    hdr = f"{'Fold':<8} {'Mult':>5} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7} {'MaxDD':>8} {'Δ Sharpe':>9}"
    print(hdr)
    print("-" * len(hdr))

    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        fold_stats = run_variant(fold_data, f"{config_label}_{fold_name}", verbose=False, **base_kwargs)
        trades_fold = fold_stats['_trades']
        if not trades_fold:
            continue

        base_pnls = [t.pnl for t in trades_fold]
        base_s = sizing_stats(trades_fold, base_pnls, "1.0x")

        print(f"  {fold_name:<6} {'1.0x':>5} {base_s['n']:>5} {base_s['sharpe']:>7.2f} "
              f"${base_s['total_pnl']:>9,.0f} {base_s['wr']:>5.1f}% ${base_s['ppt']:>6.2f} "
              f"${base_s['max_dd']:>7,.0f}      ---")

        for mult in MULTIPLIERS:
            mom_pnls = simulate_trail_momentum(trades_fold, mult)
            mom_s = sizing_stats(trades_fold, mom_pnls, f"{mult}x")
            delta = mom_s['sharpe'] - base_s['sharpe']
            print(f"  {fold_name:<6} {mult:>5.1f} {mom_s['n']:>5} {mom_s['sharpe']:>7.2f} "
                  f"${mom_s['total_pnl']:>9,.0f} {mom_s['wr']:>5.1f}% ${mom_s['ppt']:>6.2f} "
                  f"${mom_s['max_dd']:>7,.0f} {delta:>+8.2f}")

print("\n  Trail Momentum K-Fold Summary:")
print("  Pass criteria: >=5/6 folds with positive Sharpe delta, max_dd increase <15%")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-B: ORB Complete Diagnosis + NoORB K-Fold
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-B: ORB COMPLETE DIAGNOSIS — K-FOLD NoORB vs Full")
print("  Prior: NoORB Sharpe +0.24 over full sample")
print("  Method: K-Fold with/without ORB, per-strategy PnL breakdown")
print("=" * 80)

# EXP-B1: ORB strategy isolation — analyze ORB-only trades
print("\n--- B1: ORB Strategy Isolation (full sample) ---")
for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    trades_all = base_stats['_trades']
    orb_trades = [t for t in trades_all if t.strategy == 'orb']
    non_orb = [t for t in trades_all if t.strategy != 'orb']

    if orb_trades:
        orb_pnl = sum(t.pnl for t in orb_trades)
        orb_wins = sum(1 for t in orb_trades if t.pnl > 0)
        orb_wr = 100.0 * orb_wins / len(orb_trades)
        orb_ppt = orb_pnl / len(orb_trades)
        orb_avg_bars = np.mean([t.bars_held for t in orb_trades])
        orb_exit_reasons = defaultdict(int)
        for t in orb_trades:
            orb_exit_reasons[t.exit_reason] += 1
        print(f"\n  {config_label} — ORB trades:")
        print(f"    N={len(orb_trades)}, PnL=${orb_pnl:,.0f}, WR={orb_wr:.1f}%, $/t=${orb_ppt:.2f}, AvgBars={orb_avg_bars:.1f}")
        print(f"    Exit reasons: {dict(orb_exit_reasons)}")
    else:
        print(f"\n  {config_label} — No ORB trades found")

    if non_orb:
        no_pnl = sum(t.pnl for t in non_orb)
        no_wins = sum(1 for t in non_orb if t.pnl > 0)
        no_wr = 100.0 * no_wins / len(non_orb)
        no_ppt = no_pnl / len(non_orb)
        print(f"    Non-ORB: N={len(non_orb)}, PnL=${no_pnl:,.0f}, WR={no_wr:.1f}%, $/t=${no_ppt:.2f}")


# EXP-B2: NoORB K-Fold
print("\n\n--- B2: K-Fold NoORB vs Full ---")

import strategies.signals as signals_mod

_orig_get_orb = signals_mod.get_orb_strategy


class DisabledORB:
    """Stub ORB strategy that never triggers."""
    def reset_daily(self): pass
    def on_new_bar(self, bar, atr=0): return None
    def check_exit(self, pos, bar): return None
    def on_bar(self, bar, atr=0): return None


def _get_disabled_orb():
    return DisabledORB()


hdr = f"{'Config':<10} {'Fold':<8} {'Mode':<8} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7}"
print(hdr)
print("-" * len(hdr))

for config_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue

        # With ORB
        signals_mod.get_orb_strategy = _orig_get_orb
        s_full = run_variant(fold_data, f"{config_label}_{fold_name}_Full", verbose=False, **base_kwargs)

        # Without ORB
        signals_mod.get_orb_strategy = _get_disabled_orb
        s_no = run_variant(fold_data, f"{config_label}_{fold_name}_NoORB", verbose=False, **base_kwargs)

        # Restore
        signals_mod.get_orb_strategy = _orig_get_orb

        print(f"  {config_label:<10} {fold_name:<8} {'Full':<8} {s_full['n']:>5} {s_full['sharpe']:>7.2f} "
              f"${s_full['total_pnl']:>9,.0f} {s_full['win_rate']:>5.1f}% ${s_full['total_pnl']/max(s_full['n'],1):>6.2f}")
        print(f"  {config_label:<10} {fold_name:<8} {'NoORB':<8} {s_no['n']:>5} {s_no['sharpe']:>7.2f} "
              f"${s_no['total_pnl']:>9,.0f} {s_no['win_rate']:>5.1f}% ${s_no['total_pnl']/max(s_no['n'],1):>6.2f}")

print("\n  ORB Decision Criteria:")
print("  - If NoORB wins >=4/6 folds AND full-sample Sharpe delta > +0.15 → disable ORB")
print("  - If NoORB wins <3/6 folds → keep ORB")
print("  - If mixed → recheck with spread costs")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-C: Stochastic 10/90 with Spread Cost — K-Fold
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-C: STOCHASTIC 10/90 MEAN REVERSION — K-FOLD with SPREAD COST")
print("  Prior: $/t=$27.66, Sharpe decent, but only N=103 over 11 years")
print("  Method: standalone sim per fold, with $0.30 spread cost per trade")
print("=" * 80)

h1_df = data.h1_df.copy()

period, smooth_k, smooth_d = 14, 3, 3
h1_df['lowest_14'] = h1_df['Low'].rolling(period).min()
h1_df['highest_14'] = h1_df['High'].rolling(period).max()
h1_df['fast_k'] = 100 * (h1_df['Close'] - h1_df['lowest_14']) / (h1_df['highest_14'] - h1_df['lowest_14'])
h1_df['slowk'] = h1_df['fast_k'].rolling(smooth_k).mean()
h1_df['slowd'] = h1_df['slowk'].rolling(smooth_d).mean()
h1_df['stoch_cross_up'] = (h1_df['slowk'] > h1_df['slowd']) & (h1_df['slowk'].shift(1) <= h1_df['slowd'].shift(1))
h1_df['stoch_cross_down'] = (h1_df['slowk'] < h1_df['slowd']) & (h1_df['slowk'].shift(1) >= h1_df['slowd'].shift(1))

if 'EMA100' not in h1_df.columns:
    h1_df['EMA100'] = h1_df['Close'].ewm(span=100, adjust=False).mean()
if 'ATR' not in h1_df.columns:
    h1_df['tr'] = np.maximum(h1_df['High'] - h1_df['Low'],
                              np.maximum(abs(h1_df['High'] - h1_df['Close'].shift(1)),
                                         abs(h1_df['Low'] - h1_df['Close'].shift(1))))
    h1_df['ATR'] = h1_df['tr'].rolling(14).mean()


def run_stoch_sim(h1_slice, buy_th, sell_th, sl_atr, max_hold, spread_cost=0.30, risk=50.0):
    """Simulate stochastic extremes mean reversion on an H1 slice."""
    trades = []
    pos = None

    for i in range(max(100, 20), len(h1_slice)):
        row = h1_slice.iloc[i]
        close = float(row['Close'])
        atr = float(row.get('ATR', 0))
        ema100 = float(row.get('EMA100', close))
        slowk = float(row.get('slowk', 50))
        stoch_up = bool(row.get('stoch_cross_up', False))
        stoch_down = bool(row.get('stoch_cross_down', False))
        bar_time = h1_slice.index[i]

        if atr <= 0 or pd.isna(atr) or pd.isna(slowk):
            continue

        if pos is not None:
            pos['bars'] += 1
            reason = None
            exit_price = close

            if pos['dir'] == 'BUY' and close <= pos['sl']:
                reason = 'SL'
                exit_price = pos['sl']
            elif pos['dir'] == 'SELL' and close >= pos['sl']:
                reason = 'SL'
                exit_price = pos['sl']
            elif pos['bars'] >= max_hold:
                reason = 'Timeout'

            if reason:
                if pos['dir'] == 'BUY':
                    pnl_raw = (exit_price - pos['entry']) * (risk / (sl_atr * pos['atr']))
                else:
                    pnl_raw = (pos['entry'] - exit_price) * (risk / (sl_atr * pos['atr']))
                pnl = pnl_raw - spread_cost
                trades.append({'pnl': pnl, 'reason': reason, 'entry_time': pos['entry_time'],
                               'bars': pos['bars']})
                pos = None

        if pos is None:
            sl_dist = sl_atr * atr
            if slowk < buy_th and stoch_up and close > ema100:
                pos = {'dir': 'BUY', 'entry': close, 'sl': close - sl_dist,
                       'entry_time': bar_time, 'atr': atr, 'bars': 0}
            elif slowk > (100 - buy_th) and stoch_down and close < ema100:
                pos = {'dir': 'SELL', 'entry': close, 'sl': close + sl_dist,
                       'entry_time': bar_time, 'atr': atr, 'bars': 0}

    return trades


STOCH_CONFIGS = [
    ("Stoch10/90 H6", 10, 90, 4.5, 6),
    ("Stoch10/90 H8", 10, 90, 4.5, 8),
    ("Stoch10/90 H12", 10, 90, 4.5, 12),
    ("Stoch15/85 H8", 15, 85, 4.5, 8),
    ("Stoch15/85 H12", 15, 85, 4.5, 12),
    ("Stoch20/80 H8", 20, 80, 4.5, 8),
]

SPREAD_LEVELS = [0.0, 0.30, 0.50]

print("\n--- C1: Full-sample stochastic grid (with spread) ---")
hdr = f"{'Config':<20} {'Spread':>6} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7}"
print(hdr)
print("-" * len(hdr))

stoch_full_results = []
for label, buy_th, sell_th, sl_atr, max_hold in STOCH_CONFIGS:
    for spread in SPREAD_LEVELS:
        trades = run_stoch_sim(h1_df, buy_th, sell_th, sl_atr, max_hold, spread_cost=spread)
        if not trades:
            continue
        total_pnl = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        n = len(trades)
        wr = 100.0 * wins / n
        ppt = total_pnl / n

        daily = defaultdict(float)
        for t in trades:
            daily[t['entry_time'].strftime('%Y-%m-%d')] += t['pnl']
        vals = list(daily.values())
        sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0

        print(f"  {label:<20} ${spread:>5.2f} {n:>5} {sharpe:>7.2f} ${total_pnl:>9,.0f} "
              f"{wr:>5.1f}% ${ppt:>6.2f}")
        stoch_full_results.append({'label': label, 'spread': spread, 'n': n,
                                   'sharpe': sharpe, 'pnl': total_pnl, 'ppt': ppt})

# Pick best config at $0.30 spread for K-Fold
best_030 = [r for r in stoch_full_results if r['spread'] == 0.30]
if best_030:
    best_stoch = max(best_030, key=lambda x: x['sharpe'])
    print(f"\n  Best at $0.30 spread: {best_stoch['label']} (Sharpe={best_stoch['sharpe']:.2f}, $/t=${best_stoch['ppt']:.2f})")

    # Find the config
    best_cfg = None
    for label, buy_th, sell_th, sl_atr, max_hold in STOCH_CONFIGS:
        if label == best_stoch['label']:
            best_cfg = (label, buy_th, sell_th, sl_atr, max_hold)
            break

    if best_cfg:
        print(f"\n--- C2: K-Fold for best config: {best_cfg[0]} ---")
        hdr = f"{'Fold':<8} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7}"
        print(hdr)
        print("-" * len(hdr))

        fold_wins = 0
        for fold_name, start, end in FOLDS:
            ts = pd.Timestamp(start, tz='UTC')
            te = pd.Timestamp(end, tz='UTC')
            h1_fold = h1_df[(h1_df.index >= ts) & (h1_df.index < te)]
            if len(h1_fold) < 200:
                continue

            trades = run_stoch_sim(h1_fold, best_cfg[1], best_cfg[2], best_cfg[3], best_cfg[4], spread_cost=0.30)
            if not trades:
                print(f"  {fold_name:<8} {'No trades':>5}")
                continue

            total_pnl = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            n = len(trades)
            wr = 100.0 * wins / n
            ppt = total_pnl / n

            daily = defaultdict(float)
            for t in trades:
                daily[t['entry_time'].strftime('%Y-%m-%d')] += t['pnl']
            vals = list(daily.values())
            sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0

            if sharpe > 0:
                fold_wins += 1

            print(f"  {fold_name:<8} {n:>5} {sharpe:>7.2f} ${total_pnl:>9,.0f} {wr:>5.1f}% ${ppt:>6.2f}")

        print(f"\n  K-Fold: {fold_wins}/6 folds positive Sharpe")
        print("  Pass criteria: >=4/6 positive AND $/t > $10 after cost")
else:
    print("\n  No stochastic results at $0.30 spread")


# Compare with M15 RSI from same baselines
print("\n--- C3: Stochastic vs M15 RSI comparison ---")
for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    rsi_trades = [t for t in base_stats['_trades'] if t.strategy == 'm15_rsi']
    if rsi_trades:
        rsi_pnl = sum(t.pnl for t in rsi_trades)
        rsi_wins = sum(1 for t in rsi_trades if t.pnl > 0)
        rsi_n = len(rsi_trades)
        rsi_wr = 100.0 * rsi_wins / rsi_n
        rsi_ppt = rsi_pnl / rsi_n
        print(f"  {config_label} M15 RSI: N={rsi_n}, PnL=${rsi_pnl:,.0f}, WR={rsi_wr:.1f}%, $/t=${rsi_ppt:.2f}")
    else:
        print(f"  {config_label}: No M15 RSI trades")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-D: Squeeze-to-Expansion Confidence Score
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-D: SQUEEZE-TO-EXPANSION CONFIDENCE SCORE — Keltner Entry Weighting")
print("  Prior: EXP51 showed squeeze-to-expansion $/t=$6.21 vs normal $4.44 (Mega)")
print("  Method: Instead of binary filter, add BW-based confidence score to lot sizing")
print("  Logic: KC bandwidth expanding → increase lots, contracting → decrease lots")
print("=" * 80)

# Compute KC bandwidth metrics on H1
h1_bw = data.h1_df.copy()
h1_bw['kc_bw'] = (h1_bw['KC_upper'] - h1_bw['KC_lower']) / h1_bw['KC_mid']
h1_bw['kc_bw_prev3'] = h1_bw['kc_bw'].shift(3)
h1_bw['kc_bw_prev5'] = h1_bw['kc_bw'].shift(5)
h1_bw['bw_expanding_3'] = h1_bw['kc_bw'] > h1_bw['kc_bw_prev3']
h1_bw['bw_expanding_5'] = h1_bw['kc_bw'] > h1_bw['kc_bw_prev5']
h1_bw['bw_ratio_3'] = h1_bw['kc_bw'] / h1_bw['kc_bw_prev3']

# Quantile bins for BW percentile
bw_vals = h1_bw['kc_bw'].dropna()
bw_q25, bw_q50, bw_q75 = np.percentile(bw_vals, [25, 50, 75])

print(f"\n  KC Bandwidth stats: Q25={bw_q25:.4f}, Q50={bw_q50:.4f}, Q75={bw_q75:.4f}")

# Post-hoc analysis on trade list: tag each keltner trade with BW state at entry
print("\n--- D1: Trade BW-state analysis (post-hoc) ---")
for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    kc_trades = [t for t in base_stats['_trades'] if t.strategy == 'keltner']
    if not kc_trades:
        print(f"  {config_label}: No keltner trades")
        continue

    bw_groups = {'squeeze': [], 'normal': [], 'expanding': [], 'strong_exp': []}

    for t in kc_trades:
        entry_h1 = h1_bw.index.searchsorted(pd.Timestamp(t.entry_time))
        if entry_h1 <= 0 or entry_h1 >= len(h1_bw):
            continue
        idx = min(entry_h1, len(h1_bw) - 1)
        row = h1_bw.iloc[idx]
        bw = row.get('kc_bw', np.nan)
        bw_prev = row.get('kc_bw_prev3', np.nan)

        if pd.isna(bw) or pd.isna(bw_prev):
            continue

        if bw < bw_q25:
            bw_groups['squeeze'].append(t.pnl)
        elif bw > bw_q75 and bw > bw_prev:
            bw_groups['strong_exp'].append(t.pnl)
        elif bw > bw_prev:
            bw_groups['expanding'].append(t.pnl)
        else:
            bw_groups['normal'].append(t.pnl)

    print(f"\n  {config_label} Keltner trades by BW state:")
    for grp, pnls in bw_groups.items():
        n = len(pnls)
        if n == 0:
            print(f"    {grp:>12}: N=0")
            continue
        total = sum(pnls)
        avg = total / n
        wins = sum(1 for p in pnls if p > 0)
        wr = 100.0 * wins / n
        print(f"    {grp:>12}: N={n:>4}, PnL=${total:>8,.0f}, $/t=${avg:>6.2f}, WR={wr:.1f}%")


# D2: Simulate confidence-scored sizing
print("\n--- D2: BW Confidence Sizing Simulation ---")

SIZING_SCHEMES = [
    ("1.0x baseline", {'squeeze': 1.0, 'normal': 1.0, 'expanding': 1.0, 'strong_exp': 1.0}),
    ("Conservative (0.7/1.0/1.2/1.4)", {'squeeze': 0.7, 'normal': 1.0, 'expanding': 1.2, 'strong_exp': 1.4}),
    ("Aggressive (0.5/0.8/1.3/1.8)", {'squeeze': 0.5, 'normal': 0.8, 'expanding': 1.3, 'strong_exp': 1.8}),
    ("Skip squeeze (0/1/1/1.3)", {'squeeze': 0.0, 'normal': 1.0, 'expanding': 1.0, 'strong_exp': 1.3}),
    ("Expand only (0.5/0.5/1.2/1.5)", {'squeeze': 0.5, 'normal': 0.5, 'expanding': 1.2, 'strong_exp': 1.5}),
]

for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    kc_trades = [t for t in base_stats['_trades'] if t.strategy == 'keltner']
    non_kc = [t for t in base_stats['_trades'] if t.strategy != 'keltner']
    if not kc_trades:
        continue

    print(f"\n  {config_label}:")
    hdr = f"    {'Scheme':<35} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'AvgMult':>8} {'Δ Sharpe':>9}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    base_sharpe = None

    for scheme_label, mults in SIZING_SCHEMES:
        sized_pnls = []
        mult_used = []
        sized_trades_for_daily = []

        for t in kc_trades:
            entry_h1 = h1_bw.index.searchsorted(pd.Timestamp(t.entry_time))
            idx = min(max(entry_h1, 0), len(h1_bw) - 1)
            row = h1_bw.iloc[idx]
            bw = row.get('kc_bw', np.nan)
            bw_prev = row.get('kc_bw_prev3', np.nan)

            if pd.isna(bw) or pd.isna(bw_prev):
                m = 1.0
            elif bw < bw_q25:
                m = mults['squeeze']
            elif bw > bw_q75 and bw > bw_prev:
                m = mults['strong_exp']
            elif bw > bw_prev:
                m = mults['expanding']
            else:
                m = mults['normal']

            if m > 0:
                sized_pnls.append(t.pnl * m)
                mult_used.append(m)
                sized_trades_for_daily.append(t)

        # Add non-keltner trades unchanged
        all_pnls = sized_pnls + [t.pnl for t in non_kc]
        all_trade_times = sized_trades_for_daily + list(non_kc)

        daily = defaultdict(float)
        for t_obj, p in zip(all_trade_times, all_pnls):
            daily[t_obj.entry_time.strftime('%Y-%m-%d')] += p
        vals = list(daily.values())
        sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0
        total_pnl = sum(all_pnls)
        avg_mult = np.mean(mult_used) if mult_used else 1.0

        if base_sharpe is None:
            base_sharpe = sharpe

        delta = sharpe - base_sharpe
        print(f"    {scheme_label:<35} {len(all_pnls):>5} {sharpe:>7.2f} ${total_pnl:>9,.0f} "
              f"{avg_mult:>7.2f}x {delta:>+8.2f}")


# D3: K-Fold for best scheme
print("\n--- D3: K-Fold for BW Confidence Sizing ---")
print("  (Using Conservative scheme as default candidate)")

CONSERVATIVE_MULTS = {'squeeze': 0.7, 'normal': 1.0, 'expanding': 1.2, 'strong_exp': 1.4}

hdr = f"{'Config':<10} {'Fold':<8} {'Mode':<10} {'Sharpe':>7} {'PnL':>10} {'Δ Sharpe':>9}"
print(hdr)
print("-" * len(hdr))

for config_label, base_kwargs in [("Current", CURRENT), ("Mega", MEGA)]:
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue

        fold_stats = run_variant(fold_data, f"D3_{config_label}_{fold_name}", verbose=False, **base_kwargs)
        kc_trades = [t for t in fold_stats['_trades'] if t.strategy == 'keltner']
        non_kc = [t for t in fold_stats['_trades'] if t.strategy != 'keltner']

        # Baseline daily sharpe
        daily_base = defaultdict(float)
        for t in fold_stats['_trades']:
            daily_base[t.entry_time.strftime('%Y-%m-%d')] += t.pnl
        vals_base = list(daily_base.values())
        sharpe_base = np.mean(vals_base) / np.std(vals_base) * np.sqrt(252) if len(vals_base) > 1 and np.std(vals_base) > 0 else 0

        ts = pd.Timestamp(start, tz='UTC')
        te = pd.Timestamp(end, tz='UTC')
        h1_fold_bw = h1_bw[(h1_bw.index >= ts) & (h1_bw.index < te)]

        # BW confidence sizing
        all_pnls_sized = []
        all_trades_sized = []
        for t in kc_trades:
            entry_h1 = h1_fold_bw.index.searchsorted(pd.Timestamp(t.entry_time))
            idx = min(max(entry_h1, 0), len(h1_fold_bw) - 1) if len(h1_fold_bw) > 0 else 0
            if len(h1_fold_bw) == 0:
                m = 1.0
            else:
                row = h1_fold_bw.iloc[idx]
                bw = row.get('kc_bw', np.nan)
                bw_prev = row.get('kc_bw_prev3', np.nan)
                if pd.isna(bw) or pd.isna(bw_prev):
                    m = 1.0
                elif bw < bw_q25:
                    m = CONSERVATIVE_MULTS['squeeze']
                elif bw > bw_q75 and bw > bw_prev:
                    m = CONSERVATIVE_MULTS['strong_exp']
                elif bw > bw_prev:
                    m = CONSERVATIVE_MULTS['expanding']
                else:
                    m = CONSERVATIVE_MULTS['normal']

            all_pnls_sized.append(t.pnl * m)
            all_trades_sized.append(t)

        for t in non_kc:
            all_pnls_sized.append(t.pnl)
            all_trades_sized.append(t)

        daily_sized = defaultdict(float)
        for t_obj, p in zip(all_trades_sized, all_pnls_sized):
            daily_sized[t_obj.entry_time.strftime('%Y-%m-%d')] += p
        vals_sized = list(daily_sized.values())
        sharpe_sized = np.mean(vals_sized) / np.std(vals_sized) * np.sqrt(252) if len(vals_sized) > 1 and np.std(vals_sized) > 0 else 0

        delta = sharpe_sized - sharpe_base
        pnl_base = fold_stats['total_pnl']
        pnl_sized = sum(all_pnls_sized)

        print(f"  {config_label:<10} {fold_name:<8} {'Base':<10} {sharpe_base:>7.2f} ${pnl_base:>9,.0f}")
        print(f"  {config_label:<10} {fold_name:<8} {'BW-Conf':<10} {sharpe_sized:>7.2f} ${pnl_sized:>9,.0f} {delta:>+8.2f}")

print("\n  BW Confidence Pass Criteria:")
print("  - >=4/6 folds with positive Sharpe delta")
print("  - No fold with Sharpe delta < -0.50")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total_elapsed = time.time() - t_total
print("\n\n" + "=" * 80)
print("EXPERIMENT SUMMARY")
print("=" * 80)
print(f"""
  EXP-A: Trail Momentum K-Fold
    → Check if >=5/6 folds positive delta, max_dd increase <15%
    → If pass, implement as post-trade sizing rule in gold_trader.py

  EXP-B: ORB Diagnosis
    → If NoORB wins >=4/6 folds → disable ORB in live
    → If mixed → keep ORB but lower priority

  EXP-C: Stochastic 10/90
    → If K-Fold >=4/6 positive AND $/t > $10 at $0.30 cost → paper trade candidate
    → Compare with M15 RSI: only adopt if clearly superior

  EXP-D: Squeeze-to-Expansion Confidence Score
    → If >=4/6 folds positive delta → implement as BW-based lot multiplier
    → This is a WEIGHTING approach, not a binary filter (BW filter already rejected)
""")
print(f"Total runtime: {total_elapsed / 60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
