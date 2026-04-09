#!/usr/bin/env python3
"""
EXP-NEW BATCH 2 — 5 More Non-overlapping Experiments
======================================================
  EXP-E: Mega T0.5/D0.15 with Spread Cost K-Fold Validation
  EXP-F: Current config max_hold fine-tuning (16-60 bars, with cost)
  EXP-G: KC Channel Relative Position at Entry (confidence score)
  EXP-H: MFE/MAE Deep Analysis (trade efficiency, capture ratio)
  EXP-I: Consecutive Direction Pattern Analysis (BUY/SELL streaks)

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

OUTPUT_FILE = "exp_new_batch2_output.txt"


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
print("EXP-NEW BATCH 2 — 5 MORE NON-OVERLAPPING EXPERIMENTS")
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

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-04-01"),
]

# ═══════════════════════════════════════════════════════════════════════════════
# Shared baselines (no cost)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SHARED BASELINES (no spread)")
print("=" * 80)

base_cur = run_variant(data, "Current-NoCost", **CURRENT)
base_mega = run_variant(data, "Mega-NoCost", **MEGA)

print(f"\n  Current: N={base_cur['n']:,} Sharpe={base_cur['sharpe']:.2f} PnL=${base_cur['total_pnl']:,.0f}")
print(f"  Mega:    N={base_mega['n']:,} Sharpe={base_mega['sharpe']:.2f} PnL=${base_mega['total_pnl']:,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-E: Mega vs Current with Spread Cost K-Fold
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-E: MEGA vs CURRENT — WITH SPREAD COST K-FOLD")
print("  Prior: Mega no-cost Sharpe 7.66 >> Current 4.41")
print("  Question: Does Mega survive $0.30/$0.50 spread? Is it still better?")
print("=" * 80)

SPREADS = [0.0, 0.30, 0.50]

# E1: Full-sample Current vs Mega at each spread
print("\n--- E1: Full-sample spread sensitivity ---")
hdr = f"{'Config':<10} {'Spread':>7} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7} {'MaxDD':>8}"
print(hdr)
print("-" * len(hdr))

for spread in SPREADS:
    s_cur = run_variant(data, f"Current_sp{spread}", verbose=False, **CURRENT, spread_cost=spread)
    s_mega = run_variant(data, f"Mega_sp{spread}", verbose=False, **MEGA, spread_cost=spread)
    for label, s in [("Current", s_cur), ("Mega", s_mega)]:
        ppt = s['total_pnl'] / max(s['n'], 1)
        print(f"  {label:<10} ${spread:>6.2f} {s['n']:>5} {s['sharpe']:>7.2f} "
              f"${s['total_pnl']:>9,.0f} {s['win_rate']:>5.1f}% ${ppt:>6.2f} ${s['max_dd']:>7,.0f}")

# E2: K-Fold at $0.30 spread
print("\n--- E2: K-Fold at $0.30 spread ---")
hdr = f"{'Fold':<8} {'Current':>8} {'Mega':>8} {'Winner':>8} {'Δ Sharpe':>9}"
print(hdr)
print("-" * len(hdr))

cur_wins, mega_wins = 0, 0
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
        continue
    s_cur = run_variant(fold_data, f"E2_Cur_{fold_name}", verbose=False, **CURRENT, spread_cost=0.30)
    s_mega = run_variant(fold_data, f"E2_Mega_{fold_name}", verbose=False, **MEGA, spread_cost=0.30)
    delta = s_mega['sharpe'] - s_cur['sharpe']
    winner = "Mega" if delta > 0 else "Current"
    if delta > 0:
        mega_wins += 1
    else:
        cur_wins += 1
    print(f"  {fold_name:<8} {s_cur['sharpe']:>8.2f} {s_mega['sharpe']:>8.2f} {winner:>8} {delta:>+8.2f}")

print(f"\n  K-Fold score: Current={cur_wins}/6, Mega={mega_wins}/6")
print("  Decision: Mega >=5/6 wins → switch live to Mega. <4 → keep Current.")

# E3: K-Fold at $0.50 spread
print("\n--- E3: K-Fold at $0.50 spread ---")
hdr = f"{'Fold':<8} {'Current':>8} {'Mega':>8} {'Winner':>8} {'Δ Sharpe':>9}"
print(hdr)
print("-" * len(hdr))

cur_wins_50, mega_wins_50 = 0, 0
for fold_name, start, end in FOLDS:
    fold_data = data.slice(start, end)
    if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
        continue
    s_cur = run_variant(fold_data, f"E3_Cur_{fold_name}", verbose=False, **CURRENT, spread_cost=0.50)
    s_mega = run_variant(fold_data, f"E3_Mega_{fold_name}", verbose=False, **MEGA, spread_cost=0.50)
    delta = s_mega['sharpe'] - s_cur['sharpe']
    winner = "Mega" if delta > 0 else "Current"
    if delta > 0:
        mega_wins_50 += 1
    else:
        cur_wins_50 += 1
    print(f"  {fold_name:<8} {s_cur['sharpe']:>8.2f} {s_mega['sharpe']:>8.2f} {winner:>8} {delta:>+8.2f}")

print(f"\n  K-Fold score ($0.50): Current={cur_wins_50}/6, Mega={mega_wins_50}/6")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-F: Current Config Max Hold Fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-F: CURRENT CONFIG MAX HOLD FINE-TUNING (with $0.30 spread)")
print("  Prior: Timeout 60bars = $12,287 loss (largest single exit category)")
print("  EXP01 said max_hold=60 optimal for old config, re-test with Current")
print("  exit_combo_matrix only tests Mega config")
print("=" * 80)

MAX_HOLD_VALUES = [16, 20, 24, 32, 40, 0]  # 0 = default (60 from config)

# F1: Full-sample sweep
print("\n--- F1: Full-sample max_hold sweep (Current + $0.30) ---")
hdr = f"{'MaxHold':>8} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7} {'SL':>5} {'Trail':>6} {'Timeout':>8} {'TDTP':>5}"
print(hdr)
print("-" * len(hdr))

f1_results = []
for mh in MAX_HOLD_VALUES:
    label = f"MH={mh}" if mh > 0 else "MH=60(def)"
    kwargs = {**CURRENT, "spread_cost": 0.30}
    if mh > 0:
        kwargs["keltner_max_hold_m15"] = mh
    s = run_variant(data, label, verbose=False, **kwargs)

    trades = s['_trades']
    kc_trades = [t for t in trades if t.strategy == 'keltner']
    sl_n = sum(1 for t in kc_trades if t.exit_reason == 'SL')
    trail_n = sum(1 for t in kc_trades if t.exit_reason == 'Trailing')
    timeout_n = sum(1 for t in kc_trades if t.exit_reason == 'Timeout')
    tdtp_n = sum(1 for t in kc_trades if t.exit_reason == 'TimeDecayTP')

    ppt = s['total_pnl'] / max(s['n'], 1)
    mh_label = str(mh) if mh > 0 else "60(def)"
    print(f"  {mh_label:>8} {s['n']:>5} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} "
          f"{s['win_rate']:>5.1f}% ${ppt:>6.2f} {sl_n:>5} {trail_n:>6} {timeout_n:>8} {tdtp_n:>5}")
    f1_results.append({'mh': mh, 'sharpe': s['sharpe'], 'pnl': s['total_pnl']})

# F2: K-Fold for top-2 max_hold values
f1_sorted = sorted(f1_results, key=lambda x: x['sharpe'], reverse=True)
top2 = [r['mh'] for r in f1_sorted[:2] if r['mh'] != 0]
if 0 in [r['mh'] for r in f1_sorted[:2]]:
    top2.append(0)
top2 = top2[:2]

print(f"\n--- F2: K-Fold for top-2 max_hold values: {top2} ---")
hdr = f"{'MaxHold':>8} {'Fold':<8} {'Sharpe':>7} {'PnL':>10}"
print(hdr)
print("-" * len(hdr))

for mh in top2:
    fold_sharpes = []
    for fold_name, start, end in FOLDS:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        kwargs = {**CURRENT, "spread_cost": 0.30}
        if mh > 0:
            kwargs["keltner_max_hold_m15"] = mh
        mh_label = str(mh) if mh > 0 else "60(def)"
        s = run_variant(fold_data, f"F2_MH{mh_label}_{fold_name}", verbose=False, **kwargs)
        fold_sharpes.append(s['sharpe'])
        print(f"  {mh_label:>8} {fold_name:<8} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f}")
    if fold_sharpes:
        avg = np.mean(fold_sharpes)
        std = np.std(fold_sharpes)
        pos = sum(1 for s in fold_sharpes if s > 0)
        print(f"  {mh_label:>8} {'SUMMARY':<8} avg={avg:.2f} std={std:.2f} positive={pos}/{len(fold_sharpes)}")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-G: KC Channel Relative Position at Entry
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-G: KC CHANNEL RELATIVE POSITION AT ENTRY")
print("  Question: Does entry location within KC channel affect trade quality?")
print("  Method: Post-hoc, tag each Keltner trade with KC_position at entry")
print("=" * 80)

h1_df = data.h1_df.copy()

for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    kc_trades = [t for t in base_stats['_trades'] if t.strategy == 'keltner']
    if not kc_trades:
        print(f"\n  {config_label}: No keltner trades")
        continue

    print(f"\n--- G1: {config_label} Keltner entry KC-position analysis ---")

    kc_positions = []
    trade_data = []

    for t in kc_trades:
        entry_h1_idx = h1_df.index.searchsorted(pd.Timestamp(t.entry_time))
        idx = min(max(entry_h1_idx, 0), len(h1_df) - 1)
        row = h1_df.iloc[idx]
        kc_u = row.get('KC_upper', np.nan)
        kc_l = row.get('KC_lower', np.nan)
        kc_m = row.get('KC_mid', np.nan)
        atr = row.get('ATR', np.nan)

        if pd.isna(kc_u) or pd.isna(kc_l) or kc_u == kc_l:
            continue

        kc_pos = (t.entry_price - kc_l) / (kc_u - kc_l)
        kc_positions.append(kc_pos)
        trade_data.append({
            'pnl': t.pnl, 'direction': t.direction, 'bars_held': t.bars_held,
            'exit_reason': t.exit_reason, 'kc_pos': kc_pos,
            'breakout_dist': (t.entry_price - kc_u) / atr if t.direction == 'BUY' and not pd.isna(atr) and atr > 0
                            else (kc_l - t.entry_price) / atr if t.direction == 'SELL' and not pd.isna(atr) and atr > 0
                            else 0,
        })

    if not trade_data:
        continue

    df_trades = pd.DataFrame(trade_data)

    # Quantile analysis
    q_labels = ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']
    df_trades['q'] = pd.qcut(df_trades['kc_pos'], 4, labels=q_labels, duplicates='drop')

    print(f"\n  KC Position quantile analysis (N={len(df_trades)}):")
    hdr = f"    {'Quantile':<12} {'N':>5} {'$/t':>7} {'WR%':>6} {'AvgBars':>8} {'Trail%':>7} {'SL%':>5}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    for q_name in q_labels:
        grp = df_trades[df_trades['q'] == q_name]
        if len(grp) == 0:
            continue
        n = len(grp)
        ppt = grp['pnl'].mean()
        wr = 100.0 * (grp['pnl'] > 0).sum() / n
        avg_bars = grp['bars_held'].mean()
        trail_pct = 100.0 * (grp['exit_reason'] == 'Trailing').sum() / n
        sl_pct = 100.0 * (grp['exit_reason'] == 'SL').sum() / n
        print(f"    {q_name:<12} {n:>5} ${ppt:>6.2f} {wr:>5.1f}% {avg_bars:>7.1f} {trail_pct:>6.1f}% {sl_pct:>4.1f}%")

    # BUY: distance above upper band at entry
    buys = df_trades[df_trades['direction'] == 'BUY']
    sells = df_trades[df_trades['direction'] == 'SELL']

    if len(buys) > 20:
        print(f"\n  BUY breakout distance (ATR units) vs PnL:")
        buys_sorted = buys.sort_values('breakout_dist')
        thirds = np.array_split(buys_sorted, 3)
        for i, third in enumerate(thirds):
            n = len(third)
            avg_dist = third['breakout_dist'].mean()
            avg_pnl = third['pnl'].mean()
            wr = 100.0 * (third['pnl'] > 0).sum() / n
            print(f"    T{i+1} (avg dist={avg_dist:.2f} ATR): N={n}, $/t=${avg_pnl:.2f}, WR={wr:.1f}%")

    if len(sells) > 20:
        print(f"\n  SELL breakout distance (ATR units) vs PnL:")
        sells_sorted = sells.sort_values('breakout_dist')
        thirds = np.array_split(sells_sorted, 3)
        for i, third in enumerate(thirds):
            n = len(third)
            avg_dist = third['breakout_dist'].mean()
            avg_pnl = third['pnl'].mean()
            wr = 100.0 * (third['pnl'] > 0).sum() / n
            print(f"    T{i+1} (avg dist={avg_dist:.2f} ATR): N={n}, $/t=${avg_pnl:.2f}, WR={wr:.1f}%")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-H: MFE/MAE Deep Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-H: MFE/MAE DEEP ANALYSIS (Maximum Favorable/Adverse Excursion)")
print("  Purpose: Measure trade efficiency and identify exit improvement targets")
print("  Method: Replay M15 bars between entry/exit to compute per-trade MFE/MAE")
print("=" * 80)

m15_df = data.m15_df

for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    kc_trades = [t for t in base_stats['_trades'] if t.strategy == 'keltner']
    if not kc_trades:
        continue

    print(f"\n--- H1: {config_label} Keltner MFE/MAE computation ---")

    mfe_list, mae_list, pnl_list, capture_list = [], [], [], []
    exit_reasons = []
    trade_details = []

    for t in kc_trades:
        entry_ts = pd.Timestamp(t.entry_time)
        exit_ts = pd.Timestamp(t.exit_time)
        mask = (m15_df.index >= entry_ts) & (m15_df.index <= exit_ts)
        bars = m15_df[mask]

        if len(bars) == 0:
            continue

        if t.direction == 'BUY':
            max_price = bars['High'].max()
            min_price = bars['Low'].min()
            mfe = max_price - t.entry_price
            mae = t.entry_price - min_price
        else:
            max_price = bars['High'].max()
            min_price = bars['Low'].min()
            mfe = t.entry_price - min_price
            mae = max_price - t.entry_price

        mfe = max(mfe, 0)
        mae = max(mae, 0)
        capture = t.pnl / (mfe * t.lots * 100) if mfe > 0 and t.lots > 0 else 0

        mfe_list.append(mfe)
        mae_list.append(mae)
        pnl_list.append(t.pnl)
        capture_list.append(capture)
        exit_reasons.append(t.exit_reason)
        trade_details.append({
            'pnl': t.pnl, 'mfe': mfe, 'mae': mae, 'capture': capture,
            'exit_reason': t.exit_reason, 'bars_held': t.bars_held,
            'direction': t.direction,
        })

    if not trade_details:
        continue

    df_h = pd.DataFrame(trade_details)
    winners = df_h[df_h['pnl'] > 0]
    losers = df_h[df_h['pnl'] <= 0]

    print(f"\n  Overall (N={len(df_h)}):")
    print(f"    MFE: mean=${np.mean(mfe_list):.2f}, median=${np.median(mfe_list):.2f}, max=${np.max(mfe_list):.2f}")
    print(f"    MAE: mean=${np.mean(mae_list):.2f}, median=${np.median(mae_list):.2f}, max=${np.max(mae_list):.2f}")
    print(f"    Capture ratio: mean={np.mean(capture_list):.2f}, median={np.median(capture_list):.2f}")

    if len(winners) > 0:
        print(f"\n  Winners (N={len(winners)}):")
        print(f"    MFE: mean=${winners['mfe'].mean():.2f}, median=${winners['mfe'].median():.2f}")
        print(f"    MAE: mean=${winners['mae'].mean():.2f}, median=${winners['mae'].median():.2f}")
        print(f"    Capture: mean={winners['capture'].mean():.2f}, median={winners['capture'].median():.2f}")
        print(f"    Winners' worst drawdown (MAE) distribution:")
        for pct in [25, 50, 75, 90]:
            print(f"      P{pct}: ${winners['mae'].quantile(pct/100):.2f}")

    if len(losers) > 0:
        print(f"\n  Losers (N={len(losers)}):")
        print(f"    MFE: mean=${losers['mfe'].mean():.2f}, median=${losers['mfe'].median():.2f}")
        print(f"    MAE: mean=${losers['mae'].mean():.2f}, median=${losers['mae'].median():.2f}")

        # Critical: losers that had significant MFE (once profitable but turned to loss)
        h1_atr_mean = h1_df['ATR'].dropna().mean()
        thresholds_atr = [0.5, 1.0, 1.5, 2.0]
        print(f"\n  Losers that once had MFE >= X*ATR (avg ATR=${h1_atr_mean:.2f}):")
        for th in thresholds_atr:
            th_price = th * h1_atr_mean
            had_profit = losers[losers['mfe'] >= th_price]
            n_had = len(had_profit)
            pct = 100.0 * n_had / len(losers) if len(losers) > 0 else 0
            avg_loss = had_profit['pnl'].mean() if n_had > 0 else 0
            total_loss = had_profit['pnl'].sum() if n_had > 0 else 0
            print(f"    MFE >= {th:.1f}*ATR (${th_price:.1f}): N={n_had} ({pct:.1f}% of losers), "
                  f"avg loss=${avg_loss:.2f}, total=${total_loss:,.0f}")

    # By exit reason
    print(f"\n  MFE/MAE by exit reason:")
    hdr = f"    {'Reason':<15} {'N':>5} {'AvgMFE':>8} {'AvgMAE':>8} {'Capture':>8} {'Avg$/t':>8}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for reason in ['SL', 'Trailing', 'Timeout', 'TimeDecayTP', 'TP']:
        grp = df_h[df_h['exit_reason'] == reason]
        if len(grp) == 0:
            continue
        print(f"    {reason:<15} {len(grp):>5} ${grp['mfe'].mean():>7.2f} ${grp['mae'].mean():>7.2f} "
              f"{grp['capture'].mean():>7.2f} ${grp['pnl'].mean():>7.2f}")

    # Capture ratio distribution for trailing exits
    trail_trades = df_h[df_h['exit_reason'] == 'Trailing']
    if len(trail_trades) > 10:
        print(f"\n  Trailing exit capture ratio distribution (N={len(trail_trades)}):")
        for pct in [10, 25, 50, 75, 90]:
            val = trail_trades['capture'].quantile(pct / 100)
            print(f"    P{pct}: {val:.2f}")
        low_capture = trail_trades[trail_trades['capture'] < 0.3]
        print(f"    Capture < 0.30: N={len(low_capture)} ({100*len(low_capture)/len(trail_trades):.1f}%) — room for tighter trail?")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# EXP-I: Consecutive Direction Pattern Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("EXP-I: CONSECUTIVE DIRECTION PATTERN ANALYSIS")
print("  Prior: 5+ consecutive SELL → avg $/t = -$0.07, 7+ SELL → -$0.34")
print("  Method: Systematic direction streak analysis + sizing simulation")
print("  Note: 'consecutive loss sizing' already rejected; this tests DIRECTION not P&L")
print("=" * 80)

for config_label, base_stats in [("Current", base_cur), ("Mega", base_mega)]:
    trades = base_stats['_trades']
    if not trades:
        continue

    print(f"\n--- I1: {config_label} direction streak analysis ---")

    # Build direction sequence
    directions = [t.direction for t in trades]
    pnls_seq = [t.pnl for t in trades]

    # Count streak lengths and next-trade PnL
    streak_data = defaultdict(list)
    current_dir = directions[0]
    streak_len = 1

    for i in range(1, len(directions)):
        if directions[i] == current_dir:
            streak_len += 1
        else:
            streak_data[(current_dir, streak_len)].append(pnls_seq[i])
            current_dir = directions[i]
            streak_len = 1

    print(f"\n  After N consecutive same-direction trades, next trade's performance:")
    hdr = f"    {'Direction':<6} {'Streak':>6} {'NextN':>6} {'Next$/t':>8} {'NextWR%':>8}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    for direction in ['BUY', 'SELL']:
        for streak in range(1, 10):
            next_pnls = streak_data.get((direction, streak), [])
            for s2 in range(streak + 1, 15):
                next_pnls += streak_data.get((direction, s2), [])
            if len(next_pnls) < 5:
                continue
            avg_pnl = np.mean(next_pnls)
            wr = 100.0 * sum(1 for p in next_pnls if p > 0) / len(next_pnls)
            print(f"    {direction:<6} {'>=' + str(streak):>6} {len(next_pnls):>6} ${avg_pnl:>7.2f} {wr:>7.1f}%")

    # I2: BUY→SELL and SELL→BUY transition analysis
    print(f"\n--- I2: {config_label} direction switch analysis ---")
    same_dir_pnls = []
    switch_dir_pnls = []
    for i in range(1, len(trades)):
        if trades[i].direction == trades[i-1].direction:
            same_dir_pnls.append(trades[i].pnl)
        else:
            switch_dir_pnls.append(trades[i].pnl)

    if same_dir_pnls:
        print(f"    Same direction as prev: N={len(same_dir_pnls)}, "
              f"$/t=${np.mean(same_dir_pnls):.2f}, WR={100*sum(1 for p in same_dir_pnls if p>0)/len(same_dir_pnls):.1f}%")
    if switch_dir_pnls:
        print(f"    Direction switch:       N={len(switch_dir_pnls)}, "
              f"$/t=${np.mean(switch_dir_pnls):.2f}, WR={100*sum(1 for p in switch_dir_pnls if p>0)/len(switch_dir_pnls):.1f}%")

    # I3: Consecutive loss then direction analysis
    print(f"\n--- I3: {config_label} loss streak → next trade by direction ---")
    loss_streak = 0
    streak_next = defaultdict(list)
    for i in range(len(trades)):
        if i > 0:
            key = f"after_{min(loss_streak, 4)}+_losses"
            streak_next[key].append({'pnl': trades[i].pnl, 'dir': trades[i].direction})
        if trades[i].pnl <= 0:
            loss_streak += 1
        else:
            loss_streak = 0

    for key in sorted(streak_next.keys()):
        items = streak_next[key]
        if len(items) < 10:
            continue
        avg = np.mean([x['pnl'] for x in items])
        wr = 100.0 * sum(1 for x in items if x['pnl'] > 0) / len(items)
        buy_pct = 100.0 * sum(1 for x in items if x['dir'] == 'BUY') / len(items)
        print(f"    {key}: N={len(items)}, $/t=${avg:.2f}, WR={wr:.1f}%, BUY%={buy_pct:.0f}%")

    # I4: Sizing simulation — reduce lot after 5+ same direction
    print(f"\n--- I4: {config_label} direction-streak sizing simulation ---")

    schemes = [
        ("Baseline 1.0x", {}),
        ("5+ same dir → 0.7x", {5: 0.7}),
        ("5+ same dir → 0.5x", {5: 0.5}),
        ("3+ same dir → 0.8x", {3: 0.8}),
        ("Switch → 1.3x", {'switch': 1.3}),
    ]

    hdr = f"    {'Scheme':<30} {'Sharpe':>7} {'PnL':>10} {'AvgMult':>8}"
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))

    for scheme_label, rule in schemes:
        sized_pnls = []
        mults_used = []
        current_dir = None
        current_streak = 0

        for t in trades:
            if t.direction == current_dir:
                current_streak += 1
            else:
                current_streak = 1
                current_dir = t.direction

            mult = 1.0
            if 'switch' in rule and current_streak == 1 and len(sized_pnls) > 0:
                mult = rule['switch']
            else:
                for threshold, m in rule.items():
                    if isinstance(threshold, int) and current_streak >= threshold:
                        mult = m

            sized_pnls.append(t.pnl * mult)
            mults_used.append(mult)

        daily = defaultdict(float)
        for t_obj, p in zip(trades, sized_pnls):
            daily[t_obj.entry_time.strftime('%Y-%m-%d')] += p
        vals = list(daily.values())
        sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0
        total_pnl = sum(sized_pnls)
        avg_mult = np.mean(mults_used)
        print(f"    {scheme_label:<30} {sharpe:>7.2f} ${total_pnl:>9,.0f} {avg_mult:>7.2f}x")

gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total_elapsed = time.time() - t_total
print("\n\n" + "=" * 80)
print("EXPERIMENT BATCH 2 SUMMARY")
print("=" * 80)
print(f"""
  EXP-E: Mega vs Current with Spread
    → If Mega >=5/6 folds at $0.30 → switch live config
    → Key question: does Mega's tighter trail survive real spreads?

  EXP-F: Max Hold Fine-tuning
    → Find optimal max_hold for Current config with $0.30 spread
    → If different from 60(default), run K-Fold to confirm

  EXP-G: KC Position at Entry
    → If strong monotonic $/t relationship with KC position → confidence score
    → If no pattern → entry quality is not driven by channel position

  EXP-H: MFE/MAE Analysis
    → Key metrics: capture ratio, losers-with-high-MFE count
    → If many losers had MFE >= 1ATR → trail activation threshold too high
    → Direct input to trailing_evolution experiment

  EXP-I: Direction Streak Analysis
    → If 5+ same-dir streaks degrade → direction-based lot reduction
    → Note: different from rejected "consecutive loss" sizing (tests direction, not P&L)
""")
print(f"Total runtime: {total_elapsed / 60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
