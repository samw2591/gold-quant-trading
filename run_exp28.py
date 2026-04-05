#!/usr/bin/env python
"""EXP28: 事件日防御测试 — "带伞策略"穷举验证 (standalone script)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import pandas as pd
import numpy as np
from backtest import DataBundle, run_variant, C12_KWARGS
from backtest.engine import BacktestEngine, TradeRecord
from backtest.stats import calc_stats

# ================================================================
# Setup
# ================================================================
data = DataBundle.load_default()

LIVE_KWARGS = {
    **C12_KWARGS,
    "intraday_adaptive": True,
    "choppy_threshold": 0.35,
    "kc_only_threshold": 0.60,
    "spread_cost": 0.50,
}

print('Baseline run...')
baseline = run_variant(data, 'Baseline', **LIVE_KWARGS)
print(f"Baseline: N={baseline['n']}, Sharpe={baseline['sharpe']:.2f}, "
      f"PnL=${baseline['total_pnl']:.0f}, MaxDD=${baseline['max_dd']:.0f}")

# ================================================================
# Part 1: 事件日标注
# ================================================================
print("\n" + "="*80)
print("  PART 1: Event Day Classification")
print("="*80)

h1 = data.h1_df.copy()
h1['date'] = h1.index.date

daily = h1.groupby('date').agg(
    open=('Open', 'first'),
    high=('High', 'max'),
    low=('Low', 'min'),
    close=('Close', 'last'),
    atr_mean=('ATR', 'mean'),
    atr_max=('ATR', 'max'),
    adx_mean=('ADX', 'mean'),
    n_bars=('Close', 'count'),
)

daily['range'] = daily['high'] - daily['low']
daily['range_pct'] = daily['range'] / daily['close'] * 100
daily['return_pct'] = (daily['close'] / daily['open'] - 1) * 100
daily['abs_return'] = daily['return_pct'].abs()
daily['atr_20'] = daily['atr_mean'].rolling(20).mean()
daily['range_vs_atr'] = daily['range'] / daily['atr_20']

has_vix = False
try:
    macro = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'macro_history.csv'),
                        parse_dates=['date'], index_col='date')
    macro = macro[['vix']].dropna()
    macro['vix_chg'] = macro['vix'].pct_change() * 100
    macro['vix_20'] = macro['vix'].rolling(20).mean()
    daily = daily.join(macro[['vix', 'vix_chg']], how='left')
    has_vix = True
    print(f"VIX data joined: {daily['vix'].notna().sum()} days")
except Exception as e:
    print(f"No VIX data: {e}")

print(f"Total trading days: {len(daily)}")
print(f"Range stats: mean={daily['range_pct'].mean():.2f}%, "
      f"P90={daily['range_pct'].quantile(0.90):.2f}%, "
      f"P95={daily['range_pct'].quantile(0.95):.2f}%, "
      f"P99={daily['range_pct'].quantile(0.99):.2f}%")

# Event day classification
daily['is_extreme_vol'] = daily['range_vs_atr'] > 2.0
daily['is_big_move'] = daily['abs_return'] > 1.5
if has_vix:
    daily['is_vix_spike'] = daily['vix_chg'] > 15
else:
    daily['is_vix_spike'] = False

daily['is_event_day'] = daily['is_extreme_vol'] | daily['is_big_move'] | daily['is_vix_spike']

daily['prev_extreme'] = daily['is_extreme_vol'].shift(1).fillna(False)
daily['prev_big_move'] = daily['is_big_move'].shift(1).fillna(False)
daily['prev_vix_spike'] = daily['is_vix_spike'].shift(1).fillna(False)
daily['shield_day'] = daily['prev_extreme'] | daily['prev_big_move'] | daily['prev_vix_spike']

if has_vix:
    daily['high_vix'] = daily['vix'] > 25
    daily['shield_day_v2'] = daily['shield_day'] | daily['high_vix']
else:
    daily['high_vix'] = False
    daily['shield_day_v2'] = daily['shield_day']

n_event = daily['is_event_day'].sum()
n_shield = daily['shield_day'].sum()
n_shield_v2 = daily['shield_day_v2'].sum()
total = len(daily)

print(f"\n=== Event Day Statistics ===")
print(f"Total days: {total}")
print(f"Extreme vol days (range>2xATR): {daily['is_extreme_vol'].sum()} ({daily['is_extreme_vol'].mean()*100:.1f}%)")
print(f"Big move days (|ret|>1.5%):     {daily['is_big_move'].sum()} ({daily['is_big_move'].mean()*100:.1f}%)")
print(f"VIX spike days (chg>15%):       {daily['is_vix_spike'].sum()} ({daily['is_vix_spike'].mean()*100:.1f}%)")
print(f"Any event day (oracle):         {n_event} ({n_event/total*100:.1f}%)")
print(f"Shield day (prev-day signal):   {n_shield} ({n_shield/total*100:.1f}%)")
print(f"Shield v2 (+VIX>25):            {n_shield_v2} ({n_shield_v2/total*100:.1f}%)")

# Baseline trade split
trades = baseline['_trades']
event_dates = set(daily[daily['is_event_day']].index)
shield_dates = set(daily[daily['shield_day']].index)
shield_v2_dates = set(daily[daily['shield_day_v2']].index)

def split_trades_by_dates(trades, date_set):
    in_set = [t for t in trades if pd.Timestamp(t.entry_time).date() in date_set]
    out_set = [t for t in trades if pd.Timestamp(t.entry_time).date() not in date_set]
    return in_set, out_set

event_trades, normal_trades = split_trades_by_dates(trades, event_dates)
shield_trades_list, noshield_trades = split_trades_by_dates(trades, shield_dates)

def trade_stats(tlist, label):
    if not tlist:
        print(f"  {label:25s}: 0 trades")
        return
    pnl = sum(t.pnl for t in tlist)
    wr = len([t for t in tlist if t.pnl > 0]) / len(tlist) * 100
    avg = np.mean([t.pnl for t in tlist])
    print(f"  {label:25s}: N={len(tlist):>5}  PnL=${pnl:>8.0f}  $/trade=${avg:>5.2f}  WR={wr:.1f}%")

print("\n=== Trade Performance: Event Days vs Normal Days ===")
trade_stats(trades, 'ALL')
trade_stats(event_trades, 'Event days (oracle)')
trade_stats(normal_trades, 'Normal days')
print()
trade_stats(shield_trades_list, 'Shield days (prev-day)')
trade_stats(noshield_trades, 'No-shield days')

# ================================================================
# Part 2: 每把"伞"的全量代价穷举
# ================================================================
print("\n" + "="*80)
print("  PART 2: Full-Period Shield Cost")
print("="*80)

SHIELDS = [
    ('MaxPos=1',        {'max_positions': 1}),
    ('Choppy=0.40',     {'choppy_threshold': 0.40}),
    ('Choppy=0.45',     {'choppy_threshold': 0.45}),
    ('Choppy=0.50',     {'choppy_threshold': 0.50}),
    ('Cooldown=60min',  {'cooldown_hours': 1.0}),
    ('Cooldown=120min', {'cooldown_hours': 2.0}),
    ('SL=5.5',          {'sl_atr_mult': 5.5}),
    ('SL=6.0',          {'sl_atr_mult': 6.0}),
    ('Trail_tight',     {'trailing_activate_atr': 0.6, 'trailing_distance_atr': 0.15}),
    ('Trail_loose',     {'trailing_activate_atr': 1.2, 'trailing_distance_atr': 0.40}),
    ('No_V3',           {}),
]

print(f"\n{'Shield':20s} {'N':>5} {'Sharpe':>7} {'dSh':>6} {'PnL':>9} {'dPnL':>8} {'MaxDD':>7} {'dDD':>7} {'WR%':>6}")
print('-' * 85)

base_sh = baseline['sharpe']
base_pnl = baseline['total_pnl']
base_dd = baseline['max_dd']
print(f"{'Baseline':20s} {baseline['n']:>5} {base_sh:>7.2f} {'':>6} ${base_pnl:>8.0f} {'':>8} ${base_dd:>6.0f} {'':>7} {baseline['win_rate']:>5.1f}%")

shield_fullperiod = []

for sname, overrides in SHIELDS:
    if sname == 'No_V3':
        kw = {k: v for k, v in LIVE_KWARGS.items() if k != 'regime_config'}
    else:
        kw = {**LIVE_KWARGS, **overrides}

    r = run_variant(data, sname, verbose=False, **kw)
    ds = r['sharpe'] - base_sh
    dp = r['total_pnl'] - base_pnl
    dd = r['max_dd'] - base_dd

    cost_rating = 'FREE' if abs(ds) < 0.05 else ('LOW' if abs(ds) < 0.15 else ('MED' if abs(ds) < 0.30 else 'HIGH'))

    print(f"{sname:20s} {r['n']:>5} {r['sharpe']:>7.2f} {ds:>+6.2f} ${r['total_pnl']:>8.0f} {dp:>+8.0f} ${r['max_dd']:>6.0f} {dd:>+7.0f} {r['win_rate']:>5.1f}%  [{cost_rating}]")

    shield_fullperiod.append({
        'shield': sname, 'n': r['n'], 'sharpe': r['sharpe'],
        'pnl': r['total_pnl'], 'max_dd': r['max_dd'], 'wr': r['win_rate'],
        'd_sharpe': ds, 'd_pnl': dp, 'd_dd': dd, 'cost': cost_rating,
    })

print("\n=== Shields ranked by cost (Sharpe impact) ===")
for s in sorted(shield_fullperiod, key=lambda x: abs(x['d_sharpe'])):
    print(f"  {s['shield']:20s}: Sharpe {s['d_sharpe']:>+.2f}, PnL {s['d_pnl']:>+.0f}, MaxDD {s['d_dd']:>+.0f}  [{s['cost']}]")

# ================================================================
# Part 3: 事件日带伞 vs 全程默认
# ================================================================
print("\n" + "="*80)
print("  PART 3: Hybrid Shield (event-day only)")
print("="*80)

def hybrid_pnl(default_trades, shield_trades, shield_dates):
    hybrid = []
    for t in default_trades:
        entry_date = pd.Timestamp(t.entry_time).date()
        if entry_date not in shield_dates:
            hybrid.append(t)
    for t in shield_trades:
        entry_date = pd.Timestamp(t.entry_time).date()
        if entry_date in shield_dates:
            hybrid.append(t)
    hybrid.sort(key=lambda t: t.entry_time)
    return hybrid

def calc_hybrid_stats(trades_list):
    if not trades_list:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'wr': 0, 'max_dd': 0}
    pnl = sum(t.pnl for t in trades_list)
    wins = [t for t in trades_list if t.pnl > 0]
    wr = len(wins) / len(trades_list) * 100

    daily_pnl = {}
    for t in trades_list:
        d = pd.Timestamp(t.exit_time).date()
        daily_pnl[d] = daily_pnl.get(d, 0) + t.pnl
    dpnl = list(daily_pnl.values())
    sharpe = 0
    if len(dpnl) > 1 and np.std(dpnl) > 0:
        sharpe = np.mean(dpnl) / np.std(dpnl) * np.sqrt(252)

    equity = [2000]
    for t in trades_list:
        equity.append(equity[-1] + t.pnl)
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = abs(dd.min())

    return {'n': len(trades_list), 'pnl': pnl, 'sharpe': sharpe, 'wr': wr, 'max_dd': max_dd}

CANDIDATE_SHIELDS = [
    ('MaxPos=1',        {'max_positions': 1}),
    ('SL=5.5',          {'sl_atr_mult': 5.5}),
    ('SL=6.0',          {'sl_atr_mult': 6.0}),
    ('Cooldown=60min',  {'cooldown_hours': 1.0}),
    ('Trail_loose',     {'trailing_activate_atr': 1.2, 'trailing_distance_atr': 0.40}),
]

default_trades = baseline['_trades']

trigger_sets = {
    'prev_day': shield_dates,
    'prev_day+VIX25': shield_v2_dates,
    'oracle_event': event_dates,
}

print(f"\n{'Shield':20s} {'Trigger':18s} {'N':>5} {'Sharpe':>7} {'dSh':>6} {'PnL':>9} {'dPnL':>7} {'MaxDD':>7} {'dDD':>7}")
print('-' * 95)

hybrid_results = []

for sname, overrides in CANDIDATE_SHIELDS:
    kw = {**LIVE_KWARGS, **overrides}
    sr = run_variant(data, f"shield_{sname}", verbose=False, **kw)
    shield_tr = sr['_trades']

    for tname, tdates in trigger_sets.items():
        hybrid_tr = hybrid_pnl(default_trades, shield_tr, tdates)
        hs = calc_hybrid_stats(hybrid_tr)
        ds = hs['sharpe'] - base_sh
        dp = hs['pnl'] - base_pnl
        dd = hs['max_dd'] - base_dd

        winner = 'BETTER' if ds > 0.05 else ('SAME' if abs(ds) < 0.05 else 'WORSE')

        print(f"{sname:20s} {tname:18s} {hs['n']:>5} {hs['sharpe']:>7.2f} {ds:>+6.2f} ${hs['pnl']:>8.0f} {dp:>+7.0f} ${hs['max_dd']:>6.0f} {dd:>+7.0f}  [{winner}]")

        hybrid_results.append({
            'shield': sname, 'trigger': tname,
            'n': hs['n'], 'sharpe': hs['sharpe'], 'pnl': hs['pnl'],
            'max_dd': hs['max_dd'], 'd_sharpe': ds, 'd_pnl': dp, 'd_dd': dd,
            'verdict': winner,
        })

print("\n=== Best Hybrid Combos (sorted by Sharpe improvement) ===")
for h in sorted(hybrid_results, key=lambda x: x['d_sharpe'], reverse=True)[:10]:
    print(f"  {h['shield']:20s} x {h['trigger']:18s}: Sharpe {h['d_sharpe']:>+.2f}, PnL {h['d_pnl']:>+.0f}, MaxDD {h['d_dd']:>+.0f}  [{h['verdict']}]")

print("\n=== Best by MaxDD reduction ===")
for h in sorted(hybrid_results, key=lambda x: x['d_dd'])[:10]:
    print(f"  {h['shield']:20s} x {h['trigger']:18s}: MaxDD {h['d_dd']:>+.0f}, Sharpe {h['d_sharpe']:>+.2f}, PnL {h['d_pnl']:>+.0f}  [{h['verdict']}]")

# ================================================================
# Part 4: K-Fold 验证
# ================================================================
print("\n" + "="*80)
print("  PART 4: K-Fold Validation")
print("="*80)

top_combos = sorted(hybrid_results, key=lambda x: x['d_sharpe'], reverse=True)[:3]

FOLDS = [
    ('Fold1', '2015-01-01', '2016-12-31'),
    ('Fold2', '2017-01-01', '2018-12-31'),
    ('Fold3', '2019-01-01', '2020-12-31'),
    ('Fold4', '2021-01-01', '2022-12-31'),
    ('Fold5', '2023-01-01', '2024-12-31'),
    ('Fold6', '2025-01-01', '2026-04-01'),
]

kfold_results = []

for combo in top_combos:
    sname = combo['shield']
    tname = combo['trigger']

    shield_overrides = next(ov for sn, ov in CANDIDATE_SHIELDS if sn == sname)
    shield_kw = {**LIVE_KWARGS, **shield_overrides}
    trigger_dates = trigger_sets[tname]

    print(f"\n--- {sname} x {tname} ---")
    print(f"{'Fold':8s} {'Base Sh':>8} {'Hybrid Sh':>10} {'Delta':>7} {'Base PnL':>9} {'Hybrid PnL':>11} {'Winner':>7}")

    fold_data = []
    base_wins = 0
    hybrid_wins = 0

    for fname, fstart, fend in FOLDS:
        fdata = data.slice(fstart, fend)
        if len(fdata.h1_df) < 100:
            continue

        fb = run_variant(fdata, f"{fname}_base", verbose=False, **LIVE_KWARGS)
        fs = run_variant(fdata, f"{fname}_shield", verbose=False, **shield_kw)

        h_trades = hybrid_pnl(fb['_trades'], fs['_trades'], trigger_dates)
        hs = calc_hybrid_stats(h_trades)

        delta = hs['sharpe'] - fb['sharpe']
        winner = 'HYBRID' if delta > 0.05 else ('BASE' if delta < -0.05 else 'TIE')
        if winner == 'HYBRID': hybrid_wins += 1
        elif winner == 'BASE': base_wins += 1

        print(f"{fname:8s} {fb['sharpe']:>8.2f} {hs['sharpe']:>10.2f} {delta:>+7.2f} ${fb['total_pnl']:>8.0f} ${hs['pnl']:>10.0f} {winner:>7}")

        fold_data.append({'fold': fname, 'base_sharpe': fb['sharpe'], 'hybrid_sharpe': hs['sharpe'], 'delta': delta})

    avg_delta = np.mean([f['delta'] for f in fold_data]) if fold_data else 0
    print(f"  Avg delta: {avg_delta:+.2f}, Hybrid wins {hybrid_wins}/{len(fold_data)}, Base wins {base_wins}/{len(fold_data)}")

    kfold_results.append({
        'shield': sname, 'trigger': tname,
        'avg_delta': avg_delta, 'hybrid_wins': hybrid_wins,
        'total_folds': len(fold_data), 'folds': fold_data,
    })

# ================================================================
# Final Summary
# ================================================================
print("\n" + "="*80)
print("             EXP28 FINAL SUMMARY: UMBRELLA STRATEGY")
print("="*80)

print("\n1. Event Day Statistics:")
print(f"   Extreme vol days: {daily['is_extreme_vol'].sum()} ({daily['is_extreme_vol'].mean()*100:.1f}%)")
print(f"   Big move days:    {daily['is_big_move'].sum()} ({daily['is_big_move'].mean()*100:.1f}%)")
print(f"   Shield days (prev-day trigger): {n_shield} ({n_shield/total*100:.1f}%)")

print("\n2. Full-Period Shield Cost (cheapest shields):")
for s in sorted(shield_fullperiod, key=lambda x: abs(x['d_sharpe']))[:5]:
    print(f"   {s['shield']:20s}: Sharpe cost={s['d_sharpe']:>+.2f}, MaxDD change={s['d_dd']:>+.0f}  [{s['cost']}]")

print("\n3. Best Hybrid Combos (event-day only shield):")
for h in sorted(hybrid_results, key=lambda x: x['d_sharpe'], reverse=True)[:5]:
    print(f"   {h['shield']:20s} x {h['trigger']:18s}: Sharpe {h['d_sharpe']:>+.2f}, MaxDD {h['d_dd']:>+.0f}")

print("\n4. K-Fold Validation:")
for k in kfold_results:
    print(f"   {k['shield']:20s} x {k['trigger']:18s}: Avg delta {k['avg_delta']:+.2f}, wins {k['hybrid_wins']}/{k['total_folds']} folds")

print("\n5. Recommendation:")
best_kfold = max(kfold_results, key=lambda x: x['avg_delta']) if kfold_results else None
if best_kfold and best_kfold['avg_delta'] > 0.05 and best_kfold['hybrid_wins'] >= 4:
    print(f"   RECOMMEND: {best_kfold['shield']} x {best_kfold['trigger']}")
    print(f"   K-Fold: {best_kfold['hybrid_wins']}/{best_kfold['total_folds']} folds, avg Sharpe +{best_kfold['avg_delta']:.2f}")
    print(f"   Safe to implement as automatic pre-market shield.")
elif best_kfold and best_kfold['avg_delta'] > 0:
    print(f"   MARGINAL: {best_kfold['shield']} x {best_kfold['trigger']}")
    print(f"   K-Fold: {best_kfold['hybrid_wins']}/{best_kfold['total_folds']} folds, avg Sharpe +{best_kfold['avg_delta']:.2f}")
    print(f"   Effect exists but too small to justify automation. Keep manual.")
else:
    print(f"   NO SHIELD passes K-Fold validation.")
    print(f"   Current strategy's built-in defenses (V3/Adaptive/Cooldown) are sufficient.")
    print(f"   Manual intervention only for extreme events (VIX>40).")

# ================================================================
# Save results
# ================================================================
def sanitize(obj):
    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, set): return list(obj)
    return obj

save_data = {
    'event_stats': {
        'total_days': int(total),
        'extreme_vol_days': int(daily['is_extreme_vol'].sum()),
        'big_move_days': int(daily['is_big_move'].sum()),
        'shield_days': int(n_shield),
        'shield_v2_days': int(n_shield_v2),
    },
    'shield_costs': shield_fullperiod,
    'hybrid_results': hybrid_results,
    'kfold_results': sanitize(kfold_results),
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'exp28_shield_results.json')
with open(out_path, 'w') as f:
    json.dump(sanitize(save_data), f, indent=2, default=str)
print(f'\nResults saved to {out_path}')
