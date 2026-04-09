#!/usr/bin/env python3
"""
Trump Tariff / News Density as Volatility Factor (Post-hoc)
=============================================================
Analyzes whether high Trump/tariff news density predicts higher volatility
and whether trades entered during high-news periods perform differently.
Uses H1 data + simulated news density proxy (price range / ATR spikes).
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "trump_factor_output.txt"


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
print("TRUMP TARIFF / NEWS VOLATILITY FACTOR ANALYSIS")
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

h1_df = data.h1_df.copy()

# Build daily volatility metrics
d1_df = h1_df.resample('1D').agg({
    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
}).dropna()
d1_df['d_range'] = d1_df['High'] - d1_df['Low']
d1_df['d_atr14'] = d1_df['d_range'].rolling(14).mean()
d1_df['vol_ratio'] = d1_df['d_range'] / d1_df['d_atr14']
d1_df['ret'] = d1_df['Close'].pct_change() * 100
d1_df['abs_ret'] = d1_df['ret'].abs()
d1_df['gap'] = (d1_df['Open'] - d1_df['Close'].shift(1)).abs()
d1_df['gap_atr'] = d1_df['gap'] / d1_df['d_atr14']

# Trump era: 2025-01-20 onwards (inauguration)
# Pre-Trump: 2015-2025-01-19
# Tariff escalation phases (approximate):
#   Phase 1: 2025-02 to 2025-06 (initial tariffs)
#   Phase 2: 2025-10 to 2026-03 (escalation)
TRUMP_START = pd.Timestamp("2025-01-20", tz='UTC')
TARIFF_P1_START = pd.Timestamp("2025-02-01", tz='UTC')
TARIFF_P1_END = pd.Timestamp("2025-06-30", tz='UTC')
TARIFF_P2_START = pd.Timestamp("2025-10-01", tz='UTC')


def get_era(entry_time):
    ts = pd.Timestamp(entry_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    if ts < TRUMP_START:
        return "Pre-Trump"
    elif TARIFF_P1_START <= ts <= TARIFF_P1_END:
        return "Tariff_P1"
    elif ts >= TARIFF_P2_START:
        return "Tariff_P2"
    else:
        return "Trump_Other"


def to_utc(dt):
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts


def get_prev_day(entry_time):
    ts = to_utc(entry_time)
    if d1_df.index.tz is None:
        ts = ts.tz_localize(None)
    prev = d1_df.loc[:ts]
    if len(prev) < 2:
        return None
    return prev.iloc[-2]


# ── Analysis 1: Era-based performance ──
print("\n\n" + "=" * 80)
print("PART 1: ERA-BASED PERFORMANCE")
print("=" * 80)

era_stats = defaultdict(lambda: {'n': 0, 'pnl': 0.0, 'wins': 0, 'trades': []})
for t in trades:
    era = get_era(t.entry_time)
    era_stats[era]['n'] += 1
    era_stats[era]['pnl'] += t.pnl
    if t.pnl > 0:
        era_stats[era]['wins'] += 1
    era_stats[era]['trades'].append(t)

print(f"\n{'Era':<15} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6} {'Sharpe':>8}")
print("-" * 56)
for era in ["Pre-Trump", "Trump_Other", "Tariff_P1", "Tariff_P2"]:
    d = era_stats.get(era, {'n': 0, 'pnl': 0, 'wins': 0, 'trades': []})
    if d['n'] == 0:
        continue
    wr = 100.0 * d['wins'] / d['n']
    ppt = d['pnl'] / d['n']
    ts = d['trades']
    eq = [0.0]
    for t in ts:
        eq.append(eq[-1] + t.pnl)
    s = calc_stats(ts, eq)
    print(f"{era:<15} {d['n']:>6} ${d['pnl']:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}% {s['sharpe']:>8.2f}")


# ── Analysis 2: Volatility regime performance ──
print("\n\n" + "=" * 80)
print("PART 2: DAILY VOLATILITY REGIME AT ENTRY")
print("=" * 80)

vol_cats = {'low (<0.7x)': [], 'normal': [], 'high (>1.3x)': [], 'extreme (>2x)': []}
for t in trades:
    row = get_prev_day(t.entry_time)
    if row is None:
        vol_cats['normal'].append(t)
        continue
    vr = row.get('vol_ratio', 1.0)
    if pd.isna(vr):
        vr = 1.0
    if vr > 2.0:
        vol_cats['extreme (>2x)'].append(t)
    elif vr > 1.3:
        vol_cats['high (>1.3x)'].append(t)
    elif vr < 0.7:
        vol_cats['low (<0.7x)'].append(t)
    else:
        vol_cats['normal'].append(t)

print(f"\n{'Vol Regime':<18} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 50)
for cat in ['low (<0.7x)', 'normal', 'high (>1.3x)', 'extreme (>2x)']:
    bt = vol_cats[cat]
    if not bt:
        continue
    pnl = sum(t.pnl for t in bt)
    wins = sum(1 for t in bt if t.pnl > 0)
    print(f"{cat:<18} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")


# ── Analysis 3: Gap at open as news proxy ──
print("\n\n" + "=" * 80)
print("PART 3: OVERNIGHT GAP AS NEWS SHOCK PROXY")
print("=" * 80)

gap_cats = {'no gap (<0.3 ATR)': [], 'small (0.3-0.7)': [], 'big (>0.7 ATR)': []}
for t in trades:
    row = get_prev_day(t.entry_time)
    if row is None:
        continue
    ts = to_utc(t.entry_time)
    if d1_df.index.tz is None:
        ts = ts.tz_localize(None)
    today_rows = d1_df.loc[:ts]
    if len(today_rows) < 1:
        continue
    today = today_rows.iloc[-1]
    ga = today.get('gap_atr', 0)
    if pd.isna(ga):
        continue
    if ga > 0.7:
        gap_cats['big (>0.7 ATR)'].append(t)
    elif ga > 0.3:
        gap_cats['small (0.3-0.7)'].append(t)
    else:
        gap_cats['no gap (<0.3 ATR)'].append(t)

print(f"\n{'Gap Size':<22} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 55)
for cat in ['no gap (<0.3 ATR)', 'small (0.3-0.7)', 'big (>0.7 ATR)']:
    bt = gap_cats[cat]
    if not bt:
        continue
    pnl = sum(t.pnl for t in bt)
    wins = sum(1 for t in bt if t.pnl > 0)
    print(f"{cat:<22} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")


# ── Analysis 4: Consecutive high-vol days ──
print("\n\n" + "=" * 80)
print("PART 4: CONSECUTIVE HIGH-VOL DAYS (TARIFF ESCALATION PROXY)")
print("=" * 80)

d1_df['high_vol'] = d1_df['vol_ratio'] > 1.3
d1_df['consec_hv'] = 0
count = 0
for idx in d1_df.index:
    if d1_df.loc[idx, 'high_vol']:
        count += 1
    else:
        count = 0
    d1_df.loc[idx, 'consec_hv'] = count

consec_cats = {'0 days': [], '1 day': [], '2 days': [], '3+ days': []}
for t in trades:
    row = get_prev_day(t.entry_time)
    if row is None:
        consec_cats['0 days'].append(t)
        continue
    chv = row.get('consec_hv', 0)
    if pd.isna(chv):
        chv = 0
    chv = int(chv)
    if chv >= 3:
        consec_cats['3+ days'].append(t)
    elif chv == 2:
        consec_cats['2 days'].append(t)
    elif chv == 1:
        consec_cats['1 day'].append(t)
    else:
        consec_cats['0 days'].append(t)

print(f"\n{'Consec HighVol':<18} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 50)
for cat in ['0 days', '1 day', '2 days', '3+ days']:
    bt = consec_cats[cat]
    if not bt:
        continue
    pnl = sum(t.pnl for t in bt)
    wins = sum(1 for t in bt if t.pnl > 0)
    print(f"{cat:<18} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")


# ── Analysis 5: Sizing simulation ──
print("\n\n" + "=" * 80)
print("PART 5: VOLATILITY-BASED SIZING SIMULATION")
print("=" * 80)

def compute_sharpe(trades_list, pnls):
    daily = defaultdict(float)
    for t, p in zip(trades_list, pnls):
        day = t.entry_time.strftime('%Y-%m-%d')
        daily[day] += p
    vals = list(daily.values())
    if len(vals) > 1 and np.std(vals) > 0:
        return np.mean(vals) / np.std(vals) * np.sqrt(252)
    return 0

base_pnls = [t.pnl for t in trades]
base_sh = compute_sharpe(trades, base_pnls)

schemes = [
    ("Flat 1.0x (baseline)", lambda vr, chv: 1.0),
    ("HighVol 1.3x", lambda vr, chv: 1.3 if vr > 1.3 else 1.0),
    ("HighVol 0.7x (reduce)", lambda vr, chv: 0.7 if vr > 1.3 else 1.0),
    ("Extreme 0.5x", lambda vr, chv: 0.5 if vr > 2.0 else 1.0),
    ("Consec3+ 1.5x", lambda vr, chv: 1.5 if chv >= 3 else 1.0),
    ("Consec3+ 0.6x", lambda vr, chv: 0.6 if chv >= 3 else 1.0),
    ("LowVol 0.7x + HighVol 1.2x", lambda vr, chv: 0.7 if vr < 0.7 else (1.2 if vr > 1.3 else 1.0)),
]

print(f"\n{'Scheme':<35} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
print("-" * 62)

for sname, sfunc in schemes:
    pnls = []
    for t in trades:
        row = get_prev_day(t.entry_time)
        vr = float(row['vol_ratio']) if row is not None and not pd.isna(row.get('vol_ratio')) else 1.0
        chv = int(row['consec_hv']) if row is not None and not pd.isna(row.get('consec_hv')) else 0
        pnls.append(t.pnl * sfunc(vr, chv))
    sh = compute_sharpe(trades, pnls)
    print(f"{sname:<35} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh - base_sh:>+7.2f}")


print(f"\n\nTotal runtime: {(time.time()-t_total)/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
