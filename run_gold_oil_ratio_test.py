#!/usr/bin/env python3
"""
Gold/Oil Ratio as Macro Regime Signal
=======================================
Post-hoc: Analyze strategy performance segmented by Gold/Oil ratio level.
Tests whether extreme ratio levels predict different trade outcomes.
Uses CL (crude oil) data from yfinance as proxy.
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.stats import calc_stats

OUTPUT_FILE = "gold_oil_ratio_output.txt"


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
print("GOLD/OIL RATIO AS MACRO REGIME SIGNAL")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()

# Download oil data
print("\n--- Downloading CL=F (crude oil) data from yfinance ---")
try:
    oil_df = yf.download("CL=F", start="2014-01-01", end="2026-04-09", progress=False)
    if isinstance(oil_df.columns, pd.MultiIndex):
        oil_df.columns = oil_df.columns.get_level_values(0)
    oil_daily = oil_df[['Close']].rename(columns={'Close': 'oil_close'})
    oil_daily.index = pd.to_datetime(oil_daily.index)
    if oil_daily.index.tz is not None:
        oil_daily.index = oil_daily.index.tz_localize(None)
    print(f"  Oil data: {len(oil_daily)} daily bars, {oil_daily.index[0]} to {oil_daily.index[-1]}")
except Exception as e:
    print(f"  ERROR downloading oil data: {e}")
    print("  Falling back to synthetic proxy (ATR-based volatility)")
    oil_daily = None

# Load gold data
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

# Build daily gold close
h1_df = data.h1_df.copy()
gold_daily = h1_df.resample('1D').agg({'Close': 'last'}).dropna()
gold_daily.rename(columns={'Close': 'gold_close'}, inplace=True)
if gold_daily.index.tz is not None:
    gold_daily.index = gold_daily.index.tz_localize(None)

if oil_daily is not None:
    merged = gold_daily.join(oil_daily, how='inner')
    merged['go_ratio'] = merged['gold_close'] / merged['oil_close']
    merged['go_ratio_pct'] = merged['go_ratio'].rank(pct=True)
    merged['go_ratio_20d'] = merged['go_ratio'].rolling(20).mean()
    merged['go_ratio_chg'] = merged['go_ratio'].pct_change(5)
    print(f"  Merged data: {len(merged)} days")
    print(f"  Gold/Oil ratio range: {merged['go_ratio'].min():.1f} - {merged['go_ratio'].max():.1f}")
    print(f"  Current ratio: {merged['go_ratio'].iloc[-1]:.1f}")
else:
    merged = None


def to_date(dt):
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def get_ratio_at_entry(entry_time):
    if merged is None:
        return None
    d = to_date(entry_time)
    prev = merged.loc[:d]
    if len(prev) < 2:
        return None
    return prev.iloc[-2]


# ── Analysis 1: G/O ratio level bins ──
print("\n\n" + "=" * 80)
print("PART 1: TRADE PERFORMANCE BY GOLD/OIL RATIO LEVEL")
print("=" * 80)

if merged is not None:
    ratio_bins = {
        'Low (<15)': [], 'Normal (15-25)': [], 'High (25-40)': [],
        'Very High (40-60)': [], 'Extreme (>60)': []
    }
    for t in trades:
        row = get_ratio_at_entry(t.entry_time)
        if row is None:
            continue
        r = row['go_ratio']
        if pd.isna(r):
            continue
        if r < 15:
            ratio_bins['Low (<15)'].append(t)
        elif r < 25:
            ratio_bins['Normal (15-25)'].append(t)
        elif r < 40:
            ratio_bins['High (25-40)'].append(t)
        elif r < 60:
            ratio_bins['Very High (40-60)'].append(t)
        else:
            ratio_bins['Extreme (>60)'].append(t)

    print(f"\n{'G/O Ratio':<22} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print("-" * 55)
    for cat in ['Low (<15)', 'Normal (15-25)', 'High (25-40)', 'Very High (40-60)', 'Extreme (>60)']:
        bt = ratio_bins[cat]
        if not bt:
            continue
        pnl = sum(t.pnl for t in bt)
        wins = sum(1 for t in bt if t.pnl > 0)
        print(f"{cat:<22} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")


    # By percentile
    print(f"\n\nPART 2: BY RATIO PERCENTILE (within history)")
    pct_bins = {'Q1 (<25%)': [], 'Q2 (25-50%)': [], 'Q3 (50-75%)': [], 'Q4 (>75%)': []}
    for t in trades:
        row = get_ratio_at_entry(t.entry_time)
        if row is None:
            continue
        p = row.get('go_ratio_pct', 0.5)
        if pd.isna(p):
            continue
        if p < 0.25:
            pct_bins['Q1 (<25%)'].append(t)
        elif p < 0.50:
            pct_bins['Q2 (25-50%)'].append(t)
        elif p < 0.75:
            pct_bins['Q3 (50-75%)'].append(t)
        else:
            pct_bins['Q4 (>75%)'].append(t)

    print(f"\n{'Percentile':<18} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print("-" * 50)
    for cat in ['Q1 (<25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (>75%)']:
        bt = pct_bins[cat]
        if not bt:
            continue
        pnl = sum(t.pnl for t in bt)
        wins = sum(1 for t in bt if t.pnl > 0)
        print(f"{cat:<18} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")


    # By ratio change (momentum)
    print(f"\n\nPART 3: BY RATIO 5-DAY CHANGE (ratio momentum)")
    chg_bins = {'Falling (<-5%)': [], 'Stable (-5 to +5%)': [], 'Rising (>+5%)': []}
    for t in trades:
        row = get_ratio_at_entry(t.entry_time)
        if row is None:
            continue
        chg = row.get('go_ratio_chg', 0)
        if pd.isna(chg):
            continue
        if chg < -0.05:
            chg_bins['Falling (<-5%)'].append(t)
        elif chg > 0.05:
            chg_bins['Rising (>+5%)'].append(t)
        else:
            chg_bins['Stable (-5 to +5%)'].append(t)

    print(f"\n{'Ratio Change':<22} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
    print("-" * 55)
    for cat in ['Falling (<-5%)', 'Stable (-5 to +5%)', 'Rising (>+5%)']:
        bt = chg_bins[cat]
        if not bt:
            continue
        pnl = sum(t.pnl for t in bt)
        wins = sum(1 for t in bt if t.pnl > 0)
        print(f"{cat:<22} {len(bt):>6} ${pnl:>9,.0f} ${pnl/len(bt):>6.2f} {100*wins/len(bt):>5.1f}%")


    # ── Sizing simulation ──
    print(f"\n\nPART 4: SIZING SIMULATION")
    print("-" * 60)

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
        ("Flat 1.0x (baseline)", lambda row: 1.0),
        ("High ratio 1.3x", lambda row: 1.3 if row is not None and not pd.isna(row.get('go_ratio', 0)) and row['go_ratio'] > 40 else 1.0),
        ("High ratio 0.7x", lambda row: 0.7 if row is not None and not pd.isna(row.get('go_ratio', 0)) and row['go_ratio'] > 40 else 1.0),
        ("Extreme ratio 0.5x", lambda row: 0.5 if row is not None and not pd.isna(row.get('go_ratio', 0)) and row['go_ratio'] > 60 else 1.0),
        ("Rising ratio 1.3x", lambda row: 1.3 if row is not None and not pd.isna(row.get('go_ratio_chg', 0)) and row['go_ratio_chg'] > 0.05 else 1.0),
    ]

    print(f"{'Scheme':<35} {'PnL':>10} {'Sharpe':>8} {'Delta':>7}")
    print("-" * 62)
    for sname, sfunc in schemes:
        pnls = [t.pnl * sfunc(get_ratio_at_entry(t.entry_time)) for t in trades]
        sh = compute_sharpe(trades, pnls)
        print(f"{sname:<35} ${sum(pnls):>9,.0f} {sh:>8.2f} {sh - base_sh:>+7.2f}")

else:
    print("  [SKIPPED] No oil data available for analysis.")

total_elapsed = time.time() - t_total
print(f"\n\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
