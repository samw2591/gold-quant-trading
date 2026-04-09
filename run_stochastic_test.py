#!/usr/bin/env python3
"""
Stochastic Extremes Mean Reversion — H1 vs M15 RSI comparison
================================================================
Tests a Stochastic(14,3,3)-based mean reversion strategy on H1 gold data.
Compares with existing M15 RSI approach.
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

OUTPUT_FILE = "stochastic_test_output.txt"


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
print("STOCHASTIC EXTREMES MEAN REVERSION — H1 ANALYSIS")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

h1_df = data.h1_df.copy()

# Compute Stochastic(14,3,3)
print("\n--- Computing Stochastic(14,3,3) on H1 ---")
period = 14
smooth_k = 3
smooth_d = 3

h1_df['lowest_14'] = h1_df['Low'].rolling(period).min()
h1_df['highest_14'] = h1_df['High'].rolling(period).max()
h1_df['fast_k'] = 100 * (h1_df['Close'] - h1_df['lowest_14']) / (h1_df['highest_14'] - h1_df['lowest_14'])
h1_df['slowk'] = h1_df['fast_k'].rolling(smooth_k).mean()
h1_df['slowd'] = h1_df['slowk'].rolling(smooth_d).mean()
h1_df['stoch_cross_up'] = (h1_df['slowk'] > h1_df['slowd']) & (h1_df['slowk'].shift(1) <= h1_df['slowd'].shift(1))
h1_df['stoch_cross_down'] = (h1_df['slowk'] < h1_df['slowd']) & (h1_df['slowk'].shift(1) >= h1_df['slowd'].shift(1))
h1_df.dropna(subset=['slowk', 'slowd'], inplace=True)

print(f"  H1 bars with stochastic: {len(h1_df)}")

# Also need EMA100 and ATR for trend filter and stop
if 'EMA100' not in h1_df.columns:
    h1_df['EMA100'] = h1_df['Close'].ewm(span=100, adjust=False).mean()
if 'ATR' not in h1_df.columns:
    h1_df['tr'] = np.maximum(h1_df['High'] - h1_df['Low'],
                              np.maximum(abs(h1_df['High'] - h1_df['Close'].shift(1)),
                                         abs(h1_df['Low'] - h1_df['Close'].shift(1))))
    h1_df['ATR'] = h1_df['tr'].rolling(14).mean()


# ── Simulate Stochastic strategy ──
print("\n--- Simulating Stochastic Mean Reversion ---")

BUY_THRESHOLD = 20
SELL_THRESHOLD = 80
SL_ATR = 4.5
MAX_HOLD = 8  # H1 bars
RISK_PER_TRADE = 50.0

class Trade:
    def __init__(self, direction, entry_price, sl, entry_time, atr, strategy='stoch'):
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.entry_time = entry_time
        self.atr = atr
        self.strategy = strategy
        self.bars = 0
        self.pnl = 0.0
        self.exit_time = None
        self.exit_reason = ''

thresholds = [
    ("STOCH buy<20 sell>80", 20, 80),
    ("STOCH buy<15 sell>85", 15, 85),
    ("STOCH buy<10 sell>90", 10, 90),
    ("STOCH buy<25 sell>75", 25, 75),
]

hold_variants = [4, 6, 8, 12]

all_results = []

for label, buy_th, sell_th in thresholds:
    for max_h in hold_variants:
        full_label = f"{label} hold={max_h}h"
        trades_list = []
        pos = None

        for i in range(100, len(h1_df)):
            row = h1_df.iloc[i]
            close = float(row['Close'])
            atr = float(row.get('ATR', 0))
            ema100 = float(row.get('EMA100', close))
            slowk = float(row.get('slowk', 50))
            slowd = float(row.get('slowd', 50))
            stoch_up = bool(row.get('stoch_cross_up', False))
            stoch_down = bool(row.get('stoch_cross_down', False))
            bar_time = h1_df.index[i]

            if atr <= 0 or pd.isna(atr):
                continue

            # Check exit
            if pos is not None:
                pos.bars += 1
                reason = None
                exit_price = close

                if pos.direction == 'BUY' and close <= pos.sl:
                    reason = 'SL'
                    exit_price = pos.sl
                elif pos.direction == 'SELL' and close >= pos.sl:
                    reason = 'SL'
                    exit_price = pos.sl
                elif pos.bars >= max_h:
                    reason = 'Timeout'

                if reason:
                    if pos.direction == 'BUY':
                        pos.pnl = (exit_price - pos.entry_price) * (RISK_PER_TRADE / (SL_ATR * pos.atr))
                    else:
                        pos.pnl = (pos.entry_price - exit_price) * (RISK_PER_TRADE / (SL_ATR * pos.atr))
                    pos.exit_time = bar_time
                    pos.exit_reason = reason
                    trades_list.append(pos)
                    pos = None

            # Check entry
            if pos is None:
                sl_dist = SL_ATR * atr
                if slowk < buy_th and stoch_up and close > ema100:
                    pos = Trade('BUY', close, close - sl_dist, bar_time, atr)
                elif slowk > sell_th and stoch_down and close < ema100:
                    pos = Trade('SELL', close, close + sl_dist, bar_time, atr)

        # Stats
        if trades_list:
            total_pnl = sum(t.pnl for t in trades_list)
            wins = sum(1 for t in trades_list if t.pnl > 0)
            wr = 100.0 * wins / len(trades_list)
            ppt = total_pnl / len(trades_list)

            daily_pnl = defaultdict(float)
            for t in trades_list:
                daily_pnl[t.entry_time.strftime('%Y-%m-%d')] += t.pnl
            vals = list(daily_pnl.values())
            sharpe = np.mean(vals) / np.std(vals) * np.sqrt(252) if len(vals) > 1 and np.std(vals) > 0 else 0

            eq = [0.0]
            for t in trades_list:
                eq.append(eq[-1] + t.pnl)
            max_dd = min(eq[j] - max(eq[:j+1]) for j in range(len(eq)))
        else:
            total_pnl = 0
            wins = 0
            wr = 0
            ppt = 0
            sharpe = 0
            max_dd = 0

        all_results.append({
            'label': full_label,
            'n': len(trades_list),
            'pnl': total_pnl,
            'wr': wr,
            'ppt': ppt,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'trades': trades_list,
        })


# ── Run M15 RSI baseline for comparison ──
print("\n--- Running D1+3h baseline (with M15 RSI) for comparison ---")
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
base_stats = run_variant(data, "D1+3h Baseline", **MEGA_D1_3H)
rsi_trades = [t for t in base_stats['_trades'] if t.strategy == 'm15_rsi']
rsi_pnl = sum(t.pnl for t in rsi_trades)
rsi_wins = sum(1 for t in rsi_trades if t.pnl > 0)
rsi_wr = 100.0 * rsi_wins / len(rsi_trades) if rsi_trades else 0
rsi_ppt = rsi_pnl / len(rsi_trades) if rsi_trades else 0


print("\n\n" + "=" * 80)
print("STOCHASTIC STRATEGY RESULTS")
print("=" * 80)
hdr = f"{'Variant':<35} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7} {'MaxDD':>8}"
print(hdr)
print("-" * len(hdr))
for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['label']:<35} {r['n']:>5} {r['sharpe']:>7.2f} ${r['pnl']:>9,.0f} "
          f"{r['wr']:>5.1f}% ${r['ppt']:>6.2f} ${r['max_dd']:>7,.0f}")

print(f"\n\nM15 RSI COMPARISON (current live)")
print("-" * 50)
print(f"  M15 RSI: N={len(rsi_trades)}, PnL=${rsi_pnl:,.0f}, WR={rsi_wr:.1f}%, $/t=${rsi_ppt:.2f}")
if all_results:
    best = max(all_results, key=lambda x: x['sharpe'])
    print(f"  Best Stoch: {best['label']}")
    print(f"    N={best['n']}, PnL=${best['pnl']:,.0f}, WR={best['wr']:.1f}%, $/t=${best['ppt']:.2f}, Sharpe={best['sharpe']:.2f}")
    if best['ppt'] > rsi_ppt:
        print("  >>> Stochastic has higher $/trade — consider as RSI alternative or supplement.")
    else:
        print("  >>> M15 RSI performs better $/trade — keep current approach.")

total_elapsed = time.time() - t_total
print(f"\n\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
