#!/usr/bin/env python3
"""
London Session Breakout Strategy Test
========================================
Asian range (UTC 0:00-7:00) → London breakout (UTC 7:00-9:00).
Compare with existing ORB (NY session).
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

OUTPUT_FILE = "london_breakout_output.txt"


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
print("LONDON SESSION BREAKOUT STRATEGY TEST")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

m15_df = data.m15_df.copy()
h1_df = data.h1_df.copy()

# Need ATR on M15 for stop calculation
if 'ATR' not in m15_df.columns:
    m15_df['tr'] = np.maximum(m15_df['High'] - m15_df['Low'],
                               np.maximum(abs(m15_df['High'] - m15_df['Close'].shift(1)),
                                          abs(m15_df['Low'] - m15_df['Close'].shift(1))))
    m15_df['ATR'] = m15_df['tr'].rolling(56).mean()  # 14h equiv
if 'EMA100' not in m15_df.columns:
    m15_df['EMA100'] = m15_df['Close'].ewm(span=400, adjust=False).mean()  # ~100h in M15


class Trade:
    def __init__(self, direction, entry_price, sl, tp, entry_time, atr, strategy='london_bo'):
        self.direction = direction
        self.entry_price = entry_price
        self.sl = sl
        self.tp = tp
        self.entry_time = entry_time
        self.atr = atr
        self.strategy = strategy
        self.bars = 0
        self.pnl = 0.0
        self.exit_time = None
        self.exit_reason = ''


RISK_PER_TRADE = 50.0

# Variants: different parameters
configs = [
    {"label": "LB1: Asian(0-7) break(7-9) hold=8", "range_start": 0, "range_end": 7, "bo_start": 7, "bo_end": 9, "max_hold": 8, "sl_atr": 4.5, "tp_rr": 2.0, "ema_filter": True},
    {"label": "LB2: Asian(0-7) break(7-9) hold=12", "range_start": 0, "range_end": 7, "bo_start": 7, "bo_end": 9, "max_hold": 12, "sl_atr": 4.5, "tp_rr": 2.0, "ema_filter": True},
    {"label": "LB3: Asian(0-7) break(7-9) noEMA", "range_start": 0, "range_end": 7, "bo_start": 7, "bo_end": 9, "max_hold": 8, "sl_atr": 4.5, "tp_rr": 2.0, "ema_filter": False},
    {"label": "LB4: range(2-7) break(7-10) hold=8", "range_start": 2, "range_end": 7, "bo_start": 7, "bo_end": 10, "max_hold": 8, "sl_atr": 4.5, "tp_rr": 2.0, "ema_filter": True},
    {"label": "LB5: Asian SL=range, TP=2xRange", "range_start": 0, "range_end": 7, "bo_start": 7, "bo_end": 9, "max_hold": 12, "sl_atr": 0, "tp_rr": 2.0, "ema_filter": True},  # SL = range height
    {"label": "LB6: RR=1.5", "range_start": 0, "range_end": 7, "bo_start": 7, "bo_end": 9, "max_hold": 8, "sl_atr": 4.5, "tp_rr": 1.5, "ema_filter": True},
    {"label": "LB7: RR=3.0", "range_start": 0, "range_end": 7, "bo_start": 7, "bo_end": 9, "max_hold": 8, "sl_atr": 4.5, "tp_rr": 3.0, "ema_filter": True},
]


def run_london_breakout(cfg):
    trades_list = []
    pos = None
    daily_range = {}

    for i in range(200, len(m15_df)):
        row = m15_df.iloc[i]
        bar_time = m15_df.index[i]
        close = float(row['Close'])
        high = float(row['High'])
        low = float(row['Low'])
        atr = float(row.get('ATR', 0))
        ema100 = float(row.get('EMA100', close))
        hour = bar_time.hour

        if atr <= 0 or pd.isna(atr):
            continue

        day_key = bar_time.strftime('%Y-%m-%d')

        # Build Asian range
        if cfg['range_start'] <= hour < cfg['range_end']:
            if day_key not in daily_range:
                daily_range[day_key] = {'high': high, 'low': low, 'traded': False}
            else:
                daily_range[day_key]['high'] = max(daily_range[day_key]['high'], high)
                daily_range[day_key]['low'] = min(daily_range[day_key]['low'], low)

        # Check exit
        if pos is not None:
            pos.bars += 1
            reason = None
            exit_price = close

            if pos.direction == 'BUY':
                if low <= pos.sl:
                    reason = 'SL'
                    exit_price = pos.sl
                elif high >= pos.tp:
                    reason = 'TP'
                    exit_price = pos.tp
            else:
                if high >= pos.sl:
                    reason = 'SL'
                    exit_price = pos.sl
                elif low <= pos.tp:
                    reason = 'TP'
                    exit_price = pos.tp

            if not reason and pos.bars >= cfg['max_hold']:
                reason = 'Timeout'

            if reason:
                if pos.direction == 'BUY':
                    pos.pnl = (exit_price - pos.entry_price) * (RISK_PER_TRADE / (abs(pos.entry_price - pos.sl) if abs(pos.entry_price - pos.sl) > 0 else 1))
                else:
                    pos.pnl = (pos.entry_price - exit_price) * (RISK_PER_TRADE / (abs(pos.sl - pos.entry_price) if abs(pos.sl - pos.entry_price) > 0 else 1))
                pos.exit_time = bar_time
                pos.exit_reason = reason
                trades_list.append(pos)
                pos = None

        # Check entry (London breakout window)
        if pos is None and cfg['bo_start'] <= hour < cfg['bo_end']:
            dr = daily_range.get(day_key)
            if dr and not dr['traded']:
                range_h = dr['high']
                range_l = dr['low']
                range_size = range_h - range_l

                if range_size > 0.5:  # min range filter
                    if cfg['sl_atr'] > 0:
                        sl_dist = cfg['sl_atr'] * atr
                    else:
                        sl_dist = range_size  # use range as SL

                    tp_dist = sl_dist * cfg['tp_rr']

                    if close > range_h:
                        if not cfg['ema_filter'] or close > ema100:
                            dr['traded'] = True
                            pos = Trade('BUY', close, close - sl_dist, close + tp_dist, bar_time, atr)
                    elif close < range_l:
                        if not cfg['ema_filter'] or close < ema100:
                            dr['traded'] = True
                            pos = Trade('SELL', close, close + sl_dist, close - tp_dist, bar_time, atr)

    return trades_list


all_results = []
for cfg in configs:
    print(f"\n  [{cfg['label']}]", flush=True)
    t0 = time.time()
    trades_list = run_london_breakout(cfg)
    elapsed = time.time() - t0

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

        sl_cnt = sum(1 for t in trades_list if t.exit_reason == 'SL')
        tp_cnt = sum(1 for t in trades_list if t.exit_reason == 'TP')
        tm_cnt = sum(1 for t in trades_list if t.exit_reason == 'Timeout')
    else:
        total_pnl = wr = ppt = sharpe = max_dd = 0
        sl_cnt = tp_cnt = tm_cnt = 0

    all_results.append({
        'label': cfg['label'],
        'n': len(trades_list),
        'pnl': total_pnl,
        'wr': wr,
        'ppt': ppt,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'sl': sl_cnt, 'tp': tp_cnt, 'tm': tm_cnt,
        'elapsed': elapsed,
    })
    print(f"    {len(trades_list)} trades, Sharpe={sharpe:.2f}, PnL=${total_pnl:,.0f}, {elapsed:.0f}s")


# ── Compare with existing ORB ──
print("\n\n--- Running full portfolio baseline (with ORB) for comparison ---")
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
orb_trades = [t for t in base_stats['_trades'] if t.strategy == 'orb']
orb_pnl = sum(t.pnl for t in orb_trades)
orb_wr = 100.0 * sum(1 for t in orb_trades if t.pnl > 0) / len(orb_trades) if orb_trades else 0
orb_ppt = orb_pnl / len(orb_trades) if orb_trades else 0


print("\n\n" + "=" * 80)
print("LONDON BREAKOUT RESULTS")
print("=" * 80)
hdr = f"{'Variant':<40} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'WR%':>6} {'$/t':>7} {'MaxDD':>8} {'SL':>4} {'TP':>4} {'TM':>4}"
print(hdr)
print("-" * len(hdr))
for r in sorted(all_results, key=lambda x: x['sharpe'], reverse=True):
    print(f"{r['label']:<40} {r['n']:>5} {r['sharpe']:>7.2f} ${r['pnl']:>9,.0f} "
          f"{r['wr']:>5.1f}% ${r['ppt']:>6.2f} ${r['max_dd']:>7,.0f} {r['sl']:>4} {r['tp']:>4} {r['tm']:>4}")

print(f"\n\nORB COMPARISON (current live)")
print("-" * 50)
print(f"  ORB: N={len(orb_trades)}, PnL=${orb_pnl:,.0f}, WR={orb_wr:.1f}%, $/t=${orb_ppt:.2f}")
if all_results:
    best = max(all_results, key=lambda x: x['sharpe'])
    print(f"  Best London BO: {best['label']}")
    print(f"    N={best['n']}, PnL=${best['pnl']:,.0f}, WR={best['wr']:.1f}%, $/t=${best['ppt']:.2f}, Sharpe={best['sharpe']:.2f}")
    if best['sharpe'] > 1.0 and best['ppt'] > orb_ppt:
        print("  >>> London Breakout looks promising — consider replacing or supplementing ORB.")
    elif best['sharpe'] > 0:
        print("  >>> London Breakout has potential but may not justify replacing ORB.")
    else:
        print("  >>> London Breakout underperforms — keep current ORB approach.")

total_elapsed = time.time() - t_total
print(f"\n\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
