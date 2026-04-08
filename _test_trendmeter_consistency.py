#!/usr/bin/env python3
"""
Verify that backtest engine's _calc_realtime_score produces identical results
to the live IntradayTrendMeter._calc_score for the same today_bars input.

Also verify the fixed _update_intraday_score correctly extracts today's bars
using the global H1 index (not the rolling window length).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backtest import DataBundle
from backtest.engine import BacktestEngine
from intraday_trend import IntradayTrendMeter

data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
h1_df = data.h1_df

test_dates = [
    pd.Timestamp("2020-03-16").date(),  # COVID crash — high vol
    pd.Timestamp("2023-10-09").date(),  # normal day
    pd.Timestamp("2025-03-15").date(),  # recent
    pd.Timestamp("2019-06-20").date(),  # mid-year
]

print("=" * 70)
print("TEST 1: _calc_score vs _calc_realtime_score (same input)")
print("=" * 70)

meter = IntradayTrendMeter()
all_pass = True

for test_date in test_dates:
    today_bars = h1_df[h1_df.index.date == test_date]
    if len(today_bars) < 2:
        print(f"  {test_date}: skipped (only {len(today_bars)} bars)")
        continue

    live_score = meter._calc_score(today_bars)
    bt_score = BacktestEngine._calc_realtime_score(today_bars)

    match = abs(live_score - bt_score) < 1e-6
    status = "PASS" if match else "FAIL"
    if not match:
        all_pass = False
    print(f"  {test_date}: live={live_score:.3f} bt={bt_score:.3f} [{status}] ({len(today_bars)} bars)")

print(f"\n  Test 1: {'ALL PASS' if all_pass else 'FAILED'}")

print("\n" + "=" * 70)
print("TEST 2: _update_intraday_score uses correct global index")
print("=" * 70)

engine = BacktestEngine(
    data.m15_df, data.h1_df,
    intraday_adaptive=True,
    choppy_threshold=0.35,
    kc_only_threshold=0.60,
)

test2_pass = True
for test_date in test_dates:
    today_indices = engine._h1_date_map.get(test_date, [])
    if len(today_indices) < 3:
        print(f"  {test_date}: skipped (only {len(today_indices)} H1 bars)")
        continue

    for n_bars in [3, len(today_indices) // 2, len(today_indices)]:
        partial_indices = today_indices[:n_bars]
        max_idx = partial_indices[-1]

        partial_bars = h1_df.iloc[partial_indices]
        expected_score = BacktestEngine._calc_realtime_score(partial_bars)

        bar_time = h1_df.index[max_idx]
        h1_start = max(0, max_idx - 99)
        h1_window = h1_df.iloc[h1_start:max_idx + 1]

        engine._cached_date = None
        engine._cached_h1_count = 0
        engine._current_score = 0.5
        engine._current_regime = 'neutral'

        engine._update_intraday_score(h1_window, bar_time)
        actual_score = engine._current_score

        match = abs(expected_score - actual_score) < 1e-6
        if not match:
            test2_pass = False
        status = "PASS" if match else "FAIL"
        print(f"  {test_date} bars={n_bars}/{len(today_indices)}: "
              f"expected={expected_score:.3f} actual={actual_score:.3f} "
              f"regime={engine._current_regime} [{status}]")

print(f"\n  Test 2: {'ALL PASS' if test2_pass else 'FAILED'}")

print("\n" + "=" * 70)
print("TEST 3: choppy regime is actually reachable")
print("=" * 70)

choppy_count = 0
total_days = 0
sample_choppy = []

all_dates = sorted(engine._h1_date_map.keys())
for d in all_dates:
    indices = engine._h1_date_map[d]
    if len(indices) < 2:
        continue
    total_days += 1

    for n_bars in range(2, len(indices) + 1):
        partial = h1_df.iloc[indices[:n_bars]]
        score = BacktestEngine._calc_realtime_score(partial)
        if score < 0.35:
            choppy_count += 1
            if len(sample_choppy) < 5:
                sample_choppy.append((d, n_bars, len(indices), score))
            break

print(f"  Total trading days: {total_days}")
print(f"  Days with at least one choppy window: {choppy_count} ({choppy_count/total_days*100:.1f}%)")
print(f"\n  Sample choppy moments:")
for d, nb, total, sc in sample_choppy:
    print(f"    {d}: bar {nb}/{total}, score={sc:.3f}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)
