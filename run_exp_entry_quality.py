#!/usr/bin/env python3
"""
EXP-EQ: Entry Quality Filters — Anti False-Breakout
=====================================================
Based on 4/6-4/8 consecutive loss analysis, three targeted fixes:

Fix 1 — min_h1_bars_today: Don't enter when today has < N H1 bars
  Problem: bars=0 → trend_score defaults to 0.50 (neutral) → blindly allows entry
  4/6 first SELL entered at bars=0, lost $46.47

Fix 2 — ADX gray zone: When ADX is 18-25 (marginal), require higher trend_score
  Problem: ADX=20-22 passes the 18 threshold but has weak trend confirmation
  4/6 second BUY: ADX=20.2, trend_score=0.39 (barely above 0.35)
  4/8 first BUY:  ADX=22.3, lost $47.23
  4/8 second BUY: ADX=23.9, lost $18.54

Fix 3 — Escalating cooldown: After 2nd same-day loss, cooldown 4x longer
  Problem: 30min cooldown too short in persistent choppy market
  4/8: 3 losses in one day, each 30min apart

Test plan:
  Part 1: Individual effects (11yr, no spread)
  Part 2: Combined best + $0.50 spread
  Part 3: K-Fold 6-fold validation
  Part 4: Per-year breakdown

No spread for Part 1 (isolate signal quality). $0.50 for Part 2+ (realistic).
"""
import sys, os, time, json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backtest import DataBundle, run_variant, run_kfold, calc_stats, aggregate_daily_pnl
from backtest.runner import C12_KWARGS, sanitize_for_json

print("=" * 70)
print("EXP-EQ: ENTRY QUALITY FILTERS — ANTI FALSE-BREAKOUT")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
print(f"  Load time: {time.time()-t0:.1f}s")

# Current live config
LIVE = {
    **C12_KWARGS,
    "intraday_adaptive": True,
}

# ═══════════════════════════════════════════════════════════════
# Part 1: Individual Effects (no spread)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 1: INDIVIDUAL EFFECTS (no spread)")
print("=" * 70)

results_p1 = []

# A: Baseline (current live config)
r = run_variant(data, "A: Baseline (live)", **LIVE)
results_p1.append(r)

# --- Fix 1: min_h1_bars_today ---
for min_bars in [2, 3, 4, 5]:
    r = run_variant(data, f"B{min_bars}: min_bars={min_bars}",
                    **LIVE, min_h1_bars_today=min_bars)
    results_p1.append(r)

# --- Fix 2: ADX gray zone ---
# ADX in [18, 18+gray) requires trend_score >= min_score
for gray, min_sc in [(7, 0.45), (7, 0.50), (7, 0.55),
                     (10, 0.45), (10, 0.50), (10, 0.55),
                     (12, 0.50)]:
    r = run_variant(data, f"C: gray={gray} score>={min_sc:.2f}",
                    **LIVE, adx_gray_zone=gray, adx_gray_zone_min_score=min_sc)
    results_p1.append(r)

# --- Fix 3: Escalating cooldown ---
for mult in [2.0, 4.0, 8.0]:
    r = run_variant(data, f"D: esc_cd x{mult:.0f}",
                    **LIVE, escalating_cooldown=True, escalating_cooldown_mult=mult)
    results_p1.append(r)

# --- Combined: best of each ---
# min_bars=3 + gray=7/score>=0.50 + escalating x4
r = run_variant(data, "E: Combined (3+7/0.50+x4)",
                **LIVE,
                min_h1_bars_today=3,
                adx_gray_zone=7, adx_gray_zone_min_score=0.50,
                escalating_cooldown=True, escalating_cooldown_mult=4.0)
results_p1.append(r)

# min_bars=2 + gray=10/score>=0.50 + escalating x4
r = run_variant(data, "E2: Combined (2+10/0.50+x4)",
                **LIVE,
                min_h1_bars_today=2,
                adx_gray_zone=10, adx_gray_zone_min_score=0.50,
                escalating_cooldown=True, escalating_cooldown_mult=4.0)
results_p1.append(r)

print("\n" + "-" * 70)
print("PART 1 SUMMARY (no spread)")
print("-" * 70)
print(f"{'Variant':<35} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'MaxDD':>7} {'WR%':>5} "
      f"{'SkipBar':>7} {'SkipADX':>7} {'EscCD':>6}")
print("-" * 105)
base_sharpe = results_p1[0]['sharpe']
for r in results_p1:
    delta = r['sharpe'] - base_sharpe
    sign = "+" if delta >= 0 else ""
    print(f"{r['label']:<35} {r['n']:>6} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} "
          f"${r['max_dd']:>6.0f} {r.get('win_rate',0)*100:>4.1f} "
          f"{r.get('skipped_min_bars',0):>7} {r.get('skipped_adx_gray',0):>7} "
          f"{r.get('escalated_cooldowns',0):>6}  ({sign}{delta:.2f})")

# ═══════════════════════════════════════════════════════════════
# Part 2: Best variants with $0.50 spread
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: BEST VARIANTS WITH $0.50 SPREAD")
print("=" * 70)

results_p2 = []

# Baseline with spread
r = run_variant(data, "A: Baseline+sp50", **LIVE, spread_cost=0.50)
results_p2.append(r)

# Replicate top candidates from P1 with spread
for min_bars in [2, 3]:
    r = run_variant(data, f"B{min_bars}: min_bars={min_bars}+sp50",
                    **LIVE, min_h1_bars_today=min_bars, spread_cost=0.50)
    results_p2.append(r)

for gray, min_sc in [(7, 0.50), (10, 0.50)]:
    r = run_variant(data, f"C: gray={gray}/sc{min_sc}+sp50",
                    **LIVE, adx_gray_zone=gray, adx_gray_zone_min_score=min_sc,
                    spread_cost=0.50)
    results_p2.append(r)

r = run_variant(data, "D: esc_cd x4+sp50",
                **LIVE, escalating_cooldown=True, escalating_cooldown_mult=4.0,
                spread_cost=0.50)
results_p2.append(r)

r = run_variant(data, "E: Combined+sp50",
                **LIVE,
                min_h1_bars_today=3,
                adx_gray_zone=7, adx_gray_zone_min_score=0.50,
                escalating_cooldown=True, escalating_cooldown_mult=4.0,
                spread_cost=0.50)
results_p2.append(r)

r = run_variant(data, "E2: Combined2+sp50",
                **LIVE,
                min_h1_bars_today=2,
                adx_gray_zone=10, adx_gray_zone_min_score=0.50,
                escalating_cooldown=True, escalating_cooldown_mult=4.0,
                spread_cost=0.50)
results_p2.append(r)

print("\n" + "-" * 70)
print("PART 2 SUMMARY ($0.50 spread)")
print("-" * 70)
print(f"{'Variant':<35} {'N':>6} {'Sharpe':>7} {'PnL':>9} {'MaxDD':>7} {'WR%':>5} {'$/t':>6}")
print("-" * 85)
base2 = results_p2[0]['sharpe']
for r in results_p2:
    delta = r['sharpe'] - base2
    sign = "+" if delta >= 0 else ""
    dpt = r['total_pnl'] / r['n'] if r['n'] > 0 else 0
    print(f"{r['label']:<35} {r['n']:>6} {r['sharpe']:>7.2f} ${r['total_pnl']:>8.0f} "
          f"${r['max_dd']:>6.0f} {r.get('win_rate',0)*100:>4.1f} ${dpt:>5.2f}  ({sign}{delta:.2f})")

# ═══════════════════════════════════════════════════════════════
# Part 3: K-Fold 6-Fold Validation (best variant vs baseline)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: K-FOLD 6-FOLD VALIDATION ($0.50 spread)")
print("=" * 70)

# Baseline K-Fold
kf_base = run_kfold(data, {**LIVE, "spread_cost": 0.50},
                    label_prefix="Base_")

# Combined best K-Fold (using E: 3+7/0.50+x4)
kf_combined = run_kfold(data, {
    **LIVE,
    "spread_cost": 0.50,
    "min_h1_bars_today": 3,
    "adx_gray_zone": 7,
    "adx_gray_zone_min_score": 0.50,
    "escalating_cooldown": True,
    "escalating_cooldown_mult": 4.0,
}, label_prefix="Combo_")

print("\n" + "-" * 70)
print("K-FOLD RESULTS")
print("-" * 70)
print(f"{'Fold':<25} {'Base Sh':>8} {'Combo Sh':>9} {'Delta':>7} {'Winner':>8}")
print("-" * 60)
wins = 0
for b, c in zip(kf_base, kf_combined):
    delta = c['sharpe'] - b['sharpe']
    winner = "Combo" if delta > 0 else ("Base" if delta < 0 else "TIE")
    if delta > 0:
        wins += 1
    print(f"{b.get('fold','?'):<25} {b['sharpe']:>8.2f} {c['sharpe']:>9.2f} {delta:>+7.2f} {winner:>8}")

base_avg = np.mean([r['sharpe'] for r in kf_base])
combo_avg = np.mean([r['sharpe'] for r in kf_combined])
print(f"{'Average':<25} {base_avg:>8.2f} {combo_avg:>9.2f} {combo_avg-base_avg:>+7.2f} "
      f"  {wins}/{len(kf_base)} folds")

# ═══════════════════════════════════════════════════════════════
# Part 4: Per-Year Breakdown (Combined best, $0.50 spread)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: PER-YEAR BREAKDOWN (Combined best, $0.50 spread)")
print("=" * 70)

combo_trades = results_p2[-2]['_trades']  # E: Combined
base_trades = results_p2[0]['_trades']    # Baseline

years = range(2015, 2027)
print(f"{'Year':<6} {'Base N':>7} {'Base PnL':>9} {'Base Sh':>8} "
      f"{'Combo N':>7} {'Combo PnL':>9} {'Combo Sh':>8} {'Delta':>7}")
print("-" * 75)

for yr in years:
    yr_base = [t for t in base_trades
               if hasattr(t, 'entry_time') and t.entry_time.year == yr]
    yr_combo = [t for t in combo_trades
                if hasattr(t, 'entry_time') and t.entry_time.year == yr]

    b_pnl = sum(t.pnl for t in yr_base)
    c_pnl = sum(t.pnl for t in yr_combo)

    b_stats = calc_stats(yr_base, []) if yr_base else {'sharpe': 0, 'n': 0}
    c_stats = calc_stats(yr_combo, []) if yr_combo else {'sharpe': 0, 'n': 0}

    delta = c_stats['sharpe'] - b_stats['sharpe']
    print(f"{yr:<6} {b_stats['n']:>7} ${b_pnl:>8.0f} {b_stats['sharpe']:>8.2f} "
          f"{c_stats['n']:>7} ${c_pnl:>8.0f} {c_stats['sharpe']:>8.2f} {delta:>+7.2f}")

# ═══════════════════════════════════════════════════════════════
# Save results
# ═══════════════════════════════════════════════════════════════

output = {
    "part1_no_spread": sanitize_for_json(results_p1),
    "part2_with_spread": sanitize_for_json(results_p2),
    "part3_kfold_base": sanitize_for_json(kf_base),
    "part3_kfold_combo": sanitize_for_json(kf_combined),
}

out_path = "data/exp_entry_quality_results.json"
os.makedirs("data", exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"COMPLETED in {elapsed/60:.1f} minutes")
print(f"Results saved to {out_path}")
print(f"{'=' * 70}")
