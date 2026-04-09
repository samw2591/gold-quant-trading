"""Quick validation of entry quality filter parameters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS

data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2, start='2024-01-01', end='2024-07-01')

LIVE = {**C12_KWARGS, "intraday_adaptive": True}

print("\n=== TEST 1: Baseline ===")
r0 = run_variant(data, "Baseline", **LIVE)
print(f"  N={r0['n']}, Sharpe={r0['sharpe']:.2f}, PnL=${r0['total_pnl']:.0f}")
print(f"  skip_bars={r0.get('skipped_min_bars',0)}, skip_adx={r0.get('skipped_adx_gray',0)}, esc_cd={r0.get('escalated_cooldowns',0)}")

print("\n=== TEST 2: min_h1_bars=3 ===")
r1 = run_variant(data, "min_bars=3", **LIVE, min_h1_bars_today=3)
print(f"  N={r1['n']}, Sharpe={r1['sharpe']:.2f}, PnL=${r1['total_pnl']:.0f}")
print(f"  skip_bars={r1.get('skipped_min_bars',0)} <-- should be > 0")

print("\n=== TEST 3: ADX gray zone=7, score>=0.50 ===")
r2 = run_variant(data, "gray=7/sc0.50", **LIVE, adx_gray_zone=7, adx_gray_zone_min_score=0.50)
print(f"  N={r2['n']}, Sharpe={r2['sharpe']:.2f}, PnL=${r2['total_pnl']:.0f}")
print(f"  skip_adx={r2.get('skipped_adx_gray',0)} <-- should be > 0")

print("\n=== TEST 4: Escalating cooldown x4 ===")
r3 = run_variant(data, "esc_cd_x4", **LIVE, escalating_cooldown=True, escalating_cooldown_mult=4.0)
print(f"  N={r3['n']}, Sharpe={r3['sharpe']:.2f}, PnL=${r3['total_pnl']:.0f}")
print(f"  esc_cd={r3.get('escalated_cooldowns',0)}")

print("\n=== TEST 5: All combined ===")
r4 = run_variant(data, "combined", **LIVE,
                 min_h1_bars_today=3, adx_gray_zone=7, adx_gray_zone_min_score=0.50,
                 escalating_cooldown=True, escalating_cooldown_mult=4.0)
print(f"  N={r4['n']}, Sharpe={r4['sharpe']:.2f}, PnL=${r4['total_pnl']:.0f}")
print(f"  skip_bars={r4.get('skipped_min_bars',0)}, skip_adx={r4.get('skipped_adx_gray',0)}, esc_cd={r4.get('escalated_cooldowns',0)}")

print("\n=== SANITY CHECKS ===")
fails = 0
if r1.get('skipped_min_bars', 0) == 0:
    print("FAIL: min_h1_bars filter never triggered!")
    fails += 1
else:
    print(f"OK: min_bars skipped {r1['skipped_min_bars']} entries")

if r2.get('skipped_adx_gray', 0) == 0:
    print("FAIL: ADX gray zone filter never triggered!")
    fails += 1
else:
    print(f"OK: ADX gray skipped {r2['skipped_adx_gray']} entries")

if r3.get('escalated_cooldowns', 0) == 0:
    print("WARN: escalating cooldown never triggered (may be OK if few same-day double losses in 6mo)")
else:
    print(f"OK: escalated {r3['escalated_cooldowns']} cooldowns")

if r0['n'] == r1['n'] == r2['n']:
    print("FAIL: all variants have identical trade count - filters not working!")
    fails += 1
else:
    print(f"OK: trade counts differ (base={r0['n']}, bars={r1['n']}, adx={r2['n']})")

if fails == 0:
    print("\nALL CHECKS PASSED")
else:
    print(f"\n{fails} CHECK(S) FAILED")
