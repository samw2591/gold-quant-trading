"""
Migration Verification: compare old engines vs new unified engine.
Runs C12 config on both and checks that Sharpe/PnL/trade count match.
"""
import sys
import time

import config
from strategies.signals import prepare_indicators, get_orb_strategy
import strategies.signals as signals_mod

# ── Old engine imports ──
from backtest_m15 import (
    load_m15, load_h1_aligned,
    MultiTimeframeEngine, calc_stats as old_calc_stats,
    M15_CSV_PATH, H1_CSV_PATH,
)
from backtest_round2 import Round2Engine
from backtest_advanced import RegimeEngine, C12_KWARGS as OLD_C12

# ── New engine imports ──
from backtest import (
    BacktestEngine, DataBundle, run_variant, calc_stats as new_calc_stats,
    C12_KWARGS as NEW_C12, V3_REGIME, add_atr_percentile,
)


def reset_global():
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False


def run_old_round2(m15_df, h1_df, label, **kwargs):
    reset_global()
    engine = Round2Engine(m15_df, h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = old_calc_stats(trades, engine.equity_curve)
    return stats


def run_old_regime(m15_df, h1_df, label, regime_config, **kwargs):
    reset_global()
    engine = RegimeEngine(m15_df, h1_df, regime_config=regime_config, label=label, **kwargs)
    trades = engine.run()
    stats = old_calc_stats(trades, engine.equity_curve)
    return stats


def run_new(m15_df, h1_df, label, **kwargs):
    reset_global()
    engine = BacktestEngine(m15_df, h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = new_calc_stats(trades, engine.equity_curve)
    return stats


def compare(label, old_stats, new_stats):
    ok = True
    checks = [
        ('n', 'n', 0),
        ('sharpe', 'sharpe', 0.01),
        ('total_pnl', 'total_pnl', 1.0),
        ('win_rate', 'win_rate', 0.1),
        ('max_dd', 'max_dd', 1.0),
    ]
    print(f"\n  {label}:")
    for key, key2, tol in checks:
        old_val = old_stats.get(key, 0)
        new_val = new_stats.get(key2, 0)
        diff = abs(old_val - new_val)
        status = "OK" if diff <= tol else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        print(f"    {key:<12} old={old_val:>10.2f}  new={new_val:>10.2f}  diff={diff:.4f}  [{status}]")
    return ok


def main():
    print("=" * 70)
    print("  MIGRATION VERIFICATION: Old vs New Engine")
    print("=" * 70)

    # Use a small date range for speed
    START = "2024-01-01"

    print(f"\nLoading data (from {START})...")
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= f"{START}"]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    print("Preparing indicators...")
    m15_df = prepare_indicators(m15_raw)
    h1_df = prepare_indicators(h1_raw)
    h1_df = add_atr_percentile(h1_df)

    all_ok = True

    # ── Test 1: C12 (Round2Engine vs BacktestEngine) ──
    print("\n" + "=" * 50)
    print("  TEST 1: C12 Config (Round2Engine)")
    print("=" * 50)

    old = run_old_round2(m15_df, h1_df, "old-C12", **OLD_C12)
    new = run_new(m15_df, h1_df, "new-C12", **NEW_C12)
    if not compare("C12", old, new):
        all_ok = False

    # ── Test 2: C12 + Regime (RegimeEngine vs BacktestEngine) ──
    print("\n" + "=" * 50)
    print("  TEST 2: C12 + Adaptive Trail (RegimeEngine)")
    print("=" * 50)

    V3 = {
        'low': {'trail_act': 1.0, 'trail_dist': 0.35},
        'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
        'high': {'trail_act': 0.6, 'trail_dist': 0.20},
    }
    old = run_old_regime(m15_df, h1_df, "old-regime", V3, **OLD_C12)
    new = run_new(m15_df, h1_df, "new-regime", regime_config=V3, **NEW_C12)
    if not compare("Regime", old, new):
        all_ok = False

    # ── Test 3: Baseline (no overrides) ──
    print("\n" + "=" * 50)
    print("  TEST 3: Baseline (MultiTimeframeEngine)")
    print("=" * 50)

    reset_global()
    old_engine = MultiTimeframeEngine(m15_df, h1_df, label="old-baseline")
    old_trades = old_engine.run()
    old = old_calc_stats(old_trades, old_engine.equity_curve)

    new = run_new(m15_df, h1_df, "new-baseline")
    if not compare("Baseline", old, new):
        all_ok = False

    # ── Test 4: With spread cost ──
    print("\n" + "=" * 50)
    print("  TEST 4: C12 + Spread $0.50")
    print("=" * 50)

    old = run_old_round2(m15_df, h1_df, "old-spread", spread_cost=0.50, **OLD_C12)
    new = run_new(m15_df, h1_df, "new-spread", spread_cost=0.50, **NEW_C12)
    if not compare("Spread", old, new):
        all_ok = False

    # ── Test 5: RSI ADX filter ──
    print("\n" + "=" * 50)
    print("  TEST 5: RSI ADX>40 Filter")
    print("=" * 50)

    old = run_old_round2(m15_df, h1_df, "old-rsi-adx", rsi_adx_filter=40, **OLD_C12)
    new = run_new(m15_df, h1_df, "new-rsi-adx", rsi_adx_filter=40, **NEW_C12)
    if not compare("RSI ADX", old, new):
        all_ok = False

    # ── Summary ──
    print("\n" + "=" * 70)
    if all_ok:
        print("  ALL TESTS PASSED — New engine matches old engines")
    else:
        print("  SOME TESTS FAILED — Review mismatches above")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
