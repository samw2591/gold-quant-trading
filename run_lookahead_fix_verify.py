#!/usr/bin/env python3
"""
Verify look-ahead fix: compare old vs new engine behavior.
Runs a quick baseline backtest to see Sharpe before/after fix.
"""
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS

OUTPUT_FILE = "lookahead_fix_verify_output.txt"


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

print("=" * 70)
print("LOOK-AHEAD FIX VERIFICATION")
print(f"Started: {datetime.now()}")
print("=" * 70)

t0 = time.time()

print("\nLoading data...")
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

MEGA = {
    **C12_KWARGS,
    "intraday_adaptive": True,
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

CURRENT = {
    "sl_atr_mult": 4.5,
    "tp_atr_mult": 8.0,
    "keltner_adx_threshold": 18,
}

print("\n--- Post-fix baseline (look-ahead fixed) ---")

def ps(label, s):
    avg_pnl = s['total_pnl'] / s['n'] if s['n'] > 0 else 0
    print(f"    {label}: N={s['n']}, Sharpe={s['sharpe']:.2f}, "
          f"PnL=${s['total_pnl']:,.0f}, WR={s['win_rate']:.1f}%, "
          f"$/t=${avg_pnl:.2f}, MaxDD=${s['max_dd']:,.0f}")

print("\n  Current config (no cost):")
s_cur = run_variant(data, "Current-NoSpread", **CURRENT)
ps("Current $0", s_cur)

print("\n  Current config ($0.30 spread):")
s_cur30 = run_variant(data, "Current-Sp030", **CURRENT, spread_cost=0.30)
ps("Current $0.30", s_cur30)

print("\n  Current config ($0.50 spread):")
s_cur50 = run_variant(data, "Current-Sp050", **CURRENT, spread_cost=0.50)
ps("Current $0.50", s_cur50)

print("\n  Mega config (no cost):")
s_mega = run_variant(data, "Mega-NoSpread", **MEGA)
ps("Mega $0", s_mega)

print("\n  Mega config ($0.30 spread):")
s_mega30 = run_variant(data, "Mega-Sp030", **MEGA, spread_cost=0.30)
ps("Mega $0.30", s_mega30)

print("\n  Mega config ($0.50 spread):")
s_mega50 = run_variant(data, "Mega-Sp050", **MEGA, spread_cost=0.50)
ps("Mega $0.50", s_mega50)

print("\n" + "=" * 70)
print("COMPARISON WITH PRE-FIX VALUES")
print("=" * 70)
print("""
  PRE-FIX (look-ahead present):
    Current no-cost:  N=18,544  Sharpe=5.06  PnL=$35,251
    Mega no-cost:     N=21,619  Sharpe=8.42  PnL=$59,921

  POST-FIX (look-ahead removed):
    Current no-cost:  N={n_cur:>6,d}  Sharpe={sh_cur:.2f}  PnL=${pnl_cur:,.0f}
    Mega no-cost:     N={n_mega:>6,d}  Sharpe={sh_mega:.2f}  PnL=${pnl_mega:,.0f}

  Change:
    Current: Sharpe {dsh_cur:+.2f}, N {dn_cur:+d}, PnL ${dpnl_cur:+,.0f}
    Mega:    Sharpe {dsh_mega:+.2f}, N {dn_mega:+d}, PnL ${dpnl_mega:+,.0f}

  With costs:
    Current $0.30: Sharpe={sh_c30:.2f}, PnL=${pnl_c30:,.0f}
    Current $0.50: Sharpe={sh_c50:.2f}, PnL=${pnl_c50:,.0f}
    Mega $0.30:    Sharpe={sh_m30:.2f}, PnL=${pnl_m30:,.0f}
    Mega $0.50:    Sharpe={sh_m50:.2f}, PnL=${pnl_m50:,.0f}
""".format(
    n_cur=s_cur['n'], sh_cur=s_cur['sharpe'], pnl_cur=s_cur['total_pnl'],
    n_mega=s_mega['n'], sh_mega=s_mega['sharpe'], pnl_mega=s_mega['total_pnl'],
    dsh_cur=s_cur['sharpe']-5.06, dn_cur=s_cur['n']-18544,
    dpnl_cur=s_cur['total_pnl']-35251,
    dsh_mega=s_mega['sharpe']-8.42, dn_mega=s_mega['n']-21619,
    dpnl_mega=s_mega['total_pnl']-59921,
    sh_c30=s_cur30['sharpe'], pnl_c30=s_cur30['total_pnl'],
    sh_c50=s_cur50['sharpe'], pnl_c50=s_cur50['total_pnl'],
    sh_m30=s_mega30['sharpe'], pnl_m30=s_mega30['total_pnl'],
    sh_m50=s_mega50['sharpe'], pnl_m50=s_mega50['total_pnl'],
))

elapsed = time.time() - t0
print(f"Total runtime: {elapsed/60:.1f} minutes")
print(f"Completed: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nResults saved to {OUTPUT_FILE}")
