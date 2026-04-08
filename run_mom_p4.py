#!/usr/bin/env python3
"""Part 4: 6-Fold K-Fold validation — Baseline vs Combo bypass"""
import sys, os, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS, run_kfold
from backtest.engine import BacktestEngine

_original_update = BacktestEngine._update_intraday_score

def _make_bypass_update(bypass_mode="combo", upgrade_to="neutral",
                        kc_window=3, kc_min_breaks=2,
                        atr_window=5, atr_spike_mult=1.5,
                        mom_window=3, mom_atr_mult=1.5):
    def _patched(self, h1_window, bar_time):
        _original_update(self, h1_window, bar_time)
        if self._current_regime != 'choppy':
            return
        if h1_window is None or len(h1_window) < max(kc_window, atr_window, mom_window) + 1:
            return
        triggered = False
        tail = h1_window.iloc[-kc_window:]
        if bypass_mode in ("kc", "combo"):
            kc_u, kc_l = tail.get('KC_upper'), tail.get('KC_lower')
            if kc_u is not None and kc_l is not None:
                if ((tail['Close'] > kc_u) | (tail['Close'] < kc_l)).sum() >= kc_min_breaks:
                    triggered = True
        if bypass_mode in ("atr", "combo") and not triggered:
            atr_col = h1_window['ATR']
            if len(atr_col) >= atr_window + 1:
                if float(atr_col.iloc[-1]) > float(atr_col.iloc[-(atr_window+1):-1].mean()) * atr_spike_mult:
                    triggered = True
        if bypass_mode in ("mom", "combo") and not triggered:
            closes = h1_window['Close'].iloc[-(mom_window+1):]
            if len(closes) >= mom_window + 1:
                move = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))
                atr_now = float(h1_window.iloc[-1].get('ATR', 0))
                if atr_now > 0 and move > mom_atr_mult * atr_now:
                    triggered = True
        if triggered:
            self._current_regime = upgrade_to
    return _patched

t0 = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)
CURRENT = {**C12_KWARGS, "intraday_adaptive": True}

print("=" * 70)
print("PART 4: 6-FOLD K-FOLD VALIDATION")
print("=" * 70)

print("\n--- Baseline K-Fold ---")
BacktestEngine._update_intraday_score = _original_update
baseline_folds = run_kfold(data, engine_kwargs=CURRENT, n_folds=6)

print("\n--- Combo Bypass K-Fold ---")
BacktestEngine._update_intraday_score = _make_bypass_update()
combo_folds = run_kfold(data, engine_kwargs=CURRENT, n_folds=6)
BacktestEngine._update_intraday_score = _original_update

print("\n" + "=" * 70)
print("K-FOLD SUMMARY")
print("=" * 70)
if baseline_folds and combo_folds:
    b_sh = [f['sharpe'] for f in baseline_folds]
    c_sh = [f['sharpe'] for f in combo_folds]
    b_pnl = [f['total_pnl'] for f in baseline_folds]
    c_pnl = [f['total_pnl'] for f in combo_folds]
    print(f"\n  Baseline:     Avg Sharpe={np.mean(b_sh):.2f} Std={np.std(b_sh):.2f}")
    print(f"    Folds: {['%.2f'%s for s in b_sh]}")
    print(f"    PnL:   {['$%.0f'%p for p in b_pnl]}")
    print(f"\n  Combo Bypass: Avg Sharpe={np.mean(c_sh):.2f} Std={np.std(c_sh):.2f}")
    print(f"    Folds: {['%.2f'%s for s in c_sh]}")
    print(f"    PnL:   {['$%.0f'%p for p in c_pnl]}")
    print(f"\n  Delta Sharpe: {np.mean(c_sh)-np.mean(b_sh):+.3f}")
    print(f"  Positive folds: Baseline {sum(1 for s in b_sh if s>0)}/{len(b_sh)}, Combo {sum(1 for s in c_sh if s>0)}/{len(c_sh)}")
else:
    print("  K-Fold failed")

print(f"\nTotal time: {time.time()-t0:.0f}s")
