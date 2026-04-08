#!/usr/bin/env python3
"""Part 3: Parameter sensitivity (E1-E5) + Mega variants — 9 variants"""
import sys, os, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS
from backtest.engine import BacktestEngine

_original_update = BacktestEngine._update_intraday_score

def _make_bypass_update(bypass_mode="none", upgrade_to="neutral",
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
MEGA = {
    **C12_KWARGS, "intraday_adaptive": True,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}
MEGA_NO_GATE = {
    **C12_KWARGS, "intraday_adaptive": False,
    "trailing_activate_atr": 0.5, "trailing_distance_atr": 0.15,
    "regime_config": {
        'low': {'trail_act': 0.7, 'trail_dist': 0.25},
        'normal': {'trail_act': 0.5, 'trail_dist': 0.15},
        'high': {'trail_act': 0.4, 'trail_dist': 0.10},
    },
}

# (label, bypass_mode, upgrade_to, base_kwargs, bypass_params)
VARIANTS = [
    ("E1: Combo loose(1.3/1.2)", "combo", "neutral", CURRENT,
     {"atr_spike_mult": 1.3, "mom_atr_mult": 1.2}),
    ("E2: Combo default(1.5/1.5)", "combo", "neutral", CURRENT,
     {"atr_spike_mult": 1.5, "mom_atr_mult": 1.5}),
    ("E3: Combo strict(2.0/2.0)", "combo", "neutral", CURRENT,
     {"atr_spike_mult": 2.0, "mom_atr_mult": 2.0}),
    ("E4: KC win=2 min=2", "kc", "neutral", CURRENT,
     {"kc_window": 2, "kc_min_breaks": 2}),
    ("E5: Mom win=2 mult=1.0", "mom", "neutral", CURRENT,
     {"mom_window": 2, "mom_atr_mult": 1.0}),
    ("M-A: Mega Baseline", "none", "neutral", MEGA, {}),
    ("M-E: Mega+Combo", "combo", "neutral", MEGA, {}),
    ("M-G: Mega+Combo->trend", "combo", "trending", MEGA, {}),
    ("M-F: Mega No gate", "none", "neutral", MEGA_NO_GATE, {}),
]

results = []
for label, bm, ut, kw, bp in VARIANTS:
    if bm != "none" and kw.get("intraday_adaptive", False):
        BacktestEngine._update_intraday_score = _make_bypass_update(bypass_mode=bm, upgrade_to=ut, **bp)
    else:
        BacktestEngine._update_intraday_score = _original_update
    stats = run_variant(data, label, **kw)
    BacktestEngine._update_intraday_score = _original_update
    results.append((label, stats))

print("\n" + "=" * 70)
print("PART 3 RESULTS (Params + Mega)")
print("=" * 70)
print(f"{'Variant':<35} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6} {'Skip_C':>7}")
for label, s in results:
    print(f"{label:<35} {s['n']:>5} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['win_rate']*100:>5.1f}% {s['skipped_choppy']:>7}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
