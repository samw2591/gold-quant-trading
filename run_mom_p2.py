#!/usr/bin/env python3
"""Part 2: Combo bypass + No gate + Combo→trending (E,F,G) — 3 variants"""
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
NO_GATE = {**C12_KWARGS, "intraday_adaptive": False}

VARIANTS = [
    ("E: Combo bypass", "combo", "neutral", CURRENT, {}),
    ("F: No gate (upper bound)", "none", "neutral", NO_GATE, {}),
    ("G: Combo->trending", "combo", "trending", CURRENT, {}),
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
print("PART 2 RESULTS")
print("=" * 70)
print(f"{'Variant':<35} {'N':>5} {'Sharpe':>7} {'PnL':>10} {'MaxDD':>8} {'WR':>6} {'Skip_C':>7}")
for label, s in results:
    print(f"{label:<35} {s['n']:>5} {s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} ${s['max_dd']:>7,.0f} {s['win_rate']*100:>5.1f}% {s['skipped_choppy']:>7}")
print(f"\nTotal time: {time.time()-t0:.0f}s")
