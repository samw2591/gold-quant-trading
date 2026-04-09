#!/usr/bin/env python3
"""
ML Factor Importance Analysis
================================
Uses all trade data to rank factor importance via:
1. Information Coefficient (IC) — rank correlation of factor at entry vs trade PnL
2. Gradient Boosting feature importance
3. Factor interaction analysis
"""
import sys, os, time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from backtest import DataBundle, run_variant
from backtest.runner import C12_KWARGS

OUTPUT_FILE = "factor_importance_output.txt"


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
print("ML FACTOR IMPORTANCE ANALYSIS")
print(f"Started: {datetime.now()}")
print("=" * 80)

t_total = time.time()
data = DataBundle.load_custom(kc_ema=25, kc_mult=1.2)

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

print("\n--- Running baseline to get trades ---")
stats = run_variant(data, "D1+3h Baseline", **MEGA_D1_3H)
trades = stats['_trades']

h1_df = data.h1_df.copy()
m15_df = data.m15_df.copy()

# Ensure indicators exist
for df in [h1_df]:
    if 'RSI14' not in df.columns:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI14'] = 100 - 100 / (1 + rs)
    if 'RSI2' not in df.columns:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / loss
        df['RSI2'] = 100 - 100 / (1 + rs)
    if 'ATR' not in df.columns:
        tr = np.maximum(df['High'] - df['Low'],
                        np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                   abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = tr.rolling(14).mean()
    if 'atr_percentile' not in df.columns:
        df['atr_percentile'] = df['ATR'].rolling(500, min_periods=100).rank(pct=True)
    if 'ADX' not in df.columns:
        df['ADX'] = 25  # placeholder if missing


def to_utc(dt):
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def get_h1_row(entry_time):
    ts = to_utc(entry_time)
    if h1_df.index.tz is not None:
        h1_idx = h1_df.index.tz_localize(None)
    else:
        h1_idx = h1_df.index
    mask = h1_idx <= ts
    if not mask.any():
        return None
    return h1_df.loc[h1_df.index[mask][-1]]


# ── Build factor matrix ──
print("\n--- Building factor matrix from trades ---")
records = []
for t in trades:
    row = get_h1_row(t.entry_time)
    if row is None:
        continue

    close = float(row.get('Close', 0))
    ema100 = float(row.get('EMA100', close))
    kc_upper = float(row.get('KC_upper', close))
    kc_lower = float(row.get('KC_lower', close))
    kc_mid = float(row.get('KC_mid', close))

    rec = {
        'pnl': t.pnl,
        'win': 1 if t.pnl > 0 else 0,
        'strategy': t.strategy,
        'direction': t.direction,
        'hour': t.entry_time.hour,
        'dayofweek': t.entry_time.dayofweek,
        'RSI14': float(row.get('RSI14', 50)),
        'RSI2': float(row.get('RSI2', 50)),
        'ADX': float(row.get('ADX', 25)),
        'ATR': float(row.get('ATR', 0)),
        'atr_pct': float(row.get('atr_percentile', 0.5)),
        'EMA100_dist': (close - ema100) / ema100 * 100 if ema100 > 0 else 0,
        'KC_bw': (kc_upper - kc_lower) / kc_mid if kc_mid > 0 else 0,
        'KC_pos': (close - kc_lower) / (kc_upper - kc_lower) if (kc_upper - kc_lower) > 0 else 0.5,
        'Volume': float(row.get('Volume', 0)),
    }
    records.append(rec)

df_factors = pd.DataFrame(records)
print(f"  Factor matrix: {len(df_factors)} trades x {len(df_factors.columns)} features")

# ── Part 1: IC analysis ──
print("\n\n" + "=" * 80)
print("PART 1: INFORMATION COEFFICIENT (IC) — Rank correlation with PnL")
print("=" * 80)

factor_cols = ['RSI14', 'RSI2', 'ADX', 'ATR', 'atr_pct', 'EMA100_dist', 'KC_bw', 'KC_pos', 'Volume', 'hour', 'dayofweek']

print(f"\n{'Factor':<18} {'IC':>8} {'|IC|':>8} {'p-val':>8} {'Direction':<12}")
print("-" * 58)
from scipy import stats as scipy_stats

ic_results = []
for col in factor_cols:
    valid = df_factors[[col, 'pnl']].dropna()
    if len(valid) < 30:
        continue
    ic, pval = scipy_stats.spearmanr(valid[col], valid['pnl'])
    direction = "POSITIVE" if ic > 0 else "NEGATIVE"
    star = "**" if abs(ic) >= 0.05 else "*" if abs(ic) >= 0.03 else ""
    ic_results.append((col, ic, abs(ic), pval, direction, star))

ic_results.sort(key=lambda x: x[2], reverse=True)
for col, ic, abs_ic, pval, direction, star in ic_results:
    print(f"{col:<18} {ic:>+8.4f} {abs_ic:>8.4f} {pval:>8.4f} {direction:<10} {star}")

# IC by strategy
for strat in ['keltner', 'm15_rsi', 'orb']:
    sub = df_factors[df_factors['strategy'] == strat]
    if len(sub) < 30:
        continue
    print(f"\n  {strat.upper()} IC:")
    for col in factor_cols:
        valid = sub[[col, 'pnl']].dropna()
        if len(valid) < 30:
            continue
        ic, pval = scipy_stats.spearmanr(valid[col], valid['pnl'])
        if abs(ic) >= 0.03:
            print(f"    {col:<18} IC={ic:>+.4f} (p={pval:.4f})")


# ── Part 2: Gradient Boosting ──
print("\n\n" + "=" * 80)
print("PART 2: GRADIENT BOOSTING FEATURE IMPORTANCE")
print("=" * 80)

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    X = df_factors[factor_cols].fillna(0)
    y_win = df_factors['win']
    y_pnl = df_factors['pnl']

    # Classification: predict win/loss
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    cv_scores = cross_val_score(clf, X, y_win, cv=5, scoring='accuracy')
    clf.fit(X, y_win)

    print(f"\nWin/Loss Classification (5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
    print(f"\n{'Factor':<18} {'Importance':>12}")
    print("-" * 32)
    importances = sorted(zip(factor_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    for col, imp in importances:
        bar = "█" * int(imp * 100)
        print(f"{col:<18} {imp:>12.4f}  {bar}")

    # Regression: predict PnL
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    cv_r2 = cross_val_score(reg, X, y_pnl, cv=5, scoring='r2')
    reg.fit(X, y_pnl)

    print(f"\nPnL Regression (5-fold CV R²: {cv_r2.mean():.3f} ± {cv_r2.std():.3f})")
    print(f"\n{'Factor':<18} {'Importance':>12}")
    print("-" * 32)
    importances_r = sorted(zip(factor_cols, reg.feature_importances_), key=lambda x: x[1], reverse=True)
    for col, imp in importances_r:
        bar = "█" * int(imp * 100)
        print(f"{col:<18} {imp:>12.4f}  {bar}")

except ImportError:
    print("  [SKIPPED] scikit-learn not installed. Install with: pip install scikit-learn")


# ── Part 3: Factor interaction ──
print("\n\n" + "=" * 80)
print("PART 3: TOP FACTOR INTERACTIONS")
print("=" * 80)

interactions = [
    ("High ADX + High KC_bw", lambda r: r['ADX'] > 25 and r['KC_bw'] > r['KC_bw'].median()),
    ("Low ADX + Low KC_bw", lambda r: r['ADX'] <= 25 and r['KC_bw'] <= r['KC_bw'].median()),
    ("High ATR + NY session", lambda r: r['atr_pct'] > 0.7 and 14 <= r['hour'] < 21),
    ("Low ATR + Asian session", lambda r: r['atr_pct'] < 0.3 and 0 <= r['hour'] < 7),
    ("RSI14 oversold + BUY", lambda r: r['RSI14'] < 30),
    ("RSI14 overbought + SELL", lambda r: r['RSI14'] > 70),
    ("Monday trades", lambda r: r['dayofweek'] == 0),
    ("Friday trades", lambda r: r['dayofweek'] == 4),
]

med_kc = df_factors['KC_bw'].median()
print(f"\n{'Interaction':<35} {'N':>6} {'PnL':>10} {'$/t':>7} {'WR%':>6}")
print("-" * 68)

for name, cond in interactions:
    try:
        mask = df_factors.apply(lambda r: cond(r), axis=1)
        sub = df_factors[mask]
        if len(sub) < 20:
            continue
        pnl = sub['pnl'].sum()
        wins = sub['win'].sum()
        wr = 100.0 * wins / len(sub)
        ppt = pnl / len(sub)
        print(f"{name:<35} {len(sub):>6} ${pnl:>9,.0f} ${ppt:>6.2f} {wr:>5.1f}%")
    except Exception:
        pass


# ── Summary ──
print("\n\n" + "=" * 80)
print("CONCLUSIONS & RECOMMENDATIONS")
print("=" * 80)
print("\nTop factors by |IC|:")
for col, ic, abs_ic, pval, direction, star in ic_results[:5]:
    print(f"  {col}: IC={ic:+.4f} ({direction}){' — SIGNIFICANT' if abs_ic >= 0.05 else ''}")

total_elapsed = time.time() - t_total
print(f"\n\nTotal runtime: {total_elapsed/60:.1f} minutes")
print(f"Output saved to: {OUTPUT_FILE}")
print(f"Finished: {datetime.now()}")

sys.stdout = tee.stdout
tee.close()
print(f"\nDone! Results in {OUTPUT_FILE}")
