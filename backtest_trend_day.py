"""
Trend Day Filter Research
==========================
Identify "big trend days" vs "choppy days" and test whether filtering
for high-conviction environments can solve the cost-vs-profit problem.

Phase 1: Daily feature engineering + classification
Phase 2: TrendDayFilterEngine backtest (4 modes with spread)
Phase 3: Early-session predictors (T-1 / Asian session)
Phase 4: K-Fold + year-by-year validation
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from strategies.signals import get_orb_strategy, prepare_indicators
import strategies.signals as signals_mod
from backtest import TradeRecord
from backtest_m15 import (
    load_m15, load_h1_aligned, MultiTimeframeEngine,
    calc_stats, M15_CSV_PATH, H1_CSV_PATH,
)
from backtest_round2 import Round2Engine
from backtest_advanced import (
    C12_KWARGS, RegimeEngine, prepare_indicators_custom,
)
from backtest_combo_verify import V3_REGIME, add_atr_percentile

RESULTS = {}
SPREAD = 0.50


# ══════════════════════════════════════════════════════════════
# PHASE 1: Daily Feature Engineering
# ══════════════════════════════════════════════════════════════

def compute_daily_features(h1_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate H1 bars into daily features for trend-day classification."""
    rows = []
    for date, day_bars in h1_df.groupby(h1_df.index.date):
        if len(day_bars) < 4:
            continue

        day_open = float(day_bars.iloc[0]['Open'])
        day_close = float(day_bars.iloc[-1]['Close'])
        day_high = float(day_bars['High'].max())
        day_low = float(day_bars['Low'].min())
        day_range = day_high - day_low

        if day_range < 0.01:
            continue

        avg_atr = float(day_bars['ATR'].mean())
        avg_adx = float(day_bars['ADX'].mean())
        max_adx = float(day_bars['ADX'].max())

        trend_intensity = abs(day_close - day_open) / day_range
        range_vs_atr = day_range / avg_atr if avg_atr > 0 else 1.0

        # KC breakout count: bars closing above upper or below lower KC
        kc_breaks = 0
        for _, bar in day_bars.iterrows():
            if bar['Close'] > bar['KC_upper'] or bar['Close'] < bar['KC_lower']:
                kc_breaks += 1

        # EMA alignment consistency: what fraction of bars have clear trend alignment
        ema_aligned = 0
        for _, bar in day_bars.iterrows():
            if bar['EMA9'] > bar['EMA21'] > bar['EMA100']:
                ema_aligned += 1
            elif bar['EMA9'] < bar['EMA21'] < bar['EMA100']:
                ema_aligned += 1
        ema_consistency = ema_aligned / len(day_bars)

        # Asian session (UTC 0-8) range fraction
        asian = day_bars[(day_bars.index.hour >= 0) & (day_bars.index.hour < 8)]
        if len(asian) > 0:
            asian_range = float(asian['High'].max()) - float(asian['Low'].min())
            asian_frac = asian_range / day_range if day_range > 0 else 0.5
        else:
            asian_frac = 0.0

        # MACD histogram momentum: how many bars have same-sign histogram
        hist_vals = day_bars['MACD_hist'].dropna()
        if len(hist_vals) > 0:
            macd_consistency = max(
                (hist_vals > 0).sum(), (hist_vals < 0).sum()
            ) / len(hist_vals)
        else:
            macd_consistency = 0.5

        # Direction: +1 for up day, -1 for down day
        direction = 1 if day_close > day_open else -1

        rows.append({
            'date': pd.Timestamp(date),
            'day_open': day_open,
            'day_close': day_close,
            'day_high': day_high,
            'day_low': day_low,
            'day_range': day_range,
            'avg_atr': avg_atr,
            'avg_adx': avg_adx,
            'max_adx': max_adx,
            'trend_intensity': trend_intensity,
            'range_vs_atr': range_vs_atr,
            'kc_breaks': kc_breaks,
            'ema_consistency': ema_consistency,
            'asian_frac': asian_frac,
            'macd_consistency': macd_consistency,
            'direction': direction,
            'n_bars': len(day_bars),
        })

    df = pd.DataFrame(rows)
    df.set_index('date', inplace=True)
    return df


def classify_days(daily_df: pd.DataFrame,
                  trend_ti=0.55, trend_rva=1.3, trend_adx=25,
                  choppy_ti=0.30, choppy_rva=0.8) -> pd.DataFrame:
    """Classify days into trend / normal / choppy using composite score."""
    df = daily_df.copy()

    # Composite trend score (0-1 scale)
    ti_score = df['trend_intensity'].clip(0, 1)
    rva_score = (df['range_vs_atr'] / 2.0).clip(0, 1)
    adx_score = (df['avg_adx'] / 40.0).clip(0, 1)
    kc_score = (df['kc_breaks'] / df['n_bars']).clip(0, 1)
    ema_score = df['ema_consistency']
    macd_score = df['macd_consistency']

    df['trend_score'] = (
        0.25 * ti_score +
        0.20 * rva_score +
        0.20 * adx_score +
        0.15 * kc_score +
        0.10 * ema_score +
        0.10 * macd_score
    )

    # Classify using percentile-based thresholds
    p75 = df['trend_score'].quantile(0.75)
    p25 = df['trend_score'].quantile(0.25)

    df['day_regime'] = 'normal'
    df.loc[df['trend_score'] >= p75, 'day_regime'] = 'trend'
    df.loc[df['trend_score'] <= p25, 'day_regime'] = 'choppy'

    return df


def print_classification_stats(daily_df: pd.DataFrame):
    """Print statistics about the day classification."""
    print("\n  === Day Classification Statistics ===\n")

    for regime in ['trend', 'normal', 'choppy']:
        subset = daily_df[daily_df['day_regime'] == regime]
        pct = len(subset) / len(daily_df) * 100
        print(f"  {regime.upper():>8}: {len(subset):>5} days ({pct:.1f}%)")
        print(f"           Avg trend_intensity: {subset['trend_intensity'].mean():.3f}")
        print(f"           Avg range/ATR:       {subset['range_vs_atr'].mean():.3f}")
        print(f"           Avg ADX:             {subset['avg_adx'].mean():.1f}")
        print(f"           Avg KC breaks/day:   {subset['kc_breaks'].mean():.1f}")
        print(f"           Avg EMA consistency: {subset['ema_consistency'].mean():.3f}")
        print(f"           Avg day range ($):   {subset['day_range'].mean():.2f}")
        print(f"           Avg trend score:     {subset['trend_score'].mean():.3f}")
        print()

    # Feature correlation with trend_score
    features = ['trend_intensity', 'range_vs_atr', 'avg_adx', 'kc_breaks',
                 'ema_consistency', 'macd_consistency', 'asian_frac']
    print("  Feature correlations with trend_score:")
    for f in features:
        corr = daily_df[f].corr(daily_df['trend_score'])
        print(f"    {f:<22}: {corr:>+.3f}")


# ══════════════════════════════════════════════════════════════
# PHASE 2: Trend Day Filter Engine
# ══════════════════════════════════════════════════════════════

class TrendDayFilterEngine(RegimeEngine):
    """Engine that gates entries based on daily trend regime.

    Modes:
      'all'        - trade all days (baseline)
      'skip_choppy'- skip choppy days
      'trend_only' - only trade on trend days
      'hybrid'     - trend: full size, normal: half size, choppy: skip
    """

    def __init__(self, m15_df, h1_df, daily_regime_map: Dict,
                 filter_mode='all', **kwargs):
        super().__init__(m15_df, h1_df, **kwargs)
        self.daily_regime_map = daily_regime_map
        self.filter_mode = filter_mode
        self.lot_scale = 1.0
        self.skipped_by_filter = 0

    def _get_day_regime(self, bar_time) -> str:
        day = pd.Timestamp(bar_time).normalize()
        return self.daily_regime_map.get(day, 'normal')

    def _process_signals(self, signals, bar_time, source='H1'):
        regime = self._get_day_regime(bar_time)

        if self.filter_mode == 'skip_choppy' and regime == 'choppy':
            self.skipped_by_filter += len(signals)
            return
        elif self.filter_mode == 'trend_only' and regime != 'trend':
            self.skipped_by_filter += len(signals)
            return
        elif self.filter_mode == 'hybrid':
            if regime == 'choppy':
                self.skipped_by_filter += len(signals)
                return

        super()._process_signals(signals, bar_time, source)

    def _adjust_lots_for_regime(self, lots, bar_time):
        """For hybrid mode: scale lots by regime."""
        if self.filter_mode != 'hybrid':
            return lots
        regime = self._get_day_regime(bar_time)
        if regime == 'trend':
            return round(lots * 1.5, 2)
        elif regime == 'normal':
            return round(lots * 0.7, 2)
        return lots


def run_trend_filter(m15_df, h1_df, daily_regime_map, label,
                     filter_mode='all', regime_config=None,
                     spread=SPREAD, **kwargs):
    orb = get_orb_strategy()
    orb.reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    engine = TrendDayFilterEngine(
        m15_df, h1_df,
        daily_regime_map=daily_regime_map,
        filter_mode=filter_mode,
        regime_config=regime_config,
        label=label,
        spread_cost=spread,
        **kwargs,
    )
    trades = engine.run()
    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    stats['skipped_by_filter'] = engine.skipped_by_filter
    return stats, trades


def analyze_pnl_by_regime(trades: List[TradeRecord], daily_regime_map: Dict):
    """Break down trade PnL by which day-regime the trade was entered in."""
    regime_stats = {}
    for t in trades:
        day = pd.Timestamp(t.entry_time).normalize()
        regime = daily_regime_map.get(day, 'normal')
        if regime not in regime_stats:
            regime_stats[regime] = {'n': 0, 'pnl': 0, 'wins': 0}
        regime_stats[regime]['n'] += 1
        regime_stats[regime]['pnl'] += t.pnl
        if t.pnl > 0:
            regime_stats[regime]['wins'] += 1
    return regime_stats


# ══════════════════════════════════════════════════════════════
# PHASE 3: Early-Session Predictors
# ══════════════════════════════════════════════════════════════

def build_early_predictors(h1_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build T-1 and early-session features to predict today's regime.

    These are all look-ahead-free: computed from data available
    before the European/US session opens.
    """
    pred_rows = []
    dates = sorted(daily_df.index)

    for i, today in enumerate(dates):
        if i < 3:
            continue

        today_regime = daily_df.loc[today, 'day_regime']
        today_score = daily_df.loc[today, 'trend_score']

        # T-1 features
        yday = dates[i - 1]
        yday_row = daily_df.loc[yday]
        t2 = dates[i - 2]
        t2_row = daily_df.loc[t2]
        t3 = dates[i - 3]
        t3_row = daily_df.loc[t3]

        prev_trend_score = yday_row['trend_score']
        prev_adx = yday_row['avg_adx']
        prev_range_vs_atr = yday_row['range_vs_atr']
        prev_ti = yday_row['trend_intensity']

        # ADX trend over 3 days
        adx_3d_trend = yday_row['avg_adx'] - t3_row['avg_adx']
        adx_rising = 1 if (yday_row['avg_adx'] > t2_row['avg_adx'] > t3_row['avg_adx']) else 0

        # Trend score momentum
        score_3d_avg = (yday_row['trend_score'] + t2_row['trend_score'] + t3_row['trend_score']) / 3

        # Asian session features for today
        today_date = pd.Timestamp(today)
        asian_start = today_date.tz_localize('UTC')
        asian_end = asian_start + pd.Timedelta(hours=8)
        asian_bars = h1_df[(h1_df.index >= asian_start) & (h1_df.index < asian_end)]

        if len(asian_bars) >= 2:
            asian_range = float(asian_bars['High'].max()) - float(asian_bars['Low'].min())
            asian_atr = float(asian_bars['ATR'].mean())
            asian_range_vs_atr = asian_range / asian_atr if asian_atr > 0 else 1.0
            asian_adx = float(asian_bars['ADX'].mean())
            first_bar = asian_bars.iloc[0]
            gap = abs(float(first_bar['Open']) - yday_row['day_close'])
            gap_vs_atr = gap / asian_atr if asian_atr > 0 else 0
        else:
            asian_range_vs_atr = 1.0
            asian_adx = 20.0
            gap_vs_atr = 0

        pred_rows.append({
            'date': today,
            'actual_regime': today_regime,
            'actual_score': today_score,
            'prev_trend_score': prev_trend_score,
            'prev_adx': prev_adx,
            'prev_range_vs_atr': prev_range_vs_atr,
            'prev_ti': prev_ti,
            'adx_3d_trend': adx_3d_trend,
            'adx_rising': adx_rising,
            'score_3d_avg': score_3d_avg,
            'asian_range_vs_atr': asian_range_vs_atr,
            'asian_adx': asian_adx,
            'gap_vs_atr': gap_vs_atr,
        })

    return pd.DataFrame(pred_rows).set_index('date')


def evaluate_simple_predictors(pred_df: pd.DataFrame):
    """Test simple threshold-based predictors for trend-day identification."""
    results = []

    # Predictor 1: Previous day's trend score > threshold
    for thresh in [0.45, 0.50, 0.55, 0.60]:
        pred = pred_df['prev_trend_score'] >= thresh
        actual_trend = pred_df['actual_regime'] == 'trend'
        actual_choppy = pred_df['actual_regime'] == 'choppy'

        tp = (pred & actual_trend).sum()
        fp = (pred & ~actual_trend).sum()
        fn = (~pred & actual_trend).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # How often does predicting "not trend" correctly avoid choppy days?
        skip_pred = ~pred
        choppy_avoided = (skip_pred & actual_choppy).sum()
        choppy_total = actual_choppy.sum()
        avoid_rate = choppy_avoided / choppy_total if choppy_total > 0 else 0

        results.append({
            'predictor': f'prev_score >= {thresh}',
            'predicted_trend': int(pred.sum()),
            'precision': precision,
            'recall': recall,
            'choppy_avoid_rate': avoid_rate,
        })

    # Predictor 2: ADX rising over 3 days
    pred = pred_df['adx_rising'] == 1
    actual_trend = pred_df['actual_regime'] == 'trend'
    actual_choppy = pred_df['actual_regime'] == 'choppy'
    tp = (pred & actual_trend).sum()
    fp = (pred & ~actual_trend).sum()
    fn = (~pred & actual_trend).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    skip_pred = ~pred
    choppy_avoided = (skip_pred & actual_choppy).sum()
    avoid_rate = choppy_avoided / actual_choppy.sum() if actual_choppy.sum() > 0 else 0
    results.append({
        'predictor': 'adx_rising_3d',
        'predicted_trend': int(pred.sum()),
        'precision': precision,
        'recall': recall,
        'choppy_avoid_rate': avoid_rate,
    })

    # Predictor 3: Asian session range/ATR > threshold
    for thresh in [1.0, 1.2, 1.5]:
        pred = pred_df['asian_range_vs_atr'] >= thresh
        tp = (pred & actual_trend).sum()
        fp = (pred & ~actual_trend).sum()
        fn = (~pred & actual_trend).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        skip_pred = ~pred
        choppy_avoided = (skip_pred & actual_choppy).sum()
        avoid_rate = choppy_avoided / actual_choppy.sum() if actual_choppy.sum() > 0 else 0
        results.append({
            'predictor': f'asian_rva >= {thresh}',
            'predicted_trend': int(pred.sum()),
            'precision': precision,
            'recall': recall,
            'choppy_avoid_rate': avoid_rate,
        })

    # Predictor 4: 3-day avg trend score > threshold
    for thresh in [0.40, 0.45, 0.50]:
        pred = pred_df['score_3d_avg'] >= thresh
        tp = (pred & actual_trend).sum()
        fp = (pred & ~actual_trend).sum()
        fn = (~pred & actual_trend).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        skip_pred = ~pred
        choppy_avoided = (skip_pred & actual_choppy).sum()
        avoid_rate = choppy_avoided / actual_choppy.sum() if actual_choppy.sum() > 0 else 0
        results.append({
            'predictor': f'score_3d_avg >= {thresh}',
            'predicted_trend': int(pred.sum()),
            'precision': precision,
            'recall': recall,
            'choppy_avoid_rate': avoid_rate,
        })

    # Predictor 5: Overnight gap > threshold
    for thresh in [0.3, 0.5, 1.0]:
        pred = pred_df['gap_vs_atr'] >= thresh
        tp = (pred & actual_trend).sum()
        fp = (pred & ~actual_trend).sum()
        fn = (~pred & actual_trend).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        skip_pred = ~pred
        choppy_avoided = (skip_pred & actual_choppy).sum()
        avoid_rate = choppy_avoided / actual_choppy.sum() if actual_choppy.sum() > 0 else 0
        results.append({
            'predictor': f'gap_vs_atr >= {thresh}',
            'predicted_trend': int(pred.sum()),
            'precision': precision,
            'recall': recall,
            'choppy_avoid_rate': avoid_rate,
        })

    # Composite: prev_score + asian_rva
    for s_th, a_th in [(0.45, 1.0), (0.50, 1.2), (0.45, 1.2)]:
        pred = (pred_df['prev_trend_score'] >= s_th) & (pred_df['asian_range_vs_atr'] >= a_th)
        tp = (pred & actual_trend).sum()
        fp = (pred & ~actual_trend).sum()
        fn = (~pred & actual_trend).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        skip_pred = ~pred
        choppy_avoided = (skip_pred & actual_choppy).sum()
        avoid_rate = choppy_avoided / actual_choppy.sum() if actual_choppy.sum() > 0 else 0
        results.append({
            'predictor': f'prev_score>={s_th} & asian_rva>={a_th}',
            'predicted_trend': int(pred.sum()),
            'precision': precision,
            'recall': recall,
            'choppy_avoid_rate': avoid_rate,
        })

    return results


def build_predicted_regime_map(pred_df: pd.DataFrame, daily_df: pd.DataFrame,
                                predictor_fn) -> Dict:
    """Build a regime map using early-session prediction instead of hindsight.

    predictor_fn: takes a pred_df row and returns 'trend', 'normal', or 'choppy'
    For days not in pred_df, fall back to daily_df classification.
    """
    regime_map = {}
    for date, row in daily_df.iterrows():
        day = pd.Timestamp(date).normalize()
        if day.tzinfo is None:
            day = day.tz_localize('UTC')
        regime_map[day] = row['day_regime']

    for date, row in pred_df.iterrows():
        day = pd.Timestamp(date).normalize()
        if day.tzinfo is None:
            day = day.tz_localize('UTC')
        regime_map[day] = predictor_fn(row)

    return regime_map


# ══════════════════════════════════════════════════════════════
# Printing utilities
# ══════════════════════════════════════════════════════════════

def print_table(results, title=""):
    if title:
        print(f"\n  --- {title} ---")
    print(f"\n  {'Rank':<5} {'Config':<50} {'N':>6} {'Sharpe':>8} {'PnL':>10} "
          f"{'MaxDD':>8} {'DD%':>6} {'WR%':>6} {'$/trade':>8}")
    print(f"  {'-'*5} {'-'*50} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")
    for rank, r in enumerate(results, 1):
        ppt = r['pnl'] / r['n'] if r['n'] > 0 else 0
        print(f"  {rank:<5} {r['label']:<50} {r['n']:>6} {r['sharpe']:>8.2f} "
              f"${r['pnl']:>9.0f} ${r['dd']:>7.0f} {r['dd_pct']:>5.1f}% "
              f"{r['wr']:>5.1f}% ${ppt:>7.2f}")


def stats_to_row(stats, label=None):
    return {
        'label': label or stats.get('label', ''),
        'n': int(stats['n']),
        'sharpe': float(stats['sharpe']),
        'pnl': float(stats['total_pnl']),
        'wr': float(stats['win_rate']),
        'rr': float(stats['rr']),
        'dd': float(stats['max_dd']),
        'dd_pct': float(stats['max_dd_pct']),
        'skipped': int(stats.get('skipped_by_filter', 0)),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 100)
    print("  TREND DAY FILTER RESEARCH")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Spread: ${SPREAD}")
    print("=" * 100)

    # ── Load data ──
    print("\n  Loading data...", flush=True)
    m15_raw = load_m15()
    m15_raw = m15_raw[m15_raw.index >= pd.Timestamp('2015-01-01', tz='UTC')]
    h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

    m15_custom = prepare_indicators_custom(m15_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = prepare_indicators_custom(h1_raw, kc_ema=30, kc_mult=1.25)
    h1_custom = add_atr_percentile(h1_custom)

    m15_default = prepare_indicators(m15_raw)
    h1_default = prepare_indicators(h1_raw)
    h1_default = add_atr_percentile(h1_default)

    print(f"  M15: {len(m15_custom)} bars, H1: {len(h1_custom)} bars\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Daily Feature Engineering & Classification
    # ══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  PHASE 1: Daily Feature Engineering & Day Classification")
    print("=" * 100)

    t0 = time.time()
    daily_df = compute_daily_features(h1_custom)
    daily_df = classify_days(daily_df)
    print(f"\n  Computed features for {len(daily_df)} trading days ({time.time()-t0:.1f}s)")

    print_classification_stats(daily_df)

    # Build regime lookup (date -> regime string)
    daily_regime_map = {}
    for date, row in daily_df.iterrows():
        day = pd.Timestamp(date).normalize()
        if day.tzinfo is None:
            day = day.tz_localize('UTC')
        daily_regime_map[day] = row['day_regime']

    # Analyze what the existing Combo strategy earns on each regime type
    print("\n  Running Combo baseline to analyze PnL by day regime...")
    baseline_stats, baseline_trades = run_trend_filter(
        m15_custom, h1_custom, daily_regime_map,
        "A: Combo Baseline (all days)",
        filter_mode='all',
        regime_config=V3_REGIME,
        spread=SPREAD,
        **C12_KWARGS,
    )
    regime_pnl = analyze_pnl_by_regime(baseline_trades, daily_regime_map)

    print("\n  === PnL Breakdown by Day Regime (Combo, $0.50 spread) ===")
    print(f"  {'Regime':<10} {'Trades':>7} {'PnL':>10} {'$/trade':>8} {'WR%':>6}")
    print(f"  {'-'*10} {'-'*7} {'-'*10} {'-'*8} {'-'*6}")
    for regime in ['trend', 'normal', 'choppy']:
        r = regime_pnl.get(regime, {'n': 0, 'pnl': 0, 'wins': 0})
        ppt = r['pnl'] / r['n'] if r['n'] > 0 else 0
        wr = r['wins'] / r['n'] * 100 if r['n'] > 0 else 0
        print(f"  {regime:<10} {r['n']:>7} ${r['pnl']:>9.0f} ${ppt:>7.2f} {wr:>5.1f}%")
    total_n = sum(r['n'] for r in regime_pnl.values())
    total_pnl = sum(r['pnl'] for r in regime_pnl.values())
    print(f"  {'TOTAL':<10} {total_n:>7} ${total_pnl:>9.0f}")

    RESULTS['phase1'] = {
        'n_days': len(daily_df),
        'trend_days': int((daily_df['day_regime'] == 'trend').sum()),
        'normal_days': int((daily_df['day_regime'] == 'normal').sum()),
        'choppy_days': int((daily_df['day_regime'] == 'choppy').sum()),
        'regime_pnl': {k: {'n': v['n'], 'pnl': round(v['pnl'], 2), 'wins': v['wins']}
                       for k, v in regime_pnl.items()},
        'avg_trend_score_trend': float(daily_df[daily_df['day_regime'] == 'trend']['trend_score'].mean()),
        'avg_trend_score_normal': float(daily_df[daily_df['day_regime'] == 'normal']['trend_score'].mean()),
        'avg_trend_score_choppy': float(daily_df[daily_df['day_regime'] == 'choppy']['trend_score'].mean()),
    }

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Filtered Backtest (4 modes)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  PHASE 2: Trend Day Filter Backtest (4 modes, spread=${SPREAD})")
    print(f"  Using hindsight classification (oracle) to measure upper bound")
    print(f"{'='*100}")

    modes = [
        ('all', 'A: All days (baseline)'),
        ('skip_choppy', 'B: Skip choppy days'),
        ('trend_only', 'C: Trend days only'),
        ('hybrid', 'D: Hybrid (trend 1.5x, normal 0.7x, skip choppy)'),
    ]

    phase2_results = []
    phase2_regime_pnl = {}

    for mode, label in modes:
        print(f"\n  [{label}]", flush=True)
        t0 = time.time()
        stats, trades = run_trend_filter(
            m15_custom, h1_custom, daily_regime_map,
            label, filter_mode=mode,
            regime_config=V3_REGIME,
            spread=SPREAD,
            **C12_KWARGS,
        )
        elapsed = time.time() - t0
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"    N={stats['n']}, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
              f"$/trade={ppt:.2f}, skipped={stats.get('skipped_by_filter', 0)}, {elapsed:.0f}s")
        row = stats_to_row(stats, label)
        phase2_results.append(row)

        rpnl = analyze_pnl_by_regime(trades, daily_regime_map)
        phase2_regime_pnl[mode] = rpnl

    print_table(phase2_results, "Phase 2: Trend Day Filter Comparison (Oracle)")

    # Also test with C12 defaults (no KC change) to ensure filter effect isn't config-specific
    print(f"\n  --- Also testing with C12 default config ---")
    for mode, label in [('all', 'E: C12 All days'), ('skip_choppy', 'F: C12 Skip choppy'),
                         ('trend_only', 'G: C12 Trend only')]:
        label_c12 = label
        stats, trades = run_trend_filter(
            m15_default, h1_default, daily_regime_map,
            label_c12, filter_mode=mode,
            spread=SPREAD,
            **C12_KWARGS,
        )
        ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
        print(f"    {label_c12:<45} N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
              f"PnL=${stats['total_pnl']:.0f}, $/trade={ppt:.2f}")
        phase2_results.append(stats_to_row(stats, label_c12))

    RESULTS['phase2'] = phase2_results

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Early-Session Predictors
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  PHASE 3: Early-Session Predictors (Look-Ahead Free)")
    print(f"{'='*100}")

    t0 = time.time()
    pred_df = build_early_predictors(h1_custom, daily_df)
    print(f"\n  Built predictors for {len(pred_df)} days ({time.time()-t0:.1f}s)")

    predictor_results = evaluate_simple_predictors(pred_df)

    print(f"\n  === Predictor Evaluation ===")
    print(f"  {'Predictor':<40} {'Pred#':>6} {'Prec':>6} {'Recall':>7} {'ChopAvoid':>10}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")
    for r in predictor_results:
        print(f"  {r['predictor']:<40} {r['predicted_trend']:>6} {r['precision']:>6.1%} "
              f"{r['recall']:>6.1%} {r['choppy_avoid_rate']:>9.1%}")

    RESULTS['phase3_predictors'] = predictor_results

    # Test the best look-ahead-free predictor in a backtest
    # Use "skip_choppy" mode with predicted regime map
    print(f"\n  --- Backtest with predicted regimes (look-ahead-free) ---")

    def predictor_score_based(row):
        """Predict regime using prev_trend_score + asian_range_vs_atr."""
        if row['prev_trend_score'] >= 0.50 and row['asian_range_vs_atr'] >= 1.0:
            return 'trend'
        elif row['prev_trend_score'] < 0.35 and row['asian_range_vs_atr'] < 0.8:
            return 'choppy'
        return 'normal'

    def predictor_simple(row):
        """Simple: prev_trend_score threshold only."""
        if row['prev_trend_score'] >= 0.55:
            return 'trend'
        elif row['prev_trend_score'] < 0.35:
            return 'choppy'
        return 'normal'

    predictors = [
        ('P1: score+asian composite', predictor_score_based),
        ('P2: prev_score only', predictor_simple),
    ]

    phase3_bt_results = []
    for pred_name, pred_fn in predictors:
        pred_regime_map = build_predicted_regime_map(pred_df, daily_df, pred_fn)

        for mode, suffix in [('skip_choppy', 'skip choppy'), ('trend_only', 'trend only')]:
            label = f"{pred_name} [{suffix}]"
            stats, trades = run_trend_filter(
                m15_custom, h1_custom, pred_regime_map,
                label, filter_mode=mode,
                regime_config=V3_REGIME,
                spread=SPREAD,
                **C12_KWARGS,
            )
            ppt = stats['total_pnl'] / stats['n'] if stats['n'] > 0 else 0
            print(f"    {label:<55} N={stats['n']}, Sharpe={stats['sharpe']:.2f}, "
                  f"PnL=${stats['total_pnl']:.0f}, $/trade={ppt:.2f}")
            phase3_bt_results.append(stats_to_row(stats, label))

    # Add oracle baseline for comparison
    phase3_bt_results.insert(0, phase2_results[0])  # 'A: All days'

    print_table(phase3_bt_results, "Phase 3: Predicted vs Oracle vs Baseline")
    RESULTS['phase3_backtest'] = phase3_bt_results

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: K-Fold + Year-by-Year Validation
    # ══════════════════════════════════════════════════════════════
    # Find the best mode from Phase 2 (oracle) and Phase 3 (predicted)
    all_candidates = phase2_results[:4] + phase3_bt_results[1:]
    best_candidate = max(all_candidates, key=lambda x: x['sharpe'])
    print(f"\n{'='*100}")
    print(f"  PHASE 4: K-Fold + Year-by-Year Validation")
    print(f"  Best candidate: {best_candidate['label']} (Sharpe={best_candidate['sharpe']:.2f})")
    print(f"{'='*100}")

    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ]

    # Determine best mode and whether it uses predictions
    is_oracle = best_candidate['label'].startswith(('A:', 'B:', 'C:', 'D:'))
    if 'skip choppy' in best_candidate['label'] or best_candidate['label'].startswith('B:'):
        best_mode = 'skip_choppy'
    elif 'trend only' in best_candidate['label'] or best_candidate['label'].startswith('C:'):
        best_mode = 'trend_only'
    elif best_candidate['label'].startswith('D:'):
        best_mode = 'hybrid'
    else:
        best_mode = 'skip_choppy'

    use_predictor = not is_oracle
    pred_fn = predictor_score_based  # default

    if not is_oracle:
        if 'P2' in best_candidate['label']:
            pred_fn = predictor_simple

    fold_results = []
    for fold_name, test_start, test_end in folds:
        ts = pd.Timestamp(test_start, tz='UTC')
        te = pd.Timestamp(test_end, tz='UTC')
        print(f"\n  {fold_name}: {test_start} ~ {test_end}", flush=True)

        m15_w = m15_custom[(m15_custom.index >= ts) & (m15_custom.index < te)]
        h1_w = h1_custom[(h1_custom.index >= ts) & (h1_custom.index < te)]
        if len(m15_w) < 1000:
            continue

        # Build fold-specific daily features and regime map
        fold_daily = compute_daily_features(h1_w)
        fold_daily = classify_days(fold_daily)
        fold_regime_map = {}
        for date, row in fold_daily.iterrows():
            day = pd.Timestamp(date).normalize()
            if day.tzinfo is None:
                day = day.tz_localize('UTC')
            fold_regime_map[day] = row['day_regime']

        if use_predictor:
            fold_pred = build_early_predictors(h1_w, fold_daily)
            fold_regime_map = build_predicted_regime_map(fold_pred, fold_daily, pred_fn)

        # Best candidate
        stats_best, _ = run_trend_filter(
            m15_w, h1_w, fold_regime_map,
            f"Best [{fold_name}]",
            filter_mode=best_mode,
            regime_config=V3_REGIME,
            spread=SPREAD,
            **C12_KWARGS,
        )

        # Baseline (all days)
        stats_base, _ = run_trend_filter(
            m15_w, h1_w, fold_regime_map,
            f"Baseline [{fold_name}]",
            filter_mode='all',
            regime_config=V3_REGIME,
            spread=SPREAD,
            **C12_KWARGS,
        )

        fold_results.append({
            'fold': fold_name,
            'best_sharpe': float(stats_best['sharpe']),
            'best_pnl': float(stats_best['total_pnl']),
            'best_n': int(stats_best['n']),
            'base_sharpe': float(stats_base['sharpe']),
            'base_pnl': float(stats_base['total_pnl']),
            'base_n': int(stats_base['n']),
        })

    print(f"\n  {'Fold':<8} {'Best Sh':>8} {'Best PnL':>10} {'Best N':>7} "
          f"{'Base Sh':>8} {'Base PnL':>10} {'Base N':>7} {'Winner':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*7} {'-'*8}")
    best_wins = 0
    for r in fold_results:
        w = "Best" if r['best_sharpe'] > r['base_sharpe'] else "Base"
        if w == "Best":
            best_wins += 1
        print(f"  {r['fold']:<8} {r['best_sharpe']:>8.2f} ${r['best_pnl']:>9.0f} {r['best_n']:>7} "
              f"{r['base_sharpe']:>8.2f} ${r['base_pnl']:>9.0f} {r['base_n']:>7} {w:>8}")

    if fold_results:
        avg_best = np.mean([r['best_sharpe'] for r in fold_results])
        avg_base = np.mean([r['base_sharpe'] for r in fold_results])
        print(f"\n  Avg Sharpe: Best={avg_best:.2f}, Baseline={avg_base:.2f}")
        print(f"  Best wins {best_wins}/{len(fold_results)} folds")

    RESULTS['phase4_kfold'] = fold_results

    # Year-by-year analysis
    print(f"\n  --- Year-by-Year ---")
    year_results = []
    years = sorted(daily_df.index.year.unique())
    for year in years:
        ys = pd.Timestamp(f'{year}-01-01', tz='UTC')
        ye = pd.Timestamp(f'{year+1}-01-01', tz='UTC')

        m15_y = m15_custom[(m15_custom.index >= ys) & (m15_custom.index < ye)]
        h1_y = h1_custom[(h1_custom.index >= ys) & (h1_custom.index < ye)]
        if len(m15_y) < 500:
            continue

        y_daily = daily_df[(daily_df.index >= ys.tz_localize(None)) & (daily_df.index < ye.tz_localize(None))]
        if len(y_daily) < 20:
            continue
        y_daily = classify_days(y_daily)
        y_regime = {}
        for date, row in y_daily.iterrows():
            day = pd.Timestamp(date).normalize()
            if day.tzinfo is None:
                day = day.tz_localize('UTC')
            y_regime[day] = row['day_regime']

        stats_best_y, _ = run_trend_filter(
            m15_y, h1_y, y_regime,
            f"Best {year}", filter_mode=best_mode,
            regime_config=V3_REGIME, spread=SPREAD, **C12_KWARGS,
        )
        stats_base_y, _ = run_trend_filter(
            m15_y, h1_y, y_regime,
            f"Base {year}", filter_mode='all',
            regime_config=V3_REGIME, spread=SPREAD, **C12_KWARGS,
        )
        year_results.append({
            'year': year,
            'best_sharpe': float(stats_best_y['sharpe']),
            'best_pnl': float(stats_best_y['total_pnl']),
            'best_n': int(stats_best_y['n']),
            'base_sharpe': float(stats_base_y['sharpe']),
            'base_pnl': float(stats_base_y['total_pnl']),
            'base_n': int(stats_base_y['n']),
        })

    print(f"  {'Year':<6} {'Best Sh':>8} {'Best PnL':>10} {'Best N':>7} "
          f"{'Base Sh':>8} {'Base PnL':>10} {'Base N':>7}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*7} {'-'*8} {'-'*10} {'-'*7}")
    for r in year_results:
        print(f"  {r['year']:<6} {r['best_sharpe']:>8.2f} ${r['best_pnl']:>9.0f} {r['best_n']:>7} "
              f"{r['base_sharpe']:>8.2f} ${r['base_pnl']:>9.0f} {r['base_n']:>7}")

    RESULTS['phase4_yearly'] = year_results

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*100}")

    print(f"\n  Phase 1: Day Classification")
    p1 = RESULTS['phase1']
    print(f"    {p1['n_days']} total days: {p1['trend_days']} trend, "
          f"{p1['normal_days']} normal, {p1['choppy_days']} choppy")
    for regime, data in p1['regime_pnl'].items():
        ppt = data['pnl'] / data['n'] if data['n'] > 0 else 0
        print(f"    {regime}: {data['n']} trades, PnL=${data['pnl']:.0f}, $/trade=${ppt:.2f}")

    print(f"\n  Phase 2: Filter Modes (Oracle, spread=${SPREAD})")
    for r in phase2_results[:4]:
        print(f"    {r['label']:<50} Sharpe={r['sharpe']:.2f}, N={r['n']}, PnL=${r['pnl']:.0f}")

    print(f"\n  Phase 3: Predicted Regime Backtest")
    for r in phase3_bt_results[1:]:
        print(f"    {r['label']:<55} Sharpe={r['sharpe']:.2f}")

    print(f"\n  Phase 4: K-Fold Validation")
    if fold_results:
        print(f"    Best wins {best_wins}/{len(fold_results)} folds")
        print(f"    Avg Best Sharpe: {avg_best:.2f} vs Avg Baseline: {avg_base:.2f}")

    # Save results
    out = Path("data/trend_day_results.json")
    out.write_text(json.dumps(RESULTS, indent=2, default=str), encoding='utf-8')
    print(f"\n  Results saved to {out}")

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"  TOTAL: {elapsed/60:.0f} min ({elapsed/3600:.1f}h)")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
