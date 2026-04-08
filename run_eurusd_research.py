"""
EUR/USD 第二量化品类研究
========================
综合脚本: 数据下载 + ATR 研究 + 策略回测 + K-Fold 验证

EMX Pro 条件:
  Symbol: EURUSD.mx
  Contract: 100,000 EUR per standard lot
  Min lot: 0.01 (= 1,000 EUR = ~$1,000 notional)
  Spread: 1.8 pips (standard) = 0.00018
  Leverage: 1:400
  Pip value: 0.01 lot = $0.10/pip, 0.1 lot = $1.00/pip

Forex pip value: 1 pip = 0.0001 price change
  1 standard lot (100k): 1 pip = $10
  0.01 lot (1k): 1 pip = $0.10
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════
# PART 0: Download EUR/USD data via yfinance
# ═══════════════════════════════════════════════════════════════

def download_eurusd_data():
    """Download EURUSD H1 and M15 data from Dukascopy-style CSV or yfinance."""
    import yfinance as yf

    data_dir = Path("data/download")
    data_dir.mkdir(parents=True, exist_ok=True)

    h1_path = data_dir / "eurusd-h1-2019-01-01-2026-04-05.csv"
    m15_path = data_dir / "eurusd-m15-2022-01-01-2026-04-05.csv"

    # yfinance max for 1h = 730 days, for 15m = 60 days
    # We need to download in chunks for longer history

    print("=" * 80)
    print("PART 0: DOWNLOADING EUR/USD DATA")
    print("=" * 80)

    # --- H1 data: download in ~2-year chunks ---
    if h1_path.exists():
        df_check = pd.read_csv(str(h1_path))
        print(f"\n  H1 data already exists: {h1_path} ({len(df_check)} bars)")
    else:
        print("\n  Downloading EUR/USD H1 data...")
        ticker = yf.Ticker("EURUSD=X")
        chunks = []
        # yfinance allows max 730 days per request for hourly
        periods = [
            ("2019-01-01", "2020-12-31"),
            ("2021-01-01", "2022-12-31"),
            ("2023-01-01", "2024-06-30"),
            ("2024-07-01", "2026-04-06"),
        ]
        for start, end in periods:
            print(f"    {start} -> {end}...", end='', flush=True)
            try:
                df = ticker.history(start=start, end=end, interval="1h")
                if len(df) > 0:
                    chunks.append(df)
                    print(f" {len(df)} bars")
                else:
                    print(" 0 bars (skipped)")
            except Exception as e:
                print(f" ERROR: {e}")

        if chunks:
            h1_df = pd.concat(chunks)
            h1_df = h1_df[~h1_df.index.duplicated(keep='first')]
            h1_df.sort_index(inplace=True)
            # Convert to our standard CSV format
            h1_out = pd.DataFrame({
                'timestamp': (h1_df.index.astype(np.int64) // 10**6).astype(int),
                'open': h1_df['Open'].values,
                'high': h1_df['High'].values,
                'low': h1_df['Low'].values,
                'close': h1_df['Close'].values,
                'volume': h1_df['Volume'].values if 'Volume' in h1_df.columns else 0,
            })
            h1_out.to_csv(str(h1_path), index=False)
            print(f"  H1 saved: {h1_path} ({len(h1_out)} bars, {h1_df.index[0]} -> {h1_df.index[-1]})")
        else:
            print("  ERROR: No H1 data downloaded!")
            return None, None

    # --- M15 data: download in 60-day chunks ---
    if m15_path.exists():
        df_check = pd.read_csv(str(m15_path))
        print(f"  M15 data already exists: {m15_path} ({len(df_check)} bars)")
    else:
        print("\n  Downloading EUR/USD M15 data...")
        ticker = yf.Ticker("EURUSD=X")
        chunks = []
        # For M15 we can only get ~60 days per request
        import datetime as dt
        start_date = dt.date(2022, 1, 1)
        end_date = dt.date(2026, 4, 6)
        current = start_date
        while current < end_date:
            chunk_end = min(current + dt.timedelta(days=58), end_date)
            start_str = current.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")
            print(f"    {start_str} -> {end_str}...", end='', flush=True)
            try:
                df = ticker.history(start=start_str, end=end_str, interval="15m")
                if len(df) > 0:
                    chunks.append(df)
                    print(f" {len(df)} bars")
                else:
                    print(" 0 bars")
            except Exception as e:
                print(f" ERROR: {e}")
            current = chunk_end
            time.sleep(0.5)

        if chunks:
            m15_df = pd.concat(chunks)
            m15_df = m15_df[~m15_df.index.duplicated(keep='first')]
            m15_df.sort_index(inplace=True)
            m15_out = pd.DataFrame({
                'timestamp': (m15_df.index.astype(np.int64) // 10**6).astype(int),
                'open': m15_df['Open'].values,
                'high': m15_df['High'].values,
                'low': m15_df['Low'].values,
                'close': m15_df['Close'].values,
                'volume': m15_df['Volume'].values if 'Volume' in m15_df.columns else 0,
            })
            m15_out.to_csv(str(m15_path), index=False)
            print(f"  M15 saved: {m15_path} ({len(m15_out)} bars, {m15_df.index[0]} -> {m15_df.index[-1]})")
        else:
            print("  ERROR: No M15 data downloaded!")
            return None, None

    return h1_path, m15_path


# ═══════════════════════════════════════════════════════════════
# PART 1: Load and prepare data
# ═══════════════════════════════════════════════════════════════

def load_forex_csv(path, start=None, end=None):
    """Load CSV with timestamp(ms) format."""
    df = pd.read_csv(str(path))
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df['is_flat'] = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])
    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]
    return df


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_adx(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    plus_dm = high.diff()
    minus_dm = low.diff().apply(lambda x: -x)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(span=period, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(span=period, min_periods=period).mean()


def prepare_forex_indicators(df, kc_ema=25, kc_mult=1.2, ema_trend=100):
    """Calculate technical indicators for EUR/USD."""
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA100'] = df['Close'].ewm(span=ema_trend).mean()
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['KC_mid'] = df['Close'].ewm(span=kc_ema).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['RSI14'] = calc_rsi(df['Close'], 14)
    df['ADX'] = calc_adx(df, 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['atr_percentile'] = df['ATR'].rolling(500, min_periods=50).rank(pct=True).fillna(0.5)
    return df


# ═══════════════════════════════════════════════════════════════
# PART 2: ATR Study — EUR/USD vs XAUUSD
# ═══════════════════════════════════════════════════════════════

def atr_study(h1_df):
    """Analyze ATR distribution and characteristics."""
    print("\n" + "=" * 80)
    print("PART 2: ATR STUDY — EUR/USD CHARACTERISTICS")
    print("=" * 80)

    atr = h1_df['ATR'].dropna()
    atr_pips = atr * 10000  # Convert to pips

    print(f"\n  H1 ATR Distribution (pips):")
    print(f"    Mean:   {atr_pips.mean():.1f} pips")
    print(f"    Median: {atr_pips.median():.1f} pips")
    print(f"    Std:    {atr_pips.std():.1f} pips")
    print(f"    P10:    {atr_pips.quantile(0.10):.1f} pips")
    print(f"    P25:    {atr_pips.quantile(0.25):.1f} pips")
    print(f"    P75:    {atr_pips.quantile(0.75):.1f} pips")
    print(f"    P90:    {atr_pips.quantile(0.90):.1f} pips")
    print(f"    Min:    {atr_pips.min():.1f} pips")
    print(f"    Max:    {atr_pips.max():.1f} pips")

    # Daily range analysis
    daily = h1_df.resample('D').agg({'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    daily['range_pips'] = (daily['High'] - daily['Low']) * 10000
    daily = daily[daily['range_pips'] > 0]

    print(f"\n  Daily Range Distribution (pips):")
    print(f"    Mean:   {daily['range_pips'].mean():.0f} pips")
    print(f"    Median: {daily['range_pips'].median():.0f} pips")
    print(f"    P25:    {daily['range_pips'].quantile(0.25):.0f} pips")
    print(f"    P75:    {daily['range_pips'].quantile(0.75):.0f} pips")
    print(f"    P90:    {daily['range_pips'].quantile(0.90):.0f} pips")

    # Session analysis
    print(f"\n  ATR by Session (H1, pips):")
    sessions = {
        'Asia (0-8 UTC)': h1_df[h1_df.index.hour.isin(range(0, 8))],
        'London (8-14 UTC)': h1_df[h1_df.index.hour.isin(range(8, 14))],
        'NY (14-21 UTC)': h1_df[h1_df.index.hour.isin(range(14, 21))],
        'Late (21-24 UTC)': h1_df[h1_df.index.hour.isin(range(21, 24))],
    }
    for name, sdf in sessions.items():
        satr = sdf['ATR'].dropna() * 10000
        if len(satr) > 0:
            print(f"    {name:25s}: mean={satr.mean():.1f}, median={satr.median():.1f}, bars={len(satr)}")

    # ADX distribution
    adx = h1_df['ADX'].dropna()
    print(f"\n  ADX Distribution:")
    print(f"    Mean:   {adx.mean():.1f}")
    print(f"    Median: {adx.median():.1f}")
    print(f"    P25:    {adx.quantile(0.25):.1f}")
    print(f"    P75:    {adx.quantile(0.75):.1f}")
    print(f"    ADX>25: {(adx > 25).mean()*100:.1f}%")
    print(f"    ADX>18: {(adx > 18).mean()*100:.1f}%")

    # Compare with XAUUSD reference values
    print(f"\n  Comparison with XAUUSD (reference):")
    print(f"    XAUUSD H1 ATR: ~$20-25 (in dollars)")
    print(f"    EURUSD H1 ATR: ~{atr_pips.mean():.0f} pips (in pips)")
    print(f"    XAUUSD daily range: ~$30-50")
    print(f"    EURUSD daily range: ~{daily['range_pips'].mean():.0f} pips")
    print(f"    Key difference: EURUSD is 5-10x less volatile than gold per unit")

    # Risk calculation
    print(f"\n  Position Sizing for $2,000 account:")
    mean_atr_pips = atr_pips.mean()
    sl_pips = mean_atr_pips * 4.5  # Same SL ATR mult as gold
    print(f"    SL = 4.5 x ATR = {sl_pips:.0f} pips")
    print(f"    $50 risk / {sl_pips:.0f} pips = {50 / (sl_pips * 0.10):.2f} mini lots (0.01)")
    print(f"    At 0.01 lot: SL risk = {sl_pips * 0.10:.1f} USD")
    print(f"    At 0.05 lot: SL risk = {sl_pips * 0.50:.1f} USD")
    print(f"    At 0.10 lot: SL risk = {sl_pips * 1.00:.1f} USD")

    return {
        'mean_atr_pips': float(atr_pips.mean()),
        'median_atr_pips': float(atr_pips.median()),
        'daily_range_mean': float(daily['range_pips'].mean()),
        'adx_mean': float(adx.mean()),
        'adx_pct_above_18': float((adx > 18).mean()),
    }


# ═══════════════════════════════════════════════════════════════
# PART 3: Simplified Forex Backtest Engine
# ═══════════════════════════════════════════════════════════════

FOREX_PIP = 0.0001
FOREX_POINT_VALUE = {
    # pip_value per standard lot for major pairs
    'EURUSD': 10.0,   # $10 per pip per standard lot
}

class ForexPosition:
    def __init__(self, strategy, direction, entry_price, entry_time, lots,
                 sl_distance, tp_distance, entry_atr=0):
        self.strategy = strategy
        self.direction = direction
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.lots = lots
        self.sl_distance = sl_distance
        self.tp_distance = tp_distance
        self.entry_atr = entry_atr
        self.bars_held = 0
        self.extreme_price = entry_price
        self.trailing_stop_price = 0.0

        if direction == 'BUY':
            self.sl_price = entry_price - sl_distance
            self.tp_price = entry_price + tp_distance
        else:
            self.sl_price = entry_price + sl_distance
            self.tp_price = entry_price - tp_distance


class ForexTrade:
    def __init__(self, strategy, direction, entry_price, exit_price,
                 entry_time, exit_time, lots, pnl, exit_reason, bars_held):
        self.strategy = strategy
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.lots = lots
        self.pnl = pnl
        self.exit_reason = exit_reason
        self.bars_held = bars_held


class ForexBacktestEngine:
    """Simplified backtest engine for EUR/USD on H1 timeframe only.
    
    Key differences from gold engine:
    - Single timeframe (H1) for simplicity in initial research
    - Pip-based calculations instead of dollar-based
    - Different position sizing (forex lots, pip value)
    """

    def __init__(self, h1_df, *,
                 pip_value_per_lot=10.0,
                 spread_pips=1.8,
                 capital=2000,
                 risk_per_trade=50,
                 max_positions=2,
                 # Keltner params
                 kc_adx_threshold=18,
                 sl_atr_mult=4.5,
                 tp_atr_mult=8.0,
                 # Trailing
                 trail_activate_atr=0.8,
                 trail_distance_atr=0.25,
                 # V3 regime
                 regime_config=None,
                 # Timing
                 max_hold_bars=15,
                 cooldown_bars=1,
                 # RSI mean reversion
                 enable_rsi=False,
                 rsi_buy_threshold=15,
                 rsi_sell_threshold=85,
                 rsi_adx_block=40,
                 # ORB
                 enable_orb=False,
                 orb_hour_utc=14,
                 orb_range_bars=1,
                 orb_expiry_bars=6,
                 orb_sl_mult=0.75,
                 orb_tp_mult=3.0,
                 # Label
                 label=""):
        self.h1_df = h1_df
        self.pip_value_per_lot = pip_value_per_lot
        self.spread_pips = spread_pips
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions

        self.kc_adx = kc_adx_threshold
        self.sl_mult = sl_atr_mult
        self.tp_mult = tp_atr_mult
        self.trail_act = trail_activate_atr
        self.trail_dist = trail_distance_atr
        self.trail_act_base = trail_activate_atr
        self.trail_dist_base = trail_distance_atr
        self.regime_config = regime_config
        self.max_hold = max_hold_bars
        self.cooldown_bars = cooldown_bars

        self.enable_rsi = enable_rsi
        self.rsi_buy_th = rsi_buy_threshold
        self.rsi_sell_th = rsi_sell_threshold
        self.rsi_adx_block = rsi_adx_block

        self.enable_orb = enable_orb
        self.orb_hour = orb_hour_utc
        self.orb_range_bars = orb_range_bars
        self.orb_expiry_bars = orb_expiry_bars
        self.orb_sl_mult = orb_sl_mult
        self.orb_tp_mult = orb_tp_mult

        self.label = label
        self.positions = []
        self.trades = []
        self.equity_curve = []

        # ORB state
        self._orb_range_high = None
        self._orb_range_low = None
        self._orb_range_set = False
        self._orb_traded_today = False
        self._orb_window_start = None
        self._orb_current_date = None

    def _calc_lot_size(self, sl_price_dist):
        """ATR-based lot sizing."""
        sl_pips = sl_price_dist / FOREX_PIP
        if sl_pips <= 0:
            return 0.01
        # lots = risk / (sl_pips * pip_value_per_lot)
        lots = self.risk_per_trade / (sl_pips * self.pip_value_per_lot)
        lots = max(0.01, min(0.10, round(lots, 2)))
        return lots

    def run(self):
        self.positions = []
        self.trades = []
        self.equity_curve = []
        realized_pnl = 0.0
        cooldown_until = {}
        daily_losses = 0
        current_date = None

        lookback = 150
        total = len(self.h1_df)

        print(f"  Backtest: {self.h1_df.index[lookback].strftime('%Y-%m-%d')} -> "
              f"{self.h1_df.index[-1].strftime('%Y-%m-%d')}")
        print(f"  H1 bars: {total}")

        last_pct = 0
        for i in range(lookback, total):
            pct = int((i - lookback) / (total - lookback) * 100) // 10 * 10
            if pct > last_pct:
                print(f"    {pct}%...", end='', flush=True)
                last_pct = pct

            bar = self.h1_df.iloc[i]
            bar_time = self.h1_df.index[i]
            bar_date = bar_time.date()
            high = float(bar['High'])
            low = float(bar['Low'])
            close = float(bar['Close'])
            atr = float(bar['ATR']) if not pd.isna(bar['ATR']) else 0

            if bar_date != current_date:
                current_date = bar_date
                daily_losses = 0
                self._orb_range_set = False
                self._orb_traded_today = False
                self._orb_range_high = None
                self._orb_range_low = None
                self._orb_window_start = None

            if bool(bar.get('is_flat', False)):
                unrealized = sum(
                    (close - p.entry_price if p.direction == 'BUY' else p.entry_price - close)
                    / FOREX_PIP * p.lots * self.pip_value_per_lot
                    for p in self.positions
                )
                self.equity_curve.append(self.capital + realized_pnl + unrealized)
                continue

            window = self.h1_df.iloc[max(0, i - 149):i + 1]

            # --- Apply regime ---
            if self.regime_config and atr > 0:
                atr_pct = float(bar.get('atr_percentile', 0.5))
                if not pd.isna(atr_pct):
                    regime = 'low' if atr_pct < 0.30 else ('high' if atr_pct > 0.70 else 'normal')
                    rc = self.regime_config.get(regime, {})
                    self.trail_act = rc.get('trail_act', self.trail_act_base)
                    self.trail_dist = rc.get('trail_dist', self.trail_dist_base)

            # --- Check exits ---
            for pos in list(self.positions):
                pos.bars_held += 1
                reason = None
                exit_price = close

                if pos.direction == 'BUY':
                    if low <= pos.sl_price:
                        reason, exit_price = "SL", pos.sl_price
                    elif high >= pos.tp_price:
                        reason, exit_price = "TP", pos.tp_price
                else:
                    if high >= pos.sl_price:
                        reason, exit_price = "SL", pos.sl_price
                    elif low <= pos.tp_price:
                        reason, exit_price = "TP", pos.tp_price

                # Trailing stop (Keltner only)
                if not reason and pos.strategy == 'keltner' and atr > 0:
                    act_atr = self.trail_act
                    dist_atr = self.trail_dist
                    if pos.direction == 'BUY':
                        profit = high - pos.entry_price
                        pos.extreme_price = max(pos.extreme_price, high)
                    else:
                        profit = pos.entry_price - low
                        pos.extreme_price = min(pos.extreme_price, low) if pos.extreme_price > 0 else low

                    if profit >= atr * act_atr:
                        trail_d = atr * dist_atr
                        if pos.direction == 'BUY':
                            new_trail = pos.extreme_price - trail_d
                            pos.trailing_stop_price = max(pos.trailing_stop_price, new_trail)
                            if low <= pos.trailing_stop_price:
                                reason, exit_price = "Trailing", pos.trailing_stop_price
                        else:
                            new_trail = pos.extreme_price + trail_d
                            if pos.trailing_stop_price <= 0:
                                pos.trailing_stop_price = new_trail
                            else:
                                pos.trailing_stop_price = min(pos.trailing_stop_price, new_trail)
                            if high >= pos.trailing_stop_price:
                                reason, exit_price = "Trailing", pos.trailing_stop_price

                # Timeout
                if not reason and pos.bars_held >= self.max_hold:
                    reason, exit_price = f"Timeout", close

                if reason:
                    pnl_price = (exit_price - pos.entry_price) if pos.direction == 'BUY' else (pos.entry_price - exit_price)
                    pnl_pips = pnl_price / FOREX_PIP
                    pnl = pnl_pips * pos.lots * self.pip_value_per_lot
                    # Subtract spread
                    spread_cost = self.spread_pips * pos.lots * self.pip_value_per_lot
                    pnl -= spread_cost

                    self.trades.append(ForexTrade(
                        pos.strategy, pos.direction, pos.entry_price, exit_price,
                        pos.entry_time, bar_time, pos.lots, round(pnl, 2), reason, pos.bars_held
                    ))
                    self.positions.remove(pos)
                    if pnl < 0:
                        daily_losses += 1
                        cooldown_until[pos.strategy] = i + self.cooldown_bars

            # --- Check entries ---
            if daily_losses >= 5 or len(self.positions) >= self.max_positions:
                realized_pnl = sum(t.pnl for t in self.trades)
                unrealized = sum(
                    (close - p.entry_price if p.direction == 'BUY' else p.entry_price - close)
                    / FOREX_PIP * p.lots * self.pip_value_per_lot
                    for p in self.positions
                )
                self.equity_curve.append(self.capital + realized_pnl + unrealized)
                continue

            active_strats = {p.strategy for p in self.positions}
            current_dir = self.positions[0].direction if self.positions else None

            # --- Keltner signal ---
            if 'keltner' not in active_strats and atr > 0:
                if cooldown_until.get('keltner', 0) <= i:
                    adx = float(bar['ADX']) if not pd.isna(bar['ADX']) else 0
                    kc_upper = float(bar['KC_upper']) if not pd.isna(bar['KC_upper']) else 0
                    kc_lower = float(bar['KC_lower']) if not pd.isna(bar['KC_lower']) else 0
                    ema100 = float(bar['EMA100']) if not pd.isna(bar['EMA100']) else 0

                    if adx >= self.kc_adx and kc_upper > 0:
                        signal = None
                        if close > kc_upper and close > ema100:
                            signal = 'BUY'
                        elif close < kc_lower and close < ema100:
                            signal = 'SELL'

                        if signal and (current_dir is None or current_dir == signal):
                            sl_d = atr * self.sl_mult
                            tp_d = atr * self.tp_mult
                            lots = self._calc_lot_size(sl_d)
                            pos = ForexPosition('keltner', signal, close, bar_time, lots, sl_d, tp_d, atr)
                            self.positions.append(pos)
                            active_strats.add('keltner')
                            if current_dir is None:
                                current_dir = signal

            # --- RSI mean reversion ---
            if self.enable_rsi and 'm15_rsi' not in active_strats and atr > 0:
                if cooldown_until.get('m15_rsi', 0) <= i:
                    rsi2 = float(bar['RSI2']) if not pd.isna(bar['RSI2']) else 50
                    sma50 = float(bar['SMA50']) if not pd.isna(bar['SMA50']) else 0
                    ema100 = float(bar['EMA100']) if not pd.isna(bar['EMA100']) else 0
                    adx = float(bar['ADX']) if not pd.isna(bar['ADX']) else 0

                    if adx <= self.rsi_adx_block:
                        signal = None
                        if rsi2 < self.rsi_buy_th and close > sma50 and close > ema100:
                            signal = 'BUY'
                        elif rsi2 > self.rsi_sell_th and close < sma50 and close < ema100:
                            signal = 'SELL'

                        if signal and (current_dir is None or current_dir == signal):
                            sl_d = atr * self.sl_mult
                            tp_d = atr * self.tp_mult * 0.5
                            lots = self._calc_lot_size(sl_d)
                            pos = ForexPosition('m15_rsi', signal, close, bar_time, lots, sl_d, tp_d, atr)
                            self.positions.append(pos)
                            active_strats.add('m15_rsi')
                            if current_dir is None:
                                current_dir = signal

            # --- ORB ---
            if self.enable_orb and 'orb' not in active_strats and not self._orb_traded_today:
                hour = bar_time.hour
                if hour == self.orb_hour and not self._orb_range_set:
                    self._orb_range_high = high
                    self._orb_range_low = low
                    self._orb_range_set = True
                    self._orb_window_start = i

                if self._orb_range_set and not self._orb_traded_today:
                    if self._orb_window_start and (i - self._orb_window_start) <= self.orb_expiry_bars:
                        rng = self._orb_range_high - self._orb_range_low
                        if rng > 0 and cooldown_until.get('orb', 0) <= i:
                            signal = None
                            if close > self._orb_range_high:
                                signal = 'BUY'
                            elif close < self._orb_range_low:
                                signal = 'SELL'

                            if signal and (current_dir is None or current_dir == signal):
                                sl_d = max(rng * self.orb_sl_mult, atr * 1.5)
                                tp_d = rng * self.orb_tp_mult
                                lots = self._calc_lot_size(sl_d)
                                pos = ForexPosition('orb', signal, close, bar_time, lots, sl_d, tp_d, atr)
                                self.positions.append(pos)
                                self._orb_traded_today = True
                                if current_dir is None:
                                    current_dir = signal

            realized_pnl = sum(t.pnl for t in self.trades)
            unrealized = sum(
                (close - p.entry_price if p.direction == 'BUY' else p.entry_price - close)
                / FOREX_PIP * p.lots * self.pip_value_per_lot
                for p in self.positions
            )
            self.equity_curve.append(self.capital + realized_pnl + unrealized)

        # Close remaining
        if self.positions:
            last = self.h1_df.iloc[-1]
            last_time = self.h1_df.index[-1]
            last_close = float(last['Close'])
            for pos in list(self.positions):
                pnl_price = (last_close - pos.entry_price) if pos.direction == 'BUY' else (pos.entry_price - last_close)
                pnl = pnl_price / FOREX_PIP * pos.lots * self.pip_value_per_lot
                pnl -= self.spread_pips * pos.lots * self.pip_value_per_lot
                self.trades.append(ForexTrade(
                    pos.strategy, pos.direction, pos.entry_price, last_close,
                    pos.entry_time, last_time, pos.lots, round(pnl, 2), "backtest_end", pos.bars_held
                ))
            self.positions = []

        print(f" done!")
        return self.trades


def calc_forex_stats(trades, equity_curve, capital=2000):
    """Calculate performance statistics."""
    if not trades:
        return {'n': 0, 'sharpe': 0, 'total_pnl': 0, 'max_dd': 0, 'win_rate': 0}

    pnls = [t.pnl for t in trades]
    n = len(pnls)
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n if n > 0 else 0

    daily_pnls = {}
    for t in trades:
        d = t.exit_time.date() if hasattr(t.exit_time, 'date') else pd.Timestamp(t.exit_time).date()
        daily_pnls[d] = daily_pnls.get(d, 0) + t.pnl
    daily_arr = np.array(list(daily_pnls.values()))
    sharpe = 0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252)

    eq = np.array(equity_curve) if equity_curve else np.array([capital])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = float(dd.max()) if len(dd) > 0 else 0
    dd_pct = max_dd / capital * 100

    avg_pnl = total_pnl / n if n > 0 else 0

    return {
        'n': n,
        'sharpe': round(sharpe, 2),
        'total_pnl': round(total_pnl, 2),
        'max_dd': round(max_dd, 2),
        'dd_pct': round(dd_pct, 1),
        'win_rate': round(win_rate * 100, 1),
        'avg_pnl': round(avg_pnl, 2),
    }


def print_stats_table(results, title=""):
    """Print a comparison table."""
    if title:
        print(f"\n{'=' * 100}")
        print(f"  {title}")
        print(f"{'=' * 100}")
    header = f"  {'Label':<40s} {'N':>6} {'Sharpe':>8} {'PnL':>10} {'MaxDD':>8} {'DD%':>6} {'WR%':>6} {'$/t':>7}"
    print(header)
    print("  " + "-" * 95)
    for r in results:
        label = r.get('label', '?')
        print(f"  {label:<40s} {r['n']:>6} {r['sharpe']:>8.2f} ${r['total_pnl']:>9.0f} ${r['max_dd']:>7.0f} "
              f"{r['dd_pct']:>5.1f}% {r['win_rate']:>5.1f}% ${r['avg_pnl']:>6.2f}")


# ═══════════════════════════════════════════════════════════════
# PART 4-6: Strategy Backtests
# ═══════════════════════════════════════════════════════════════

V3_REGIME = {
    'low': {'trail_act': 1.0, 'trail_dist': 0.35},
    'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
    'high': {'trail_act': 0.6, 'trail_dist': 0.20},
}


def run_single(h1_df, label, **kwargs):
    """Run one backtest and return stats."""
    print(f"\n  [{label}]")
    engine = ForexBacktestEngine(h1_df, label=label, **kwargs)
    trades = engine.run()
    stats = calc_forex_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['_trades'] = trades
    stats['_equity'] = engine.equity_curve

    # Strategy breakdown
    strat_map = {}
    for t in trades:
        s = t.strategy
        if s not in strat_map:
            strat_map[s] = {'n': 0, 'pnl': 0, 'wins': 0}
        strat_map[s]['n'] += 1
        strat_map[s]['pnl'] += t.pnl
        if t.pnl > 0:
            strat_map[s]['wins'] += 1
    stats['_strats'] = strat_map

    print(f"    {stats['n']} trades, Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
          f"MaxDD=${stats['max_dd']:.0f}, WR={stats['win_rate']:.1f}%")
    for sn, sv in strat_map.items():
        wr = sv['wins'] / sv['n'] * 100 if sv['n'] > 0 else 0
        print(f"      {sn}: {sv['n']} trades, PnL=${sv['pnl']:.0f}, WR={wr:.1f}%")

    return stats


def run_kfold(h1_df, label_prefix, n_folds=6, **kwargs):
    """K-Fold cross validation."""
    # Determine year range
    years = sorted(h1_df.index.year.unique())
    min_year = years[0]
    max_year = years[-1]

    folds = []
    fold_years = 2
    y = min_year
    fold_num = 1
    while y < max_year and fold_num <= n_folds:
        y_end = min(y + fold_years, max_year + 1)
        start = pd.Timestamp(f"{y}-01-01", tz='UTC')
        end = pd.Timestamp(f"{y_end}-01-01", tz='UTC')
        fold_df = h1_df[(h1_df.index >= start) & (h1_df.index < end)]
        if len(fold_df) > 500:
            folds.append((f"Fold{fold_num}({y}-{y_end-1})", fold_df))
            fold_num += 1
        y = y_end

    results = []
    for fname, fdf in folds:
        fdf_prep = prepare_forex_indicators(fdf)
        stats = run_single(fdf_prep, f"{label_prefix}_{fname}", **kwargs)
        stats['fold'] = fname
        results.append(stats)
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    start_time = datetime.now()
    print(f"{'=' * 80}")
    print(f"EUR/USD QUANTITATIVE RESEARCH")
    print(f"Started: {start_time}")
    print(f"{'=' * 80}")

    # --- PART 0: Download data ---
    h1_path, m15_path = download_eurusd_data()
    if h1_path is None:
        print("FATAL: Could not download data!")
        sys.exit(1)

    # --- PART 1: Load and prepare ---
    print(f"\n{'=' * 80}")
    print("PART 1: LOADING DATA")
    print(f"{'=' * 80}")

    h1_raw = load_forex_csv(h1_path)
    print(f"  H1 raw: {len(h1_raw)} bars, {h1_raw.index[0]} -> {h1_raw.index[-1]}")

    h1_df = prepare_forex_indicators(h1_raw)
    print(f"  H1 with indicators: {len(h1_df)} bars")
    print(f"  Columns: {list(h1_df.columns[:10])}...")

    # --- PART 2: ATR Study ---
    atr_info = atr_study(h1_df)

    # --- PART 3: Keltner Channel Backtest ---
    print(f"\n{'=' * 80}")
    print("PART 3: KELTNER CHANNEL STRATEGY ON EUR/USD")
    print(f"{'=' * 80}")

    keltner_results = []

    # 3a: Gold-equivalent params (baseline)
    keltner_results.append(run_single(h1_df, "KC Gold-params",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 3b: Wider SL (forex tends to have more noise)
    keltner_results.append(run_single(h1_df, "KC SL=6.0 TP=10.0",
        kc_adx_threshold=18, sl_atr_mult=6.0, tp_atr_mult=10.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 3c: Different KC widths
    for kc_m in [1.0, 1.5, 2.0]:
        h1_custom = prepare_forex_indicators(h1_raw, kc_mult=kc_m)
        keltner_results.append(run_single(h1_custom, f"KC mult={kc_m}",
            kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
            trail_activate_atr=0.8, trail_distance_atr=0.25,
            regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 3d: Different ADX thresholds
    for adx in [15, 20, 25]:
        keltner_results.append(run_single(h1_df, f"KC ADX>={adx}",
            kc_adx_threshold=adx, sl_atr_mult=4.5, tp_atr_mult=8.0,
            trail_activate_atr=0.8, trail_distance_atr=0.25,
            regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 3e: Mega trail (from gold research)
    keltner_results.append(run_single(h1_df, "KC Mega Trail(0.5/0.15)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.5, trail_distance_atr=0.15,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # 3f: No spread (theoretical max)
    keltner_results.append(run_single(h1_df, "KC No-spread (theoretical)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=0.0, max_hold_bars=15))

    print_stats_table(keltner_results, "KELTNER STRATEGY RESULTS")

    # --- PART 4: ORB Strategy ---
    print(f"\n{'=' * 80}")
    print("PART 4: ORB STRATEGY ON EUR/USD")
    print(f"{'=' * 80}")

    orb_results = []

    # London open ORB (8 UTC)
    orb_results.append(run_single(h1_df, "ORB London(8 UTC)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        enable_orb=True, orb_hour_utc=8, orb_expiry_bars=6,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # NY open ORB (14 UTC)
    orb_results.append(run_single(h1_df, "ORB NY(14 UTC)",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        enable_orb=True, orb_hour_utc=14, orb_expiry_bars=6,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    # Both ORBs (London + NY)
    orb_results.append(run_single(h1_df, "ORB London+NY",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        enable_orb=True, orb_hour_utc=8, orb_expiry_bars=6,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    print_stats_table(orb_results, "ORB STRATEGY RESULTS")

    # --- PART 5: Mean Reversion (RSI) ---
    print(f"\n{'=' * 80}")
    print("PART 5: MEAN REVERSION (RSI) ON EUR/USD")
    print(f"{'=' * 80}")

    mr_results = []

    # RSI only
    mr_results.append(run_single(h1_df, "RSI2 15/85 only",
        kc_adx_threshold=999, enable_rsi=True,
        rsi_buy_threshold=15, rsi_sell_threshold=85, rsi_adx_block=40,
        sl_atr_mult=3.0, tp_atr_mult=4.0,
        trail_activate_atr=1.0, trail_distance_atr=0.3,
        spread_pips=1.8, max_hold_bars=8))

    # RSI with wider thresholds
    mr_results.append(run_single(h1_df, "RSI2 10/90 only",
        kc_adx_threshold=999, enable_rsi=True,
        rsi_buy_threshold=10, rsi_sell_threshold=90, rsi_adx_block=40,
        sl_atr_mult=3.0, tp_atr_mult=4.0,
        trail_activate_atr=1.0, trail_distance_atr=0.3,
        spread_pips=1.8, max_hold_bars=8))

    # RSI + Keltner combo
    mr_results.append(run_single(h1_df, "KC + RSI2 15/85",
        kc_adx_threshold=18, enable_rsi=True,
        rsi_buy_threshold=15, rsi_sell_threshold=85, rsi_adx_block=40,
        sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15))

    print_stats_table(mr_results, "MEAN REVERSION RESULTS")

    # --- PART 6: K-Fold Validation (best config) ---
    print(f"\n{'=' * 80}")
    print("PART 6: K-FOLD CROSS-VALIDATION")
    print(f"{'=' * 80}")

    # Find best Keltner config
    best_kc = max(keltner_results, key=lambda r: r['sharpe'])
    print(f"\n  Best Keltner config: {best_kc['label']} (Sharpe={best_kc['sharpe']:.2f})")

    # K-Fold on the gold-equivalent params (most comparable)
    print(f"\n  K-Fold: KC Gold-params")
    kfold_base = run_kfold(h1_raw, "KC-base",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.8, trail_distance_atr=0.25,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15)
    print_stats_table(kfold_base, "K-Fold: KC Gold-params")

    # K-Fold on Mega trail
    print(f"\n  K-Fold: KC Mega Trail")
    kfold_mega = run_kfold(h1_raw, "KC-mega",
        kc_adx_threshold=18, sl_atr_mult=4.5, tp_atr_mult=8.0,
        trail_activate_atr=0.5, trail_distance_atr=0.15,
        regime_config=V3_REGIME, spread_pips=1.8, max_hold_bars=15)
    print_stats_table(kfold_mega, "K-Fold: KC Mega Trail")

    # --- PART 7: Year-by-Year ---
    print(f"\n{'=' * 80}")
    print("PART 7: YEAR-BY-YEAR BREAKDOWN")
    print(f"{'=' * 80}")

    best_trades = best_kc.get('_trades', [])
    if best_trades:
        yearly = {}
        for t in best_trades:
            y = t.exit_time.year if hasattr(t.exit_time, 'year') else pd.Timestamp(t.exit_time).year
            if y not in yearly:
                yearly[y] = {'n': 0, 'pnl': 0, 'wins': 0}
            yearly[y]['n'] += 1
            yearly[y]['pnl'] += t.pnl
            if t.pnl > 0:
                yearly[y]['wins'] += 1

        print(f"\n  Best config: {best_kc['label']}")
        print(f"  {'Year':<8} {'N':>6} {'PnL':>10} {'WR%':>8} {'$/trade':>8}")
        print(f"  {'-'*44}")
        for y in sorted(yearly.keys()):
            v = yearly[y]
            wr = v['wins'] / v['n'] * 100 if v['n'] > 0 else 0
            avg = v['pnl'] / v['n'] if v['n'] > 0 else 0
            print(f"  {y:<8} {v['n']:>6} ${v['pnl']:>9.0f} {wr:>7.1f}% ${avg:>7.2f}")

    # --- Summary ---
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 80}")
    print(f"RESEARCH COMPLETE")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 80}")

    # Save results summary
    summary = {
        'timestamp': str(datetime.now()),
        'atr_study': atr_info,
        'keltner_results': [{k: v for k, v in r.items() if not k.startswith('_')} for r in keltner_results],
        'orb_results': [{k: v for k, v in r.items() if not k.startswith('_')} for r in orb_results],
        'mr_results': [{k: v for k, v in r.items() if not k.startswith('_')} for r in mr_results],
    }

    out_path = Path("data/eurusd_research_results.json")
    with open(str(out_path), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")
