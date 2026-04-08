"""Check if any trading signals would have fired during the 02:58-03:13 gap"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
from strategies.signals import prepare_indicators, scan_all_signals
from intraday_trend import IntradayTrendMeter

# Get recent H1 data
ticker = yf.Ticker("GC=F")
h1 = ticker.history(period="5d", interval="1h")
if h1.empty:
    print("No data from yfinance")
    exit()

h1 = prepare_indicators(h1)
print(f"H1 data: {h1.index[0]} to {h1.index[-1]}, {len(h1)} bars\n")

# Show last 30 bars with indicators
print("Recent H1 bars:")
print(f"  {'Time':<25} {'Close':>8} {'ADX':>6} {'KC_up':>8} {'KC_lo':>8} {'EMA9':>8} {'EMA21':>8} {'EMA100':>8}")
for i in range(-30, 0):
    if abs(i) > len(h1):
        continue
    row = h1.iloc[i]
    t = str(h1.index[i])[:19]
    adx = row.get('ADX', 0)
    kcu = row.get('KC_upper', 0)
    kcl = row.get('KC_lower', 0)
    print(f"  {t:<25} {row['Close']:>8.2f} {adx:>6.1f} {kcu:>8.2f} {kcl:>8.2f} {row.get('EMA9',0):>8.2f} {row.get('EMA21',0):>8.2f} {row.get('EMA100',0):>8.2f}")

# Check signals at each bar
print("\nSignal scan for last 24 bars:")
for i in range(-24, 0):
    if abs(i) > len(h1):
        continue
    idx = len(h1) + i
    window = h1.iloc[max(0, idx-100):idx+1]
    if len(window) < 50:
        continue
    signals = scan_all_signals(window, 'H1')
    t = str(h1.index[idx])[:19]
    close = h1.iloc[idx]['Close']
    if signals:
        for s in signals:
            strat = s.get('strategy', '')
            dirn = s.get('direction', '')
            reason = str(s.get('reason', '')).encode('ascii', 'replace').decode()
            print(f"  {t} Close={close:.2f} -> SIGNAL: {strat} {dirn} reason={reason}")

meter2 = IntradayTrendMeter()
print("\nTrendMeter progression for Apr 7:")
apr7 = h1[h1.index.date == pd.Timestamp("2026-04-07").date()]
for i in range(2, len(apr7)+1):
    partial = h1[h1.index <= apr7.index[i-1]]
    partial_today = partial[partial.index.date == pd.Timestamp("2026-04-07").date()]
    if len(partial_today) >= 2:
        sc = meter2._calc_score(partial_today)
        regime = meter2._classify(sc)
        t = str(apr7.index[i-1])[:19]
        c = apr7.iloc[i-1]['Close']
        print(f"  {t} bars={i} Close={c:.2f} score={sc:.3f} ({regime})")

# Check TrendMeter for today
meter = IntradayTrendMeter()
score = meter.update(h1)
print(f"\nCurrent TrendMeter: score={score:.3f} regime={meter.get_regime()} bars={meter.get_bar_count()}")
