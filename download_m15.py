"""
Download complete XAUUSD M15 data from Dukascopy in 6-month chunks.
Merge into a single CSV at the end.
"""
import subprocess
import sys
import os
from pathlib import Path
import pandas as pd

OUT_DIR = Path("data/download/m15_full")
MERGED_FILE = Path("data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv")

CHUNKS = [
    ("2015-01-01", "2015-07-01"),
    ("2015-07-01", "2016-01-01"),
    ("2016-01-01", "2016-07-01"),
    ("2016-07-01", "2017-01-01"),
    ("2017-01-01", "2017-07-01"),
    ("2017-07-01", "2018-01-01"),
    ("2018-01-01", "2018-07-01"),
    ("2018-07-01", "2019-01-01"),
    ("2019-01-01", "2019-07-01"),
    ("2019-07-01", "2020-01-01"),
    ("2020-01-01", "2020-07-01"),
    ("2020-07-01", "2021-01-01"),
    ("2021-01-01", "2021-07-01"),
    ("2021-07-01", "2022-01-01"),
    ("2022-01-01", "2022-07-01"),
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
    ("2024-07-01", "2025-01-01"),
    ("2025-01-01", "2025-07-01"),
    ("2025-07-01", "2026-04-01"),
]

MAX_RETRIES = 3


def download_chunk(date_from: str, date_to: str) -> bool:
    cmd = (
        f'npx dukascopy-node -i xauusd -from {date_from} -to {date_to} '
        f'-t m15 -p bid -f csv -v -dir "{OUT_DIR}"'
    )
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{date_from} -> {date_to}] attempt {attempt}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=300, shell=True,
                encoding='utf-8', errors='replace',
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if result.returncode == 0 and "File saved" in stdout:
                print("OK")
                return True
            else:
                err = (stderr.strip() or stdout.strip())[-100:]
                print(f"FAIL ({err})")
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
    return False


def merge_csvs():
    print("\nMerging CSVs...")
    files = sorted(OUT_DIR.glob("*.csv"))
    if not files:
        print("  No CSV files found!")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
        print(f"  {f.name}: {len(df)} rows")

    merged = pd.concat(frames, ignore_index=True)
    merged.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    merged.sort_values('timestamp', inplace=True)
    merged.reset_index(drop=True, inplace=True)

    if 'volume' in merged.columns:
        merged.rename(columns={'volume': 'Volume'}, inplace=True)

    merged.to_csv(MERGED_FILE, index=False)
    print(f"\n  Merged: {len(merged)} rows -> {MERGED_FILE}")

    ts_start = pd.to_datetime(merged['timestamp'].iloc[0], unit='ms', utc=True)
    ts_end = pd.to_datetime(merged['timestamp'].iloc[-1], unit='ms', utc=True)
    print(f"  Range: {ts_start} -> {ts_end}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading XAUUSD M15 data to {OUT_DIR}")
    print(f"Total chunks: {len(CHUNKS)}\n")

    failed = []
    for date_from, date_to in CHUNKS:
        if not download_chunk(date_from, date_to):
            failed.append((date_from, date_to))

    if failed:
        print(f"\n  WARNING: {len(failed)} chunks failed:")
        for f, t in failed:
            print(f"    {f} -> {t}")
    else:
        print(f"\n  All {len(CHUNKS)} chunks downloaded successfully!")

    merge_csvs()


if __name__ == "__main__":
    main()
