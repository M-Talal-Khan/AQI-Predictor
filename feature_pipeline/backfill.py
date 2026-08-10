"""
Historical backfill - seeds the feature store with years of real hourly data.

The original scaffold assumed you would hand-download a Kaggle CSV, because
AQICN's free tier has no history endpoint. That is no longer necessary: this
pulls history directly from Open-Meteo, which serves hourly CAMS air quality
for Lahore back to 2022-09-01 and matching hourly reanalysis weather, both
without an API key. That is ~34k hourly rows, far more than needed to train.

The --csv path is retained for the case where you want to seed from a
ground-station dataset instead (see load_csv).

Usage:
    python feature_pipeline/backfill.py                      # Open-Meteo, full history
    python feature_pipeline/backfill.py --start 2024-01-01   # shorter range
    python feature_pipeline/backfill.py --local-only         # skip Hopsworks
    python feature_pipeline/backfill.py --csv history.csv --timestamp-col date --aqi-col aqi
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

import numpy as np
import pandas as pd

from config import AQ_ARCHIVE_START, FORECAST_HORIZONS
from feature_pipeline.features import build_features, RAW_AQ_COLUMNS
from feature_pipeline.openmeteo_aq_client import get_aq_upto_now
from feature_pipeline.weather_forecast_client import get_weather_upto_now
from feature_pipeline.store import write_features, backend_name


def load_csv(path: str, timestamp_col: str, aqi_col: str) -> pd.DataFrame:
    """Optional alternative seed: a ground-station CSV instead of Open-Meteo."""
    raw = pd.read_csv(path)
    df = pd.DataFrame()
    df["observed_at"] = pd.to_datetime(raw[timestamp_col])
    df["aqi"] = pd.to_numeric(raw[aqi_col], errors="coerce")

    for col in RAW_AQ_COLUMNS:
        if col == "aqi":
            continue
        df[col] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan

    return df.sort_values("observed_at").reset_index(drop=True)


# --------------------------------------------------------------------------
# data quality
# --------------------------------------------------------------------------

def audit(aq: pd.DataFrame, engineered: pd.DataFrame) -> None:
    """
    Sanity-check the pull before anything downstream trusts it.

    The prompt specifically asked to spot-check for a systematic one-directional
    bias before believing downstream metrics, so this reports the distribution
    against known-real expectations for Lahore rather than just row counts. If
    these numbers look wrong, every R2 below is meaningless.
    """
    print("\n" + "=" * 62)
    print("BACKFILL AUDIT")
    print("=" * 62)

    span = aq["observed_at"].max() - aq["observed_at"].min()
    expected = int(span.total_seconds() // 3600) + 1
    print(f"Rows fetched      : {len(aq):,}")
    print(f"Date range        : {aq['observed_at'].min()} -> {aq['observed_at'].max()}")
    print(f"Span              : {span.days} days ({expected:,} hourly slots)")
    print(f"Coverage          : {len(aq) / expected:.1%} of hourly slots present")

    print("\n--- missingness by column (raw AQ pull) ---")
    for col in RAW_AQ_COLUMNS:
        if col in aq.columns:
            n_null = aq[col].isna().sum()
            print(f"  {col:<6} {n_null:>7,} null  ({n_null / len(aq):>6.2%})")

    aqi = aq["aqi"].dropna()
    print("\n--- AQI distribution ---")
    print(f"  mean {aqi.mean():.1f} | median {aqi.median():.1f} | std {aqi.std():.1f}")
    print(f"  min {aqi.min():.0f} | p05 {aqi.quantile(.05):.0f} | p95 "
          f"{aqi.quantile(.95):.0f} | max {aqi.max():.0f}")

    # Bias check. Lahore genuinely averages "Unhealthy" annually, with a severe
    # Nov-Jan smog season. A mean near 50, or a winter that is not the worst
    # season, would mean we pulled the wrong location or a broken series.
    print("\n--- bias / plausibility checks ---")
    monthly = aq.assign(m=aq["observed_at"].dt.month).groupby("m")["aqi"].mean()
    worst, best = monthly.idxmax(), monthly.idxmin()
    print(f"  worst month = {worst} ({monthly.max():.0f}), "
          f"best month = {best} ({monthly.min():.0f})")

    checks = [
        ("mean AQI in plausible Lahore range (80-250)", 80 <= aqi.mean() <= 250),
        ("winter (Nov-Jan) is the worst season", worst in (11, 12, 1)),
        ("monsoon (Jul-Sep) is not the worst season", worst not in (7, 8, 9)),
        ("has genuine hazardous episodes (max > 300)", aqi.max() > 300),
        ("not degenerate (std > 25)", aqi.std() > 25),
        ("no impossible values (min >= 0)", aqi.min() >= 0),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        all_ok &= ok

    print("\n--- engineered frame ---")
    print(f"  {len(engineered):,} rows x {len(engineered.columns)} columns")
    for h in FORECAST_HORIZONS:
        tgt = f"aqi_target_{h}h"
        if tgt in engineered.columns:
            usable = engineered[tgt].notna().sum()
            print(f"  {tgt:<18} {usable:,} non-null targets")

    fc_cols = [c for c in engineered.columns if c.startswith("fc")]
    print(f"  forecast-weather features: {len(fc_cols)}")

    print("\n" + ("AUDIT PASSED" if all_ok else "AUDIT HAS FAILURES - investigate before training"))
    print("=" * 62 + "\n")


# --------------------------------------------------------------------------

def run(start_date: str, csv_path: str = None, timestamp_col: str = "date",
        aqi_col: str = "aqi", local_only: bool = False, chunk_size: int = 5000):
    print(f"Storage backend: {backend_name()}")

    if csv_path:
        print(f"Loading air quality from CSV: {csv_path}")
        aq = load_csv(csv_path, timestamp_col, aqi_col)
    else:
        print(f"Fetching air quality history from Open-Meteo ({start_date} -> now)...")
        aq = get_aq_upto_now(start_date)
    print(f"  {len(aq):,} AQ rows")

    print("Fetching matching weather history from Open-Meteo...")
    weather = get_weather_upto_now(start_date)
    print(f"  {len(weather):,} weather rows")

    print("Building features (incl. per-horizon forecast weather)...")
    engineered = build_features(aq, weather, with_targets=True)
    engineered["station"] = "open-meteo-cams" if not csv_path else "csv-backfill"

    audit(aq, engineered)

    print(f"Writing {len(engineered):,} rows in chunks of {chunk_size}...")
    for start in range(0, len(engineered), chunk_size):
        chunk = engineered.iloc[start:start + chunk_size]
        write_features(chunk, prefer_local=local_only)
        print(f"  rows {start:,}-{start + len(chunk):,}")

    print("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the feature store with AQI history")
    parser.add_argument("--start", default=AQ_ARCHIVE_START,
                        help=f"start date YYYY-MM-DD (default {AQ_ARCHIVE_START})")
    parser.add_argument("--csv", default=None, help="optional CSV seed instead of Open-Meteo")
    parser.add_argument("--timestamp-col", default="date")
    parser.add_argument("--aqi-col", default="aqi")
    parser.add_argument("--local-only", action="store_true", help="skip Hopsworks, write parquet only")
    args = parser.parse_args()

    run(args.start, args.csv, args.timestamp_col, args.aqi_col, args.local_only)
