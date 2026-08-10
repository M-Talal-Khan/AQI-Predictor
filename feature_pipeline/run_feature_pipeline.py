"""
Entrypoint for the HOURLY feature pipeline (GitHub Actions, see
.github/workflows/feature_pipeline.yml).

Flow:
  1. Pull a recent window of AQ observations (not just the current hour).
  2. Pull weather covering that window AND the next ~5 days of forecast.
  3. Rebuild features over the window - which gives the newest rows real
     forecast-weather features for +24/+48/+72h.
  4. Upsert the window into the feature store.

WHY A WINDOW AND NOT A SINGLE ROW
---------------------------------
The scaffold fetched exactly one reading per run and wrote one row. Two
problems with that: a single missed run (Actions outage, API blip) leaves a
permanent hole that nothing ever repairs, and lag features for the new row
depend on history being complete. Re-fetching a rolling window and upserting
makes every run self-healing - a gap is filled by the next successful run, and
revised CAMS values overwrite earlier provisional ones.

WHY WEATHER IS FETCHED INTO THE FUTURE
--------------------------------------
The whole point of the design: the row for "now" needs weather valid at now+24,
+48 and +72h to be predictable at all. Those hours have not happened, so they
come from the forecast endpoint. See features.add_forecast_weather_features.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import date, timedelta

import pandas as pd

from feature_pipeline.features import build_features
from feature_pipeline.openmeteo_aq_client import fetch_aq_live, trim_to_now
from feature_pipeline.weather_forecast_client import fetch_weather_forecast
from feature_pipeline.store import write_features, backend_name, hopsworks_available
from config import FORECAST_HORIZONS

# Long enough to cover the longest lag/rolling window (168h) plus slack, so
# the newest rows get fully-populated history features rather than NaNs.
WINDOW_PAST_DAYS = 14


def run(local_only: bool = False, past_days: int = WINDOW_PAST_DAYS,
        strict: bool = None):
    # In CI the local parquet mirror lives on a throwaway container filesystem,
    # so a Hopsworks write that quietly fails leaves nothing behind while the
    # job still reports success. Default to strict whenever running under
    # GitHub Actions; stay lenient for interactive local runs.
    if strict is None:
        strict = os.getenv("GITHUB_ACTIONS") == "true" and not local_only

    print(f"Storage backend: {backend_name()}  (strict={strict})")
    if not local_only and not hopsworks_available():
        print("  NOTE: hopsworks not importable or HOPSWORKS_API_KEY unset - "
              "writes will not reach the feature store")

    print(f"Fetching AQ observations (last {past_days} days)...")
    # trim_to_now: the live endpoint returns whole days, so later hours of today
    # are forecast rather than observed - they must not enter the feature store
    # as observations. Weather, by contrast, is deliberately kept into the future.
    aq = trim_to_now(fetch_aq_live(past_days=past_days, forecast_days=1))
    observed = aq[aq["aqi"].notna()]
    print(f"  {len(aq)} rows, {len(observed)} with AQI, "
          f"latest observation {observed['observed_at'].max()}")

    print("Fetching weather (past window + 5-day forecast)...")
    weather = fetch_weather_forecast(forecast_days=5, past_days=past_days)
    print(f"  {len(weather)} rows, {weather['observed_at'].min()} -> {weather['observed_at'].max()}")

    print("Building features...")
    engineered = build_features(aq, weather, with_targets=True)
    engineered["station"] = "open-meteo-cams"

    # Keep only rows that actually have an observed AQI. Rows past the last
    # observation exist in the frame (they carry forecast weather) but have no
    # AQI, so storing them would pollute the feature group with empty history.
    engineered = engineered[engineered["aqi"].notna()]

    latest = engineered.tail(1)
    if latest.empty:
        raise SystemExit("No usable rows built - AQ source returned nothing observed.")

    row = latest.iloc[0]
    print(f"\nLatest row: {row['observed_at']}  AQI={row['aqi']:.0f}")
    for h in FORECAST_HORIZONS:
        col = f"fc{h}_temperature_2m"
        wind = f"fc{h}_wind_speed_10m"
        if col in row and pd.notna(row[col]):
            print(f"  +{h}h forecast weather: temp={row[col]:.1f}C wind={row[wind]:.1f}km/h")
        else:
            print(f"  +{h}h forecast weather: MISSING - forecast horizon too short?")

    # Sanity: the newest row must carry forecast features or the model has
    # nothing horizon-specific to predict from.
    missing = [h for h in FORECAST_HORIZONS if pd.isna(row.get(f"fc{h}_temperature_2m"))]
    if missing:
        print(f"  WARNING: horizons {missing} lack forecast weather on the newest row")

    print(f"\nWriting {len(engineered)} rows (upsert) to the feature store...")
    write_features(engineered, prefer_local=local_only, strict=strict)
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--local-only", action="store_true", help="skip Hopsworks, parquet only")
    p.add_argument("--past-days", type=int, default=WINDOW_PAST_DAYS)
    a = p.parse_args()
    run(local_only=a.local_only, past_days=a.past_days)
