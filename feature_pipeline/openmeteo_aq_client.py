"""
Open-Meteo CAMS air-quality client - the PRIMARY source of AQI observations.

WHY NOT AQICN (the originally scaffolded source)
------------------------------------------------
AQICN's only Lahore station ("Lahore US Embassy", uid 11765) has stopped
reporting. Verified on 2026-08-10 with a valid token:

  /feed/lahore/           -> AQI 34 stamped 2025-02-18 (~18 months stale)
  /search/?keyword=lahore -> that one station, aqi = "-" (no current value)
  /map/bounds/ (Lahore bbox) -> 0 stations
  /feed/geo:31.52;74.36/  -> silently returns Narela, Delhi, INDIA

That last one is the dangerous failure: AQICN's geo endpoint does not error on
a coverage gap, it quietly hands back the nearest station it does have, which
is 400km away in another country. An hourly pipeline pointed at it would look
healthy while filling the feature store with stale or foreign readings.

So AQI observations come from Open-Meteo's CAMS product instead. It is
modelled/reanalysed rather than a physical monitor, which is a real limitation
(documented in REPORT.md), but it is complete, current, and actually about
Lahore. aqicn_client.py is kept as an optional cross-check, not a dependency.

Endpoints used (no API key required):
  archive mode  - start_date/end_date, 2022-09-01 .. ~5 days ago
  live mode     - past_days/forecast_days, covers now and +4 days
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from config import (
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    AQ_VARS,
    OPENMETEO_AQ_URL,
    AQ_ARCHIVE_START,
    now_local_naive,
)

# Open-Meteo's CAMS names vs the column names used across this project.
# Renaming here keeps the AQICN-shaped vocabulary (pm25/o3/no2/so2/co) that
# features.py, the feature group schema and the webapp already speak.
AQ_RENAME = {
    "us_aqi": "aqi",
    "pm2_5": "pm25",
    "pm10": "pm10",
    "ozone": "o3",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}


class AQClientError(RuntimeError):
    pass


def _get_json(params: dict, retries: int = 3, timeout: int = 90) -> dict:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(OPENMETEO_AQ_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    rate-limited by Open-Meteo AQ, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    raise AQClientError(f"Open-Meteo AQ request failed after {retries} tries: {last_err}")


def _to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise AQClientError(f"Unexpected AQ payload shape: {list(payload)[:10]}")

    df = pd.DataFrame(hourly).rename(columns={"time": "observed_at", **AQ_RENAME})
    df["observed_at"] = pd.to_datetime(df["observed_at"])

    # NaNs are left as NaNs on purpose - see module docstring of features.py.
    # Silently dropping or zero-filling a pollutant gap would bias the model.
    return df.sort_values("observed_at").reset_index(drop=True)


def fetch_aq_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """Historical hourly air quality, inclusive of both dates ('YYYY-MM-DD')."""
    return _to_frame(_get_json({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AQ_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }))


def fetch_aq_live(past_days: int = 7, forecast_days: int = 4) -> pd.DataFrame:
    """
    Recent + near-future hourly air quality.

    The hourly feature pipeline uses this instead of a single "current reading"
    call: pulling a window means one missed pipeline run self-heals on the next
    run rather than leaving a permanent hole in the feature store.
    """
    return _to_frame(_get_json({
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(AQ_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }))


def trim_to_now(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows later than the current hour.

    The live endpoint returns whole days, so a call made at 15:00 also hands
    back 16:00-23:00 - which are CAMS *forecast* values, not observations.
    Storing those as observed AQI would mean training targets built from a
    model's own forecast rather than from what actually happened, and would
    make the most recent rows look artificially predictable. Observations only.

    "Now" comes from config.now_local_naive(), NOT datetime.now(): the data is
    Asia/Karachi wall time, and GitHub Actions runners are UTC. Using the
    machine clock silently drops the five newest hours on every CI run.
    """
    now = pd.Timestamp(now_local_naive()).floor("h")
    return df[df["observed_at"] <= now].reset_index(drop=True)


def fetch_aq_archive_chunked(start_date: str, end_date: str, chunk_days: int = 365) -> pd.DataFrame:
    """Year-at-a-time pull so one failed chunk doesn't cost the whole backfill."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        print(f"  AQ archive {cursor} -> {chunk_end}")
        frames.append(fetch_aq_archive(cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["observed_at"]).sort_values("observed_at").reset_index(drop=True)


def get_aq_upto_now(start_date: str = AQ_ARCHIVE_START) -> pd.DataFrame:
    """
    Full observed AQ history from `start_date` to the present hour.

    Same archive-lags-reality stitch as the weather client: the dated archive
    mode trails real time by several days, so the tail is filled from live mode.
    """
    today = now_local_naive().date()  # city-local date, not the runner's UTC date
    archive_end = today - timedelta(days=8)

    frames = []
    if datetime.strptime(start_date, "%Y-%m-%d").date() <= archive_end:
        frames.append(fetch_aq_archive_chunked(start_date, archive_end.isoformat()))

    print("  AQ recent-gap fill from live endpoint (past_days=14)")
    frames.append(fetch_aq_live(past_days=14, forecast_days=1))

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["observed_at"], keep="first")
    out = out.sort_values("observed_at").reset_index(drop=True)
    return trim_to_now(out)


def fetch_current_reading() -> dict:
    """
    Single most recent observed hour, as a flat dict.
    Shape-compatible with aqicn_client.parse_reading() so the hourly pipeline
    can treat either source interchangeably.
    """
    df = fetch_aq_live(past_days=2, forecast_days=1)
    now = pd.Timestamp(now_local_naive()).floor("h")
    observed = df[(df["observed_at"] <= now) & df["aqi"].notna()]
    if observed.empty:
        raise AQClientError("No non-null AQ observation at or before the current hour")

    row = observed.iloc[-1].to_dict()
    row["station"] = "open-meteo-cams"
    return row


if __name__ == "__main__":
    print("--- live ---")
    cur = fetch_current_reading()
    print({k: cur[k] for k in ["observed_at", "station", "aqi", "pm25", "pm10"]})

    print("\n--- archive sample ---")
    a = fetch_aq_archive("2024-01-01", "2024-01-03")
    print(a.head(3).to_string())
    print(f"rows={len(a)} cols={list(a.columns)}")
