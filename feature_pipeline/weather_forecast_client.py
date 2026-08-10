"""
Open-Meteo weather client - historical archive AND forward forecast.

WHY THIS MODULE EXISTS (the core modelling fix)
-----------------------------------------------
Lag/rolling features built from PAST AQI and PAST weather cannot predict AQI
24-72h ahead well, because they carry no information about the weather that is
COMING. Whether Lahore's AQI is 90 or 320 three days from now is driven mostly
by what the atmosphere does between now and then - wind speed, rainfall, and
how low the boundary layer sits - none of which is knowable from history alone.

So for a prediction made at time t for target t+H, we feed the model the
*forecast* weather valid at t+H, not just the observed weather at t.

TRAIN/SERVE CONSISTENCY
-----------------------
Open-Meteo's archive and forecast endpoints serve the *same variable names in
the same units* (verified - see config.WEATHER_VARS). That is what makes this
work: at training time we read weather valid at t+H out of the archive, at
serving time we read weather valid at t+H out of the forecast. Same columns,
same units, same physical quantities.

One honest caveat, documented in REPORT.md: at training time the "forecast"
weather is actually reanalysis (i.e. what the weather truly did), while at
serving time it is a genuine forecast carrying its own error. This is the
standard "perfect prognosis" setup in air-quality forecasting. It makes offline
metrics slightly optimistic relative to live performance, and the gap grows
with horizon since weather forecasts degrade with lead time.

No API key is required for any Open-Meteo endpoint.
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
    WEATHER_VARS,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_FORECAST_URL,
    now_local_naive,
)


class WeatherClientError(RuntimeError):
    pass


def _get_json(url: str, params: dict, retries: int = 3, timeout: int = 20) -> dict:
    """
    Open-Meteo is free but rate-limited; a burst of archive calls can get a 429.
    Back off and retry rather than losing a long backfill to one transient error.

    timeout/backoff are modest (worst case ~70s across all retries), not the
    many-minutes ceiling a longer per-attempt timeout here would allow. This
    function serves both the batch pipeline and the live dashboard - on the
    dashboard, build_live_frame calls this AND the AQ client sequentially, so a
    generous per-call ceiling compounds into minutes of a blank, stuck-looking
    page rather than a prompt, visible error.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = 6 * (attempt + 1)
                print(f"    rate-limited by Open-Meteo, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise WeatherClientError(f"Open-Meteo request failed after {retries} tries: {last_err}")


def _hourly_to_frame(payload: dict, time_col: str = "observed_at") -> pd.DataFrame:
    """
    Open-Meteo returns hourly data column-wise:
        {"hourly": {"time": [...], "temperature_2m": [...], ...}}
    Flatten that into a tidy DataFrame. Missing values arrive as JSON null and
    become real NaN here - we deliberately do NOT fill them at this layer, so
    that gaps stay visible to the feature/training code instead of being
    silently invented.
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherClientError(f"Unexpected Open-Meteo payload shape: {list(payload)[:10]}")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": time_col})
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(time_col).reset_index(drop=True)


def fetch_weather_archive(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Historical (reanalysis) hourly weather for Lahore, inclusive of both dates.
    Dates are 'YYYY-MM-DD' strings in local (Asia/Karachi) time.

    Used to build the *training* copy of the forecast-weather features.
    """
    payload = _get_json(
        OPENMETEO_ARCHIVE_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(WEATHER_VARS),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": TIMEZONE,
        },
    )
    return _hourly_to_frame(payload)


def fetch_weather_forecast(forecast_days: int = 5, past_days: int = 2) -> pd.DataFrame:
    """
    Forward weather forecast for Lahore, hourly.

    `past_days` also pulls recent past hours from the same endpoint, which
    matters at serving time: the forecast endpoint's "now" can lag by an hour,
    and having a couple of days of overlap lets the webapp line up observed
    history with forecast future without a seam.

    forecast_days=5 comfortably covers our +72h horizon.
    """
    payload = _get_json(
        OPENMETEO_FORECAST_URL,
        {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "hourly": ",".join(WEATHER_VARS),
            "forecast_days": forecast_days,
            "past_days": past_days,
            "timezone": TIMEZONE,
        },
    )
    return _hourly_to_frame(payload)


def fetch_weather_archive_chunked(
    start_date: str, end_date: str, chunk_days: int = 365
) -> pd.DataFrame:
    """
    The archive endpoint will happily serve multi-year ranges, but a single
    ~4-year hourly request for 9 variables is a big response and a single point
    of failure. Chunk it by year so a hiccup costs one chunk, not the whole pull.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        print(f"  weather archive {cursor} -> {chunk_end}")
        frames.append(fetch_weather_archive(cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)

    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["observed_at"]).sort_values("observed_at").reset_index(drop=True)


def get_weather_upto_now(start_date: str) -> pd.DataFrame:
    """
    Weather from `start_date` right up to the present, stitched across the two
    endpoints.

    This exists because the archive (reanalysis) endpoint lags real time by
    roughly 5 days. The forecast endpoint with past_days fills that trailing
    gap. Without this stitch, every training run would be blind to the most
    recent week - exactly the week the live model has to predict from.
    """
    today = now_local_naive().date()  # city-local date, not the runner's UTC date
    archive_end = today - timedelta(days=6)

    frames = []
    if datetime.strptime(start_date, "%Y-%m-%d").date() <= archive_end:
        frames.append(fetch_weather_archive_chunked(start_date, archive_end.isoformat()))

    print("  weather recent-gap fill from forecast endpoint (past_days=10)")
    recent = fetch_weather_forecast(forecast_days=1, past_days=10)
    frames.append(recent)

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["observed_at"], keep="first")
    return out.sort_values("observed_at").reset_index(drop=True)


if __name__ == "__main__":
    # Smoke test: prove both endpoints work and agree on columns.
    print("--- archive sample ---")
    a = fetch_weather_archive("2024-01-01", "2024-01-03")
    print(a.head(3).to_string())
    print(f"rows={len(a)} cols={list(a.columns)}")

    print("\n--- forecast sample ---")
    f = fetch_weather_forecast()
    print(f.head(3).to_string())
    print(f"rows={len(f)} cols={list(f.columns)}")

    assert set(a.columns) == set(f.columns), "archive/forecast column mismatch!"
    print("\nOK: archive and forecast expose identical columns.")
