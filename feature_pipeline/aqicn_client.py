"""
Thin client around the AQICN (waqi.info) feed API.

AQICN's free API only exposes the CURRENT reading per station - it has
no historical query endpoint on the free tier. That's why the backfill
strategy (see backfill.py) is "poll now, run hourly, accumulate history
yourself" rather than "pull N days of history in one call."
"""
import requests
from datetime import datetime, timezone

from config import AQICN_BASE_URL, AQICN_TOKEN, AQICN_CITY_SLUG


class AQICNError(RuntimeError):
    pass


def fetch_current_reading(city_slug: str = AQICN_CITY_SLUG) -> dict:
    """
    Hits GET /feed/{city}/?token=... and returns the raw JSON 'data' block.
    Raises AQICNError on non-'ok' status or network failure.
    """
    if not AQICN_TOKEN:
        raise AQICNError(
            "AQICN_TOKEN is not set. Get a free token at "
            "https://aqicn.org/data-platform/token/ and put it in your .env"
        )

    url = f"{AQICN_BASE_URL}/feed/{city_slug}/"
    resp = requests.get(url, params={"token": AQICN_TOKEN}, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "ok":
        raise AQICNError(f"AQICN API returned status={payload.get('status')}: {payload}")

    return payload["data"]


def parse_reading(raw: dict) -> dict:
    """
    Flattens the nested AQICN response into a single-level dict of
    raw fields. This is the ONLY function that should know about
    AQICN's specific JSON shape - everything downstream works off
    this flat dict.

    AQICN 'iaqi' block has keys like:
      pm25, pm10, o3, no2, so2, co, t (temp), h (humidity),
      p (pressure), w (wind), dew
    Not all stations report all pollutants - missing ones become None.
    """
    iaqi = raw.get("iaqi", {})

    def _get(key):
        v = iaqi.get(key)
        return v.get("v") if v else None

    # AQICN gives 'time.iso' e.g. "2026-08-10T14:00:00+05:00" - this is
    # the station's last update time, which is what we timestamp the row with.
    time_block = raw.get("time", {})
    reading_time = time_block.get("iso")
    if reading_time:
        observed_at = datetime.fromisoformat(reading_time)
    else:
        observed_at = datetime.now(timezone.utc)

    return {
        "observed_at": observed_at,
        "station": raw.get("city", {}).get("name"),
        "aqi": raw.get("aqi"),  # overall composite AQI - this is our TARGET
        "pm25": _get("pm25"),
        "pm10": _get("pm10"),
        "o3": _get("o3"),
        "no2": _get("no2"),
        "so2": _get("so2"),
        "co": _get("co"),
        "temperature": _get("t"),
        "humidity": _get("h"),
        "pressure": _get("p"),
        "wind": _get("w"),
        "dew_point": _get("dew"),
    }


def fetch_and_parse(city_slug: str = AQICN_CITY_SLUG) -> dict:
    raw = fetch_current_reading(city_slug)
    return parse_reading(raw)
