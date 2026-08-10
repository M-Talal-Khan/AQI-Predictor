"""
Central config for the AQI predictor.
Keep all magic numbers / city info / thresholds here so every
script (feature pipeline, training, webapp) reads the same values.
"""
import os

# Load .env if present so local runs pick up secrets without exporting them by
# hand. In GitHub Actions / Streamlit Cloud there is no .env file and the values
# come from real environment variables instead - load_dotenv is a no-op there
# and never overrides an already-set variable.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:  # python-dotenv is optional
    pass

# --- City ---
CITY_NAME = "Lahore"
# AQICN station search string - waqi.info lets you query by city name directly,
# e.g. https://api.waqi.info/feed/lahore/?token=...
AQICN_CITY_SLUG = "lahore"
# Lahore coordinates - used by the Open-Meteo clients (weather + AQ history),
# which are location-based rather than station-based.
LATITUDE = 31.5204
LONGITUDE = 74.3587
TIMEZONE = "Asia/Karachi"

# --- API ---
AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")  # get a free token at https://aqicn.org/data-platform/token/
AQICN_BASE_URL = "https://api.waqi.info"

# --- Open-Meteo (no API key required) ---
# Three separate hosts, all free for non-commercial use:
#   archive   -> reanalysis weather, 1940..~5 days ago
#   forecast  -> weather forecast, now..+16 days
#   air-quality -> CAMS air quality, 2022-07-29..+5 days (incl. us_aqi)
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Weather variables pulled from Open-Meteo. The SAME names are served by both
# the archive and the forecast endpoints - that is deliberate and load-bearing:
# it means the historical weather we train on and the forecast weather we serve
# on are the same physical quantities in the same units, so a model trained on
# one can be fed the other without a translation layer.
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "dew_point_2m",
    "boundary_layer_height",  # strong AQI driver - low BLH traps pollution
    "cloud_cover",
]

# Air-quality variables pulled from Open-Meteo's CAMS archive for the backfill.
AQ_VARS = [
    "us_aqi",
    "pm2_5",
    "pm10",
    "ozone",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
]

# Earliest date the CAMS air-quality archive has data for.
AQ_ARCHIVE_START = "2022-09-01"

# --- Hopsworks ---
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "aqi_predictor")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_feature_view"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "aqi_forecast_model"

# --- Feature engineering ---
# how many past hourly AQI readings to use as lag features
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 72]
ROLLING_WINDOWS = [3, 6, 24, 72, 168]  # hours, for rolling mean/std/min/max

# --- Forecast horizon ---
FORECAST_DAYS = 3
# Hours ahead we forecast. One model is trained per horizon, each fed that
# horizon's own forecast-weather block (see feature_pipeline/features.py).
FORECAST_HORIZONS = (24, 48, 72)

# --- Hazard levels (US EPA AQI breakpoints) ---
AQI_CATEGORIES = [
    (0, 50, "Good"),
    (51, 100, "Moderate"),
    (101, 150, "Unhealthy for Sensitive Groups"),
    (151, 200, "Unhealthy"),
    (201, 300, "Very Unhealthy"),
    (301, 500, "Hazardous"),
]
HAZARD_ALERT_THRESHOLD = 151  # trigger alert at "Unhealthy" and above

# Official US EPA category colours, used by the dashboard so the colour coding
# matches what people see on any other AQI service.
AQI_CATEGORY_COLORS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
    "Unknown": "#9E9E9E",
}

# Plain-language guidance shown alongside the forecast.
AQI_CATEGORY_ADVICE = {
    "Good": "Air quality is satisfactory. Outdoor activity is fine.",
    "Moderate": "Acceptable, but unusually sensitive people should limit prolonged exertion outdoors.",
    "Unhealthy for Sensitive Groups": "Children, older adults and people with asthma or heart conditions should limit prolonged outdoor exertion.",
    "Unhealthy": "Everyone may experience effects. Limit prolonged outdoor exertion; sensitive groups should stay indoors.",
    "Very Unhealthy": "Health alert. Avoid outdoor exertion; keep windows closed and run air purifiers if available.",
    "Hazardous": "Emergency conditions. Everyone should remain indoors and avoid all physical activity outdoors.",
    "Unknown": "No guidance available.",
}


def category_color(aqi_value: float) -> str:
    return AQI_CATEGORY_COLORS.get(categorize_aqi(aqi_value), "#9E9E9E")


def category_advice(aqi_value: float) -> str:
    return AQI_CATEGORY_ADVICE.get(categorize_aqi(aqi_value), "")


def categorize_aqi(aqi_value: float) -> str:
    for low, high, label in AQI_CATEGORIES:
        if low <= aqi_value <= high:
            return label
    return "Hazardous" if aqi_value > 500 else "Unknown"
