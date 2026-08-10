"""
Feature engineering.

THE CENTRAL IDEA
----------------
Predicting AQI 24-72h ahead from lagged AQI alone is close to hopeless past the
first few hours. Lag/rolling features describe where pollution has BEEN; what
decides where it goes over the next three days is the weather that is COMING -
a monsoon burst scrubs the air, a collapsed night-time boundary layer traps it.

So for every row at time t and every horizon H we attach the weather valid at
t+H, plus summaries of the weather over the interval between them. Those are
the `fc{H}_*` columns. Everything else is "as of t" history.

ONE CODE PATH FOR TRAIN AND SERVE
---------------------------------
build_features() takes an AQ frame and a weather frame and does not care
whether the weather rows are past or future. At training time the weather frame
is archive data, so shifting -24h lands on real observed weather. At serving
time the weather frame has the Open-Meteo forecast appended, so the identical
shift lands on forecast weather. Same columns, same code, no separate serving
path to drift out of sync with training.

Rows whose t+H weather is missing get NaN features; rows whose t+H AQI has not
happened yet get NaN targets. Both are dropped at training time, and the latter
is exactly the row we want to predict at serving time.

NaN POLICY
----------
Gaps are interpolated only when short (<= MAX_INTERP_GAP_HOURS) and are
otherwise left as NaN. Long gaps are real missing data - inventing values there
would put fabricated history into lag features and quietly bias the model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from config import LAG_HOURS, ROLLING_WINDOWS, WEATHER_VARS, FORECAST_HORIZONS

# Pollutant/observation columns carried from the AQ source.
RAW_AQ_COLUMNS = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co"]

# Short gaps get linearly interpolated; anything longer stays NaN.
MAX_INTERP_GAP_HOURS = 3

# Weather variables summarised over the lead-time window, and how.
# precipitation is summed (total rainfall clears particulates); boundary layer
# height takes its minimum (the worst trapping moment in the window dominates);
# the rest are averaged.
WINDOW_AGGS = {
    "precipitation": "sum",
    "boundary_layer_height": "min",
    "wind_speed_10m": "mean",
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
}


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def to_hourly_grid(df: pd.DataFrame, time_col: str = "observed_at") -> pd.DataFrame:
    """
    Snap to a gap-free hourly index.

    This is load-bearing rather than cosmetic: every lag, rolling and forecast
    feature below is built with .shift(n), which means "n ROWS". That only
    equals "n hours" if the index is complete and evenly spaced. Feed this
    pipeline a frame with missing hours and every lag silently misaligns.
    """
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.drop_duplicates(subset=[time_col]).sort_values(time_col)

    full = pd.date_range(out[time_col].min(), out[time_col].max(), freq="h")
    out = out.set_index(time_col).reindex(full)
    out.index.name = time_col
    return out.reset_index()


def merge_aq_and_weather(aq: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join AQ observations with weather on the hourly grid.

    Outer, not inner, on purpose: at serving time the weather frame runs ~72h
    past the last AQ observation, and those future-weather rows are precisely
    what the forecast features are built from. An inner join would discard them
    and leave nothing to predict with.
    """
    aq = aq.copy()
    weather = weather.copy()
    aq["observed_at"] = pd.to_datetime(aq["observed_at"])
    weather["observed_at"] = pd.to_datetime(weather["observed_at"])

    keep_aq = ["observed_at"] + [c for c in RAW_AQ_COLUMNS if c in aq.columns]
    keep_w = ["observed_at"] + [c for c in WEATHER_VARS if c in weather.columns]

    merged = pd.merge(aq[keep_aq], weather[keep_w], on="observed_at", how="outer")
    return to_hourly_grid(merged)


def interpolate_short_gaps(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Bridge gaps of <= MAX_INTERP_GAP_HOURS; leave longer ones as NaN."""
    out = df.copy()
    columns = columns or [c for c in RAW_AQ_COLUMNS + WEATHER_VARS if c in out.columns]
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].interpolate(
            method="linear", limit=MAX_INTERP_GAP_HOURS, limit_area="inside"
        )
    return out


# --------------------------------------------------------------------------
# feature blocks
# --------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, time_col: str = "observed_at") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[time_col])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Cyclical encodings so the model sees hour 23 and hour 0 as adjacent,
    # and December as adjacent to January - Lahore's smog season straddles
    # the year boundary, so that wrap-around genuinely matters here.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def _add_wind_direction_components(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    Wind direction is circular - 359 deg and 1 deg are 2 deg apart, but as a raw
    number they look 358 apart. Split into u/v components instead. Direction
    matters for Lahore specifically: north-westerlies carry crop-burning smoke
    in from Punjab during the post-harvest season.
    """
    if col not in df.columns:
        return df
    rad = np.deg2rad(pd.to_numeric(df[col], errors="coerce"))
    df[f"{prefix}_sin"] = np.sin(rad)
    df[f"{prefix}_cos"] = np.cos(rad)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only AQI dynamics. Assumes a gap-free hourly grid (see to_hourly_grid)."""
    df = df.copy()

    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag)

    for window in ROLLING_WINDOWS:
        roll = df["aqi"].rolling(window=window, min_periods=max(2, window // 2))
        df[f"aqi_rolling_mean_{window}h"] = roll.mean()
        df[f"aqi_rolling_std_{window}h"] = roll.std()
        df[f"aqi_rolling_max_{window}h"] = roll.max()
        df[f"aqi_rolling_min_{window}h"] = roll.min()

    df["aqi_change_rate_1h"] = df["aqi"].diff(1)
    df["aqi_change_rate_3h"] = df["aqi"].diff(3)
    df["aqi_change_rate_24h"] = df["aqi"].diff(24)

    # Where does the current level sit against the recent norm? Helps the model
    # distinguish "high and rising" from "high but already decaying".
    df["aqi_vs_24h_mean"] = df["aqi"] - df["aqi_rolling_mean_24h"]

    # Same-hour-yesterday / same-hour-last-week: AQI is strongly diurnal, so
    # these are much stronger baselines than a plain 24h mean.
    df["aqi_lag_24h_same_hour"] = df["aqi"].shift(24)
    df["aqi_lag_168h_same_hour"] = df["aqi"].shift(168)

    for pollutant in ["pm25", "pm10", "no2", "o3", "co", "so2"]:
        if pollutant in df.columns:
            df[f"{pollutant}_rolling_mean_24h"] = (
                df[pollutant].rolling(24, min_periods=6).mean()
            )

    return df


def add_current_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Weather as observed at t, plus 24h context."""
    df = df.copy()
    df = _add_wind_direction_components(df, "wind_direction_10m", "wind_dir")

    for var in ["wind_speed_10m", "boundary_layer_height", "temperature_2m"]:
        if var in df.columns:
            df[f"{var}_rolling_mean_24h"] = df[var].rolling(24, min_periods=6).mean()

    if "precipitation" in df.columns:
        df["precip_sum_24h"] = df["precipitation"].rolling(24, min_periods=6).sum()

    # Ventilation index ~ how vigorously the atmosphere can disperse pollution.
    # The product of mixing depth and wind speed is a standard air-quality
    # diagnostic and is far more informative than either term alone.
    if "boundary_layer_height" in df.columns and "wind_speed_10m" in df.columns:
        df["ventilation_index"] = df["boundary_layer_height"] * df["wind_speed_10m"]

    return df


def add_forecast_weather_features(
    df: pd.DataFrame, horizons=FORECAST_HORIZONS
) -> pd.DataFrame:
    """
    THE KEY FEATURE BLOCK.

    For each horizon H, attach to row t:
      fc{H}_<var>            weather valid exactly at t+H
      fc{H}_<var>_delta      how much that differs from now (the trend)
      fc{H}_<var>_w24_<agg>  the var summarised over the 24h ending at t+H
      fc{H}_precip_total     cumulative rainfall over the whole interval (t, t+H]
      fc{H}_ventilation      mixing depth x wind speed at t+H

    Negative shifts look like leakage but are not: these are all WEATHER
    columns, never AQI. Weather at t+H is legitimately knowable at time t from
    a forecast, which is the entire premise. No AQI value at or after t+H ever
    enters the feature set - that would be leakage, and is what the targets are.

    Requires a gap-free hourly grid so shift(-H) means exactly H hours.
    """
    df = df.copy()
    available = [v for v in WEATHER_VARS if v in df.columns]

    for h in horizons:
        for var in available:
            if var == "wind_direction_10m":
                # circular - shift then decompose, never average raw degrees
                shifted = df[var].shift(-h)
                rad = np.deg2rad(pd.to_numeric(shifted, errors="coerce"))
                df[f"fc{h}_wind_dir_sin"] = np.sin(rad)
                df[f"fc{h}_wind_dir_cos"] = np.cos(rad)
                continue

            df[f"fc{h}_{var}"] = df[var].shift(-h)
            df[f"fc{h}_{var}_delta"] = df[f"fc{h}_{var}"] - df[var]

        # 24h window ending at the target hour
        for var, how in WINDOW_AGGS.items():
            if var not in df.columns:
                continue
            roll = df[var].rolling(24, min_periods=6)
            agg = getattr(roll, how)()
            df[f"fc{h}_{var}_w24_{how}"] = agg.shift(-h)

        # everything that happens between now and the target
        if "precipitation" in df.columns:
            df[f"fc{h}_precip_total"] = (
                df["precipitation"].rolling(h, min_periods=max(6, h // 4)).sum().shift(-h)
            )
        if "wind_speed_10m" in df.columns:
            df[f"fc{h}_wind_mean_lead"] = (
                df["wind_speed_10m"].rolling(h, min_periods=max(6, h // 4)).mean().shift(-h)
            )
        if "boundary_layer_height" in df.columns:
            df[f"fc{h}_blh_min_lead"] = (
                df["boundary_layer_height"].rolling(h, min_periods=max(6, h // 4)).min().shift(-h)
            )

        if f"fc{h}_boundary_layer_height" in df.columns and f"fc{h}_wind_speed_10m" in df.columns:
            df[f"fc{h}_ventilation"] = (
                df[f"fc{h}_boundary_layer_height"] * df[f"fc{h}_wind_speed_10m"]
            )

    return df


def add_forecast_targets(df: pd.DataFrame, horizon_hours=FORECAST_HORIZONS) -> pd.DataFrame:
    """
    Target for row t at horizon H is the AQI actually observed at t+H.
    Rows near the end of the frame get NaN targets because t+H has not happened
    yet - those are dropped at training time and are the rows we predict at
    serving time.
    """
    df = df.copy()
    for h in horizon_hours:
        df[f"aqi_target_{h}h"] = df["aqi"].shift(-h)
    return df


# --------------------------------------------------------------------------
# public entrypoints
# --------------------------------------------------------------------------

def build_features(
    aq: pd.DataFrame,
    weather: pd.DataFrame,
    horizons=FORECAST_HORIZONS,
    with_targets: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: raw AQ + weather in -> model-ready feature frame out.

    Used identically by backfill, the hourly pipeline and the webapp. Pass a
    weather frame that extends H hours past the last AQ observation to get
    usable forecast features for the most recent rows.
    """
    df = merge_aq_and_weather(aq, weather)
    df = interpolate_short_gaps(df)
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_current_weather_features(df)
    df = add_forecast_weather_features(df, horizons)
    if with_targets:
        df = add_forecast_targets(df, horizons)
    return df


def horizon_feature_columns(df: pd.DataFrame, horizon: int, horizons=FORECAST_HORIZONS) -> list:
    """
    Feature columns for one horizon's model: every shared history feature, plus
    only that horizon's own `fc{H}_*` block.

    Excludes the identifier/target columns, and `aqi` itself - `aqi` at time t
    is legitimate input, but it is already exposed as aqi_lag_* / rolling
    features, and keeping the bare column invites confusion with the target.
    """
    other_prefixes = tuple(f"fc{h}_" for h in horizons if h != horizon)
    exclude = {"observed_at", "station", "aqi"} | {f"aqi_target_{h}h" for h in horizons}

    cols = []
    for c in df.columns:
        if c in exclude or c.startswith(other_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


# Backwards-compatible alias for the original scaffold's entrypoint name.
def build_feature_row(history_df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(history_df)
    df = add_lag_and_rolling_features(df)
    return df
