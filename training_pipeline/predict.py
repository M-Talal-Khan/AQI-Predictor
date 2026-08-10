"""
Serving-side prediction: turn "right now" into three forecasts.

Shared by the Streamlit dashboard and any CLI use, so there is exactly one
place that knows how to go from live APIs to a prediction. The webapp
deliberately holds no feature logic of its own - if it did, it would drift out
of step with training the first time features changed.

The live feature frame is built by the SAME build_features() used in training.
The only difference is the weather frame carries Open-Meteo forecast rows past
the last AQ observation, so the newest row's fc{H}_* columns are populated from
a real forecast instead of from history.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS
from feature_pipeline.features import build_features
from feature_pipeline.openmeteo_aq_client import fetch_aq_live, trim_to_now
from feature_pipeline.weather_forecast_client import fetch_weather_forecast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

# 14 days back covers the longest history window (168h rolling) with slack.
LIVE_PAST_DAYS = 14


def load_bundle(horizon: int, model_dir: str = MODEL_DIR):
    """
    Load a trained model bundle. Returns None if that horizon has no model yet,
    so the dashboard can degrade to "no model" instead of crashing.
    """
    path = os.path.join(model_dir, f"model_{horizon}h.pkl")
    if not os.path.exists(path):
        return None

    bundle = joblib.load(path)
    if bundle.get("type") != "tensorflow_mlp":
        return bundle

    # Loading a Keras model can fail where it was not written - TF is pinned to
    # 2.18 in CI/Streamlit Cloud (protobuf compatibility with hopsworks) while
    # local dev runs 2.21, and TF may be absent entirely. Fall back to the
    # sklearn model saved next to it rather than taking the dashboard down.
    try:
        import tensorflow as tf
        keras_path = bundle.get("keras_path")
        # Re-resolve relative to model_dir: the path baked in at training time
        # points at the training machine, which is not where serving runs.
        if not keras_path or not os.path.exists(keras_path):
            keras_path = os.path.join(model_dir, f"model_{horizon}h.keras")
        bundle["model"] = tf.keras.models.load_model(keras_path, compile=False)
        return bundle
    except Exception as e:
        fallback_path = os.path.join(model_dir, f"model_{horizon}h_fallback.pkl")
        if os.path.exists(fallback_path):
            print(f"[{horizon}h] Keras load failed ({type(e).__name__}); "
                  f"using non-TF fallback model.")
            fb = joblib.load(fallback_path)
            fb["degraded_from"] = "tensorflow_mlp"
            return fb
        raise


def build_live_frame(past_days: int = LIVE_PAST_DAYS) -> pd.DataFrame:
    """
    Fetch live AQ + forecast weather and build the feature frame.
    Targets are omitted - t+H has not happened, that is the whole point.
    """
    aq = trim_to_now(fetch_aq_live(past_days=past_days, forecast_days=1))
    weather = fetch_weather_forecast(forecast_days=5, past_days=past_days)

    frame = build_features(aq, weather, with_targets=False)
    # Only rows with an observed AQI can serve as a prediction origin.
    return frame[frame["aqi"].notna()].reset_index(drop=True)


def build_frame_from_store(hours: int = 24 * 30) -> pd.DataFrame:
    """
    Read recent engineered rows straight out of the Hopsworks feature store.

    This is the point of having a feature store: the hourly pipeline already
    computed every feature, including the fc{H}_* forecast-weather block, at
    ingestion time using the forecast that was valid then. Serving from those
    rows means the dashboard predicts from exactly the values that were
    written, with no chance of the serving path recomputing something subtly
    different from what training saw.

    Returns an empty frame if Hopsworks is unavailable, so the caller can fall
    back to building features live.
    """
    from feature_pipeline.store import hopsworks_available

    if not hopsworks_available():
        return pd.DataFrame()

    try:
        from feature_pipeline.store import read_all_features

        df = read_all_features()
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["observed_at"] = pd.to_datetime(df["observed_at"])
        # The store round-trips timestamps as UTC-labelled; the values are
        # Asia/Karachi wall time (see config.now_local_naive). Strip the label
        # rather than convert, or every timestamp shifts by five hours.
        if getattr(df["observed_at"].dtype, "tz", None) is not None:
            df["observed_at"] = df["observed_at"].dt.tz_localize(None)

        df = df.sort_values("observed_at").reset_index(drop=True)
        df = df[df["aqi"].notna()]
        return df.tail(hours).reset_index(drop=True)
    except Exception as e:
        print(f"feature-store read failed ({type(e).__name__}: {e}); "
              f"falling back to live API")
        return pd.DataFrame()


def load_bundle_from_registry(horizon: int, model_dir: str = MODEL_DIR):
    """
    Fetch a model from the Hopsworks model registry instead of the repo.

    Downloads into `model_dir` and then hands off to load_bundle, so the
    TensorFlow-optional fallback logic still applies. Callers should cache this
    (the dashboard wraps it in @st.cache_resource) - the brief is explicit that
    the registry must not be hit on every page load, and a download per render
    would be far worse than a local read.
    """
    try:
        from feature_pipeline.store import get_project
        from config import MODEL_NAME

        mr = get_project().get_model_registry()
        models = mr.get_models(f"{MODEL_NAME}_{horizon}h")
        if not models:
            return None
        best = max(models, key=lambda m: m.version)
        path = best.download()

        for fname in os.listdir(path):
            if fname.startswith(f"model_{horizon}h"):
                shutil_copy(os.path.join(path, fname), os.path.join(model_dir, fname))
        return load_bundle(horizon, model_dir)
    except Exception as e:
        print(f"registry fetch failed for {horizon}h ({type(e).__name__}: {e}); "
              f"using the repo copy")
        return load_bundle(horizon, model_dir)


def shutil_copy(src, dst):
    import shutil
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def latest_usable_row(frame: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Newest row whose features are all present.

    Usually the very last row, but if the forecast window is short or a feature
    is missing there, stepping back a few hours yields a valid prediction
    origin rather than a NaN-poisoned one. Predicting from silently-NaN inputs
    is how a dashboard ends up confidently displaying nonsense.
    """
    have = [c for c in feature_cols if c in frame.columns]
    complete = frame.dropna(subset=have)
    if complete.empty:
        return pd.DataFrame()
    return complete.tail(1)


def predict_horizon(bundle: dict, row: pd.DataFrame) -> float:
    X = row[bundle["feature_cols"]]
    if bundle.get("scaler") is not None:
        X = bundle["scaler"].transform(X)
    pred = bundle["model"].predict(X, verbose=0) if bundle["type"] == "tensorflow_mlp" \
        else bundle["model"].predict(X)
    return float(np.asarray(pred).flatten()[0])


def forecast_now(frame: pd.DataFrame = None, model_dir: str = MODEL_DIR,
                 loader=None) -> dict:
    """
    Returns:
      {
        "frame": full live feature frame,
        "current": {"observed_at":..., "aqi":...},
        "predictions": {24: {...}, 48: {...}, 72: {...}},
      }
    Each prediction carries the timestamp it is valid for, so the dashboard
    plots it at the right place on the x-axis rather than assuming +H from now.

    `loader` overrides how model bundles are obtained. The dashboard passes its
    @st.cache_resource-wrapped loader here so models are deserialised once per
    process instead of on every page view - reloading a Keras model per render
    costs seconds and buys nothing, since models only change when the daily
    training job reruns.
    """
    load = loader or (lambda h: load_bundle(h, model_dir))

    if frame is None:
        frame = build_live_frame()
    if frame.empty:
        raise RuntimeError("No live rows with an observed AQI could be built.")

    last = frame.iloc[-1]
    out = {
        "frame": frame,
        "current": {"observed_at": last["observed_at"], "aqi": float(last["aqi"])},
        "predictions": {},
    }

    for h in FORECAST_HORIZONS:
        bundle = load(h)
        if bundle is None:
            continue
        row = latest_usable_row(frame, bundle["feature_cols"])
        if row.empty:
            continue

        origin = row.iloc[0]["observed_at"]
        out["predictions"][h] = {
            "value": predict_horizon(bundle, row),
            "valid_at": origin + pd.Timedelta(hours=h),
            "origin": origin,
            "model_type": bundle["type"],
            "metrics": bundle.get("metrics", {}),
            "trained_at": bundle.get("trained_at"),
        }

    return out


if __name__ == "__main__":
    res = forecast_now()
    c = res["current"]
    print(f"Current: {c['observed_at']}  AQI={c['aqi']:.0f}")
    for h, p in res["predictions"].items():
        r2 = p["metrics"].get("r2")
        print(f"  +{h}h -> {p['value']:6.1f}  (valid {p['valid_at']}, "
              f"{p['model_type']}, test R2={r2:.3f})")
