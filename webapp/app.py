"""
Streamlit dashboard for the Lahore AQI predictor.

    streamlit run webapp/app.py

CACHING STRATEGY (as specified)
-------------------------------
- Models are loaded once per process with @st.cache_resource. Reloading a
  Keras model on every page view would add seconds to each interaction and
  gains nothing: the model only changes when the daily training job reruns.
- Live features are cached with @st.cache_data(ttl=3600). The upstream data is
  hourly, so refetching more often than that costs API quota and returns the
  same numbers. The cache key includes the current hour, so it also refreshes
  promptly when a new hour's data lands rather than at an arbitrary offset.

All feature construction lives in training_pipeline/predict.py, which uses the
same build_features() as training. The dashboard contains no feature logic of
its own, so it cannot drift away from what the models were trained on.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CITY_NAME,
    FORECAST_HORIZONS,
    HAZARD_ALERT_THRESHOLD,
    AQI_CATEGORY_COLORS,
    categorize_aqi,
    category_color,
    category_advice,
    now_local_naive,
)
from training_pipeline.predict import (
    MODEL_DIR,
    build_live_frame,
    forecast_now,
    load_bundle,
)
from training_pipeline.explain import load_saved

st.set_page_config(
    page_title=f"{CITY_NAME} AQI Forecast",
    page_icon="🌫️",
    layout="wide",
)


# --------------------------------------------------------------------------
# cached data / model access
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Fetching live air quality and weather forecast...")
def load_live_frame(hour_key: str) -> pd.DataFrame:
    """hour_key busts the cache when the clock rolls into a new hour."""
    return build_live_frame()


@st.cache_resource
def get_bundle(horizon: int):
    return load_bundle(horizon)


@st.cache_data(ttl=3600)
def get_importance(horizon: int):
    return load_saved(os.path.join(MODEL_DIR, f"shap_{horizon}h.json"))


def current_hour_key() -> str:
    # City-local hour, not the server's. Streamlit Cloud runs UTC, and the data
    # is Asia/Karachi wall time - keying on UTC would roll the cache over at the
    # wrong moment relative to when new hourly data actually lands.
    return now_local_naive().strftime("%Y-%m-%dT%H")


# --------------------------------------------------------------------------
# presentation helpers
# --------------------------------------------------------------------------

def aqi_badge(label: str, value: float, sub: str = "") -> str:
    color = category_color(value)
    cat = categorize_aqi(value)
    # Dark text on the light end of the scale, white on the dark end, so the
    # number stays readable across the whole EPA palette.
    text = "#111" if cat in ("Good", "Moderate", "Unhealthy for Sensitive Groups") else "#fff"
    return f"""
    <div style="background:{color};color:{text};border-radius:12px;padding:14px 16px;
                text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.18);">
      <div style="font-size:0.85rem;font-weight:600;opacity:.85;">{label}</div>
      <div style="font-size:2.4rem;font-weight:700;line-height:1.1;">{value:.0f}</div>
      <div style="font-size:0.82rem;font-weight:600;">{cat}</div>
      <div style="font-size:0.72rem;opacity:.8;">{sub}</div>
    </div>"""


def category_legend() -> str:
    chips = "".join(
        f'<span style="background:{c};color:#111;padding:2px 8px;border-radius:10px;'
        f'font-size:0.7rem;margin-right:5px;display:inline-block;">{n}</span>'
        for n, c in AQI_CATEGORY_COLORS.items() if n != "Unknown"
    )
    return f'<div style="margin:6px 0 2px;">{chips}</div>'


# --------------------------------------------------------------------------

def main():
    st.title(f"🌫️ {CITY_NAME} — AQI forecast")
    st.caption(
        "Hourly CAMS air quality + Open-Meteo weather forecast. Each horizon has its own "
        "model, fed the forecast weather valid at that horizon."
    )

    try:
        frame = load_live_frame(current_hour_key())
    except Exception as e:
        st.error(f"Could not fetch live data: {e}")
        st.stop()

    if frame.empty:
        st.warning("No live data available yet. Run the feature pipeline first.")
        st.stop()

    try:
        # get_bundle is @st.cache_resource-wrapped, so models deserialise once
        # per process rather than on every rerun.
        result = forecast_now(frame=frame, loader=get_bundle)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    current = result["current"]
    preds = result["predictions"]

    # ---- headline row ----------------------------------------------------
    st.markdown(category_legend(), unsafe_allow_html=True)
    cols = st.columns(1 + len(FORECAST_HORIZONS))
    obs_time = pd.to_datetime(current["observed_at"])
    cols[0].markdown(
        aqi_badge("NOW", current["aqi"], obs_time.strftime("%d %b, %H:%M")),
        unsafe_allow_html=True,
    )

    for i, h in enumerate(FORECAST_HORIZONS):
        col = cols[i + 1]
        if h not in preds:
            col.info(f"+{h}h\n\nno model yet")
            continue
        p = preds[h]
        col.markdown(
            aqi_badge(f"+{h}h", p["value"], pd.to_datetime(p["valid_at"]).strftime("%d %b, %H:%M")),
            unsafe_allow_html=True,
        )

    # ---- hazard alert ----------------------------------------------------
    breaches = {h: p for h, p in preds.items() if p["value"] >= HAZARD_ALERT_THRESHOLD}
    st.write("")
    if breaches:
        worst_h = max(breaches, key=lambda h: breaches[h]["value"])
        worst = breaches[worst_h]
        cat = categorize_aqi(worst["value"])
        horizons_txt = ", ".join(f"+{h}h" for h in sorted(breaches))
        st.error(
            f"**⚠ Hazard alert — {cat}.** Forecast AQI reaches "
            f"**{worst['value']:.0f}** at +{worst_h}h "
            f"({pd.to_datetime(worst['valid_at']).strftime('%a %d %b, %H:%M')}). "
            f"Threshold breached at: {horizons_txt}.\n\n{category_advice(worst['value'])}"
        )
    elif preds:
        st.success(
            f"No hazard alert. All forecast horizons stay below the "
            f"'Unhealthy' threshold of {HAZARD_ALERT_THRESHOLD}."
        )

    # ---- forecast chart --------------------------------------------------
    st.subheader("Forecast")
    hist = frame.tail(24 * 7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["observed_at"], y=hist["aqi"], mode="lines",
        name="Observed AQI", line=dict(color="#1f77b4", width=2),
    ))

    if preds:
        ordered = sorted(preds)
        fx = [obs_time] + [pd.to_datetime(preds[h]["valid_at"]) for h in ordered]
        fy = [current["aqi"]] + [preds[h]["value"] for h in ordered]
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="markers+lines", name="Forecast",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=10),
        ))

    fig.add_hline(
        y=HAZARD_ALERT_THRESHOLD, line_dash="dot", line_color="#8B0000",
        annotation_text=f"Hazard threshold ({HAZARD_ALERT_THRESHOLD})",
        annotation_position="top left",
    )
    fig.update_layout(
        height=420, yaxis_title="US AQI", xaxis_title="",
        hovermode="x unified", margin=dict(t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    st.plotly_chart(fig, width='stretch')

    # ---- model quality ---------------------------------------------------
    with st.expander("Model performance on held-out test data", expanded=False):
        rows = []
        for h in FORECAST_HORIZONS:
            if h not in preds:
                continue
            m = preds[h]["metrics"]
            rows.append({
                "Horizon": f"+{h}h",
                "Model": preds[h]["model_type"],
                "RMSE": round(m.get("rmse", float("nan")), 2),
                "MAE": round(m.get("mae", float("nan")), 2),
                "R²": round(m.get("r2", float("nan")), 3),
                "Trained": (preds[h].get("trained_at") or "")[:16].replace("T", " "),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            st.caption(
                "Scored on the most recent 20% of history, split chronologically "
                "(never randomly — a random split on a time series leaks the answer)."
            )

    # ---- historical trend ------------------------------------------------
    st.subheader("Recent trend")
    days = st.radio("Window", [7, 14, 30], index=0, horizontal=True,
                    format_func=lambda d: f"last {d} days")
    recent = frame.tail(24 * days)

    tfig = go.Figure()
    tfig.add_trace(go.Scatter(
        x=recent["observed_at"], y=recent["aqi"], mode="lines",
        name="AQI", line=dict(color="#1f77b4", width=1.6),
        fill="tozeroy", fillcolor="rgba(31,119,180,.12)",
    ))
    if "aqi_rolling_mean_24h" in recent.columns:
        tfig.add_trace(go.Scatter(
            x=recent["observed_at"], y=recent["aqi_rolling_mean_24h"],
            mode="lines", name="24h mean", line=dict(color="#ff7f0e", width=2),
        ))
    tfig.update_layout(height=320, yaxis_title="US AQI", hovermode="x unified",
                       margin=dict(t=20, b=10),
                       legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    st.plotly_chart(tfig, width='stretch')

    c1, c2, c3, c4 = st.columns(4)
    aqi_recent = recent["aqi"].dropna()
    c1.metric(f"{days}-day mean", f"{aqi_recent.mean():.0f}")
    c2.metric(f"{days}-day max", f"{aqi_recent.max():.0f}")
    c3.metric(f"{days}-day min", f"{aqi_recent.min():.0f}")
    unhealthy = (aqi_recent >= HAZARD_ALERT_THRESHOLD).mean() * 100
    c4.metric("Hours 'Unhealthy'+", f"{unhealthy:.0f}%")

    # ---- explainability --------------------------------------------------
    st.subheader("What's driving the forecast")
    horizon_choice = st.selectbox("Horizon", list(FORECAST_HORIZONS),
                                  format_func=lambda h: f"+{h}h")
    imp = get_importance(horizon_choice)

    if imp is None:
        st.info("Feature importances are computed by the training pipeline. "
                "Run `python training_pipeline/train.py` to produce them.")
    else:
        table, method = imp
        table = table.sort_values("mean_abs_shap", ascending=True).tail(15)
        colors = ["#d62728" if f.startswith("fc") else "#1f77b4" for f in table["feature"]]
        ifig = go.Figure(go.Bar(
            x=table["mean_abs_shap"], y=table["feature"],
            orientation="h", marker_color=colors,
        ))
        ifig.update_layout(
            height=460, xaxis_title="mean |SHAP| (impact on predicted AQI)",
            margin=dict(t=10, b=10, l=10),
        )
        st.plotly_chart(ifig, width='stretch')
        st.caption(
            f"Method: {method}. **Red bars are forecast-weather features** "
            f"(`fc{horizon_choice}_*`) — weather predicted for the target hour. "
            "Blue bars are history-based features."
        )

    # ---- current conditions ---------------------------------------------
    with st.expander("Current pollutant and weather readings"):
        last = frame.iloc[-1]
        pollutants = {"PM2.5": "pm25", "PM10": "pm10", "O₃": "o3",
                      "NO₂": "no2", "SO₂": "so2", "CO": "co"}
        weather = {"Temp (°C)": "temperature_2m", "Humidity (%)": "relative_humidity_2m",
                   "Wind (km/h)": "wind_speed_10m", "Pressure (hPa)": "surface_pressure",
                   "Mixing depth (m)": "boundary_layer_height"}

        pc = st.columns(len(pollutants))
        for (label, col), c in zip(pollutants.items(), pc):
            c.metric(label, f"{last[col]:.1f}" if pd.notna(last.get(col)) else "—")

        wc = st.columns(len(weather))
        for (label, col), c in zip(weather.items(), wc):
            c.metric(label, f"{last[col]:.1f}" if pd.notna(last.get(col)) else "—")

    st.caption(
        f"Data: Open-Meteo CAMS air quality + Open-Meteo weather forecast · "
        f"Latest observation {obs_time:%Y-%m-%d %H:%M} (Asia/Karachi) · "
        f"Page rendered {now_local_naive():%H:%M} PKT"
    )


if __name__ == "__main__":
    main()
