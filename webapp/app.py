"""
Streamlit dashboard for the Lahore AQI predictor.

    streamlit run webapp/app.py

VISUAL LAYER
------------
theme.py owns all presentation (a light frosted-glass "bento card" layout,
a sticky pill navbar, a live summary dock, Plotly theming). It is purely
cosmetic and fails silently, so a styling problem can never take the
data/prediction flow down.

CACHING STRATEGY (as specified)
-------------------------------
- Models are loaded once per process with @st.cache_resource.
- Live features are cached with @st.cache_data(ttl=3600).
- All feature construction lives in training_pipeline/predict.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `import theme`

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Bridge Streamlit secrets into the environment BEFORE importing config
try:
    for _key in ("HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT_NAME", "AQICN_TOKEN"):
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:
    pass  # no secrets.toml configured - fall back to the live API path

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
    build_frame_from_store,
    forecast_now,
    load_bundle,
    load_bundle_from_registry,
)
from training_pipeline.explain import load_saved
from feature_pipeline.store import hopsworks_available
import theme

st.set_page_config(
    page_title=f"{CITY_NAME} AQI Forecast",
    page_icon="🌫️",
    layout="wide",
)
theme.inject_theme()


# --------------------------------------------------------------------------
# cached data / model access
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading air quality data...")
def load_frame(hour_key: str):
    """
    Feature store first, live API second.
    """
    df = build_frame_from_store()
    if not df.empty:
        return df, "hopsworks"
    return build_live_frame(), "live-api"


@st.cache_resource
def get_bundle(horizon: int):
    """
    Cached so models deserialise once per process, never per page view.
    """
    if hopsworks_available():
        return load_bundle_from_registry(horizon)
    return load_bundle(horizon)


@st.cache_data(ttl=3600)
def get_importance(horizon: int):
    return load_saved(os.path.join(MODEL_DIR, f"shap_{horizon}h.json"))


def current_hour_key() -> str:
    return now_local_naive().strftime("%Y-%m-%dT%H")


# --------------------------------------------------------------------------
# presentation helpers
# --------------------------------------------------------------------------

def aqi_badge(label: str, value: float, sub: str = "", hazard: bool = False) -> str:
    color = category_color(value)
    cat = categorize_aqi(value)
    return theme.glass_tile(label, value, cat, color, sub, hazard=hazard)


def category_legend() -> str:
    return theme.glass_legend(AQI_CATEGORY_COLORS)


# --------------------------------------------------------------------------
# Main Application
# --------------------------------------------------------------------------

def main():
    # 1. Floating Pill Navigation Bar
    theme.render_navbar(CITY_NAME)

    # 2. Grand Hero Display
    theme.render_hero(CITY_NAME)

    try:
        frame, source = load_frame(current_hour_key())
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    if frame.empty:
        st.warning("No live data available yet. Run the feature pipeline first.")
        st.stop()

    try:
        result = forecast_now(frame=frame, loader=get_bundle)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    current = result["current"]
    preds = result["predictions"]
    obs_time = pd.to_datetime(current["observed_at"])
    last_row = frame.iloc[-1].to_dict()

    # 3. Signature Live "AQI Dock" (Centerpiece Hero Widget)
    theme.render_live_dock(current, preds, last_row, obs_time, CITY_NAME)

    # 4. Multi-Horizon Forecast Bento Card
    with st.container(border=True):
        st.subheader("72-Hour Horizon Forecasts")
        st.markdown(category_legend(), unsafe_allow_html=True)
        cols = st.columns(1 + len(FORECAST_HORIZONS))
        cols[0].markdown(
            aqi_badge("NOW", current["aqi"], obs_time.strftime("%d %b, %H:%M")),
            unsafe_allow_html=True,
        )

        breaches = {h: p for h, p in preds.items() if p["value"] >= HAZARD_ALERT_THRESHOLD}

        for i, h in enumerate(FORECAST_HORIZONS):
            col = cols[i + 1]
            if h not in preds:
                col.info(f"+{h}h\n\nno model yet")
                continue
            p = preds[h]
            col.markdown(
                aqi_badge(f"+{h}h", p["value"],
                         pd.to_datetime(p["valid_at"]).strftime("%d %b, %H:%M"),
                         hazard=(h in breaches)),
                unsafe_allow_html=True,
            )

        # Hazard Alert Banner
        if breaches:
            worst_h = max(breaches, key=lambda h: breaches[h]["value"])
            worst = breaches[worst_h]
            cat = categorize_aqi(worst["value"])
            horizons_txt = ", ".join(f"+{h}h" for h in sorted(breaches))
            title = f"Hazard Alert — {cat} (AQI {worst['value']:.0f} at +{worst_h}h)"
            desc = (
                f"Forecast breaches the hazard threshold ({HAZARD_ALERT_THRESHOLD}) at {horizons_txt}. "
                f"Valid {pd.to_datetime(worst['valid_at']).strftime('%a %d %b, %H:%M')}. "
                f"{category_advice(worst['value'])}"
            )
            theme.render_hazard_banner(True, title, desc)
        elif preds:
            theme.render_hazard_banner(
                False,
                "Optimal Air Quality Window",
                f"All forecast horizons stay below the 'Unhealthy' threshold of {HAZARD_ALERT_THRESHOLD}."
            )

    # 5. Forecast Trajectory Chart Bento
    with st.container(border=True):
        st.markdown('<div id="forecast" style="scroll-margin-top:90px"></div>', unsafe_allow_html=True)
        st.subheader("Forecast Trajectory")
        st.caption("Historical 7-day observation paired with multi-horizon machine learning forecasts.")
        
        hist = frame.tail(24 * 7)
        fig = go.Figure()
        
        # Historical line
        fig.add_trace(go.Scatter(
            x=hist["observed_at"], y=hist["aqi"], mode="lines",
            name="Observed AQI",
            line=dict(color="#0284c7", width=2.6),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.06)",
        ))

        # Forecast line
        if preds:
            ordered = sorted(preds)
            fx = [obs_time] + [pd.to_datetime(preds[h]["valid_at"]) for h in ordered]
            fy = [current["aqi"]] + [preds[h]["value"] for h in ordered]
            fig.add_trace(go.Scatter(
                x=fx, y=fy, mode="markers+lines", name="72h Forecast",
                line=dict(color="#ec4899", width=2.8, dash="dot"),
                marker=dict(size=9, color="#ec4899", symbol="circle"),
            ))

        fig.add_hline(
            y=HAZARD_ALERT_THRESHOLD, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Hazard Threshold ({HAZARD_ALERT_THRESHOLD})",
            annotation_position="top left",
            annotation_font=dict(color="#ef4444", size=11),
        )
        fig.update_layout(
            height=400, yaxis_title="US AQI", xaxis_title="",
            hovermode="x unified", margin=dict(t=25, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(theme.style_fig(fig), width='stretch')

        # Model Performance expander
        with st.expander("Model Performance Metrics on Held-Out Test Set", expanded=False):
            rows = []
            for h in FORECAST_HORIZONS:
                if h not in preds:
                    continue
                m = preds[h]["metrics"]
                rows.append({
                    "Horizon": f"+{h}h",
                    "Architecture": preds[h]["model_type"],
                    "RMSE": round(m.get("rmse", float("nan")), 2),
                    "MAE": round(m.get("mae", float("nan")), 2),
                    "R² Score": round(m.get("r2", float("nan")), 3),
                    "Trained At": (preds[h].get("trained_at") or "")[:16].replace("T", " "),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
                st.caption(
                    "Scored on the most recent 20% chronological test split. "
                    "Models are evaluated against real CAMS ground-truth observations."
                )

    # 6. Recent Historical Trend Bento
    with st.container(border=True):
        st.markdown('<div id="trend" style="scroll-margin-top:90px"></div>', unsafe_allow_html=True)
        st.subheader("Historical AQI Trend")
        
        c_head, c_opt = st.columns([2, 1])
        with c_opt:
            days = st.radio("Time Window", [7, 14, 30], index=0, horizontal=True,
                            format_func=lambda d: f"{d} Days")
        
        recent = frame.tail(24 * days)

        tfig = go.Figure()
        tfig.add_trace(go.Scatter(
            x=recent["observed_at"], y=recent["aqi"], mode="lines",
            name="Hourly AQI", line=dict(color="#0ea5e9", width=2.2),
            fill="tozeroy", fillcolor="rgba(14, 165, 233, 0.08)",
        ))
        if "aqi_rolling_mean_24h" in recent.columns:
            tfig.add_trace(go.Scatter(
                x=recent["observed_at"], y=recent["aqi_rolling_mean_24h"],
                mode="lines", name="24h Rolling Mean", line=dict(color="#f59e0b", width=2.4),
            ))
        tfig.update_layout(
            height=320, yaxis_title="US AQI", hovermode="x unified",
            margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
        )
        st.plotly_chart(theme.style_fig(tfig), width='stretch')

        c1, c2, c3, c4 = st.columns(4)
        aqi_recent = recent["aqi"].dropna()
        c1.metric(f"{days}-Day Mean AQI", f"{aqi_recent.mean():.0f}")
        c2.metric(f"{days}-Day Peak AQI", f"{aqi_recent.max():.0f}")
        c3.metric(f"{days}-Day Min AQI", f"{aqi_recent.min():.0f}")
        unhealthy = (aqi_recent >= HAZARD_ALERT_THRESHOLD).mean() * 100
        c4.metric("Hazard Hours", f"{unhealthy:.0f}%")

    # 7. Explainability & SHAP Drivers Bento
    with st.container(border=True):
        st.markdown('<div id="drivers" style="scroll-margin-top:90px"></div>', unsafe_allow_html=True)
        st.subheader("Model Feature Drivers & Explainability")
        st.caption("SHAP (SHapley Additive exPlanations) values quantifying the exact impact of each weather feature.")
        
        horizon_choice = st.selectbox("Select Forecast Horizon", list(FORECAST_HORIZONS),
                                      format_func=lambda h: f"+{h} Hours Ahead")
        imp = get_importance(horizon_choice)

        if imp is None:
            st.info("Feature importances are computed during pipeline training. Run training to populate SHAP values.")
        else:
            table, method = imp
            table = table.sort_values("mean_abs_shap", ascending=True).tail(12)
            colors = ["#ec4899" if f.startswith("fc") else "#0284c7" for f in table["feature"]]
            ifig = go.Figure(go.Bar(
                x=table["mean_abs_shap"], y=table["feature"],
                orientation="h", marker=dict(color=colors, line=dict(width=0)),
            ))
            ifig.update_layout(
                height=420, xaxis_title="Mean |SHAP Value| (Impact on predicted AQI score)",
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(theme.style_fig(ifig), width='stretch')
            st.caption(
                f"Attribution Method: {method}. "
                f"**Pink bars are forecast-weather features (`fc{horizon_choice}_*`)** predicting atmospheric conditions at target hour. "
                "Blue bars represent recent historical trend features."
            )

    # 8. Sensor Telemetry & Atmospheric Conditions Bento
    with st.container(border=True):
        st.markdown('<div id="telemetry" style="scroll-margin-top:90px"></div>', unsafe_allow_html=True)
        st.subheader("Atmospheric Telemetry & Pollutant Readings")
        st.caption(f"Real-time sensor telemetry for {CITY_NAME} captured via Open-Meteo CAMS atmospheric models.")

        last = frame.iloc[-1]
        pollutants = {
            "PM2.5 (µg/m³)": "pm25",
            "PM10 (µg/m³)": "pm10",
            "O₃ (µg/m³)": "o3",
            "NO₂ (µg/m³)": "no2",
            "SO₂ (µg/m³)": "so2",
            "CO (µg/m³)": "co"
        }
        weather = {
            "Temperature": "temperature_2m",
            "Relative Humidity": "relative_humidity_2m",
            "Wind Speed (10m)": "wind_speed_10m",
            "Surface Pressure": "surface_pressure",
            "Boundary Layer Height": "boundary_layer_height"
        }

        st.markdown("##### 🔬 Pollutant Concentrations")
        pc = st.columns(len(pollutants))
        for (label, col), c in zip(pollutants.items(), pc):
            val = last.get(col)
            c.metric(label, f"{val:.1f}" if pd.notna(val) else "—")

        st.write("")
        st.markdown("##### 🌤️ Meteorological Variables")
        wc = st.columns(len(weather))
        unit_map = {
            "Temperature": " °C",
            "Relative Humidity": " %",
            "Wind Speed (10m)": " km/h",
            "Surface Pressure": " hPa",
            "Boundary Layer Height": " m"
        }
        for (label, col), c in zip(weather.items(), wc):
            val = last.get(col)
            suffix = unit_map.get(label, "")
            c.metric(label, f"{val:.1f}{suffix}" if pd.notna(val) else "—")

    # 9. Attribution & Footer
    if source == "hopsworks":
        source_label = "Hopsworks Feature Store (Automated Hourly Pipeline)"
    else:
        source_label = "Live Open-Meteo APIs (Direct Fallback)"

    footer_html = (
        f'<div class="dock-footer">'
        f'<div><strong>{CITY_NAME}</strong> AQI Forecast</div>'
        f'<div>Source: {source_label} · CAMS Air Quality &amp; ECMWF Weather Models</div>'
        f'<div>Latest Observation: {obs_time:%Y-%m-%d %H:%M} PKT · Page Rendered: {now_local_naive():%H:%M} PKT</div>'
        f'</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
