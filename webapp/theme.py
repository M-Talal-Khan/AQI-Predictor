"""
Visual theme: a light "bento card" dashboard - frosted-glass panels on a flat
neutral canvas, a sticky pill navbar, and a compact live-summary strip above
the main content.

Purely cosmetic. Nothing here touches data, features, or predictions - app.py
still computes every number exactly as before, this module only restyles how
it's presented. inject_theme() never raises, so a styling problem can never
take the dashboard down with it.

No external font/icon requests: an earlier version of this theme pulled Inter
from Google Fonts via a render-blocking `@import`, which risks stalling first
paint for any visitor with latency or a failure reaching that host. System
font stacks only.
"""
from datetime import datetime
import streamlit as st

CSS = """
<style>
:root {
  --bg-canvas: #edeceb;
  --card-bg: rgba(255, 255, 255, 0.94);
  --card-border: rgba(255, 255, 255, 0.98);
  --card-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.06), 0 4px 12px rgba(0, 0, 0, 0.03);
  --text-main: #0f172a;
  --text-muted: #475569;
  --text-dim: #64748b;
}

html {
  scroll-behavior: smooth !important;
}

/* Base canvas + a soft glowing perimeter vignette */
html, body, [data-testid="stApp"] {
  background-color: var(--bg-canvas) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  -webkit-font-smoothing: antialiased;
  margin: 0;
  padding: 0;
}

/* Soft glowing perimeter vignette around the viewport edge */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 999999;
  box-shadow: 
    inset 0 0 60px 15px rgba(59, 130, 246, 0.42),
    inset 0 0 140px 45px rgba(99, 102, 241, 0.22);
}

[data-testid="stAppViewContainer"] {
  background-color: var(--bg-canvas) !important;
  background-image: none !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
  box-shadow: none !important;
  display: none !important;
}

[data-testid="stAppViewContainer"] > .main {
  background: transparent !important;
  padding-top: 1rem !important;
}

.block-container {
  max-width: 1140px !important;
  padding-top: 1.5rem !important;
  padding-bottom: 5rem !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}

/* High Contrast Typography */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  color: #0f172a !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  scroll-margin-top: 90px !important;
}

/* p/li/label only - NOT span. An earlier version included span here,
   which broke every custom-colored <span> in the injected HTML components
   below (the live-dock horizon numbers, telemetry readings, and category
   legend pills all set their own inline color intentionally) - a directly-
   matching !important rule beats a plain inline style regardless of the
   inline style'''s higher base specificity, so those custom colors were
   silently overridden to this one dark slate value everywhere, including
   white-on-dark category pills like "Hazardous", which became hard to read.
   :not() would also work, but omitting span outright is simpler and nothing
   here relies on a bare <span> getting this default. */
p, li, label, .stMarkdown p {
  color: #334155 !important;
}

.stCaption, [data-testid="stCaptionContainer"] {
  color: #64748b !important;
  font-weight: 500 !important;
}

/* Floating Frosted Glass Navbar */
.dock-nav-outer {
  display: flex;
  justify-content: center;
  position: sticky;
  top: 14px;
  z-index: 99999;
  margin-bottom: 2rem;
  width: 100%;
}

.dock-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  background: rgba(255, 255, 255, 0.90);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.98);
  border-radius: 9999px;
  padding: 8px 18px 8px 14px;
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.02);
  transition: all 0.25s ease;
  max-width: 780px;
  width: 100%;
}

.dock-nav:hover {
  box-shadow: 0 16px 42px rgba(0, 0, 0, 0.12);
  transform: translateY(-1px);
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
}

.brand-icon {
  width: 32px;
  height: 32px;
  background: #0d0d0d;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  color: #ffffff;
}

.brand-title {
  font-size: 0.98rem;
  font-weight: 700;
  color: #0d0d0d;
  letter-spacing: -0.02em;
}

.nav-links {
  display: flex;
  align-items: center;
  gap: 16px;
}

.nav-link {
  font-size: 0.86rem;
  font-weight: 600;
  color: #4b5563;
  text-decoration: none;
  transition: color 0.18s ease;
}

.nav-link:hover {
  color: #000000;
}

.nav-btn-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  background: #0d0d0d;
  color: #ffffff !important;
  padding: 6px 14px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 600;
  text-decoration: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.18);
}

.nav-live-dot {
  width: 7px;
  height: 7px;
  background-color: #22c55e;
  border-radius: 50%;
  display: inline-block;
  box-shadow: 0 0 8px #22c55e;
  animation: live-pulse 2s infinite;
}

@keyframes live-pulse {
  0% { transform: scale(0.95); opacity: 0.8; }
  50% { transform: scale(1.25); opacity: 1; }
  100% { transform: scale(0.95); opacity: 0.8; }
}

/* Hero Section */
.hero-wrapper {
  text-align: center;
  padding: 1rem 1rem 1.8rem;
  max-width: 840px;
  margin: 0 auto;
}

.hero-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.95);
  padding: 6px 16px;
  border-radius: 999px;
  font-size: 0.84rem;
  font-weight: 600;
  color: #1f2937;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
  margin-bottom: 1.2rem;
}

.hero-title {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-size: 3.8rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.04em !important;
  line-height: 1.06 !important;
  color: #0a0a0a !important;
  margin: 0 0 1.2rem 0 !important;
}

.hero-subtitle {
  font-size: 1.15rem !important;
  font-weight: 400 !important;
  color: #475569 !important;
  line-height: 1.6 !important;
  max-width: 680px;
  margin: 0 auto 1.5rem auto !important;
}

.hero-tags {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 2rem;
}

.tag-capsule {
  background: rgba(255, 255, 255, 0.85);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.95);
  padding: 5px 14px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  color: #475569;
  box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}

/* Live AQI Dock */
.aqi-dock-container {
  display: flex;
  justify-content: center;
  margin: 0 auto 2.8rem auto;
  width: 100%;
}

.aqi-dock {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: #121216;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 26px;
  padding: 12px 18px;
  box-shadow: 0 24px 60px -12px rgba(0, 0, 0, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.22);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  max-width: 980px;
  width: 100%;
}

.aqi-dock:hover {
  transform: translateY(-2px);
  box-shadow: 0 30px 70px -12px rgba(0, 0, 0, 0.42);
}

.dock-widget {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 18px;
  padding: 10px 14px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  color: #ffffff;
  min-height: 72px;
}

.dock-widget-clock {
  min-width: 130px;
  text-align: left;
}
.dock-clock-time {
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.1;
  color: #ffffff;
}
.dock-clock-date {
  font-size: 0.74rem;
  color: #9ca3af;
  margin-top: 3px;
  font-weight: 600;
}

.dock-widget-aqi {
  min-width: 165px;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
}
.dock-aqi-num {
  font-size: 2.1rem;
  font-weight: 900;
  letter-spacing: -0.04em;
  line-height: 1;
}
.dock-aqi-meta {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.dock-aqi-pill {
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  display: inline-block;
  text-shadow: 0 1px 2px rgba(0,0,0,0.25);
  box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.dock-aqi-label {
  font-size: 0.68rem;
  color: #9ca3af;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
}

.dock-widget-weather {
  min-width: 130px;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;
}
.dock-weather-icon {
  font-size: 1.7rem;
}
.dock-weather-info {
  display: flex;
  flex-direction: column;
}
.dock-weather-temp {
  font-size: 1.25rem;
  font-weight: 800;
  color: #ffffff;
  line-height: 1.1;
}
.dock-weather-sub {
  font-size: 0.72rem;
  color: #9ca3af;
  font-weight: 500;
}

.dock-widget-forecast {
  display: flex;
  align-items: center;
  gap: 8px;
}
.dock-horizon-pill {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 11px;
  background: rgba(255, 255, 255, 0.06);
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  min-width: 58px;
}
.dock-h-name {
  font-size: 0.68rem;
  color: #9ca3af;
  font-weight: 600;
}
.dock-h-val {
  font-size: 1.1rem;
  font-weight: 800;
  margin-top: 1px;
}

.dock-widget-telemetry {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
}
.dock-telem-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.dock-telem-lbl {
  font-size: 0.65rem;
  color: #9ca3af;
  font-weight: 600;
}
.dock-telem-val {
  font-size: 0.9rem;
  font-weight: 800;
  color: #e5e7eb;
}

/* Bento Card Styling */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 255, 255, 0.94) !important;
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid rgba(255, 255, 255, 0.98) !important;
  border-radius: 26px !important;
  box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.06), 0 4px 12px rgba(0, 0, 0, 0.03) !important;
  padding: 22px 26px !important;
  margin-bottom: 2rem !important;
}

/* Modern Frosted Metric Tiles */
.dock-bento-tile {
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  padding: 20px 16px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.7);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.8);
  backdrop-filter: blur(16px);
  transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease;
}

.dock-bento-tile:hover {
  transform: translateY(-4px) scale(1.015);
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.12);
}

.dock-bento-tile .tile-label {
  font-size: 0.82rem;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.9;
  margin-bottom: 4px;
}

.dock-bento-tile .tile-val {
  font-size: 2.8rem;
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -0.04em;
  margin: 4px 0;
}

.dock-bento-tile .tile-cat {
  font-size: 0.88rem;
  font-weight: 800;
}

.dock-bento-tile .tile-sub {
  font-size: 0.76rem;
  opacity: 0.85;
  margin-top: 4px;
  font-weight: 600;
}

@keyframes hazard-pulse {
  0%, 100% { box-shadow: 0 8px 24px rgba(239, 68, 68, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.8); }
  50%      { box-shadow: 0 12px 38px rgba(239, 68, 68, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.8); }
}

.dock-bento-tile.tile-hazard {
  animation: hazard-pulse 2.2s ease-in-out infinite;
}

/* Category Legend Pills */
.dock-pill-row {
  margin: 4px 0 16px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.dock-category-pill {
  padding: 5px 14px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.6);
  display: inline-block;
}

/* Hazard Alert Card */
.dock-alert-banner {
  border-radius: 20px;
  padding: 16px 20px;
  display: flex;
  align-items: flex-start;
  gap: 14px;
  margin-top: 14px;
  backdrop-filter: blur(14px);
  border: 1px solid rgba(255, 255, 255, 0.8);
  box-shadow: 0 8px 24px rgba(0,0,0,0.04);
}

.alert-danger {
  background: linear-gradient(135deg, rgba(254, 226, 226, 0.95), rgba(254, 242, 242, 0.9));
  border-left: 5px solid #ef4444;
  color: #991b1b;
}

.alert-success {
  background: linear-gradient(135deg, rgba(220, 252, 231, 0.95), rgba(240, 253, 244, 0.9));
  border-left: 5px solid #22c55e;
  color: #166534;
}

.alert-icon {
  font-size: 1.4rem;
  line-height: 1;
}

.alert-body {
  font-size: 0.92rem;
  line-height: 1.5;
}

.alert-title {
  font-weight: 700;
  margin-bottom: 2px;
}

/* Apple Segment Controls (Radio Buttons) */
div[role="radiogroup"] {
  background: #e2e8f0 !important;
  padding: 4px !important;
  border-radius: 999px !important;
  display: inline-flex !important;
  gap: 4px !important;
  border: 1px solid #cbd5e1 !important;
}

div[role="radiogroup"] label {
  background: transparent !important;
  border: none !important;
  border-radius: 999px !important;
  padding: 6px 16px !important;
  margin: 0 !important;
  font-size: 0.84rem !important;
  font-weight: 600 !important;
  color: #475569 !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

div[role="radiogroup"] label:hover {
  color: #0f172a !important;
}

div[role="radiogroup"] label[data-checked="true"],
div[role="radiogroup"] label:has(input:checked) {
  background: #ffffff !important;
  color: #0f172a !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
  font-weight: 700 !important;
}

div[role="radiogroup"] label span,
div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {
  color: inherit !important;
  font-weight: inherit !important;
}

/* Selectbox Styling (High Contrast) */
[data-baseweb="select"] > div {
  background-color: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  border-radius: 14px !important;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
  color: #0f172a !important;
  font-weight: 600 !important;
}

[data-baseweb="select"] span, [data-baseweb="select"] div {
  color: #0f172a !important;
}

[data-baseweb="select"] svg {
  fill: #0f172a !important;
}

div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
  background-color: #ffffff !important;
  color: #0f172a !important;
  border-radius: 12px !important;
}

li[role="option"]:hover, li[aria-selected="true"] {
  background-color: #f1f5f9 !important;
  color: #0f172a !important;
}

/* Streamlit Metrics */
[data-testid="stMetric"] {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 20px !important;
  padding: 16px 18px !important;
  box-shadow: 0 4px 14px rgba(0,0,0,0.03) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(0,0,0,0.06) !important;
}

[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
  color: #0f172a !important;
  font-weight: 900 !important;
  font-size: 1.85rem !important;
  letter-spacing: -0.03em !important;
}

[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
  color: #64748b !important;
  font-weight: 700 !important;
  font-size: 0.78rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}

/* Expanders */
[data-testid="stExpander"] {
  background: #ffffff !important;
  border: 1px solid #e2e8f0 !important;
  border-radius: 18px !important;
  box-shadow: 0 4px 14px rgba(0,0,0,0.03) !important;
  overflow: hidden !important;
}

[data-testid="stExpander"] summary {
  color: #0f172a !important;
  font-weight: 700 !important;
  padding: 14px 18px !important;
}

[data-testid="stExpander"] summary:hover {
  color: #0284c7 !important;
}

[data-testid="stExpander"] summary svg {
  fill: #0f172a !important;
}

[data-testid="stExpanderDetails"] {
  padding: 14px 18px !important;
  background: #ffffff !important;
  color: #334155 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border-radius: 14px !important;
  border: 1px solid #e2e8f0 !important;
  background: #ffffff !important;
}

/* Footer */
.dock-footer {
  text-align: center;
  padding: 2.5rem 1rem 1rem;
  color: #64748b;
  font-size: 0.84rem;
  line-height: 1.6;
}
.dock-footer a {
  color: #0d0d0d;
  font-weight: 700;
  text-decoration: none;
}
</style>
"""


def inject_theme() -> None:
    """Call once near the top of the page. Never raises."""
    try:
        st.markdown(CSS, unsafe_allow_html=True)
    except Exception:
        pass


def render_navbar(city_name: str) -> None:
    """Renders the floating frosted pill navigation bar without markdown indentation."""
    nav_html = (
        f'<div class="dock-nav-outer">'
        f'<div class="dock-nav">'
        f'<a href="#top" class="nav-brand">'
        f'<div class="brand-icon">🌫️</div>'
        f'<div><span class="brand-title">{city_name} AQI</span></div>'
        f'</a>'
        f'<div class="nav-links">'
        f'<a href="#overview" class="nav-link">Overview</a>'
        f'<a href="#forecast" class="nav-link">Forecast 72h</a>'
        f'<a href="#trend" class="nav-link">Trend</a>'
        f'<a href="#drivers" class="nav-link">SHAP Drivers</a>'
        f'<a href="#telemetry" class="nav-link">Sensors</a>'
        f'</div>'
        f'<div><span class="nav-btn-pill"><span class="nav-live-dot"></span> Sync: Live</span></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(nav_html, unsafe_allow_html=True)


def render_hero(city_name: str) -> None:
    """Renders the hero display at the top of the page."""
    hero_html = (
        f'<div class="hero-wrapper" id="top" style="scroll-margin-top:90px">'
        f'<div class="hero-chip">Air Quality Forecasting</div>'
        f'<h1 class="hero-title">24-72 hour air quality<br>forecasts for {city_name}.</h1>'
        f'<p class="hero-subtitle">Real-time CAMS air quality, per-horizon machine learning forecasts, '
        f'and live atmospheric telemetry for {city_name}.</p>'
        f'<div class="hero-tags">'
        f'<span class="tag-capsule">Hopsworks Feature Store</span>'
        f'<span class="tag-capsule">XGBoost + TensorFlow</span>'
        f'<span class="tag-capsule">CAMS Air Quality</span>'
        f'<span class="tag-capsule">Open-Meteo Forecast</span>'
        f'<span class="tag-capsule">SHAP Explainability</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_live_dock(current: dict, preds: dict, last_row: dict, obs_time: datetime, city_name: str) -> None:
    """Renders the signature interactive Live AQI Dock widget with clean compact HTML."""
    time_str = obs_time.strftime("%I:%M %p")
    date_str = obs_time.strftime("%a, %d %b")

    aqi_val = current.get("aqi", 0.0)
    from config import categorize_aqi, category_color
    cat_name = categorize_aqi(aqi_val)
    cat_col = category_color(aqi_val)
    text_color = "#10131f" if cat_name in ("Good", "Moderate", "Unhealthy for Sensitive Groups") else "#ffffff"

    temp = last_row.get("temperature_2m", None)
    temp_str = f"{temp:.0f}°C" if temp is not None else "—"
    hum = last_row.get("relative_humidity_2m", None)
    hum_str = f"{hum:.0f}%" if hum is not None else "—"
    wind = last_row.get("wind_speed_10m", None)
    wind_str = f"{wind:.0f} km/h" if wind is not None else "—"

    # Mini forecast horizon items
    f_items = []
    for h in [24, 48, 72]:
        if h in preds:
            val = preds[h]["value"]
            c_col = category_color(val)
            f_items.append(
                f'<div class="dock-horizon-pill">'
                f'<span class="dock-h-name">+{h}h</span>'
                f'<span class="dock-h-val" style="color:{c_col};">{val:.0f}</span>'
                f'</div>'
            )
    f_items_html = "".join(f_items)

    pm25 = last_row.get("pm25", None)
    pm25_str = f"{pm25:.1f}" if pm25 is not None else "—"
    pm10 = last_row.get("pm10", None)
    pm10_str = f"{pm10:.1f}" if pm10 is not None else "—"

    dock_html = (
        f'<div class="aqi-dock-container" id="overview" style="scroll-margin-top:90px">'
        f'<div class="aqi-dock">'
        f'<div class="dock-widget dock-widget-clock">'
        f'<div class="dock-clock-time">{time_str}</div>'
        f'<div class="dock-clock-date">{date_str} · {city_name}</div>'
        f'</div>'
        f'<div class="dock-widget dock-widget-aqi">'
        f'<div class="dock-aqi-num" style="color:{cat_col};">{aqi_val:.0f}</div>'
        f'<div class="dock-aqi-meta">'
        f'<span class="dock-aqi-pill" style="background:{cat_col};color:{text_color};">{cat_name}</span>'
        f'<span class="dock-aqi-label">Current US AQI</span>'
        f'</div>'
        f'</div>'
        f'<div class="dock-widget dock-widget-weather">'
        f'<div class="dock-weather-icon">🌫️</div>'
        f'<div class="dock-weather-info">'
        f'<div class="dock-weather-temp">{temp_str}</div>'
        f'<div class="dock-weather-sub">{hum_str} hum · {wind_str} wind</div>'
        f'</div>'
        f'</div>'
        f'<div class="dock-widget dock-widget-forecast">{f_items_html}</div>'
        f'<div class="dock-widget dock-widget-telemetry">'
        f'<div class="dock-telem-item"><span class="dock-telem-lbl">PM2.5</span><span class="dock-telem-val">{pm25_str}</span></div>'
        f'<div style="width:1px;height:24px;background:rgba(255,255,255,0.15);"></div>'
        f'<div class="dock-telem-item"><span class="dock-telem-lbl">PM10</span><span class="dock-telem-val">{pm10_str}</span></div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(dock_html, unsafe_allow_html=True)


def style_fig(fig):
    """
    Apply the theme colors to a Plotly figure layout only - trace colors and
    data are untouched, so this cannot change what a chart shows, only how it looks.
    """
    try:
        fig.update_layout(
            paper_bgcolor="rgba(255, 255, 255, 0.0)",
            plot_bgcolor="rgba(248, 250, 252, 0.75)",
            font=dict(
                color="#0f172a",
                family="'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                size=12
            ),
            xaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.07)",
                zerolinecolor="rgba(0, 0, 0, 0.1)",
                tickfont=dict(color="#475569", size=11, family="'Inter', sans-serif"),
                title_font=dict(color="#0f172a", size=12, family="'Inter', sans-serif"),
                showline=False,
            ),
            yaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.07)",
                zerolinecolor="rgba(0, 0, 0, 0.1)",
                tickfont=dict(color="#475569", size=11, family="'Inter', sans-serif"),
                title_font=dict(color="#0f172a", size=12, family="'Inter', sans-serif"),
                showline=False,
            ),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.85)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                font=dict(color="#0f172a", size=11, family="'Inter', sans-serif")
            ),
            hoverlabel=dict(
                bgcolor="#0f172a",
                font_color="#ffffff",
                font_size=12,
                font_family="'Inter', -apple-system, sans-serif",
                bordercolor="rgba(255, 255, 255, 0.2)"
            ),
        )
    except Exception:
        pass
    return fig


def glass_tile(label: str, value: float, category: str, color: str, sub: str = "",
               hazard: bool = False) -> str:
    """Bento-style tile for a forecast horizon."""
    text_color = "#0f172a" if category in ("Good", "Moderate", "Unhealthy for Sensitive Groups") else "#ffffff"
    hazard_class = " tile-hazard" if hazard else ""
    return (
        f'<div class="dock-bento-tile{hazard_class}" style="background:linear-gradient(145deg, {color}ee, {color}c0);color:{text_color};">'
        f'<div class="tile-label">{label}</div>'
        f'<div class="tile-val">{value:.0f}</div>'
        f'<div class="tile-cat">{category}</div>'
        f'<div class="tile-sub">{sub}</div>'
        f'</div>'
    )


def glass_legend(items: dict) -> str:
    """Category legend pills."""
    light_bg = ("Good", "Moderate", "Unhealthy for Sensitive Groups")
    chips = "".join(
        f'<span class="dock-category-pill" style="background:{color};'
        f'color:{"#0f172a" if name in light_bg else "#ffffff"};">{name}</span>'
        for name, color in items.items() if name != "Unknown"
    )
    return f'<div class="dock-pill-row">{chips}</div>'


def render_hazard_banner(is_hazard: bool, title: str, description: str) -> None:
    """Renders a hazard/success banner."""
    cls = "alert-danger" if is_hazard else "alert-success"
    icon = "⚠" if is_hazard else "✓"
    html = (
        f'<div class="dock-alert-banner {cls}">'
        f'<div class="alert-icon">{icon}</div>'
        f'<div class="alert-body">'
        f'<div class="alert-title">{title}</div>'
        f'<div>{description}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
