# Lahore AQI Forecasting — Project Report

Forecasting Lahore's Air Quality Index 24, 48 and 72 hours ahead, with an
hourly feature pipeline, a daily training pipeline, and a public dashboard.

**Headline result — both quality targets met:**

| Target | Required | Achieved |
|---|---|---|
| Overall mean R² | > 0.70 | **0.743** ✅ |
| Weakest horizon (any) | > 0.60 | **0.684** ✅ |

Every horizon also beats its best naive baseline by 24–32% RMSE, which matters
more than the raw R²: an R² that merely matches persistence is not a forecast.

---

## 1. Data sources

### 1.1 The AQICN problem (and why the source changed)

The project was scaffolded around AQICN (waqi.info). A valid token was obtained
and tested, and **AQICN has no usable live data for Lahore.** Verified
2026-08-10:

| Probe | Result |
|---|---|
| `GET /feed/lahore/` | AQI **34**, timestamped **2025-02-18** — ~18 months stale |
| `GET /search/?keyword=lahore` | One station ("Lahore US Embassy", uid 11765), `aqi: "-"` |
| `GET /map/bounds/` over Lahore | **0 stations** |
| `GET /feed/geo:31.52;74.36/` | Returns **Narela, Delhi, India** — ~400 km away |

The last row is the dangerous one. AQICN's geo endpoint does not error when it
has no coverage; it silently substitutes the nearest station it *does* have,
which is in another country. An hourly pipeline pointed at it would have
reported success indefinitely while filling the feature store with stale or
foreign readings, and every metric downstream would have been meaningless. An
AQI of 34 for Lahore is also implausible on its face — Lahore is one of the
most polluted cities in the world, and its true annual mean is ~154.

**Decision:** move observations to Open-Meteo's CAMS air-quality product.
`aqicn_client.py` is retained and functional as an optional cross-check, but
nothing in the pipeline depends on it.

### 1.2 Sources actually used

| Source | Role | Coverage | Key |
|---|---|---|---|
| Open-Meteo **Air Quality** (CAMS) | AQI + pollutants, history & live | 2022-09-01 → now, hourly | none |
| Open-Meteo **Archive** (ERA5) | Historical weather | 1940 → ~5 days ago, hourly | none |
| Open-Meteo **Forecast** | Weather forecast (the core feature) | now → +16 days, hourly | none |
| AQICN | Optional cross-check only | — | token |

This also replaced the originally-suggested Kaggle/OpenAQ manual CSV route.
OpenAQ's Pakistan stations report mostly PM2.5 with sparse other pollutants;
CAMS gives a complete, gap-free hourly record of all seven variables, which
matters because lag features on a gappy series silently misalign.

**Backfill audit** (`feature_pipeline/backfill.py` runs this automatically):

```
Rows fetched : 34,560      Coverage : 100.0% of hourly slots
Range        : 2022-09-01 -> 2026-08-10
AQI          : mean 153.9 | median 154.0 | std 47.8 | min 12 | max 538
Nulls        : 0 across aqi, pm25, pm10, o3, no2, so2, co
Worst month  : January (220)      Best month : April (111)
[PASS] mean AQI in plausible Lahore range      [PASS] winter is worst season
[PASS] monsoon is not the worst season         [PASS] has hazardous episodes
[PASS] not degenerate (std > 25)               [PASS] no impossible values
```

The bias checks are the point here: a dataset that put the pollution peak in
July, or averaged 50, would indicate the wrong location or a broken series.
January-worst / April-best is the correct Lahore signature (crop-residue
burning plus winter thermal inversion).

---

## 2. Feature engineering

### 2.1 The core problem

Lag and rolling features describe where pollution **has been**. What decides
where it goes over the next three days is the weather that is **coming**. The
EDA notebook measures how quickly past AQI stops being informative:

| Horizon | Autocorrelation r | r² (ceiling for persistence) |
|---|---|---|
| +24h | 0.807 | 0.652 |
| +48h | 0.667 | 0.445 |
| +72h | 0.595 | 0.354 |

At 72 hours, past AQI alone explains ~35% of variance. No amount of model
tuning recovers information that is not in the features.

### 2.2 The fix

For a prediction made at time *t* for target *t+H*, the model is given the
**weather forecast valid at t+H** — not only weather observed at *t*. Per
horizon (`fc24_*`, `fc48_*`, `fc72_*`), 27 features each:

- **Point forecast at t+H**: temperature, humidity, pressure, wind speed,
  wind direction (as sin/cos), precipitation, dew point, boundary layer
  height, cloud cover
- **Deltas**: forecast value minus current value — the trend, not just the level
- **24h window ending at t+H**: mean wind, min boundary layer height,
  total precipitation, mean temperature/humidity
- **Whole lead-time window (t, t+H]**: cumulative precipitation, mean wind,
  minimum mixing depth
- **Ventilation index at t+H**: `boundary_layer_height × wind_speed` — the
  standard air-quality dispersion diagnostic, far more informative than either
  term alone

Plus shared history features (100 total per horizon): AQI lags out to 168h,
rolling mean/std/min/max over 3/6/24/72/168h, change rates, pollutant rolling
means, cyclical hour/month/day-of-year encodings, and current weather.

### 2.3 Does it actually work? (ablation)

Claimed benefits should be measured, not asserted. `training_pipeline/ablation.py`
retrains LightGBM with identical hyperparameters, varying only the feature set:

| Horizon | History only | + forecast weather | Δ R² |
|---|---|---|---|
| +24h | 0.767 | **0.820** | +0.053 |
| +48h | 0.526 | **0.665** | +0.140 |
| +72h | 0.444 | **0.654** | **+0.210** |

**The gain grows with horizon**, exactly as the theory predicts — the further
ahead you forecast, the less history tells you and the more the coming weather
decides. Critically, **without forecast weather both 48h (0.526) and 72h
(0.444) fall below the project's 0.60 floor.** This single design choice is
what makes the project pass.

A note on method: comparing *univariate* correlations of current vs forecast
weather against the target does **not** demonstrate this, because weather is
itself strongly autocorrelated — temperature at *t* and at *t+72h* have nearly
identical marginal correlations with the target. Only the ablation isolates the
contribution. (An earlier draft of the EDA notebook used the correlation
comparison and it showed ~zero gain; it was replaced with the ablation.)

SHAP confirms it independently — the top features for the long horizons are all
forecast-weather:

- **48h**: `fc48_precip_total`, `fc48_blh_min_lead`, `aqi_change_rate_3h`
- **72h**: `fc72_precip_total`, `fc72_wind_speed_10m_w24_mean`, `fc72_boundary_layer_height_w24_min`

### 2.4 Correctness details that matter

- **Gap-free hourly grid.** Every lag/rolling/forecast feature uses `.shift(n)`,
  which means *n rows*. That equals *n hours* only on a complete index, so the
  frame is reindexed to a full hourly range first. Without this, every lag
  silently misaligns across gaps.
- **No leakage.** Negative shifts are applied *only* to weather columns.
  Weather at t+H is legitimately knowable at *t* from a forecast — that is the
  premise. No AQI value at or after t+H ever enters the feature set; those are
  the targets.
- **Circular encoding.** Wind direction is split into sin/cos components; 359°
  and 1° are 2° apart, not 358.
- **NaN policy.** Gaps ≤3h are interpolated; longer gaps stay NaN and are
  dropped at training time. Zero-filling a pollutant gap would put fabricated
  history into the lag features.
- **Observations only.** The live endpoint returns whole days, so a call at
  15:00 also returns 16:00–23:00 — which are *forecast*, not observed. Those
  are trimmed before storage; otherwise training targets would be built from a
  model's own forecast and recent rows would look artificially predictable.

---

## 3. Models and results

Trained per horizon: Ridge, Random Forest, LightGBM, XGBoost, TensorFlow MLP.
Split **chronologically** — the oldest 80% trains, the newest 20% tests
(2025-11-26 → 2026-08-08, which includes a full winter smog season). A random
split would put hour *t+1* in train and *t* in test; with 24h-autocorrelated AQI
that leaks the answer and yields flattering numbers that collapse in production.

Early stopping for LightGBM/XGBoost/TF uses a validation slice taken from the
**end** of the training period, never from the test period.

### Full results

| Horizon | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| **24h** | **xgboost** ⭐ | **21.75** | **15.81** | **0.833** |
| 24h | tensorflow_mlp | 22.14 | 16.05 | 0.827 |
| 24h | lightgbm | 22.60 | 16.82 | 0.820 |
| 24h | random_forest | 23.48 | 17.58 | 0.806 |
| 24h | ridge | 23.79 | 17.16 | 0.800 |
| 24h | *baseline: persistence* | 31.55 | 21.34 | 0.649 |
| 24h | *baseline: same hour yesterday* | 38.97 | 27.34 | 0.465 |
| **48h** | **tensorflow_mlp** ⭐ | **29.91** | **22.75** | **0.684** |
| 48h | xgboost | 30.72 | 22.96 | 0.667 |
| 48h | lightgbm | 30.78 | 22.73 | 0.665 |
| 48h | ridge | 31.00 | 23.00 | 0.661 |
| 48h | random_forest | 31.36 | 23.02 | 0.653 |
| 48h | *baseline: persistence* | 39.18 | 27.56 | 0.458 |
| 48h | *baseline: same hour yesterday* | 41.89 | 30.65 | 0.381 |
| **72h** | **tensorflow_mlp** ⭐ | **28.59** | **21.28** | **0.712** |
| 72h | xgboost | 30.66 | 22.90 | 0.669 |
| 72h | lightgbm | 31.31 | 23.18 | 0.654 |
| 72h | random_forest | 31.49 | 23.21 | 0.650 |
| 72h | ridge | 32.86 | 24.54 | 0.619 |
| 72h | *baseline: persistence* | 41.96 | 30.72 | 0.380 |
| 72h | *baseline: same hour yesterday* | 44.66 | 33.02 | 0.297 |

The persistence baseline is AQI at *t* carried forward — the strongest fair
version. An earlier draft mistakenly used AQI at *t−1h*, a handicapped
baseline that overstated the models' advantage by roughly two percentage
points; the numbers above are from the corrected comparison.

### Selected models

| Horizon | Model | R² | RMSE | MAE | RMSE lift vs best baseline |
|---|---|---|---|---|---|
| +24h | XGBoost | 0.833 | 21.75 | 15.81 | +31.1% |
| +48h | TensorFlow MLP | 0.684 | 29.91 | 22.75 | +23.7% |
| +72h | TensorFlow MLP | 0.712 | 28.59 | 21.28 | +31.9% |

**Mean R² = 0.743** (target >0.70 ✅) · **weakest = 0.684** (target >0.60 ✅)

### Observations

- **Adding gradient boosting was worthwhile**, as the brief anticipated:
  XGBoost wins outright at 24h and beats Random Forest at every horizon.
- **The MLP wins the long horizons.** With forecast-weather features the
  relationship becomes smooth and interaction-heavy, which suits a dense
  network; trees have to approximate it with axis-aligned splits.
- **72h (0.712) scores slightly above 48h (0.684).** This is mildly
  counter-intuitive. It is not an error: the 48h and 72h models are separate
  fits with independent random initialisation, and the difference (~0.03) is
  within the run-to-run variance of the MLP — across repeated runs the two
  horizons traded places. It should not be read as "72h is genuinely easier
  than 48h." The tree models, which are far more deterministic, do order the
  horizons as expected (xgboost: 0.833 / 0.667 / 0.669).
- **Recursive multi-step forecasting was not needed.** The brief suggested it
  as a fallback if 72h R² stayed under 0.60; forecast-weather features took 72h
  to 0.712 directly, so the added complexity and error-compounding risk of
  feeding predictions back in was not justified.

---

## 4. Architecture

```
Open-Meteo CAMS (AQI) ─┐
                       ├─> features.py ─> Hopsworks Feature Store ─┐
Open-Meteo weather ────┘    (per-horizon    (+ local parquet mirror) │
   archive + FORECAST        fc{H}_* block)                          │
                                                                     v
                                                        training_pipeline/train.py
                                                     (Ridge/RF/LGBM/XGB/TF, daily)
                                                                     │
                                              Hopsworks Model Registry + models/
                                                                     │
                                                                     v
                                                    Streamlit dashboard (webapp/)
```

**Local parquet mirror.** Every store read/write goes through
`feature_pipeline/store.py`, which uses Hopsworks when available and a parquet
file otherwise. This exists because the `hopsworks` SDK **cannot be
pip-installed on Windows** — it depends on `pyjks` → `twofish`, which ships
source-only (no wheels for any platform) and requires MSVC C++ build tools. It
installs cleanly on Linux, which is what CI and Streamlit Cloud run. The mirror
also makes training reproducible offline and degrades a Hopsworks outage into a
local write rather than a failed pipeline run.

**Windows `/tmp` shim.** The Hopsworks client materialises certificates into a
hardcoded `/tmp`, which on Windows resolves to `<drive>:\tmp` and usually does
not exist, killing login with a `FileNotFoundError`. `store.py` creates the
directory before login.

**Dependency conflict (important).** `hopsworks` pins `protobuf<5.0.0` in every
published version; TensorFlow ≥2.19 requires `protobuf>=6.31.1`. Those ranges
do not intersect — `pip install hopsworks tensorflow` with an unpinned TF gives
either a resolver failure or a silently broken TensorFlow. `requirements.txt`
therefore pins **`tensorflow==2.18.1`**, the newest TF whose protobuf range
(`>=3.20.3,<6.0.0dev`) overlaps hopsworks at protobuf 4.25.x. Anyone bumping
TensorFlow must re-check this.

Because local development runs Python 3.13 / TF 2.21 while CI pins TF 2.18, a
`.keras` file written locally is not guaranteed to load in CI. Training
therefore also saves the best **non-TensorFlow** model as
`model_{H}h_fallback.pkl`, and the serving layer falls back to it if the Keras
load fails — a small accuracy cost instead of a dashboard outage.

**Serving caching** (as specified): models load once per process via
`@st.cache_resource`; live features use `@st.cache_data(ttl=3600)` keyed on the
current hour, so the cache refreshes when new hourly data lands rather than at
an arbitrary offset. All feature construction lives in `predict.py` and reuses
the training-time `build_features()`, so the dashboard cannot drift from what
the models were trained on.

**Self-healing hourly pipeline.** The hourly job re-fetches a 14-day window and
upserts, rather than writing a single row. A missed run (Actions outage, API
blip) is repaired by the next successful run instead of leaving a permanent
hole in the lag features.

---

## 5. Limitations

1. **CAMS is modelled, not measured.** Open-Meteo's air quality is a reanalysis/
   chemical-transport product, not a physical monitor. It is complete, current
   and genuinely about Lahore — but it is not a ground-truth sensor reading.
   The honest alternative was a dead station, or a monitor 400 km away in India.
2. **Perfect-prognosis optimism.** At training time the "forecast" weather is
   reanalysis — what the weather actually did. At serving time it is a real
   forecast carrying its own error. Live accuracy will therefore be somewhat
   below these offline numbers, and the gap widens with horizon as weather
   forecasts degrade. Quantifying this needs archived forecast runs, which
   Open-Meteo's free tier does not expose for past dates.
3. **One test period.** Metrics come from a single chronological split
   (2025-11-26 → 2026-08-08). Rolling-origin cross-validation would give
   tighter error bars.
4. **Point forecasts only.** No prediction intervals. For a health-alert
   product, "AQI 180 ± 40" is more actionable than "AQI 180"; quantile
   regression would be the natural next step.
5. **Single location.** Features are for one coordinate pair. Lahore has real
   intra-city variation that a single point cannot capture.
6. **48h/72h ordering.** As noted above, the two long horizons are within
   run-to-run noise of each other rather than cleanly ordered.

---

## 6. Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env          # add HOPSWORKS_API_KEY (+ optional AQICN_TOKEN)

python feature_pipeline/backfill.py            # ~34k rows + audit
python training_pipeline/train.py              # trains, scores, registers
python training_pipeline/ablation.py           # forecast-weather contribution
streamlit run webapp/app.py                    # dashboard
```

On Windows, add `--local-only` to the backfill and use the **backfill** GitHub
Actions workflow to seed Hopsworks (see §4 for why).

Automation: `feature_pipeline.yml` hourly, `training_pipeline.yml` daily,
`backfill.yml` manual one-off. Required repo secrets: `HOPSWORKS_API_KEY`,
`HOPSWORKS_PROJECT_NAME`, and optionally `AQICN_TOKEN`.

---

## 7. Dashboard

`webapp/app.py` provides: current AQI with EPA category colouring, the three
forecast horizons, a hazard alert banner with health guidance, a forecast chart
against the hazard threshold, a 7/14/30-day trend with summary statistics,
model performance on held-out data, SHAP feature importance (forecast-weather
features highlighted in red), and current pollutant/weather readings.

*Screenshots of the deployed dashboard to be added once Streamlit Cloud
deployment is complete.*
