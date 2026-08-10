# Lahore AQI Predictor

Serverless end-to-end pipeline forecasting Lahore's Air Quality Index
**24h / 48h / 72h ahead**, with a Hopsworks feature store + model registry,
GitHub Actions scheduling, and a Streamlit dashboard.

### ▶ Live dashboard: https://aqi-predictor-gysxgouemt7ar72ptwcema.streamlit.app

![Dashboard](docs/screenshots/dashboard.png)

| Horizon | Model | R² | RMSE | MAE | vs best naive baseline |
|---|---|---|---|---|---|
| +24h | XGBoost | **0.833** | 21.75 | 15.81 | +31.1% RMSE |
| +48h | TensorFlow MLP | **0.684** | 29.91 | 22.75 | +23.7% RMSE |
| +72h | TensorFlow MLP | **0.712** | 28.59 | 21.28 | +31.9% RMSE |

Mean R² **0.743** (target > 0.70) · weakest horizon **0.684** (target > 0.60).
Scored on the most recent 20% of ~34.5k hourly rows, split chronologically.

Full write-up: **[REPORT.md](REPORT.md)** · Analysis: **[notebooks/eda.ipynb](notebooks/eda.ipynb)**

## The core idea

Lag features describe where pollution *has been*. What decides where it goes
over the next three days is the weather that is *coming*. So each horizon's
model is fed the **weather forecast valid at that horizon** — the `fc{H}_*`
feature block — not just historical weather.

Measured contribution ([`training_pipeline/ablation.py`](training_pipeline/ablation.py)):

| Horizon | History only | + forecast weather | Δ R² |
|---|---|---|---|
| +24h | 0.767 | 0.820 | +0.053 |
| +48h | 0.526 | 0.665 | +0.140 |
| +72h | 0.444 | 0.654 | **+0.210** |

The gain grows with horizon. **Without it, 48h and 72h fall below the 0.60
floor** — this one design choice is what makes the project pass.

## Architecture

```
Open-Meteo CAMS (AQI) ─┐
                       ├─> features.py ─> Hopsworks Feature Store ─┐
Open-Meteo weather ────┘    (per-horizon    (+ local parquet mirror)│
   archive + FORECAST        fc{H}_* block)                         │
                                                                    v
                                                       training_pipeline/train.py
                                                    (Ridge/RF/LGBM/XGB/TF, daily)
                                                                    │
                                             Hopsworks Model Registry + models/
                                                                    │
                                                                    v
                                                   Streamlit dashboard (webapp/)
```

## Data source note

The project was scaffolded around AQICN, but **AQICN has no live data for
Lahore** — its only station stopped reporting (returns an 18-month-old value),
a bounding-box query returns zero stations, and the geo endpoint silently
substitutes a station in **Delhi, India**. Observations therefore come from
Open-Meteo's CAMS product (no API key, complete hourly history back to
2022-09-01). `aqicn_client.py` is kept as an optional cross-check.
See [REPORT.md §1](REPORT.md) for the verification.

## Setup

There are two requirement sets:

| File | For | Python |
|---|---|---|
| `requirements.txt` | the Streamlit dashboard | any 3.9+ |
| `requirements-pipeline.txt` | feature pipeline, training, backfill | **3.9–3.12** |

The split keeps the dashboard from having to install TensorFlow just to draw
charts. It does include `hopsworks`, but gated to `python_version < "3.13"`, so
the dashboard reads the feature store and model registry where the SDK can be
installed and falls back to the live APIs plus the repo's model copies where it
cannot. Likewise, where a Keras model won't load it uses the non-TF model saved
beside it. The footer states which source is actually in use.

The pipeline set needs Python **3.9–3.12**: TensorFlow is pinned to 2.18.1 for
`protobuf` compatibility with `hopsworks`, and that release ships no wheels for
3.13+ — pip reports `No matching distribution found for tensorflow==2.18.1`
without mentioning your interpreter version.

```bash
pip install -r requirements-pipeline.txt   # full stack
cp .env.example .env      # add HOPSWORKS_API_KEY + HOPSWORKS_PROJECT_NAME
```

Only Hopsworks needs credentials — every data source is keyless.

```bash
python feature_pipeline/backfill.py        # ~34.5k rows + data-quality audit
python training_pipeline/train.py          # train, score, register
python training_pipeline/ablation.py       # forecast-weather contribution
streamlit run webapp/app.py                # dashboard
```

### On Windows

The `hopsworks` SDK **cannot be pip-installed on Windows** — it depends on
`pyjks` → `twofish`, which ships source-only and needs MSVC C++ build tools.
Everything still works: `feature_pipeline/store.py` falls back to a local
parquet mirror, so backfill, training and the dashboard all run unchanged. To
populate Hopsworks itself, use the **One-off historical backfill** GitHub
Actions workflow, which runs on Linux.

```bash
python feature_pipeline/backfill.py --local-only
```

## Automation

| Workflow | Schedule | Purpose |
|---|---|---|
| `feature_pipeline.yml` | hourly (:15) | Fetch latest AQ + forecast weather, upsert features |
| `training_pipeline.yml` | daily (03:00 UTC) | Retrain, register best model, commit artifacts |
| `backfill.yml` | manual | One-off seed of the full history |

Repo secrets: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`, optionally `AQICN_TOKEN`.

The hourly job re-fetches a **14-day window and upserts** rather than writing a
single row, so a missed run self-heals on the next one instead of leaving a
permanent hole in the lag features.

## Deploying the dashboard (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → **New app**, select the repo, branch
   `main`, main file `webapp/app.py`.
3. Add `HOPSWORKS_API_KEY` and `HOPSWORKS_PROJECT_NAME` under **Secrets** to
   serve from the feature store and model registry. Without them the app falls
   back to the live APIs and the models committed in the repo — it still works,
   it just isn't reading the store. The footer states which source is in use.
4. Choose **Python 3.11** under **Advanced settings** if you want the
   feature-store path. Streamlit Cloud otherwise provisions Python 3.14, where
   `hopsworks` cannot install at all — `confluent-kafka` publishes no cp314
   wheels — and the app will run on the live-API fallback instead.

Streamlit Cloud fixes the Python version **when the app is created**; it cannot
be changed later from the settings page, so a new app is needed to switch. The
build never breaks either way: `hopsworks` is marker-gated, so on 3.14 it is
simply skipped. Verified in both configurations — against the real project the
footer reads `Source: Hopsworks feature store`; with the SDK absent (pandas
3.0.5 / numpy 2.5.2 / scikit-learn 1.9, no TensorFlow) it reads
`Source: live Open-Meteo APIs` and serves the non-TF fallback models. Zero
exceptions in both.

The daily training workflow commits refreshed models, and Streamlit Cloud
redeploys on push, so the deployed app keeps up to date on its own.

## Project layout

```
config.py                             city, coords, thresholds, feature constants
feature_pipeline/
  openmeteo_aq_client.py              AQI observations (primary source)
  weather_forecast_client.py          weather archive + FORECAST  <- the key fix
  aqicn_client.py                     optional cross-check only
  features.py                         time/lag/rolling/forecast-weather features
  store.py                            Hopsworks + local parquet fallback
  run_feature_pipeline.py             hourly entrypoint
  backfill.py                         historical seed + data-quality audit
  sync_to_hopsworks.py                push local mirror to the feature store
training_pipeline/
  train.py                            Ridge/RF/LightGBM/XGBoost/TF MLP per horizon
  predict.py                          live feature build + serving
  explain.py                          SHAP importance (precomputed at train time)
  ablation.py                         measures the forecast-weather contribution
  register_models.py                  standalone model-registry upload
webapp/app.py                         Streamlit dashboard
notebooks/eda.ipynb                   EDA
```

## Dependency notes

Three constraints are load-bearing here; changing any of them silently breaks
something, so they are worth knowing before touching the requirement files.

**TensorFlow is pinned to 2.18.1** in `requirements-pipeline.txt`. `hopsworks`
pins `protobuf<5` in every published version, while TensorFlow ≥2.19 needs
`protobuf>=6.31` — the ranges don't intersect. 2.18.1 allows
`protobuf>=3.20.3,<6.0.0dev`, which overlaps at 4.25.x.

**Streamlit is capped below 1.61** in `requirements.txt`. From 1.61 it requires
`protobuf>=5.26.1`, which again collides with the `hopsworks` pin — uncapped,
the dashboard can never reach the feature store.

**`hopsworks[python]`, not bare `hopsworks`,** in the pipeline set. Reads go over
Arrow Flight, but feature-group *writes* go through Kafka, and hsfs gates that
path on `HAS_CONFLUENT_KAFKA`. `confluent-kafka` ships only in that extra, so
without it `fg.insert()` raises `ModuleNotFoundError` while reads keep working —
which presents as a pipeline running green while the feature store never
advances.
