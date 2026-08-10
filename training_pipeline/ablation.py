"""
Ablation: how much do the forecast-weather features actually buy?

The claim driving this project's design is that feeding each horizon's model
the weather forecast valid at that horizon beats using history alone. A
univariate correlation check cannot settle that - current and future weather
are themselves strongly correlated, so their marginal correlations with the
target look nearly identical. The honest test is to hold everything else fixed,
remove the fc{H}_* block, retrain, and compare.

Three feature sets per horizon:
  history_only    lags, rollings, pollutants, time, and weather AS OF t
  forecast_only   history minus current weather, plus the fc{H}_* block
  full            everything (what the shipped models use)

LightGBM throughout - fast, and identical hyperparameters across arms so the
only thing varying is the feature set.

    python training_pipeline/ablation.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

from config import FORECAST_HORIZONS, WEATHER_VARS
from feature_pipeline.store import read_all_features
from feature_pipeline.features import add_forecast_targets, horizon_feature_columns
from training_pipeline.train import chronological_split, evaluate, MODEL_DIR

VAL_FRAC = 0.1


def split_feature_sets(cols: list, horizon: int) -> dict:
    """Partition a horizon's columns into the three arms described above."""
    fc_prefix = f"fc{horizon}_"
    fc_cols = [c for c in cols if c.startswith(fc_prefix)]

    # "current weather" = the raw weather vars at t and their derived rollings.
    weather_roots = tuple(WEATHER_VARS) + ("wind_dir_", "ventilation_index", "precip_sum_")
    cur_weather = [
        c for c in cols
        if not c.startswith(fc_prefix) and c.startswith(weather_roots)
    ]

    history_only = [c for c in cols if c not in fc_cols]
    forecast_only = [c for c in cols if c not in cur_weather]
    return {
        "history_only": history_only,
        "forecast_only": forecast_only,
        "full": cols,
    }


def fit_eval(df, feature_cols, target_col):
    import lightgbm as lgb

    data = df.dropna(subset=feature_cols + [target_col])
    train_df, test_df = chronological_split(data)
    cut = int(len(train_df) * (1 - VAL_FRAC))
    fit_df, val_df = train_df.iloc[:cut], train_df.iloc[cut:]

    model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, num_leaves=64,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(
        fit_df[feature_cols], fit_df[target_col],
        eval_set=[(val_df[feature_cols], val_df[target_col])],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return evaluate(test_df[target_col], model.predict(test_df[feature_cols])), len(feature_cols)


def run():
    df = read_all_features()
    if not any(f"aqi_target_{h}h" in df.columns for h in FORECAST_HORIZONS):
        df = add_forecast_targets(df, FORECAST_HORIZONS)

    results = {}
    print(f"{'Horizon':<9}{'Feature set':<16}{'n_feat':>7}{'RMSE':>9}{'MAE':>9}{'R2':>8}")
    print("-" * 58)

    for h in FORECAST_HORIZONS:
        target_col = f"aqi_target_{h}h"
        cols = horizon_feature_columns(df, h, FORECAST_HORIZONS)
        arms = split_feature_sets(cols, h)

        results[h] = {}
        for name in ["history_only", "forecast_only", "full"]:
            metrics, n = fit_eval(df, arms[name], target_col)
            results[h][name] = metrics
            print(f"{str(h) + 'h':<9}{name:<16}{n:>7}{metrics['rmse']:>9.2f}"
                  f"{metrics['mae']:>9.2f}{metrics['r2']:>8.3f}")
        print("-" * 58)

    print("\nForecast-weather contribution (full vs history_only):")
    for h in FORECAST_HORIZONS:
        a, b = results[h]["history_only"], results[h]["full"]
        d_r2 = b["r2"] - a["r2"]
        d_rmse = (1 - b["rmse"] / a["rmse"]) * 100
        print(f"  +{h}h: R2 {a['r2']:.3f} -> {b['r2']:.3f} "
              f"({d_r2:+.3f}), RMSE {d_rmse:+.1f}%")

    out = os.path.join(MODEL_DIR, "ablation.json")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nsaved {out}")
    return results


if __name__ == "__main__":
    run()
