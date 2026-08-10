"""
Training pipeline. Run daily by GitHub Actions.

Flow:
  1. Pull engineered features (Hopsworks, or local parquet mirror).
  2. For each horizon, select that horizon's feature block and its target.
  3. Split chronologically - never randomly. A random split on a time series
     puts hour t+1 in train and hour t in test; with 24h-autocorrelated AQI
     that leaks the answer and produces R2 ~0.99 that collapses in production.
  4. Train Ridge / RandomForest / LightGBM / XGBoost / TF MLP.
  5. Score against two naive baselines so the numbers mean something.
  6. Register the best model per horizon.

WHY BASELINES ARE REPORTED
--------------------------
An R2 of 0.75 sounds good until you notice that "tomorrow's AQI equals today's"
also scores 0.70. Every horizon is therefore scored against persistence and
same-hour-yesterday, and the summary reports the lift over the better of them.
A model that cannot beat persistence is not a forecast, it is an echo.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from feature_pipeline.store import read_all_features, get_project, hopsworks_available
from feature_pipeline.features import add_forecast_targets, horizon_feature_columns
from config import MODEL_NAME, FORECAST_HORIZONS

warnings.filterwarnings("ignore", category=UserWarning)

HORIZONS = FORECAST_HORIZONS
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

TEST_FRAC = 0.2
VAL_FRAC = 0.1  # carved off the END of train, for early stopping


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def chronological_split(df: pd.DataFrame, test_frac: float = TEST_FRAC):
    """Oldest rows train, newest rows test. Order is never shuffled."""
    df = df.sort_values("observed_at").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def build_tf_model(input_dim: int):
    import tensorflow as tf
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
    return model


def naive_baselines(train_df, test_df, target_col) -> dict:
    """
    persistence          : AQI at t+H will equal AQI at t
    same_hour_yesterday  : AQI at t+H will equal AQI at t-24h (diurnal echo)
    """
    out = {}
    y_test = test_df[target_col]

    # Must be `aqi` (the value at t), NOT aqi_lag_1h. Using the 1h-old value
    # would be a deliberately handicapped baseline and would overstate how much
    # the models actually add - the one number this whole comparison exists to
    # report honestly.
    if "aqi" in test_df.columns:
        out["baseline_persistence"] = {"metrics": evaluate(y_test, test_df["aqi"])}
    if "aqi_lag_24h_same_hour" in test_df.columns:
        out["baseline_same_hour_yesterday"] = {
            "metrics": evaluate(y_test, test_df["aqi_lag_24h_same_hour"])
        }
    return out


# --------------------------------------------------------------------------
# per-horizon training
# --------------------------------------------------------------------------

def train_for_horizon(df: pd.DataFrame, horizon: int, quick: bool = False) -> dict:
    target_col = f"aqi_target_{horizon}h"
    feature_cols = horizon_feature_columns(df, horizon, HORIZONS)

    data = df.dropna(subset=feature_cols + [target_col])
    if len(data) < 200:
        print(f"  [{horizon}h] only {len(data)} usable rows - skipping")
        return None

    train_df, test_df = chronological_split(data)
    val_cut = int(len(train_df) * (1 - VAL_FRAC))
    fit_df, val_df = train_df.iloc[:val_cut], train_df.iloc[val_cut:]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    X_fit, y_fit = fit_df[feature_cols], fit_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    print(f"  [{horizon}h] {len(feature_cols)} features | "
          f"train {len(train_df):,} ({train_df['observed_at'].min().date()} -> "
          f"{train_df['observed_at'].max().date()}) | "
          f"test {len(test_df):,} ({test_df['observed_at'].min().date()} -> "
          f"{test_df['observed_at'].max().date()})")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Ridge -----------------------------------------------------------
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_s, y_train)
    results["ridge"] = {"model": ridge, "scaler": scaler,
                        "metrics": evaluate(y_test, ridge.predict(X_test_s))}

    # --- Random Forest ---------------------------------------------------
    rf = RandomForestRegressor(
        n_estimators=100 if quick else 300,
        max_depth=None, min_samples_leaf=2,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = {"model": rf, "scaler": None,
                                "metrics": evaluate(y_test, rf.predict(X_test))}

    # --- LightGBM --------------------------------------------------------
    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, num_leaves=64,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        lgbm.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        results["lightgbm"] = {"model": lgbm, "scaler": None,
                               "metrics": evaluate(y_test, lgbm.predict(X_test)),
                               "best_iteration": int(lgbm.best_iteration_ or 0)}
    except Exception as e:
        print(f"  [{horizon}h] LightGBM failed: {e}")

    # --- XGBoost ---------------------------------------------------------
    try:
        import xgboost as xgb
        xgbm = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
            random_state=42, n_jobs=-1, early_stopping_rounds=100,
            eval_metric="rmse", tree_method="hist",
        )
        xgbm.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        results["xgboost"] = {"model": xgbm, "scaler": None,
                              "metrics": evaluate(y_test, xgbm.predict(X_test)),
                              "best_iteration": int(getattr(xgbm, "best_iteration", 0) or 0)}
    except Exception as e:
        print(f"  [{horizon}h] XGBoost failed: {e}")

    # --- TensorFlow MLP --------------------------------------------------
    if not quick:
        try:
            import tensorflow as tf
            X_fit_s = scaler.transform(X_fit)
            X_val_s = scaler.transform(X_val)
            tf_model = build_tf_model(X_train_s.shape[1])
            tf_model.fit(
                X_fit_s, y_fit,
                validation_data=(X_val_s, y_val),
                epochs=200, batch_size=128, verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0),
                ],
            )
            results["tensorflow_mlp"] = {
                "model": tf_model, "scaler": scaler,
                "metrics": evaluate(y_test, tf_model.predict(X_test_s, verbose=0).flatten()),
            }
        except Exception as e:
            print(f"  [{horizon}h] TensorFlow model failed: {e}")

    # --- baselines -------------------------------------------------------
    baselines = naive_baselines(train_df, test_df, target_col)

    trained = {k: v for k, v in results.items()}
    best_name = min(trained, key=lambda n: trained[n]["metrics"]["rmse"])

    all_metrics = {n: r["metrics"] for n, r in results.items()}
    all_metrics.update({n: r["metrics"] for n, r in baselines.items()})

    return {
        "horizon": horizon,
        "best_name": best_name,
        "best": trained[best_name],
        "all_metrics": all_metrics,
        "feature_cols": feature_cols,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_start": str(test_df["observed_at"].min()),
        "test_end": str(test_df["observed_at"].max()),
        # Kept for the SHAP pass in save_and_register. Tail of the test set, so
        # importances describe behaviour on recent unseen data rather than on
        # data the model was fit to.
        "explain_X": X_test.tail(300),
        "explain_y": y_test.tail(300),
        "all_models": trained,
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_results_table(results: list) -> str:
    """Prints, and returns as markdown for REPORT.md, the metrics table."""
    lines = []
    header = f"| {'Horizon':<8} | {'Model':<28} | {'RMSE':>8} | {'MAE':>8} | {'R2':>7} |"
    sep = "|" + "-" * 10 + "|" + "-" * 30 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 9 + "|"
    lines += [header, sep]

    for res in results:
        if not res:
            continue
        h = res["horizon"]
        ordered = sorted(res["all_metrics"].items(), key=lambda kv: kv[1]["rmse"])
        for name, m in ordered:
            star = " *BEST*" if name == res["best_name"] else ""
            label = f"{name}{star}"
            lines.append(f"| {str(h) + 'h':<8} | {label:<28} | {m['rmse']:>8.2f} | "
                         f"{m['mae']:>8.2f} | {m['r2']:>7.3f} |")
        lines.append(sep)

    table = "\n".join(lines)
    print("\n" + "=" * 84)
    print("RESULTS - all models, all horizons (test = most recent 20%, chronological)")
    print("=" * 84)
    print(table)
    return table


def print_summary(results: list) -> dict:
    print("\n" + "=" * 84)
    print("SUMMARY vs TARGET (overall R2 > 0.70, weakest horizon > 0.60)")
    print("=" * 84)

    r2s = []
    summary = {}
    for res in results:
        if not res:
            continue
        h = res["horizon"]
        best_m = res["all_metrics"][res["best_name"]]
        base = min(
            (m["rmse"] for n, m in res["all_metrics"].items() if n.startswith("baseline_")),
            default=None,
        )
        lift = (1 - best_m["rmse"] / base) * 100 if base else float("nan")
        r2s.append(best_m["r2"])
        summary[f"{h}h"] = {
            "best_model": res["best_name"], **best_m,
            "rmse_lift_vs_best_baseline_pct": round(lift, 1),
        }
        flag = "OK " if best_m["r2"] >= 0.60 else "LOW"
        print(f"  [{flag}] {h}h: {res['best_name']:<16} R2={best_m['r2']:.3f}  "
              f"RMSE={best_m['rmse']:.2f}  MAE={best_m['mae']:.2f}  "
              f"({lift:+.1f}% RMSE vs best naive baseline)")

    overall = float(np.mean(r2s)) if r2s else float("nan")
    weakest = float(np.min(r2s)) if r2s else float("nan")
    print(f"\n  Overall mean R2 : {overall:.3f}   (target > 0.70) "
          f"{'PASS' if overall > 0.70 else 'FAIL'}")
    print(f"  Weakest horizon : {weakest:.3f}   (target > 0.60) "
          f"{'PASS' if weakest > 0.60 else 'FAIL'}")
    print("=" * 84)

    summary["_overall"] = {"mean_r2": overall, "weakest_r2": weakest,
                           "target_mean_r2": 0.70, "target_weakest_r2": 0.60,
                           "trained_at": datetime.now().isoformat(timespec="seconds")}
    return summary


# --------------------------------------------------------------------------
# persistence / registry
# --------------------------------------------------------------------------

def save_and_register(result: dict, register: bool = True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    horizon = result["horizon"]
    best = result["best"]
    model_path = os.path.join(MODEL_DIR, f"model_{horizon}h.pkl")

    bundle = {
        "type": result["best_name"],
        "scaler": best["scaler"],
        "feature_cols": result["feature_cols"],
        "metrics": result["all_metrics"][result["best_name"]],
        "horizon": horizon,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    if result["best_name"] == "tensorflow_mlp":
        # .keras is the portable format - a SavedModel directory is tied more
        # tightly to the TF version that wrote it, and CI pins a different TF
        # than local dev does (see requirements.txt).
        keras_path = os.path.join(MODEL_DIR, f"model_{horizon}h.keras")
        best["model"].save(keras_path)
        bundle["keras_path"] = keras_path

        # Ship the best NON-TensorFlow model alongside it. Local dev runs
        # Python 3.13 (TF 2.21) while CI and Streamlit Cloud pin TF 2.18 for
        # protobuf compatibility with hopsworks, so a .keras file written here
        # is not guaranteed to load there. Rather than have the dashboard fail
        # to serve, it falls back to a pure-sklearn model that pickles and
        # loads identically across all three environments. The cost is a small
        # accuracy drop; the alternative is an outage.
        alt = {n: r for n, r in result["all_models"].items() if n != "tensorflow_mlp"}
        if alt:
            alt_name = min(alt, key=lambda n: alt[n]["metrics"]["rmse"])
            joblib.dump(
                {"model": alt[alt_name]["model"], "scaler": alt[alt_name]["scaler"],
                 "type": alt_name, "feature_cols": result["feature_cols"],
                 "metrics": alt[alt_name]["metrics"], "horizon": horizon,
                 "trained_at": bundle["trained_at"]},
                os.path.join(MODEL_DIR, f"model_{horizon}h_fallback.pkl"),
            )
            print(f"  saved non-TF fallback ({alt_name}, "
                  f"R2={alt[alt_name]['metrics']['r2']:.3f})")
    else:
        bundle["model"] = best["model"]

    joblib.dump(bundle, model_path)

    with open(os.path.join(MODEL_DIR, f"metrics_{horizon}h.json"), "w") as f:
        json.dump(result["all_metrics"], f, indent=2)

    print(f"  saved {model_path} ({result['best_name']})")

    # Global feature importance, computed once here so the dashboard never has
    # to run SHAP on a page load. See training_pipeline/explain.py.
    try:
        from training_pipeline.explain import compute_and_save

        X_exp = result["explain_X"]
        if best["scaler"] is not None:
            X_exp = pd.DataFrame(
                best["scaler"].transform(X_exp),
                columns=result["feature_cols"], index=X_exp.index,
            )
        table = compute_and_save(
            best["model"], X_exp, result["best_name"],
            result["feature_cols"],
            os.path.join(MODEL_DIR, f"shap_{horizon}h.json"),
        )
        top3 = ", ".join(table["feature"].head(3))
        print(f"  SHAP top features: {top3}")
    except Exception as e:
        print(f"  WARNING: feature-importance step failed: {e}")

    if not register:
        return
    if not hopsworks_available():
        print("  (Hopsworks not available - skipping registry upload)")
        return

    try:
        mr = get_project().get_model_registry()
        hw_model = mr.python.create_model(
            name=f"{MODEL_NAME}_{horizon}h",
            metrics=result["all_metrics"][result["best_name"]],
            description=f"Best model ({result['best_name']}) for {horizon}h Lahore AQI forecast",
        )
        hw_model.save(model_path)
        print(f"  registered as {MODEL_NAME}_{horizon}h in Hopsworks model registry")
    except Exception as e:
        print(f"  WARNING: Hopsworks registry upload failed: {e}")


# --------------------------------------------------------------------------

def run(quick: bool = False, register: bool = True):
    print("Loading features...")
    df = read_all_features()
    if df.empty:
        raise SystemExit("Feature store is empty - run feature_pipeline/backfill.py first.")
    print(f"  {len(df):,} rows x {len(df.columns)} cols "
          f"({df['observed_at'].min()} -> {df['observed_at'].max()})")

    if not any(f"aqi_target_{h}h" in df.columns for h in HORIZONS):
        print("Targets absent - rebuilding them...")
        df = add_forecast_targets(df, HORIZONS)

    results = []
    for horizon in HORIZONS:
        print(f"\n=== {horizon}h horizon ===")
        res = train_for_horizon(df, horizon, quick=quick)
        if res:
            for name, m in sorted(res["all_metrics"].items(), key=lambda kv: kv[1]["rmse"]):
                print(f"    {name:<30} RMSE={m['rmse']:>7.2f}  MAE={m['mae']:>7.2f}  R2={m['r2']:>6.3f}")
            print(f"    -> BEST: {res['best_name']}")
            save_and_register(res, register=register)
        results.append(res)

    table = print_results_table(results)
    summary = print_summary(results)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(MODEL_DIR, "metrics_table.md"), "w") as f:
        f.write(table + "\n")

    return results, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="smaller forest, skip TF - for fast iteration")
    p.add_argument("--no-register", action="store_true", help="skip Hopsworks registry upload")
    a = p.parse_args()
    run(quick=a.quick, register=not a.no_register)
