"""
Upload already-trained model artifacts to the Hopsworks model registry.

train.py registers automatically when the SDK is importable. This standalone
script exists for the case where it is not - notably Windows local development,
where `hopsworks` cannot be pip-installed into the same environment as
TensorFlow (see requirements.txt and REPORT.md §4). Training runs in the main
environment; registration runs from wherever the SDK is available, against the
artifacts already sitting in models/.

    python training_pipeline/register_models.py
    python training_pipeline/register_models.py --horizons 24 72
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from config import MODEL_NAME, FORECAST_HORIZONS
from feature_pipeline.store import get_project, hopsworks_available

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")


def register(horizon: int, mr) -> bool:
    model_path = os.path.join(MODEL_DIR, f"model_{horizon}h.pkl")
    metrics_path = os.path.join(MODEL_DIR, f"metrics_{horizon}h.json")

    if not os.path.exists(model_path):
        print(f"  [{horizon}h] no artifact at {model_path} - skipping")
        return False

    # Read metadata from the JSON sidecars rather than unpickling the bundle.
    # Unpickling would import the model's own library (xgboost, lightgbm,
    # tensorflow...), which defeats the purpose: this script has to run in the
    # minimal environment where the Hopsworks SDK is installable, and that
    # environment deliberately does not carry the training stack.
    summary_path = os.path.join(MODEL_DIR, "summary.json")
    best_type, metrics, trained_at = "unknown", {}, "unknown"

    if os.path.exists(summary_path):
        entry = json.load(open(summary_path)).get(f"{horizon}h", {})
        best_type = entry.get("best_model", best_type)
        metrics = {k: v for k, v in entry.items() if isinstance(v, (int, float))}
        trained_at = json.load(open(summary_path)).get("_overall", {}).get("trained_at", trained_at)

    if not metrics and os.path.exists(metrics_path):
        metrics = json.load(open(metrics_path)).get(best_type, {})

    bundle = {"type": best_type, "trained_at": trained_at}

    # The registry only accepts scalar metrics.
    clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    hw_model = mr.python.create_model(
        name=f"{MODEL_NAME}_{horizon}h",
        metrics=clean,
        description=(f"Best model ({best_type}) for {horizon}h Lahore AQI forecast. "
                     f"Trained {bundle.get('trained_at', 'unknown')}."),
    )

    # Upload the whole models/ payload for this horizon: the bundle, the Keras
    # weights if it is a neural net, and the non-TF fallback, so a consumer can
    # serve the model even where TensorFlow will not load (see predict.py).
    artifacts = [model_path]
    for extra in (f"model_{horizon}h.keras", f"model_{horizon}h_fallback.pkl",
                  f"metrics_{horizon}h.json", f"shap_{horizon}h.json"):
        path = os.path.join(MODEL_DIR, extra)
        if os.path.exists(path):
            artifacts.append(path)

    upload_dir = os.path.join(MODEL_DIR, f"_upload_{horizon}h")
    os.makedirs(upload_dir, exist_ok=True)
    import shutil
    for a in artifacts:
        shutil.copy2(a, upload_dir)

    hw_model.save(upload_dir)
    shutil.rmtree(upload_dir, ignore_errors=True)

    r2 = clean.get("r2")
    print(f"  [{horizon}h] registered {MODEL_NAME}_{horizon}h ({best_type}"
          + (f", R2={r2:.3f}" if r2 is not None else "") + ")")
    return True


def run(horizons=FORECAST_HORIZONS):
    if not hopsworks_available():
        raise SystemExit(
            "Hopsworks SDK unavailable or HOPSWORKS_API_KEY unset.\n"
            "On Windows the SDK will not pip-install alongside TensorFlow - run "
            "this from Linux/WSL, or let the training GitHub Actions workflow "
            "handle registration."
        )

    mr = get_project().get_model_registry()
    print(f"Registering {len(horizons)} model(s) from {MODEL_DIR}")
    n = sum(register(h, mr) for h in horizons)
    print(f"Done - {n} model(s) registered.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", type=int, nargs="*", default=list(FORECAST_HORIZONS))
    a = p.parse_args()
    run(a.horizons)
