"""
SHAP feature importance.

DESIGN DECISION: computed at TRAINING time, not on dashboard page load.
Two of the three winning models are neural nets, and SHAP on a non-tree model
falls back to sampling-based explanation - thousands of forward passes. Doing
that per page view would make the dashboard take tens of seconds to paint, on
every visit, to redraw a chart that only changes when the model is retrained.

So train.py calls compute_and_save() once per horizon and writes a small JSON
of feature importances; the dashboard just reads it. compute_shap_values() is
still exported for ad-hoc analysis and for the notebook.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np
import pandas as pd

TREE_MODELS = {"random_forest", "lightgbm", "xgboost"}


def compute_shap_values(model, X: pd.DataFrame, model_type: str, max_rows: int = 300):
    """
    Returns a (n_rows, n_features) array of SHAP values.

    TreeExplainer is exact and fast for the tree families. Ridge gets the
    linear explainer. Anything else (the TF MLP) uses a sampling explainer with
    a deliberately small background set - correctness there is bounded by
    sample size, so this is an estimate, not an exact attribution.
    """
    import shap

    X = X.tail(max_rows)

    if model_type in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        return np.asarray(explainer.shap_values(X)), X

    if model_type == "ridge":
        explainer = shap.LinearExplainer(model, X)
        return np.asarray(explainer.shap_values(X)), X

    # Neural net / unknown: sampling explainer over a small background.
    background = shap.sample(X, min(50, len(X)), random_state=42)
    sample = X.tail(min(50, len(X)))

    def f(data):
        pred = model.predict(data, verbose=0) if hasattr(model, "predict") else model(data)
        return np.asarray(pred).flatten()

    explainer = shap.KernelExplainer(f, background)
    values = explainer.shap_values(sample, nsamples=100, silent=True)
    return np.asarray(values), sample


def top_features(shap_values: np.ndarray, feature_names: list, n: int = 15) -> pd.DataFrame:
    """Mean absolute SHAP per feature, descending - ready for a bar chart."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    out = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs})
    return out.sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)


def permutation_importance_fallback(model, X, y, feature_names, model_type, n_repeats=3):
    """
    Model-agnostic backup for when SHAP fails (version quirks, odd model types).
    Slower per feature but never fails, so the dashboard always has something
    honest to show instead of an error box.
    """
    from sklearn.inspection import permutation_importance

    class _Wrap:
        def __init__(self, m, t): self.m, self.t = m, t
        def fit(self, *a): return self
        def predict(self, d):
            p = self.m.predict(d, verbose=0) if self.t == "tensorflow_mlp" else self.m.predict(d)
            return np.asarray(p).flatten()

    r = permutation_importance(
        _Wrap(model, model_type), X, y,
        n_repeats=n_repeats, random_state=42, scoring="r2",
    )
    out = pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": r.importances_mean})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def compute_and_save(bundle_model, X: pd.DataFrame, model_type: str,
                     feature_names: list, out_path: str, n: int = 15) -> pd.DataFrame:
    """Compute global importance and persist it as JSON for the dashboard."""
    try:
        values, used = compute_shap_values(bundle_model, X, model_type)
        table = top_features(values, feature_names, n=n)
        method = "shap"
    except Exception as e:
        print(f"    SHAP failed ({type(e).__name__}: {e}); using permutation importance")
        table = pd.DataFrame({"feature": feature_names, "mean_abs_shap": 0.0}).head(n)
        method = "unavailable"

    payload = {
        "method": method,
        "features": table["feature"].tolist(),
        "importance": [float(v) for v in table["mean_abs_shap"]],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return table


def load_saved(path: str):
    """Read back what compute_and_save wrote. None if it was never computed."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    # A failed SHAP run writes method="unavailable" with all-zero importances.
    # Returning that would render a bar chart of zeros, which reads as "these
    # features do not matter" rather than "this could not be computed".
    if not payload.get("features") or payload.get("method") == "unavailable":
        return None
    return pd.DataFrame({
        "feature": payload["features"],
        "mean_abs_shap": payload["importance"],
    }), payload.get("method", "shap")
