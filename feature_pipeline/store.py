"""
Feature storage: Hopsworks feature store, with a local parquet mirror.

WHY THERE IS A LOCAL MIRROR
---------------------------
The `hopsworks` SDK cannot be pip-installed on Windows/Python 3.13: it depends
on pyjks -> twofish, which ships source-only (no wheels for any platform) and
needs MSVC C++ build tools to compile. It installs cleanly on Linux, which is
what GitHub Actions and Streamlit Cloud both run.

Rather than make local development impossible, every read/write goes through
this module, which uses Hopsworks when it is importable and configured and
falls back to a parquet file under data/ when it is not. Same function
signatures either way, so no caller needs to know which backend is live.

This is also a genuine resilience win in CI: a Hopsworks outage degrades the
hourly pipeline to a local write instead of failing the run outright.

A second, subtler reason to keep the mirror: it makes training reproducible
offline, so the model results in REPORT.md can be regenerated without network
access or a live Hopsworks project.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOCAL_FEATURES_PATH = os.path.join(DATA_DIR, "features.parquet")


# --------------------------------------------------------------------------
# backend detection
# --------------------------------------------------------------------------

def hopsworks_available() -> bool:
    """True only if the SDK imports AND a key is configured."""
    if not HOPSWORKS_API_KEY:
        return False
    try:
        import hopsworks  # noqa: F401
        return True
    except Exception:
        return False


def _windows_tmp_shim():
    """
    The hopsworks client materialises its certificates into a hardcoded '/tmp'.
    On Windows that resolves to <current drive>:\\tmp, which usually does not
    exist, and login dies with FileNotFoundError deep inside _materialize_certs.
    Creating the directory is the whole fix. No-op everywhere else.
    """
    if os.name != "nt":
        return
    try:
        os.makedirs(os.path.join(os.path.splitdrive(os.getcwd())[0], os.sep, "tmp"),
                    exist_ok=True)
    except OSError:
        pass  # not fatal - login will report the real problem if this mattered


def get_project():
    import hopsworks
    _windows_tmp_shim()
    if not HOPSWORKS_API_KEY:
        raise RuntimeError(
            "HOPSWORKS_API_KEY is not set. Create a free project at "
            "https://app.hopsworks.ai and put the API key in your .env"
        )
    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT_NAME,
    )


def get_feature_store():
    return get_project().get_feature_store()


def get_or_create_feature_group(fs, primary_key=("observed_at",), description=""):
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=list(primary_key),
        event_time="observed_at",
        description=description or "Hourly AQI + weather + forecast-weather features for Lahore",
        online_enabled=False,
        # Explicit, not left to the default. hopsworks 5.x defaults to DELTA,
        # whose backing library (hops-deltalake) only publishes wheels for
        # Linux and macOS-arm64 - so the default silently works in CI and fails
        # on Windows with "delta library is not installed". HUDI is available
        # everywhere and gives the upsert-on-primary-key semantics this
        # pipeline depends on (the hourly job re-writes overlapping rows).
        time_travel_format="HUDI",
        # Statistics off. Hopsworks runs a separate Spark job to profile every
        # column on each insert; across 160 feature columns that job is slow
        # and fails outright on the free tier, which then marks the whole
        # insert execution FAILED even though the data materialised correctly.
        # We do not consume Hopsworks statistics anywhere - the backfill audit
        # and the EDA notebook cover data quality - so this is pure overhead.
        statistics_config={"enabled": False, "histograms": False,
                           "correlations": False, "exact_uniqueness": False},
    )


# --------------------------------------------------------------------------
# local parquet backend
# --------------------------------------------------------------------------

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hopsworks feature names must be lowercase and free of special characters,
    and every column has to carry a concrete dtype. Doing this in one place
    means the local mirror and the remote store hold identical schemas, so
    switching backends never changes what a model sees.
    """
    out = df.copy()
    out.columns = [c.lower().replace("-", "_").replace(" ", "_") for c in out.columns]
    out["observed_at"] = pd.to_datetime(out["observed_at"])

    for col in out.columns:
        if col in ("observed_at", "station"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    if "station" in out.columns:
        out["station"] = out["station"].astype(str)
    return out


def write_features_local(df: pd.DataFrame) -> str:
    """Upsert into the parquet mirror, keyed on observed_at."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = _sanitize(df)

    if os.path.exists(LOCAL_FEATURES_PATH):
        existing = pd.read_parquet(LOCAL_FEATURES_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        # keep="last" so a re-run corrects an earlier row rather than duplicating
        combined = combined.drop_duplicates(subset=["observed_at"], keep="last")
    else:
        combined = df

    combined = combined.sort_values("observed_at").reset_index(drop=True)
    combined.to_parquet(LOCAL_FEATURES_PATH, index=False)
    return LOCAL_FEATURES_PATH


def read_features_local() -> pd.DataFrame:
    if not os.path.exists(LOCAL_FEATURES_PATH):
        return pd.DataFrame()
    df = pd.read_parquet(LOCAL_FEATURES_PATH)
    return df.sort_values("observed_at").reset_index(drop=True)


# --------------------------------------------------------------------------
# public API - backend-agnostic
# --------------------------------------------------------------------------

def write_features(df: pd.DataFrame, prefer_local: bool = False):
    """
    Write engineered rows to whichever backend is available.

    Always writes the local mirror. That is deliberate: the mirror doubles as
    the offline training source, so it must stay current even when Hopsworks
    is the primary store.
    """
    path = write_features_local(df)
    print(f"  local mirror updated: {path} ({len(df)} rows in)")

    if prefer_local or not hopsworks_available():
        return None

    try:
        fs = get_feature_store()
        fg = get_or_create_feature_group(fs)
        fg.insert(_sanitize(df), write_options={"wait_for_job": True})
        print(f"  wrote {len(df)} rows to Hopsworks feature group "
              f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
        return fg
    except Exception as e:
        # Do not fail the pipeline on a store outage - the local mirror has the
        # data and the next run will re-sync.
        print(f"  WARNING: Hopsworks write failed, local mirror still updated: {e}")
        return None


def read_all_features() -> pd.DataFrame:
    """
    Read the full feature history, preferring Hopsworks and falling back to the
    local mirror (also used if Hopsworks returns fewer rows, which happens
    while a backfill job is still materialising).
    """
    local = read_features_local()

    if not hopsworks_available():
        return local

    try:
        fs = get_feature_store()
        fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
        remote = fg.read()
        if remote is not None and len(remote) >= len(local):
            return remote.sort_values("observed_at").reset_index(drop=True)
        print(f"  Hopsworks returned {0 if remote is None else len(remote)} rows vs "
              f"{len(local)} local - using local mirror")
        return local
    except Exception as e:
        print(f"  WARNING: Hopsworks read failed, using local mirror: {e}")
        return local


def backend_name() -> str:
    return "hopsworks" if hopsworks_available() else "local-parquet"


if __name__ == "__main__":
    print(f"Active backend: {backend_name()}")
    df = read_all_features()
    if df.empty:
        print("No features stored yet - run backfill.py first.")
    else:
        print(f"{len(df)} rows, {df['observed_at'].min()} -> {df['observed_at'].max()}")
        print(f"{len(df.columns)} columns")
