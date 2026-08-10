"""
Push the local parquet mirror into the Hopsworks feature store.

Why this exists as its own script: the backfill fetches ~34k rows from
Open-Meteo and audits them, which is slow and rate-limited. Once that has been
done and validated locally, re-running the whole fetch just to populate
Hopsworks would be wasteful and would hit the API again for data we already
hold. This uploads what is already on disk.

    python feature_pipeline/sync_to_hopsworks.py              # everything
    python feature_pipeline/sync_to_hopsworks.py --since 2025-01-01
    python feature_pipeline/sync_to_hopsworks.py --chunk-size 2000

Note for Windows users: the hopsworks SDK cannot be pip-installed on Windows
(pyjks -> twofish needs MSVC build tools). Either run this from Linux/WSL, or
use the one-off `backfill` GitHub Actions workflow, which does the same job on
ubuntu-latest.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import pandas as pd

from feature_pipeline.store import (
    read_features_local,
    get_feature_store,
    get_or_create_feature_group,
    hopsworks_available,
    _sanitize,
)
from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION


def run(since: str = None, chunk_size: int = 5000):
    if not hopsworks_available():
        raise SystemExit(
            "Hopsworks SDK unavailable or HOPSWORKS_API_KEY unset.\n"
            "On Windows the SDK will not install - use the 'One-off historical "
            "backfill' GitHub Actions workflow instead."
        )

    df = read_features_local()
    if df.empty:
        raise SystemExit("Local mirror is empty - run feature_pipeline/backfill.py first.")

    if since:
        df = df[df["observed_at"] >= pd.Timestamp(since)]

    df = _sanitize(df).sort_values("observed_at").reset_index(drop=True)
    print(f"Uploading {len(df):,} rows x {len(df.columns)} cols "
          f"({df['observed_at'].min()} -> {df['observed_at'].max()})")

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    total = 0
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        # wait_for_job on the final chunk only: waiting on every chunk
        # serialises ~7 materialisation jobs and turns minutes into an hour.
        last = start + chunk_size >= len(df)
        fg.insert(chunk, write_options={"wait_for_job": last})
        total += len(chunk)
        print(f"  {total:,}/{len(df):,} rows sent")

    print(f"\nDone. Feature group {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} updated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--since", default=None, help="only upload rows from this date")
    p.add_argument("--chunk-size", type=int, default=5000)
    a = p.parse_args()
    run(a.since, a.chunk_size)
