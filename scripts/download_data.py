"""Download PaySim into data/raw/paysim.parquet.

Tries the canonical Kaggle dataset first (needs KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json),
then falls back to a public Hugging Face mirror that needs no login.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = RAW / "paysim.parquet"
KAGGLE_REF = "ealaxi/paysim1"
HF_REPO = "purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection"
HF_FILE = "financial_txns.parquet"
EXPECTED_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def from_kaggle() -> pd.DataFrame | None:
    has_creds = os.getenv("KAGGLE_KEY") or (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not has_creds:
        print("No Kaggle credentials, skipping Kaggle.")
        return None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_REF, path=RAW, unzip=True)
        csv = next(RAW.glob("*.csv"))
        return pd.read_csv(csv)
    except Exception as exc:  # network / auth problems -> fall back
        print(f"Kaggle download failed ({exc}), falling back.")
        return None


def from_hf() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", local_dir=RAW)
    return pd.read_parquet(path)


def main() -> None:
    if OUT.exists():
        print(f"{OUT} already exists.")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    df = from_kaggle()
    source = "kaggle"
    if df is None:
        df, source = from_hf(), "huggingface"
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        sys.exit(f"Downloaded data is missing PaySim columns: {missing}")
    df = df[EXPECTED_COLS]
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df):,} rows from {source} -> {OUT}")
    print(f"Fraud rows: {int(df['isFraud'].sum()):,}")


if __name__ == "__main__":
    main()
