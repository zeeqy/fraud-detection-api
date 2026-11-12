"""Load the trained bundle once and score transactions with it."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from fraud.explain import Explainer
from fraud.features import SCORED_TYPES, build_features

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
PUBLIC_KEYS = [
    "model_name", "version", "trained_at", "data", "split", "features", "threshold",
    "cost_model", "test", "test_ablation_no_drain", "shap_mean_abs",
]  # fmt: skip


def _unscored(txn_type: str) -> dict:
    return {
        "fraud_probability": 0.0,
        "decision": "allow",
        "scored": False,
        "reasons": [
            {
                "feature": "type",
                "impact": 0.0,
                "direction": "lowers risk",
                "text": f"{txn_type} transactions had no fraud in the training data, "
                "so they are not scored",
            }
        ],
    }


class FraudModel:
    def __init__(self, path: str | Path | None = None):
        # joblib uses pickle: only load the bundle this repo trains, never a user-supplied file.
        bundle = joblib.load(path or os.getenv("MODEL_PATH") or DEFAULT_PATH)
        self.model = bundle["model"]
        self.info = bundle["info"]
        self.threshold = float(self.info["threshold"])
        self.version = str(self.info["version"])
        self.explainer = Explainer(self.model)

    def predict(self, rows: list[dict], top_k: int = 3) -> list[dict]:
        """Score rows that use PaySim column names (see Transaction.to_raw). Order is kept."""
        results = {i: _unscored(r["type"]) for i, r in enumerate(rows)}
        idx = [i for i, r in enumerate(rows) if r["type"] in SCORED_TYPES]
        if idx:
            scored = [rows[i] for i in idx]
            probs = self.model.predict_proba(build_features(pd.DataFrame(scored)))[:, 1]
            reasons = self.explainer.reasons(scored, top_k=top_k)
            for i, p, rs in zip(idx, probs, reasons, strict=True):
                results[i] = {
                    "fraud_probability": round(float(p), 6),
                    "decision": "flag" if p >= self.threshold else "allow",
                    "scored": True,
                    "reasons": rs,
                }
        return [results[i] for i in range(len(rows))]

    def public_info(self) -> dict:
        return {k: self.info[k] for k in PUBLIC_KEYS if k in self.info}
