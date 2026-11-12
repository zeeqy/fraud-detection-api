"""SHAP explanations turned into plain-English reasons."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap

from fraud.features import FEATURE_COLS, build_features, describe


class Explainer:
    def __init__(self, model):
        self.model = model
        self._explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            # shap warns that LightGBM output is a list of arrays; handled just below.
            warnings.filterwarnings("ignore", message="LightGBM binary classifier")
            vals = self._explainer.shap_values(X[FEATURE_COLS])
        if isinstance(vals, list):  # older shap: one array per class
            vals = vals[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, 1]
        return vals

    def reasons(self, rows: list[dict], top_k: int = 3) -> list[list[dict]]:
        X = build_features(pd.DataFrame(rows))
        vals = self.shap_values(X)
        out = []
        for row, sv in zip(rows, vals, strict=True):
            order = np.argsort(-np.abs(sv))[:top_k]
            out.append(
                [
                    {
                        "feature": FEATURE_COLS[i],
                        "impact": round(float(sv[i]), 4),
                        "direction": "raises risk" if sv[i] > 0 else "lowers risk",
                        "text": f"{describe(FEATURE_COLS[i], row)} "
                        f"({'raises' if sv[i] > 0 else 'lowers'} risk)",
                    }
                    for i in order
                ]
            )
        return out
