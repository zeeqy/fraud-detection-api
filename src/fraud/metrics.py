"""Imbalance-aware metrics and the cost model used for threshold tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class CostModel:
    """Business cost assumptions (PaySim units treated as MYR).

    missed fraud  -> we lose the transaction amount
    any alert     -> analyst review + customer friction, a flat fee per flagged transaction
    """

    review_cost: float = 15.0
    missed_fraud_multiplier: float = 1.0

    def total(self, y_true, y_score, amount, threshold: float) -> dict:
        y_true = np.asarray(y_true).astype(bool)
        flagged = np.asarray(y_score) >= threshold
        amount = np.asarray(amount, dtype=float)
        fn_loss = amount[y_true & ~flagged].sum() * self.missed_fraud_multiplier
        review = flagged.sum() * self.review_cost
        return {
            "threshold": float(threshold),
            "missed_fraud_loss": float(fn_loss),
            "review_cost": float(review),
            "total_cost": float(fn_loss + review),
            "alerts": int(flagged.sum()),
            "tp": int((y_true & flagged).sum()),
            "fp": int((~y_true & flagged).sum()),
            "fn": int((y_true & ~flagged).sum()),
        }

    def sweep(self, y_true, y_score, amount, thresholds=None) -> list[dict]:
        if thresholds is None:
            fine = np.arange(1, 10) / 1000  # 0.001-0.009 for low-precision models
            thresholds = np.round(np.concatenate([fine, np.arange(1, 100) / 100]), 3)
        return [self.total(y_true, y_score, amount, t) for t in thresholds]

    def best(self, y_true, y_score, amount, thresholds=None, tol: float = 0.001) -> dict:
        """Cheapest threshold. When several are within `tol` (relative) of the minimum cost,
        prefer the one catching the most fraud, then the one raising the fewest alerts."""
        rows = self.sweep(y_true, y_score, amount, thresholds)
        floor = min(r["total_cost"] for r in rows) * (1 + tol)
        near = [r for r in rows if r["total_cost"] <= floor]
        return max(near, key=lambda r: (r["tp"], -r["alerts"]))


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def score_report(y_true, y_score) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "recall_at_p90": recall_at_precision(y_true, y_score, 0.90),
        "recall_at_p95": recall_at_precision(y_true, y_score, 0.95),
    }
