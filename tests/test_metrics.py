import numpy as np

from fraud.metrics import CostModel, recall_at_precision, score_report


def test_cost_total_counts():
    y = [1, 1, 0, 0]
    s = [0.9, 0.1, 0.8, 0.2]
    amt = [100.0, 50.0, 10.0, 10.0]
    r = CostModel(review_cost=5).total(y, s, amt, 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["alerts"]) == (1, 1, 1, 2)
    assert r["missed_fraud_loss"] == 50.0
    assert r["review_cost"] == 10.0
    assert r["total_cost"] == 60.0


def test_best_prefers_more_fraud_caught_on_ties():
    y = np.array([1, 0, 0])
    s = np.array([0.3, 0.1, 0.05])
    amt = np.array([0.0, 1.0, 1.0])  # zero-amount fraud: cost is identical either way
    best = CostModel(review_cost=0).best(y, s, amt, thresholds=[0.2, 0.5])
    assert best["threshold"] == 0.2 and best["tp"] == 1


def test_recall_at_precision_perfect():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    assert recall_at_precision(y, s, 0.95) == 1.0
    assert score_report(y, s)["pr_auc"] == 1.0
