"""Train, compare and package the fraud model.

    uv run python -m fraud.train

Writes models/model.joblib (served by the API), reports/metrics.json and reports/figures/*.png,
and logs every experiment to a local MLflow store (sqlite:///mlflow.db).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fraud.features import FEATURE_COLS, build_features, filter_scored  # noqa: E402
from fraud.metrics import CostModel, score_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "paysim.parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
SEED = 42
ABLATION_DROP = ["drains_account", "amount_to_balance"]
LGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
)


def load_splits():
    df = filter_scored(pd.read_parquet(DATA)).reset_index(drop=True)
    X, y = build_features(df), df["isFraud"].to_numpy()
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEED)
    return df, X, y, tr, va, te


def fit_lgb(Xtr, ytr, Xva, yva, **extra):
    model = lgb.LGBMClassifier(**LGB_PARAMS, **extra)
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def logreg(**kw):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))


def experiments(X, y, tr, va):
    Xtr, ytr, Xva, yva = X.iloc[tr], y[tr], X.iloc[va], y[va]
    # SMOTE up to a 1:10 ratio rather than 1:1: enough minority signal, far less synthetic data.
    sm_X, sm_y = SMOTE(sampling_strategy=0.1, random_state=SEED).fit_resample(Xtr, ytr)
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    abl_tr, abl_va = Xtr.drop(columns=ABLATION_DROP), Xva.drop(columns=ABLATION_DROP)
    return {
        "logreg_classweight": lambda: logreg(class_weight="balanced").fit(Xtr, ytr),
        "logreg_smote": lambda: logreg().fit(sm_X, sm_y),
        "lgbm_unweighted": lambda: fit_lgb(Xtr, ytr, Xva, yva),
        "lgbm_classweight": lambda: fit_lgb(Xtr, ytr, Xva, yva, scale_pos_weight=spw),
        "lgbm_smote": lambda: fit_lgb(sm_X, sm_y, Xva, yva),
        "lgbm_ablation_no_drain": lambda: fit_lgb(abl_tr, ytr, abl_va, yva),
        "logreg_ablation_no_drain": lambda: logreg(class_weight="balanced").fit(abl_tr, ytr),
    }


def predict(model, X):
    cols = getattr(model, "feature_name_", None) or getattr(model, "feature_names_in_", None)
    cols = list(cols) if cols is not None else FEATURE_COLS
    return model.predict_proba(X[cols])[:, 1]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120, bbox_inches="tight")
    plt.close("all")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("paysim-fraud")

    df, X, y, tr, va, te = load_splits()
    print(f"scored rows {len(df):,}  train {len(tr):,}  val {len(va):,}  test {len(te):,}")
    results, models = {}, {}
    for name, build in experiments(X, y, tr, va).items():
        with mlflow.start_run(run_name=name):
            t0 = time.time()
            model = build()
            fit_s = time.time() - t0
            rep = score_report(y[va], predict(model, X.iloc[va]))
            mlflow.log_params({"model": name, "n_train": len(tr), "fit_seconds": round(fit_s, 1)})
            if hasattr(model, "best_iteration_"):
                mlflow.log_param("best_iteration", model.best_iteration_)
            mlflow.log_metrics({f"val_{k}": v for k, v in rep.items()})
        results[name], models[name] = rep, model
        line = "  ".join(f"{k}={v:.4f}" for k, v in rep.items())
        iters = getattr(model, "best_iteration_", "-")
        print(f"{name:24s} {line}  iters={iters}  ({fit_s:.0f}s)")

    # Serve a LightGBM model: TreeExplainer gives exact, fast SHAP values per request.
    # Prefer the unweighted model unless another LightGBM variant beats it by > 0.002 PR-AUC:
    # class weights and SMOTE shift the predicted probabilities away from the real base rate,
    # and the API returns a probability.
    candidates = [n for n in results if n.startswith("lgbm") and "ablation" not in n]
    top = max(candidates, key=lambda n: results[n]["pr_auc"])
    gap = results[top]["pr_auc"] - results["lgbm_unweighted"]["pr_auc"]
    best_name = top if gap > 0.002 else "lgbm_unweighted"
    best = models[best_name]
    print(f"selected: {best_name} (top by val PR-AUC: {top}, gap {gap:.4f})")

    # Robustness: time-based split (train on first 500 hourly steps, test on the rest).
    early = (df["step"] <= 500).to_numpy()
    t_model = fit_lgb(X[early], y[early], X[~early], y[~early])
    time_split = score_report(y[~early], predict(t_model, X[~early]))
    time_split["test_fraud_rate"] = float(y[~early].mean())
    print("time split:", time_split)

    cost = CostModel()
    amt = df["amount"].to_numpy()
    p_va, p_te = predict(best, X.iloc[va]), predict(best, X.iloc[te])
    sweep = cost.sweep(y[va], p_va, amt[va])
    thr = cost.best(y[va], p_va, amt[va])["threshold"]
    # Same analysis on the ablation model, where the threshold is a real trade-off.
    abl = models["lgbm_ablation_no_drain"]
    pa_va, pa_te = predict(abl, X.iloc[va]), predict(abl, X.iloc[te])
    abl_sweep = cost.sweep(y[va], pa_va, amt[va])
    abl_thr = cost.best(y[va], pa_va, amt[va])["threshold"]
    test_rep = score_report(y[te], p_te)

    # Figures -----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in ["logreg_classweight", best_name, "lgbm_ablation_no_drain"]:
        PrecisionRecallDisplay.from_predictions(
            y[te], predict(models[n], X.iloc[te]), name=n, ax=ax
        )
    ax.set_title("Precision-recall, test set")
    save(fig, "pr_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    panels = [
        (sweep, thr, f"Served model ({best_name})"),
        (abl_sweep, abl_thr, "Ablation: no account-drain features"),
    ]
    for ax, (rows, t, title) in zip(axes, panels, strict=True):
        sw = pd.DataFrame(rows)
        ax.plot(
            sw.threshold, sw.missed_fraud_loss / 1e6, label="missed fraud loss", lw=4, alpha=0.5
        )
        ax.plot(
            sw.threshold,
            sw.review_cost / 1e6,
            label=f"review cost (RM{cost.review_cost:.0f}/alert)",
        )
        ax.plot(sw.threshold, sw.total_cost / 1e6, label="total", lw=2.5, color="black")
        ax.axvline(t, ls="--", color="red", label=f"chosen threshold {t:.3f}")
        ax.set_xlabel("decision threshold (log scale)")
        ax.set_xscale("log")
        ax.set_ylabel("cost on validation set (RM million)")
        ax.set_yscale("log")
        ax.set_title(title)
    axes[0].legend()
    save(fig, "cost_curve.png")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y[te],
        (p_te >= thr).astype(int),
        display_labels=["legit", "fraud"],
        ax=ax,
        values_format=",",
        colorbar=False,
    )
    ax.set_title(f"Test confusion matrix @ {thr:.3f}")
    save(fig, "confusion_matrix.png")

    sample = X.iloc[te].sample(5000, random_state=SEED)
    sv = shap.TreeExplainer(best).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(sv, sample, show=False, max_display=11)
    plt.title("SHAP: global feature impact")
    save(plt.gcf(), "shap_summary.png")
    mean_abs = np.abs(sv).mean(0).round(4).tolist()
    importance = dict(sorted(zip(FEATURE_COLS, mean_abs, strict=True), key=lambda kv: -kv[1]))

    # Package -----------------------------------------------------------------------------
    info = {
        "model_name": best_name,
        "version": datetime.now(UTC).strftime("%Y%m%d%H%M"),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": "PaySim (6.36M synthetic mobile-money txns), TRANSFER + CASH_OUT only",
        "split": {
            "train": len(tr),
            "val": len(va),
            "test": len(te),
            "method": "stratified 70/15/15",
        },
        "features": FEATURE_COLS,
        "threshold": thr,
        "cost_model": {"review_cost_per_alert": cost.review_cost, "missed_fraud": "txn amount"},
        "validation": results,
        "test": test_rep,
        "test_logreg_baseline": score_report(
            y[te], predict(models["logreg_classweight"], X.iloc[te])
        ),
        "test_logreg_ablation_no_drain": score_report(
            y[te], predict(models["logreg_ablation_no_drain"], X.iloc[te])
        ),
        "test_ablation_no_drain": score_report(
            y[te], predict(models["lgbm_ablation_no_drain"], X.iloc[te])
        ),
        "test_cost_at_threshold": cost.total(y[te], p_te, amt[te], thr),
        "test_cost_at_0.5": cost.total(y[te], p_te, amt[te], 0.5),
        "test_cost_no_model": float(amt[te][y[te] == 1].sum()),
        "ablation_threshold": abl_thr,
        "ablation_test_cost_at_threshold": cost.total(y[te], pa_te, amt[te], abl_thr),
        "ablation_test_cost_at_0.5": cost.total(y[te], pa_te, amt[te], 0.5),
        "time_split_check": time_split,
        "shap_mean_abs": importance,
    }
    joblib.dump({"model": best, "info": info}, MODELS / "model.joblib", compress=3)
    (REPORTS / "metrics.json").write_text(json.dumps(info, indent=2))
    with mlflow.start_run(run_name=f"final_{best_name}"):
        mlflow.log_params({"model": best_name, "threshold": thr})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_rep.items()})
        mlflow.log_metric("test_total_cost", info["test_cost_at_threshold"]["total_cost"])
        mlflow.log_artifacts(str(FIG), "figures")
    keys = [
        "test",
        "test_logreg_baseline",
        "test_ablation_no_drain",
        "threshold",
        "test_cost_at_threshold",
        "test_cost_at_0.5",
        "test_cost_no_model",
        "shap_mean_abs",
    ]
    print(json.dumps({k: info[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
