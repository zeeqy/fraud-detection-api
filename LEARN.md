# LEARN: Fraud Detection API

## 1. What it does
It trains a model that flags fraudulent e-wallet transfers using PaySim, a 6.36M-row synthetic mobile-money dataset.
The model runs behind a FastAPI service that returns a fraud probability, a flag/allow decision and the top 3 reasons in plain English.
The decision threshold is set by a business cost model (missed fraud vs analyst review), not by the default 0.5.

## 2. Decisions
| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Data source | Kaggle, fall back to HF mirror | Kaggle only | Kaggle needs an API token; the mirror's row and fraud counts match exactly, so anyone can reproduce it |
| Rows used | TRANSFER + CASH_OUT only | All 5 types | Other types have zero fraud; the API short-circuits them to "allow" |
| Dropped columns | `newbalance*`, `isFlaggedFraud`, IDs | Keep everything | Post-transaction balances aren't known before authorisation and leak the label; the flag is a rule output |
| Split | Stratified random 70/15/15 + time-split check | Time split only | PaySim legit volume collapses after day 17 (simulator artifact), so a time split mostly measures that. The time check (0.9998) confirms the result holds |
| Model served | LightGBM unweighted | SMOTE / class-weighted LightGBM, logreg | SMOTE gain was 0.0004 PR-AUC (noise) and it shifts probabilities. Class weights hurt (0.89). Trees give exact, fast SHAP |
| Early stopping metric | Log-loss | PR-AUC | PR-AUC saturated after 3 trees, giving a degenerate model whose scores never passed 0.5 |
| SMOTE ratio | 1:10 | 1:1 | Enough minority signal with ~10x less synthetic data |
| Threshold | Min business cost on validation; ties go to more fraud caught | 0.5 default | 0.5 is arbitrary; cost is what the business cares about |
| Explanations | TreeSHAP top 3, mapped to sentences | Feature importance only | Analysts need per-transaction reasons, not global ones |
| Serving format | joblib bundle (model + threshold + metadata) | MLflow model registry | One file, so the Docker image needs no MLflow server. MLflow is used for experiment tracking |
| Docker | Multi-stage uv build to slim runtime, non-root, healthcheck | Single stage | Smaller image and less attack surface; ready for AWS in project 05 |

## 3. Concepts to understand
- **Class imbalance:** 0.3% fraud means "always legit" scores 99.7% accuracy. Fix the metric (PR-AUC) and optionally the training (weights, SMOTE). `src/fraud/train.py`
- **PR-AUC (average precision):** area under the precision-recall curve. It ignores true negatives, so it isn't inflated by millions of easy legit rows. Random baseline = the fraud rate. `src/fraud/metrics.py`
- **Recall at fixed precision:** "if we accept that 95% of alerts must be real fraud, what share of fraud do we catch?" It maps directly to analyst workload. `src/fraud/metrics.py`
- **Data leakage:** using information unavailable at prediction time. Here, post-transaction balances. It looks great offline and fails in production. `notebooks/01_eda.ipynb`
- **SMOTE:** creates synthetic minority rows by interpolating between fraud neighbours. Train set only, never on validation or test. `src/fraud/train.py`
- **Class weights / scale_pos_weight:** make each fraud row count more in the loss. Both this and SMOTE shift probabilities upward. `src/fraud/train.py`
- **Cost-based threshold:** total cost = sum of missed fraud amounts + RM15 × alerts. Sweep thresholds and pick the minimum. `src/fraud/metrics.py`
- **SHAP (TreeSHAP):** splits one prediction into additive per-feature contributions in log-odds. Positive values push towards fraud. `src/fraud/explain.py`
- **Ablation:** retrain without a feature to measure how much the result depends on it. Here PR-AUC drops from 0.997 to 0.837. `src/fraud/train.py`
- **Train/serve skew:** different feature code in training and serving. Avoided by one `build_features()` used by both. `src/fraud/features.py`, `api/model_loader.py`
- **Pydantic validation:** request schema rejects bad input with a 422 before it reaches the model. `api/schemas.py`
- **Early stopping:** stop adding trees when validation loss stops improving. The metric choice matters (see decisions). `src/fraud/train.py`

## 4. Interview questions
1. **Why not accuracy?** With 0.3% fraud, predicting "legit" always is 99.7% accurate and catches nothing. I used PR-AUC, recall at 90/95% precision, and the confusion matrix at the chosen threshold.
2. **Your PR-AUC is 0.997. Isn't that suspicious?** Yes. It comes from PaySim's simulator: 97.6% of fraud empties the account and legit transactions almost never do. I ran an ablation without that feature and PR-AUC drops to 0.837. That's the number I'd expect to be closer to reality, and I say so in the README.
3. **What leakage did you find?** The post-transaction balances. 90% of legit rows don't reconcile (old − amount ≠ new) versus 0.5% of fraud, so the "balance error" feature many Kaggle notebooks use is almost a label. A pre-authorisation API can't see post-transaction balances anyway, so I dropped them. I also dropped `isFlaggedFraud`, which is a rule output that fires on 16 rows.
4. **Did SMOTE help?** No measurable gain: +0.0004 PR-AUC on validation, which is noise. Class weights actually hurt LightGBM (0.89). I served the unweighted model because it keeps probabilities closer to the real base rate.
5. **How did you choose the threshold?** On validation, I swept thresholds and minimised missed-fraud amount plus RM15 per alert. For the served model it barely matters. For the ablation model, moving from 0.5 to the cost-optimal threshold cuts test cost from RM75M to RM5.3M, but alerts go from ~1k to ~31k, so in practice analyst capacity would cap it.
6. **Why LightGBM if logistic regression is just as good?** On this data, logistic regression is within 0.001 PR-AUC, and I'd say that openly. Remove the account-drain features, though, and LightGBM scores 0.84 PR-AUC against logistic regression's 0.29, so it handles the harder, non-linear patterns much better. TreeSHAP also gives exact explanations quickly. For a regulated bank, logistic regression would be a defensible choice too.
7. **How does the API explain a decision?** TreeSHAP computes each feature's contribution to that one prediction. I take the top 3 by absolute value and map each to a sentence like "transaction empties the sender's account (raises risk)".
8. **How do you avoid training/serving mismatch?** One `build_features()` function in `src/fraud/features.py` is imported by both the training script and the API. Tests cover it.
9. **What happens with a PAYMENT transaction?** It's never fraudulent in the training data, so the API returns `scored: false`, probability 0 and "allow", with that reason. It doesn't push out-of-distribution data through the model.
10. **How would you take this to production?** Use real data with customer history and velocity features, calibrate probabilities, optimise under an alert budget, add drift and false-positive monitoring per customer segment, version models in a registry, and deploy the container to AWS (planned in project 05).

## 5. Weaknesses and how to say them
- **Synthetic data.** "It's PaySim, a simulator, so the fraud pattern is far cleaner than reality. I quantified that with an ablation: without the account-drain signal, PR-AUC drops from 0.997 to 0.84."
- **No behavioural features.** "Real fraud models lean on customer history, device and velocity. PaySim doesn't support that, so this project is about the pipeline and the decision process, not state-of-the-art features."
- **Cost model is simple.** "Missed fraud = amount and review = flat RM15, and PaySim units are treated as RM. It ignores analyst capacity and chargebacks. I'd add an alert budget next."
- **No fairness measurement.** "PaySim has no demographics, so I couldn't measure fairness. I flagged which features could hurt certain users and recommend per-segment monitoring."
- **Docker image not built locally.** "Docker Desktop wasn't healthy on my laptop, so CI builds and smoke-tests the image on GitHub Actions."

## 6. Resume bullets
- **[ML]** Built a fraud detection model on 6.36M PaySim transactions comparing logistic regression and LightGBM (class weights, SMOTE). Reached 0.997 PR-AUC and used an ablation to show performance falls to 0.84 without the simulator's leaky signal.
- **[SWE]** Shipped the model as a FastAPI service (single and batch scoring, health and model-info endpoints) with Pydantic validation, 26 pytest tests, a slim non-root Docker image and GitHub Actions CI. Scores with SHAP reasons in ~75 ms.
- **[Data]** Tuned the fraud decision threshold with a business cost model (missed fraud vs RM15 review cost). On the realistic model, cost-based thresholding cut test-set losses from RM75M to RM5.3M compared with the 0.5 default.
- **[AI]** Added per-transaction explainability by converting TreeSHAP contributions into plain-English top-3 reasons returned with every API decision, so analysts can act on and defend each flag.
- **[Cloud]** Containerised the service with a multi-stage uv build, healthcheck and docker-compose, ready for AWS deployment.
