# 04 Fraud Detection API

Transaction fraud model served as a real API with explanations. Project 05 will deploy this to AWS, so keep it container-ready.
Tracks shown: ML, SWE, fintech.

## Data
PaySim (synthetic mobile money transactions, Kaggle). Fits e-wallet context. If unavailable, use the ULB credit card fraud dataset and say so.
Don't commit the raw dataset; add a download script.

## Modelling
1. EDA notebook: fraud rate, patterns by transaction type, amount, balance anomalies. Watch for leakage columns in PaySim and explain what you dropped.
2. Feature engineering in a reusable `src/features.py` (same code in training and serving).
3. Models: logistic regression baseline vs LightGBM/XGBoost. Handle imbalance (class weights, compare with SMOTE).
4. Metrics: PR-AUC, recall at fixed precision, confusion matrix. Not accuracy.
5. Threshold tuning with a cost matrix (missed fraud cost vs false alarm cost) and show the business trade-off.
6. SHAP: global importance + per-transaction explanation.
7. Track experiments with MLflow (local).

## Serving
- FastAPI: `POST /predict` (single), `POST /predict/batch`, `GET /health`, `GET /model-info`. Response includes fraud probability, decision, top 3 SHAP reasons in plain English.
- Pydantic validation, pytest tests for API + features.
- Dockerfile (slim, non-root), docker-compose for local run.
- Small Streamlit or HTML demo page that calls the API.

## README
Business framing (what a fintech fraud team needs), model card (data, metrics, limitations, fairness note), API examples with curl.
