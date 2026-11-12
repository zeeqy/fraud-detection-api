"""Fraud scoring API.

uv run uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.model_loader import FraudModel
from api.schemas import BatchRequest, BatchResponse, Health, Prediction, Transaction


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = FraudModel()
    except FileNotFoundError:
        app.state.model = None
    yield


app = FastAPI(
    title="PaySim Fraud Detection API",
    version="1.0.0",
    description="Scores e-wallet transactions for fraud and explains each decision with SHAP.",
    lifespan=lifespan,
)
# The demo page may be served from another origin. Restrict allow_origins in a real deployment.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"])


def get_model(request: Request) -> FraudModel:
    model = request.app.state.model
    if model is None:
        raise HTTPException(503, "Model not loaded. Run `uv run python -m fraud.train` first.")
    return model


def _predict(model: FraudModel, txns: list[Transaction]) -> list[Prediction]:
    out = model.predict([t.to_raw() for t in txns])
    return [
        Prediction(
            transaction_id=t.transaction_id,
            threshold=model.threshold,
            model_version=model.version,
            **r,
        )
        for t, r in zip(txns, out, strict=True)
    ]


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    loaded = request.app.state.model is not None
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.get("/model-info")
def model_info(request: Request) -> dict:
    return get_model(request).public_info()


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction, request: Request) -> Prediction:
    return _predict(get_model(request), [txn])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    preds = _predict(get_model(request), batch.transactions)
    return BatchResponse(predictions=preds, flagged=sum(p.decision == "flag" for p in preds))
