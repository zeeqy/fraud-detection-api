import pytest
from fastapi.testclient import TestClient

from api.main import app

FRAUDY = {
    "transaction_id": "t1",
    "type": "TRANSFER",
    "amount": 250000.0,
    "oldbalance_orig": 250000.0,
    "oldbalance_dest": 0.0,
    "hour": 3,
}
NORMAL = {
    "transaction_id": "t2",
    "type": "CASH_OUT",
    "amount": 1800.0,
    "oldbalance_orig": 52000.0,
    "oldbalance_dest": 120000.0,
    "hour": 14,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert 0 < info["threshold"] < 1
    assert "drains_account" in info["features"]
    assert info["test"]["pr_auc"] > 0.9


def test_predict_flags_account_drain(client):
    body = client.post("/predict", json=FRAUDY).json()
    assert body["decision"] == "flag"
    assert body["fraud_probability"] > 0.5
    assert body["scored"] is True
    assert len(body["reasons"]) == 3
    assert any("empties" in r["text"] for r in body["reasons"])


def test_predict_allows_normal(client):
    body = client.post("/predict", json=NORMAL).json()
    assert body["decision"] == "allow"
    assert body["fraud_probability"] < body["threshold"]


def test_unscored_type_short_circuits(client):
    body = client.post("/predict", json={**NORMAL, "type": "PAYMENT"}).json()
    assert body["scored"] is False
    assert body["decision"] == "allow"
    assert body["fraud_probability"] == 0.0


def test_batch_keeps_order(client):
    txns = [FRAUDY, {**NORMAL, "type": "DEBIT", "transaction_id": "t3"}, NORMAL]
    body = client.post("/predict/batch", json={"transactions": txns}).json()
    assert [p["transaction_id"] for p in body["predictions"]] == ["t1", "t3", "t2"]
    assert [p["decision"] for p in body["predictions"]] == ["flag", "allow", "allow"]
    assert [p["scored"] for p in body["predictions"]] == [True, False, True]
    assert body["flagged"] == 1


@pytest.mark.parametrize(
    "patch",
    [{"amount": -5}, {"hour": 24}, {"type": "WIRE"}, {"unexpected": 1}, {"oldbalance_orig": "x"}],
)
def test_validation_errors(client, patch):
    assert client.post("/predict", json={**FRAUDY, **patch}).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
