from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TxnType = Literal["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]


class Transaction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "transaction_id": "txn-001",
                "type": "TRANSFER",
                "amount": 250000.0,
                "oldbalance_orig": 250000.0,
                "oldbalance_dest": 0.0,
                "hour": 3,
            }
        },
    )

    transaction_id: str | None = Field(None, max_length=64)
    type: TxnType
    amount: float = Field(..., ge=0, le=1e9, description="Transaction amount")
    oldbalance_orig: float = Field(..., ge=0, description="Sender balance before the transaction")
    oldbalance_dest: float = Field(..., ge=0, description="Receiver balance before the transaction")
    hour: int = Field(..., ge=0, le=23, description="Hour of day of the transaction (0-23)")

    def to_raw(self) -> dict:
        """Map API field names to the PaySim column names used by fraud.features."""
        return {
            "step": self.hour,
            "type": self.type,
            "amount": self.amount,
            "oldbalanceOrg": self.oldbalance_orig,
            "oldbalanceDest": self.oldbalance_dest,
        }


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class Reason(BaseModel):
    feature: str
    impact: float = Field(..., description="SHAP value in log-odds; positive raises fraud risk")
    direction: Literal["raises risk", "lowers risk"]
    text: str


class Prediction(BaseModel):
    transaction_id: str | None
    fraud_probability: float
    decision: Literal["flag", "allow"]
    threshold: float
    scored: bool = Field(..., description="False if the type is never fraudulent and was skipped")
    reasons: list[Reason]
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    flagged: int


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
