"""Feature engineering shared by training and serving.

Only information available *before* a transaction is authorised is used:
type, amount, hour, and the pre-transaction balances. Post-transaction balances
(newbalanceOrig/newbalanceDest), the rule flag (isFlaggedFraud) and account IDs are dropped.
See notebooks/01_eda.ipynb for why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fraud only ever occurs in these two types in PaySim. Other types are short-circuited
# to "allow" by the API instead of being scored.
SCORED_TYPES = ("TRANSFER", "CASH_OUT")

RAW_INPUT_COLS = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
DROPPED_COLS = [
    "newbalanceOrig",
    "newbalanceDest",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
]

FEATURE_COLS = [
    "is_transfer",
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_balance",
    "drains_account",
    "orig_balance_zero",
    "dest_balance_zero",
    "hour",
    "is_night",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw PaySim-style rows into model features. Pure function, no fitted state."""
    missing = set(RAW_INPUT_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")

    amount = df["amount"].astype(float)
    bal_orig = df["oldbalanceOrg"].astype(float)
    bal_dest = df["oldbalanceDest"].astype(float)
    hour = (df["step"].astype(int) % 24).astype(int)

    out = pd.DataFrame(index=df.index)
    out["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    out["amount"] = amount
    out["log_amount"] = np.log1p(amount)
    out["oldbalanceOrg"] = bal_orig
    out["oldbalanceDest"] = bal_dest
    # Capped so a zero balance does not produce inf; 1.0 means "sends everything".
    out["amount_to_balance"] = np.clip(amount / np.maximum(bal_orig, 1.0), 0, 100)
    out["drains_account"] = ((bal_orig > 0) & (np.abs(amount - bal_orig) < 0.01)).astype(int)
    out["orig_balance_zero"] = (bal_orig == 0).astype(int)
    out["dest_balance_zero"] = (bal_dest == 0).astype(int)
    out["hour"] = hour
    out["is_night"] = (hour < 7).astype(int)
    return out[FEATURE_COLS]


def filter_scored(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the transaction types the model is trained on."""
    return df[df["type"].isin(SCORED_TYPES)]


def describe(feature: str, row: dict) -> str:
    """Plain-English description of one feature's value for one raw transaction."""
    v = build_features(pd.DataFrame([row])).iloc[0][feature]
    money = "{:,.2f}".format
    texts = {
        "is_transfer": f"transaction type is {row['type']}",
        "amount": f"amount is {money(row['amount'])}",
        "log_amount": f"amount is {money(row['amount'])}",
        "oldbalanceOrg": f"sender balance before the transaction is {money(v)}",
        "oldbalanceDest": f"receiver balance before the transaction is {money(v)}",
        "amount_to_balance": f"amount is {v:.0%} of the sender's balance",
        "drains_account": "transaction empties the sender's account"
        if v
        else "sender keeps part of their balance",
        "orig_balance_zero": "sender balance was zero" if v else "sender had a non-zero balance",
        "dest_balance_zero": "receiver account had a zero balance"
        if v
        else "receiver account already held funds",
        "hour": f"made at {int(v):02d}:00",
        "is_night": "made at night (00:00-06:59)" if v else "made during daytime hours",
    }
    return texts[feature]
