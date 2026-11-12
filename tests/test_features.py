import pandas as pd
import pytest

from fraud.features import FEATURE_COLS, build_features, describe, filter_scored


def row(**kw):
    base = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "oldbalanceOrg": 1000.0,
        "oldbalanceDest": 0.0,
    }
    base.update(kw)
    return base


def test_columns_and_order():
    X = build_features(pd.DataFrame([row()]))
    assert list(X.columns) == FEATURE_COLS


def test_drain_and_zero_flags():
    f = build_features(pd.DataFrame([row()])).iloc[0]
    assert f.drains_account == 1
    assert f.dest_balance_zero == 1
    assert f.orig_balance_zero == 0
    assert f.amount_to_balance == pytest.approx(1.0)


def test_partial_send_not_drain():
    f = build_features(pd.DataFrame([row(amount=10.0, oldbalanceDest=5.0)])).iloc[0]
    assert f.drains_account == 0
    assert f.dest_balance_zero == 0


def test_zero_balance_no_inf():
    f = build_features(pd.DataFrame([row(oldbalanceOrg=0.0)])).iloc[0]
    assert f.amount_to_balance == 100  # capped
    assert f.drains_account == 0


@pytest.mark.parametrize("step,hour,night", [(1, 1, 1), (24, 0, 1), (31, 7, 0), (743, 23, 0)])
def test_hour(step, hour, night):
    f = build_features(pd.DataFrame([row(step=step)])).iloc[0]
    assert f.hour == hour and f.is_night == night


def test_missing_column_raises():
    with pytest.raises(ValueError, match="oldbalanceDest"):
        build_features(
            pd.DataFrame([{"step": 1, "type": "TRANSFER", "amount": 1.0, "oldbalanceOrg": 1.0}])
        )


def test_filter_scored():
    df = pd.DataFrame([row(type=t) for t in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]])
    assert set(filter_scored(df)["type"]) == {"TRANSFER", "CASH_OUT"}


def test_describe_is_plain_english():
    assert describe("drains_account", row()) == "transaction empties the sender's account"
    assert describe("amount", row(amount=1234.5)) == "amount is 1,234.50"
    assert describe("hour", row(step=3)) == "made at 03:00"
