"""Streamlit demo that calls the fraud API.

API_URL=http://localhost:8000 uv run --group demo streamlit run demo/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

PRESETS = {
    "Suspicious: night transfer emptying the account": dict(
        type="TRANSFER", amount=250000.0, oldbalance_orig=250000.0, oldbalance_dest=0.0, hour=3
    ),
    "Normal: small daytime cash-out": dict(
        type="CASH_OUT", amount=1800.0, oldbalance_orig=52000.0, oldbalance_dest=120000.0, hour=14
    ),
    "Payment (not scored)": dict(
        type="PAYMENT", amount=120.0, oldbalance_orig=900.0, oldbalance_dest=0.0, hour=12
    ),
}

st.set_page_config(page_title="E-wallet Fraud Check", page_icon="🛡️", layout="centered")
st.title("🛡️ E-wallet fraud check")
st.caption(
    "LightGBM trained on PaySim (synthetic mobile-money data), served by FastAPI, "
    "explained with SHAP."
)


@st.cache_data(ttl=60)
def model_info():
    return requests.get(f"{API_URL}/model-info", timeout=5).json()


try:
    info = model_info()
except requests.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it with `docker compose up`.")
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write(f"**{info['model_name']}** · v{info['version']}")
    st.metric("Test PR-AUC", f"{info['test']['pr_auc']:.3f}")
    st.metric(
        "PR-AUC without account-drain features", f"{info['test_ablation_no_drain']['pr_auc']:.3f}"
    )
    st.metric("Decision threshold", f"{info['threshold']:.3f}")
    st.caption("Threshold minimises missed-fraud loss + RM15 per alert on validation data.")

preset = st.selectbox("Start from an example", list(PRESETS))
p = PRESETS[preset]
with st.form("txn"):
    c1, c2 = st.columns(2)
    types = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    txn_type = c1.selectbox("Type", types, index=types.index(p["type"]))
    hour = c2.slider("Hour of day", 0, 23, p["hour"])
    amount = c1.number_input("Amount", min_value=0.0, value=p["amount"], step=100.0)
    bal_o = c2.number_input("Sender balance before", min_value=0.0, value=p["oldbalance_orig"])
    bal_d = c1.number_input("Receiver balance before", min_value=0.0, value=p["oldbalance_dest"])
    submitted = st.form_submit_button("Score transaction", type="primary")

if submitted:
    body = dict(
        type=txn_type, amount=amount, oldbalance_orig=bal_o, oldbalance_dest=bal_d, hour=hour
    )
    r = requests.post(f"{API_URL}/predict", json=body, timeout=10)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()
    res = r.json()
    flagged = res["decision"] == "flag"
    c1, c2 = st.columns(2)
    c1.metric("Fraud probability", f"{res['fraud_probability']:.2%}")
    c2.metric("Decision", "🚩 FLAG for review" if flagged else "✅ ALLOW")
    st.subheader("Top reasons")
    for reason in res["reasons"]:
        arrow = "🔺" if reason["direction"] == "raises risk" else "🔻"
        st.write(f"{arrow} {reason['text']}")
    with st.expander("Raw API response"):
        st.json(res)
