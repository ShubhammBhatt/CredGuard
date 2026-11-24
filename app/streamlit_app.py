import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

MERCHANT_CATEGORIES = [
    "grocery",
    "dining",
    "retail",
    "utilities",
    "travel",
    "electronics",
]

PRESETS = {
    "Legit Grocery Swipe": {
        "transaction_id": "TXN_LEGIT_01",
        "amount": 35.0,
        "hour_of_day": 14,
        "merchant_category": "grocery",
        "is_foreign": 0,
        "is_online": 0,
        "distance_from_home_km": 4.5,
        "tx_count_last_24h": 1,
        "card_age_months": 48,
    },
    "Borderline Review Case": {
        "transaction_id": "TXN_REVIEW_02",
        "amount": 180.0,
        "hour_of_day": 19,
        "merchant_category": "travel",
        "is_foreign": 0,
        "is_online": 1,
        "distance_from_home_km": 85.0,
        "tx_count_last_24h": 3,
        "card_age_months": 18,
    },
    "High-Risk Foreign Electronics": {
        "transaction_id": "TXN_FRAUD_03",
        "amount": 520.0,
        "hour_of_day": 2,
        "merchant_category": "electronics",
        "is_foreign": 1,
        "is_online": 1,
        "distance_from_home_km": 240.5,
        "tx_count_last_24h": 6,
        "card_age_months": 8,
    },
}

st.set_page_config(
    page_title="CreditGuard AI – Fraud Risk Console",
    page_icon="🛡️",
    layout="wide",
)

# Initialize session state for form fields once
if "form_values" not in st.session_state:
    st.session_state.form_values = PRESETS["High-Risk Foreign Electronics"].copy()

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


def apply_preset(preset_name: str) -> None:
    """Populate session_state from a sample scenario and sync widget keys."""
    preset = PRESETS[preset_name].copy()
    st.session_state.form_values = preset
    for key, val in preset.items():
        st.session_state[f"input_{key}"] = val


# Ensure individual widget keys exist on first render
for k, v in st.session_state.form_values.items():
    widget_key = f"input_{k}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = v

st.title("CreditGuard AI – Fraud Risk Console")
st.caption(
    "Score transactions through FastAPI, inspect risk decisions (Approve / Flag for Review / Block), "
    "and monitor SQLite audit logs."
)

# Backend health check
backend_ok = False
health_info = {}
try:
    resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
    if resp.status_code == 200:
        backend_ok = True
        health_info = resp.json()
except requests.RequestException:
    backend_ok = False

if backend_ok:
    st.success(
        f"Backend Connected (`{API_BASE_URL}`) | Active Model: `{health_info.get('best_model', 'unknown')}`"
    )
else:
    st.error(
        f"Backend unreachable at `{API_BASE_URL}`. Start FastAPI with: `uvicorn src.api:app --reload`"
    )

# Fetch recent predictions once so KPIs and table stay consistent
recent_items = []
history_ok = False
try:
    hist_resp = requests.get(f"{API_BASE_URL}/predictions/recent", params={"limit": 100}, timeout=5)
    if hist_resp.status_code == 200:
        history_ok = True
        recent_items = hist_resp.json().get("items", [])
except requests.RequestException:
    history_ok = False

# Top KPI summary strip from logged predictions
total_logged = len(recent_items)
approved_count = sum(1 for row in recent_items if row.get("risk_decision") == "Approve")
review_count = sum(1 for row in recent_items if row.get("risk_decision") == "Flag for Review")
blocked_count = sum(1 for row in recent_items if row.get("risk_decision") == "Block")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Logged Transactions", total_logged)
k2.metric("Approved", approved_count)
k3.metric("Flagged for Review", review_count)
k4.metric("Blocked", blocked_count)

st.divider()

left_col, right_col = st.columns([1.05, 1.35], gap="large")

with left_col:
    with st.container(border=True):
        st.subheader("1. Score a Transaction")
        st.caption("Load a quick scenario or customize fields below:")

        p_cols = st.columns(3)
        for idx, preset_label in enumerate(PRESETS.keys()):
            with p_cols[idx]:
                st.button(
                    preset_label,
                    key=f"btn_{preset_label}",
                    on_click=apply_preset,
                    args=(preset_label,),
                    use_container_width=True,
                )

        with st.form("fraud_prediction_form"):
            transaction_id = st.text_input("Transaction ID", key="input_transaction_id")
            amount = st.number_input(
                "Amount", min_value=0.01, max_value=1_000_000.0, step=10.0, key="input_amount"
            )
            hour_of_day = st.slider(
                "Hour of Day (0–23)", min_value=0, max_value=23, key="input_hour_of_day"
            )
            merchant_category = st.selectbox(
                "Merchant Category", options=MERCHANT_CATEGORIES, key="input_merchant_category"
            )

            c1, c2 = st.columns(2)
            with c1:
                is_foreign = st.selectbox(
                    "Is Foreign Transaction?",
                    options=[0, 1],
                    format_func=lambda x: f"{x} ({'Yes' if x else 'No'})",
                    key="input_is_foreign",
                )
            with c2:
                is_online = st.selectbox(
                    "Is Online Transaction?",
                    options=[0, 1],
                    format_func=lambda x: f"{x} ({'Yes' if x else 'No'})",
                    key="input_is_online",
                )

            distance_from_home_km = st.number_input(
                "Distance from Home (km)",
                min_value=0.0,
                max_value=50_000.0,
                step=5.0,
                key="input_distance_from_home_km",
            )
            tx_count_last_24h = st.number_input(
                "Transactions in Last 24h",
                min_value=0,
                max_value=500,
                step=1,
                key="input_tx_count_last_24h",
            )
            card_age_months = st.number_input(
                "Card Age (Months)",
                min_value=0,
                max_value=600,
                step=1,
                key="input_card_age_months",
            )

            submitted = st.form_submit_button(
                "Evaluate Transaction Risk", type="primary", use_container_width=True
            )

        if submitted:
            payload = {
                "transaction_id": transaction_id.strip() or "TXN_MANUAL",
                "amount": float(amount),
                "hour_of_day": int(hour_of_day),
                "merchant_category": merchant_category,
                "is_foreign": int(is_foreign),
                "is_online": int(is_online),
                "distance_from_home_km": float(distance_from_home_km),
                "tx_count_last_24h": int(tx_count_last_24h),
                "card_age_months": int(card_age_months),
            }

            try:
                res = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
                if res.status_code == 200:
                    st.session_state.latest_result = res.json()
                    st.rerun()
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except requests.RequestException as exc:
                st.error(f"Failed to reach FastAPI backend: {exc}")

        if st.session_state.latest_result is not None:
            data = st.session_state.latest_result
            decision = data["risk_decision"]
            prob = float(data["fraud_probability"])

            st.markdown("### Decision Result")
            if decision == "Approve":
                st.success(
                    f"**APPROVE** — Fraud Probability: `{prob:.4f}` (Log ID #{data['record_id']})"
                )
            elif decision == "Flag for Review":
                st.warning(
                    f"**FLAG FOR REVIEW** — Fraud Probability: `{prob:.4f}` (Log ID #{data['record_id']})"
                )
            else:
                st.error(
                    f"**BLOCK** — Fraud Probability: `{prob:.4f}` (Log ID #{data['record_id']})"
                )

            st.progress(min(max(prob, 0.0), 1.0), text=f"Fraud Risk Score: {prob:.2%}")
            st.json(data)

with right_col:
    with st.container(border=True):
        st.subheader("2. Recent Logged Predictions (SQLite)")
        limit = st.slider("Display recent rows", min_value=5, max_value=100, value=15, step=5)

        if history_ok and recent_items:
            hist_df = pd.DataFrame(recent_items[:limit])
            st.dataframe(
                hist_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "fraud_probability": st.column_config.ProgressColumn(
                        "Fraud Prob",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    ),
                    "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                },
            )
        elif history_ok:
            st.info("No predictions logged yet. Submit a transaction on the left to log your first record.")
        else:
            st.info("Start the FastAPI backend to load recent SQLite prediction logs.")