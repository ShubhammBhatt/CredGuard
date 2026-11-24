import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.database import fetch_recent_predictions, init_db, log_prediction

MODEL_PATH = Path("artifacts/model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=64, examples=["TXN_10001"])
    amount: float = Field(..., gt=0, examples=[520.0])
    hour_of_day: int = Field(..., ge=0, le=23, examples=[2])
    merchant_category: str = Field(..., min_length=1, examples=["electronics"])
    is_foreign: int = Field(..., ge=0, le=1, examples=[1])
    is_online: int = Field(..., ge=0, le=1, examples=[1])
    distance_from_home_km: float = Field(..., ge=0, examples=[240.5])
    tx_count_last_24h: int = Field(..., ge=0, examples=[6])
    card_age_months: int = Field(..., ge=0, examples=[8])


class PredictionResponse(BaseModel):
    record_id: int
    transaction_id: str
    fraud_probability: float
    risk_decision: Literal["Approve", "Flag for Review", "Block"]
    model_used: str
    thresholds: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite and load model/metadata once when the API server starts."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python src/train.py` before starting the API."
        )

    init_db()
    app.state.model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        app.state.metadata = json.load(f)

    yield


app = FastAPI(
    title="CreditGuard AI API",
    description="Real-time credit card fraud scoring, risk routing, and SQLite audit logging.",
    version="1.0.0",
    lifespan=lifespan,
)


def classify_risk(probability: float, thresholds: dict) -> Literal["Approve", "Flag for Review", "Block"]:
    """Map fraud probability (0.0 to 1.0) to a business risk decision."""
    approve_max = float(thresholds.get("approve_max", 0.30))
    flag_max = float(thresholds.get("flag_max", 0.75))

    if probability < approve_max:
        return "Approve"
    if probability <= flag_max:
        return "Flag for Review"
    return "Block"


@app.get("/health")
def health_check() -> dict:
    """Simple liveness endpoint to verify API and loaded model status."""
    return {
        "status": "ok",
        "best_model": app.state.metadata.get("best_model"),
        "features": app.state.metadata.get("features"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(payload: TransactionRequest) -> PredictionResponse:
    """Score a single transaction, determine risk decision, and log to SQLite."""
    metadata = app.state.metadata
    model = app.state.model

    allowed_categories = set(metadata.get("merchant_categories", []))
    if allowed_categories and payload.merchant_category not in allowed_categories:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid merchant_category '{payload.merchant_category}'. "
                f"Allowed values: {sorted(allowed_categories)}"
            ),
        )

    # Build a single-row DataFrame matching exact training feature column order
    feature_cols: list[str] = metadata["features"]
    row_dict = {
        "amount": payload.amount,
        "hour_of_day": payload.hour_of_day,
        "merchant_category": payload.merchant_category,
        "is_foreign": payload.is_foreign,
        "is_online": payload.is_online,
        "distance_from_home_km": payload.distance_from_home_km,
        "tx_count_last_24h": payload.tx_count_last_24h,
        "card_age_months": payload.card_age_months,
    }
    input_df = pd.DataFrame([[row_dict[col] for col in feature_cols]], columns=feature_cols)

    # Column 1 of predict_proba is the probability of class 1 (fraud)
    fraud_prob = float(model.predict_proba(input_df)[0, 1])
    thresholds = metadata.get("risk_thresholds", {"approve_max": 0.30, "flag_max": 0.75})
    decision = classify_risk(fraud_prob, thresholds)

    record_id = log_prediction(
        {
            "transaction_id": payload.transaction_id,
            "amount": payload.amount,
            "hour_of_day": payload.hour_of_day,
            "merchant_category": payload.merchant_category,
            "is_foreign": payload.is_foreign,
            "is_online": payload.is_online,
            "distance_from_home_km": payload.distance_from_home_km,
            "tx_count_last_24h": payload.tx_count_last_24h,
            "card_age_months": payload.card_age_months,
            "fraud_probability": fraud_prob,
            "risk_decision": decision,
        }
    )

    return PredictionResponse(
        record_id=record_id,
        transaction_id=payload.transaction_id,
        fraud_probability=round(fraud_prob, 4),
        risk_decision=decision,
        model_used=metadata.get("best_model", "unknown"),
        thresholds={
            "approve_max": float(thresholds.get("approve_max", 0.30)),
            "flag_max": float(thresholds.get("flag_max", 0.75)),
        },
    )


@app.get("/predictions/recent")
def recent_predictions(limit: int = Query(default=25, ge=1, le=200)) -> dict:
    """Return the most recently logged predictions from SQLite."""
    items = fetch_recent_predictions(limit=limit)
    return {"count": len(items), "items": items}