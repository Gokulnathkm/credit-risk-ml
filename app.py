import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

MODEL_PATH = "models/credit_risk_model.pkl"

# ==============================
# App Initialization
# ==============================

app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0.0"
)

# ==============================
# Load Model Safely
# ==============================

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Model file not found. Please run: python src/train.py first."
    )

saved_obj = joblib.load(MODEL_PATH)
model = saved_obj["model"]
trained_at = saved_obj["trained_at"]
roc_auc = saved_obj["roc_auc"]
feature_count = saved_obj["feature_count"]

# ==============================
# Input Schema
# ==============================

class CreditInput(BaseModel):
    data: Dict[str, Any]


# ==============================
# Routes
# ==============================

@app.get("/")
def root():
    return {
        "message": "Credit Risk API running",
        "model_trained_at": trained_at,
        "roc_auc": roc_auc,
        "feature_count": feature_count
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now()}

@app.post("/predict")
def predict(payload: CreditInput):

    try:
        # Get expected training columns from pipeline
        expected_columns = model.named_steps["preprocessor"].feature_names_in_

        # Build full row with all expected columns
        full_input = {}

        for col in expected_columns:
            full_input[col] = payload.data.get(col, None)

        input_df = pd.DataFrame([full_input])

        probability = model.predict_proba(input_df)[0][1]

        if probability < 0.3:
            risk = "Low Risk"
        elif probability < 0.7:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        return {
            "default_probability": round(float(probability), 4),
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    