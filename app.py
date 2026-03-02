from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load trained pipeline
model = joblib.load("models/credit_risk_model.pkl")


# Input Schema (only key business fields required)
class CreditInput(BaseModel):
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    FLAG_MOBIL: int
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    FLAG_EMAIL: int
    OCCUPATION_TYPE: str


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API Running"}


@app.post("/predict")
def predict(data: CreditInput):
    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])

        # Get expected columns from trained pipeline
        expected_columns = model.named_steps["preprocessor"].feature_names_in_

        # Add missing columns as NaN
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # Ensure correct order
        input_df = input_df[expected_columns]

        # Predict
        probability = model.predict_proba(input_df)[0][1]
        prediction = int(probability > 0.5)

        return {
            "default_probability": float(probability),
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}