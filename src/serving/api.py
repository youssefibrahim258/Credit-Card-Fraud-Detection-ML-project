"""
FastAPI application for Credit Card Fraud Detection.

This module exposes a REST endpoint that receives transaction data
and returns fraud probability and prediction using a trained ML model.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.serving.inference import predict


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0"
)


class TransactionInput(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

# Endpoint
@app.post("/predict")
def predict_fraud(data: TransactionInput):
    result = predict(data.dict())

    return {
        "fraud_probability": result["probability"],
        "fraud_prediction": result["prediction"]
    }
