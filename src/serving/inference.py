"""
Inference module for Credit Card Fraud Detection.

This module:
- Loads trained model from MLflow Model Registry
- Applies same preprocessing used during training
- Ensures feature order consistency
- Returns fraud probability and prediction
"""
import os 
import pandas as pd
import joblib

from src.features.preprocessing import apply_log_transform



BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(
    os.path.join(MODELS_DIR, "model.pkl"))



FEATURE_COLUMNS = [
    "Time", "Amount",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8",
    "V9", "V10", "V11", "V12", "V13", "V14", "V15",
    "V16", "V17", "V18", "V19", "V20", "V21", "V22",
    "V23", "V24", "V25", "V26", "V27", "V28"
]



def _transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Same preprocessing as training
    df = apply_log_transform(df)

    # Ensure column order
    df = df[FEATURE_COLUMNS]

    return df



def predict(input_dict: dict) -> dict:
    """
    Returns:
        {
            "probability": float,
            "prediction": int
        }
    """

    df = pd.DataFrame([input_dict])
    df = _transform(df)

    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= 0.5)

    return {
        "probability": prob,
        "prediction": pred
    }
