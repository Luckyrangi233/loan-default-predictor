from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/xgboost_model.pkl")

@app.get("/")
def home():
    return {"message": "FastAPI Loan Prediction API Running"}

@app.post("/predict")
def predict(data: dict):

    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "risk_score": float(probability)
    }