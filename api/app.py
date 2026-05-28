from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = joblib.load("models/xgboost_model.pkl")

@app.route("/")
def home():
    return "Loan Default Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "risk_score": float(probability)
    })

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)