import joblib
import numpy as np

model = joblib.load("models/xgboost_model.pkl")

sample = np.array([[50000,10000,700,5,0.2]])

prediction = model.predict(sample)

print(prediction)