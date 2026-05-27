import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import joblib

df = pd.read_csv("../data/loan_data.csv")

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier()

model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:,1]

score = roc_auc_score(y_test, preds)

print(score)

joblib.dump(model, "../models/xgboost_model.pkl")