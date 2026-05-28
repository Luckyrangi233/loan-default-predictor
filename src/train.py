import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load small sample
df = pd.read_csv(
    "data/loan_data.csv",
    nrows=5000,
    low_memory=False
)

print(df.head())

# Keep only numeric columns
df = df.select_dtypes(include=["number"])

# Fill missing values
df = df.fillna(0)

# Create dummy target
df["loan_status"] = (df.iloc[:, 0] > 0).astype(int)

# Features and target
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

score = accuracy_score(y_test, preds)

print("Accuracy:", score)

# Create models folder if missing
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/xgboost_model.pkl")

print("Model saved successfully!")