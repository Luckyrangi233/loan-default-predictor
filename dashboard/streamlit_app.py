import streamlit as st
import joblib
import numpy as np

st.title("Loan Default Prediction Dashboard")

model = joblib.load("models/xgboost_model.pkl")

income = st.number_input("Income", value=50000)
loan_amount = st.number_input("Loan Amount", value=10000)
credit_score = st.number_input("Credit Score", value=700)
employment_years = st.number_input("Employment Years", value=5)

dti_ratio = loan_amount / income

if st.button("Predict"):

    features = np.array([
        [income, loan_amount, credit_score, employment_years, dti_ratio]
    ])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1]

    st.write("Prediction:", prediction)
    st.write("Risk Score:", probability)