import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../models/churn_model.pkl")

st.title("Customer Churn Prediction System")

st.write("""
This application predicts whether a telecom customer
is likely to churn or stay based on customer details.
""")

# Inputs
tenure = st.slider("Tenure Months", 0, 72, 12)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1500.0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.selectbox(
    "Dependents",
    ["No", "Yes"]
)

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

if st.button("Predict Churn"):

    input_data = pd.DataFrame({
        'Tenure Months': [tenure],
        'Monthly Charges': [monthly_charges],
        'Total Charges': [total_charges],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Senior Citizen_Yes': [1 if senior == "Yes" else 0],
        'Partner_Yes': [1 if partner == "Yes" else 0],
        'Dependents_Yes': [1 if dependents == "Yes" else 0],
        'Phone Service_Yes': [1 if phone_service == "Yes" else 0],
        'Internet Service_Fiber optic': [1 if internet == "Fiber optic" else 0],
        'Internet Service_No': [1 if internet == "No" else 0],
        'Contract_One year': [1 if contract == "One year" else 0],
        'Contract_Two year': [1 if contract == "Two year" else 0],
        'Payment Method_Credit card (automatic)': [1 if payment == "Credit card (automatic)" else 0],
        'Payment Method_Electronic check': [1 if payment == "Electronic check" else 0],
        'Payment Method_Mailed check': [1 if payment == "Mailed check" else 0]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    stay_prob = probability[0][0] * 100
    churn_prob = probability[0][1] * 100

    st.write(f"### Stay Probability: {stay_prob:.2f}%")
    st.write(f"### Churn Probability: {churn_prob:.2f}%")

    if prediction[0] == 1:
        st.error("⚠ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")