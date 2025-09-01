import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("fraud_detection_pipeline.pkl")   # make sure extension matches saved file

st.title("💳 Fraud Detection App")

# Collect inputs
amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, format="%.2f")
newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, format="%.2f")
oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, format="%.2f")
newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, format="%.2f")
balanceDiffOrig = st.number_input("Balance Diff Origin", format="%.2f")
balancedDiffDest = st.number_input("Balance Diff Destination", format="%.2f")

# Transaction type (the one that gave "PAYMENT" issue)
type_options = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
type_selected = st.selectbox("Transaction Type", type_options)

# Predict button
if st.button("Predict Fraud"):
    # Prepare input as dataframe
    input_data = pd.DataFrame([{
        "type": type_selected,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": balanceDiffOrig,
        "balancedDiffDest": balancedDiffDest
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # fraud probability
    
    if prediction == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Confidence: {proba:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {proba:.2f})")
