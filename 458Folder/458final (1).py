
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn # This is needed for the pickle file to load!

# Load the trained model
import os

# Get the path to the directory where this script lives
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, "458final.pkl")

with open(file_path, "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #e6f7ff; padding: 10px; color: #007bff;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant's Details")

# --- Input fields for features ---

# Numerical inputs
fico_score = st.slider("FICO Score", min_value=500, max_value=850, value=650, step=1)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=1000, max_value=100000, value=5000, step=100)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=0, max_value=30000, value=1500, step=100)
ever_bankrupt = st.radio("Ever Bankrupt or Foreclosed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Categorical inputs
reason_options = ['cover_an_unexpected_cost', 'credit_card_refinancing', 'debt_conslidation', 'home_improvement', 'major_purchase', 'other']
reason = st.selectbox("Reason for Loan", reason_options)

fico_score_group_options = ['excellent', 'fair', 'good', 'poor', 'very_good']
fico_score_group = st.selectbox("FICO Score Group", fico_score_group_options)

employment_status_options = ['full_time', 'part_time', 'unemployed']
employment_status = st.selectbox("Employment Status", employment_status_options)

employment_sector_options = ['communication_services', 'consumer_discretionary', 'consumer_staples', 'energy', 'financials', 'health_care', 'industrials', 'information_technology', 'materials', 'real_estate', 'utilities']
employment_sector = st.selectbox("Employment Sector", employment_sector_options)

lender_options = ['A', 'B', 'C']
lender = st.selectbox("Preferred Lender", lender_options)

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt],
    "Reason": [reason],
    "Fico_Score_group": [fico_score_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction (similar to preprocessing in notebook) ---
# Identify categorical and numerical columns for Streamlit input processing
categorical_cols_input = ['Reason', 'Fico_Score_group', 'Employment_Status', 'Employment_Sector', 'Lender']
numerical_cols_input = ['FICO_score', 'Monthly_Gross_Income', 'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose']

# One-hot encode categorical features
input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols_input)

# Ensure all expected columns from model training are present, fill missing with 0
# and match order
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match the training data
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Predict Loan Approval"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]
    prediction_proba = model.predict_proba(input_data_encoded)[0]

    # Display result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **Loan Approved!** ✅")
        st.write(f"Probability of Approval: {prediction_proba[1]:.2f}")
    else:
        st.error(f"The model predicts: **Loan Denied.** ❌")
        st.write(f"Probability of Approval: {prediction_proba[1]:.2f}")
        st.write(f"Probability of Denial: {prediction_proba[0]:.2f}")
