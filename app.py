import pickle
import pandas as pd
import streamlit as st

# 1. Page configuration and title
st.set_page_config(page_title="Credit Card Prediction App", page_icon="💳")
st.title("💳 Credit Card / Property Approval Prediction")
st.write("Enter the applicant's details below to get the prediction:")

# 2. Load the saved model
with open("logistic.pkl", "rb") as f:
    model = pickle.load(f)

# 3. Create input form with exact 7 column names
col1, col2 = st.columns(2)

with col1:
    gender = st.number_input("Gender (CODE_GENDER: 0 or 1)", value=0)
    income = st.number_input("Total Annual Income (AMT_INCOME_TOTAL)", value=100000.0)
    education = st.number_input("Education Type (NAME_EDUCATION_TYPE)", value=1)
    family_status = st.number_input("Family Status (NAME_FAMILY_STATUS)", value=3)

with col2:
    days_birth = st.number_input("Days Birth (DAYS_BIRTH)", value=-15000)
    days_employed = st.number_input("Days Employed (DAYS_EMPLOYED)", value=-2000)
    fam_members = st.number_input("Family Members (CNT_FAM_MEMBERS)", value=2)

# 4. Prediction button
if st.button("Predict Result"):
    # Create DataFrame with exact feature names
    input_df = pd.DataFrame(
        [[gender, income, education, family_status, days_birth, days_employed, fam_members]],
        columns=[
            "CODE_GENDER",
            "AMT_INCOME_TOTAL",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_FAM_MEMBERS",
        ],
    )

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Display result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ **Yes (1)** — Probability: {probability[1] * 100:.2f}%")
    else:
        st.error(f"❌ **No (0)** — Probability: {probability[0] * 100:.2f}%")
