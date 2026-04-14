import streamlit as st
import pandas as pd
import joblib

# Load model and files
model = joblib.load("KNN_HEART.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Title
st.title("❤️ Heart Disease Prediction App")

st.markdown("Please enter the following details to predict the risk of heart disease:")

# User Inputs
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.slider("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Predict Button
if st.button("Predict"):

    # Step 1: Raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingECG": resting_ecg,
        "ExerciseAngina": exercise_angina,
        "ST_Slope": st_slope
    }

    # Step 2: Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Step 3: Apply encoding (VERY IMPORTANT)
    input_df = pd.get_dummies(input_df)

    # Step 4: Match training columns (MOST IMPORTANT 🔥)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Step 5: Apply scaler (on full dataframe)
    input_scaled = scaler.transform(input_df)

    # Step 6: Prediction
    prediction = model.predict(input_scaled)[0]

    # Step 7: Output
    if prediction == 1:
        st.error("⚠️ High risk of heart disease. Please consult a doctor.")
    else:
        st.success("✅ Low risk of heart disease. Keep up the healthy lifestyle!")

    # Optional: Probability 🔥
    try:
        prob = model.predict_proba(input_scaled)[0][1]
        st.write(f"Risk Probability: {round(prob*100, 2)}%")
    except:
        pass