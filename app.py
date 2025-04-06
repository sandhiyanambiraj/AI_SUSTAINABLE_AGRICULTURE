# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model_utils import load_and_preprocess_data, predict

# Load models and encoder
yield_model = joblib.load("yield_model.pkl")
sustain_model = joblib.load("sustainability_model.pkl")
_, label_encoder = load_and_preprocess_data("farmer_advisor_dataset.csv")

st.title("🌾 AI for Sustainable Agriculture")

st.markdown("Predict **Crop Yield** and **Sustainability Score** based on your farm data.")

# Input form
with st.form("input_form"):
    crop_type = st.selectbox("Crop Type", label_encoder.classes_)
    soil_pH = st.slider("Soil pH", 4.0, 9.0, 6.5)
    soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 30.0)
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0)
    fertilizer = st.slider("Fertilizer Usage (kg)", 0.0, 300.0, 100.0)
    pesticide = st.slider("Pesticide Usage (kg)", 0.0, 50.0, 10.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    crop_encoded = label_encoder.transform([crop_type])[0]
    input_data = pd.DataFrame([[
        soil_pH, soil_moisture, temperature, rainfall,
        crop_encoded, fertilizer, pesticide
    ]], columns=[
        "Soil_pH", "Soil_Moisture", "Temperature_C", "Rainfall_mm",
        "Crop_Type", "Fertilizer_Usage_kg", "Pesticide_Usage_kg"
    ])
    
    crop_yield, sustain_score = predict(input_data, yield_model, sustain_model)

    st.success(f"🌽 **Predicted Crop Yield:** {crop_yield:.2f} tons")
    st.success(f"🌱 **Predicted Sustainability Score:** {sustain_score:.2f}/100")
