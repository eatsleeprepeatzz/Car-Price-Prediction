import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("xgboost_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🚗 Used Car Price Prediction")

# User inputs
name = st.selectbox("Car Name", label_encoders["Name"].classes_)
location = st.selectbox("Location", label_encoders["Location"].classes_)
fuel = st.selectbox("Fuel Type", label_encoders["Fuel_Type"].classes_)
transmission = st.selectbox("Transmission", label_encoders["Transmission"].classes_)
owner = st.selectbox("Owner Type", label_encoders["Owner_Type"].classes_)

year = st.number_input("Year", min_value=1990, max_value=2025)
kilometers = st.number_input("Kilometers Driven", min_value=0)
mileage = st.number_input("Mileage")
engine = st.number_input("Engine (CC)")
power = st.number_input("Power (bhp)")
seats = st.number_input("Seats", min_value=2, max_value=10)

# Encode inputs
input_data = {
    "Name": label_encoders["Name"].transform([name])[0],
    "Location": label_encoders["Location"].transform([location])[0],
    "Fuel_Type": label_encoders["Fuel_Type"].transform([fuel])[0],
    "Transmission": label_encoders["Transmission"].transform([transmission])[0],
    "Owner_Type": label_encoders["Owner_Type"].transform([owner])[0],
    "Year": year,
    "Kilometers_Driven": kilometers,
    "Mileage": mileage,
    "Engine": engine,
    "Power": power,
    "Seats": seats
}

input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {prediction:,.2f} Lakhs")
