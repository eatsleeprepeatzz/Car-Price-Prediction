import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------
# Custom CSS for Better UI
# --------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Encoders
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgboost_gridsearch_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, label_encoders

model, label_encoders = load_artifacts()

# --------------------------------------------------
# App Header
# --------------------------------------------------
st.title("🚗 Used Car Price Prediction System")
st.markdown(
    "Predict the **resale price of a used car** using a **machine-learning powered XGBoost model**."
)

st.divider()

# --------------------------------------------------
# Input Sections
# --------------------------------------------------
st.subheader("🔧 Car Specifications")

col1, col2 = st.columns(2)

with col1:
    name = st.selectbox(
        "Car Model",
        label_encoders["Name"].classes_
    )

    location = st.selectbox(
        "Location",
        label_encoders["Location"].classes_
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        label_encoders["Fuel_Type"].classes_
    )

    transmission = st.selectbox(
        "Transmission",
        label_encoders["Transmission"].classes_
    )

    owner_type = st.selectbox(
        "Owner Type",
        label_encoders["Owner_Type"].classes_
    )

with col2:
    year = st.number_input(
        "Manufacturing Year",
        min_value=1990,
        max_value=2025,
        value=2015
    )

    kilometers_driven = st.number_input(
        "Kilometers Driven",
        min_value=0,
        step=1000
    )

    mileage = st.number_input(
        "Mileage (km/l or km/kg)",
        min_value=0.0,
        step=0.1
    )

    engine = st.number_input(
        "Engine Capacity (CC)",
        min_value=500.0,
        step=50.0
    )

    power = st.number_input(
        "Power (bhp)",
        min_value=20.0,
        step=5.0
    )

    seats = st.number_input(
        "Number of Seats",
        min_value=2,
        max_value=10,
        value=5
    )

st.divider()

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if st.button("🔮 Predict Car Price"):
    try:
        # Encode categorical inputs
        input_data = {
            "Name": label_encoders["Name"].transform([name])[0],
            "Location": label_encoders["Location"].transform([location])[0],
            "Year": year,
            "Kilometers_Driven": kilometers_driven,
            "Fuel_Type": label_encoders["Fuel_Type"].transform([fuel_type])[0],
            "Transmission": label_encoders["Transmission"].transform([transmission])[0],
            "Owner_Type": label_encoders["Owner_Type"].transform([owner_type])[0],
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])

        # 🔥 Ensure correct feature order
        input_df = input_df[model.feature_names_in_]

        # Predict
        prediction = model.predict(input_df)[0]

        st.success(
            f"💰 **Estimated Car Price:** ₹ {prediction:,.2f} Lakhs"
        )

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Built with ❤️ using **Python, XGBoost & Streamlit** | "
    "Machine Learning 2025 – Section B"
)
