import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ================= HEADER =================
st.header('Road Accident Severity Prediction Using Machine Learning')

data = '''This project predicts accident severity—slight, serious, or fatal—using key features like driver demographics, vehicle type, and weather/road conditions. Authorities can use this to improve road safety and reduce casualties.'''
st.subheader(data)
st.image('https://www.pioneeredge.in/wp-content/uploads/2022/11/accident.jpg')

# ================= LOAD MODEL =================
with open('road_accident_severity_pred.pkl', 'rb') as f:
    severity_model = pickle.load(f)

# ================= LOAD ORIGINAL DATA =================
df = pd.read_csv("road.csv")  # CSV with original values

# ================= MODEL FEATURES =================
FEATURES = [
    'Type_of_vehicle',
    'Age_band_of_driver',
    'Driving_experience',
    'Weather_conditions',
    'Light_conditions',
    'Type_of_collision'
]

# ================= ORDINAL ENCODING DICTIONARIES =================
# Must match what you used during model training
ordinal_maps = {
    'Type_of_vehicle': {'Bicycle': 0, 'Scooter': 1, 'Motorcycle': 2, 'Car': 3, 'Truck': 4, 'Bus': 5},
    'Age_band_of_driver': {'18-25': 0, '26-35': 1, '36-50': 2, '50+': 3},
    'Driving_experience': {'0-2': 0, '3-5': 1, '6-10': 2, '10+': 3},
    'Weather_conditions': {'Clear': 0, 'Rain': 1, 'Fog': 2, 'Storm': 3},
    'Light_conditions': {'Daylight': 0, 'Night': 1},
    'Type_of_collision': {'Rear-end': 0, 'Side-impact': 1, 'Head-on': 2, 'Multiple vehicles': 3}
}

st.sidebar.header("Input Key Accident Features")
st.sidebar.image(
    'https://static.vecteezy.com/system/resources/previews/000/554/213/original/exclamation-mark-vector-icon.jpg', 
    use_container_width=True
)

# ================= COLLECT USER INPUTS =================
input_data = {}
for feature in FEATURES:
    options = list(ordinal_maps[feature].keys())
    selected = st.sidebar.selectbox(f"Select {feature}", options)
    input_data[feature] = ordinal_maps[feature][selected]

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ================= PREDICTION =================
if st.sidebar.button("Predict Accident Severity"):
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    pred = severity_model.predict(input_df)[0]
    pred_int = int(pred)  # Fix KeyError by ensuring key is int
    severity_map = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
    st.success(f"Predicted Accident Severity: {severity_map[pred_int]}")

st.markdown('Designed by: Aaryan Bhardwaj')
