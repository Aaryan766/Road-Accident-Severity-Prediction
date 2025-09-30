import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# ================= HEADER =================
st.header('Road Accident Severity Prediction Using Machine Learning')

data = '''This project predicts accident severity—slight, serious, or fatal—using key features like driver demographics, vehicle type, road and weather conditions. Authorities can use this to improve road safety and reduce casualties.'''
st.subheader(data)
st.image('https://www.pioneeredge.in/wp-content/uploads/2022/11/accident.jpg')

st.sidebar.write("Sidebar test label")
option = st.sidebar.selectbox("Test select", ["A", "B"])
st.sidebar.write(f"You selected: {option}")

# ================= LOAD MODEL =================
with open('road_accident_severity_pred.pkl', 'rb') as f:
    severity_model = pickle.load(f)

# ================= LOAD PREPROCESSED DATA =================
df = pd.read_csv("road_preprocessed.csv")

# ====== Define feature list used during training ======
MODEL_FEATURES = ['your_feature_1', 'your_feature_2', 'your_feature_3', 'your_feature_4']  # Replace with actual features

top_features = [
    'Type_of_collision',
    'Type_of_vehicle',
    'Age_band_of_driver',
    'Driving_experience',
    'Light_conditions',
    'Weather_conditions',
    'Road_surface_conditions'
]

def find_onehot_columns(df, prefix):
    return [col for col in df.columns if col.startswith(prefix + '_')]

st.sidebar.header("Input Key Accident Features")
st.sidebar.image('https://static.vecteezy.com/system/resources/previews/000/554/213/original/exclamation-mark-vector-icon.jpg', use_container_width=True)

full_input = pd.DataFrame(columns=df.columns, index=[0])

numeric_cols = df.select_dtypes(include=[np.number]).columns
full_input[numeric_cols] = df[numeric_cols].median()

cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    full_input[col] = df[col].mode()[0]

for feature in top_features:
    onehot_cols = find_onehot_columns(df, feature)
    if onehot_cols:
        options = [col.replace(f"{feature}_", "") for col in onehot_cols]
        selected = st.sidebar.selectbox(f"Select {feature}", options)
        for col in onehot_cols:
            full_input[col] = 0
        full_input[f"{feature}_{selected}"] = 1
    elif feature in df.columns and (df[feature].dtype in ['int64', 'float64']):
        min_val, max_val = df[feature].min(), df[feature].max()
        val = st.sidebar.slider(f"Select {feature}", float(min_val), float(max_val), float(df[feature].median()))
        full_input[feature] = val
    elif feature in df.columns:
        choices = df[feature].dropna().unique()
        selected = st.sidebar.selectbox(f"Select {feature}", choices)
        full_input[feature] = selected

for col in ['Type_of_collision', 'Type_of_vehicle', 'Light_conditions', 'Weather_conditions', 'Road_surface_conditions']:
    if col in full_input.columns:
        full_input.drop(columns=[col], inplace=True)

full_input = full_input.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
full_input = full_input.reindex(columns=MODEL_FEATURES, fill_value=0)

progress_bar = st.progress(0)
placeholder = st.empty()
place = st.empty()


for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

# Use model output string directly
pred = severity_model.predict(full_input)[0]
st.write(f"Raw model output: {pred}")

st.success(f"Predicted Accident Severity: {pred}")

st.markdown('Designed by: Aaryan Bhardwaj')
