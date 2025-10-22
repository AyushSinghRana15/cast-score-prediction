import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load model and scaler
model = load_model('csat_ann_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("CSAT Prediction Dashboard")

# Collect inputs
channel_name = st.number_input("Channel Name (int)", min_value=0, max_value=10)
category = st.number_input("Category (int)", min_value=0, max_value=10)
sub_category = st.number_input("Sub-category (int)", min_value=0, max_value=10)
agent_name = st.number_input("Agent Name (int)", min_value=0)
supervisor = st.number_input("Supervisor (int)", min_value=0)
manager = st.number_input("Manager (int)", min_value=0)
tenure_bucket = st.number_input("Tenure Bucket (int)", min_value=0, max_value=10)
agent_shift = st.number_input("Agent Shift (int)", min_value=0, max_value=10)

if st.button('Predict CSAT'):
    features = np.array([[channel_name, category, sub_category, agent_name, supervisor, manager, tenure_bucket, agent_shift]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    st.write(f"Predicted CSAT Score: {prediction[0][0]:.2f}")
