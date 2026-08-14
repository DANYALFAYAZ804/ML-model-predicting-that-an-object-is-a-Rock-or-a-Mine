import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('rock_vs_mine_model.pkl')

st.title("⚓ Sonar Signal Classifier: Rock vs. Mine")
st.write("Enter 60 frequency values (comma-separated) or test with sample data.")

# Input layout
input_data = st.text_area(
    "60 Frequency Readings", 
    placeholder="0.02, 0.0371, 0.0428, 0.0207, ..."
)

# Sample presets for quick testing
if st.button("Load Sample Rock Data"):
    input_data = "0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.6393,0.5787,0.4847,0.3441,0.201,0.2107,0.1911,0.125,0.2104,0.1235,0.035,0.076,0.04,0.03,0.015,0.02,0.015,0.01,0.005,0.002,0.001,0.002,0.003,0.004,0.002,0.002,0.001,0.002,0.003,0.001,0.002"

if st.button("Classify Signal"):
    try:
        # Convert raw string input to numpy array
        features = np.array([float(x.strip()) for x in input_data.split(',')]).reshape(1, -1)
        
        if features.shape[1] != 60:
            st.error(f"Expected 60 feature values, but received {features.shape[1]}.")
        else:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            if prediction == 'R':
                st.success(f"**Result: Rock** (Confidence: {probability[1]:.2%})")
            else:
                st.error(f"**Result: Mine** (Confidence: {probability[0]:.2%})")
    except Exception as e:
        st.error(f"Invalid input format: {e}")