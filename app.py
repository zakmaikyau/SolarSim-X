import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page Config
st.set_page_config(page_title="SolarSim-X", layout="wide")
st.title("☀️ SolarSim-X: ML-Physics Hybrid Simulator")

# Sidebar Inputs
st.sidebar.header("Input Parameters")
eg = st.sidebar.slider("Bandgap (eV)", 0.8, 2.0, 1.34)
l_abs = st.sidebar.slider("Thickness (um)", 0.5, 5.0, 2.0)
log_n = st.sidebar.slider("Log Doping (cm-3)", 14.0, 18.0, 16.0)
tau = st.sidebar.slider("Lifetime (us)", 0.1, 50.0, 10.0)
temp = st.sidebar.slider("Temperature (K)", 273, 350, 300)
irr = st.sidebar.slider("Irradiance (W/m2)", 200, 1000, 1000)

# Load Model and Predict
try:
    model = joblib.load('solarsim_ml_model.joblib')
    
    input_df = pd.DataFrame({
        'E_g': [eg],
        'L_absorber_um': [l_abs],
        'log_N': [log_n],
        'tau_rec_us': [tau],
        'T_cell': [temp],
        'G_irradiance': [irr]
    })
    
    res = model.predict(input_df)[0]
    
    # Display Results
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Voc (V)", round(res[0], 3))
    col2.metric("Jsc (mA/cm2)", round(res[1], 2))
    col3.metric("FF", round(res[2], 3))
    col4.metric("Efficiency (%)", round(res[3], 2))

except FileNotFoundError:
    st.error("Model file not found. Run train_model.py first.")
