import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# FLAT STRUCTURE: Expect all files in the SAME directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'best_random_forest.joblib')
data_path = os.path.join(current_dir, 'processed_bike_data.csv')

# Load model
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Critical Error: Model file 'best_random_forest.joblib' not found in root.")
    st.stop()

st.set_page_config(page_title="Bike Sharing Demand Predictor", layout="centered")

st.title("🚲 Bike Sharing Rental Demand Prediction")
st.markdown("""
Predict the total number of bike rentals based on weather, time, and seasonal factors.
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    # Time factors
    hr = st.sidebar.slider("Hour (0-23)", 0, 23, 12)
    mnth = st.sidebar.slider("Month (1-12)", 1, 12, 6)
    weekday = st.sidebar.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x])
    yr = st.sidebar.selectbox("Year", options=[0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    
    # Flags
    holiday = st.sidebar.checkbox("Is it a Holiday?")
    workingday = st.sidebar.checkbox("Is it a Working Day?", value=True)
    
    # Weather
    temp = st.sidebar.slider("Temperature (0-1)", 0.0, 1.0, 0.5)
    atemp = st.sidebar.slider("Feeling Temperature (0-1)", 0.0, 1.0, 0.5)
    hum = st.sidebar.slider("Humidity (0-1)", 0.0, 1.0, 0.5)
    windspeed = st.sidebar.slider("Windspeed (0-1)", 0.0, 1.0, 0.1)
    
    season = st.sidebar.selectbox("Season", ["springer", "summer", "fall", "winter"])
    weather_sit = st.sidebar.selectbox("Weather Situation", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    
    # Create the feature dict
    data = {
        'yr': [yr],
        'holiday': [1 if holiday else 0],
        'workingday': [1 if workingday else 0],
        'temp': [temp],
        'atemp': [atemp],
        'hum': [hum],
        'windspeed': [windspeed],
        'hr_sin': [np.sin(2 * np.pi * hr / 24)],
        'hr_cos': [np.cos(2 * np.pi * hr / 24)],
        'mnth_sin': [np.sin(2 * np.pi * mnth / 12)],
        'mnth_cos': [np.cos(2 * np.pi * mnth / 12)],
        'weekday_sin': [np.sin(2 * np.pi * weekday / 7)],
        'weekday_cos': [np.cos(2 * np.pi * weekday / 7)],
        'day_sin': [np.sin(2 * np.pi * 15 / 31)],
        'day_cos': [np.cos(2 * np.pi * 15 / 31)],
    }
    
    # One-Hot Encoding for Season
    data['season_springer'] = [1 if season == 'springer' else 0]
    data['season_summer'] = [1 if season == 'summer' else 0]
    data['season_winter'] = [1 if season == 'winter' else 0]
    
    # One-Hot Encoding for Weather
    data['weather_Heavy Rain'] = [1 if weather_sit == 'Heavy Rain' else 0]
    data['weather_Light Snow'] = [1 if weather_sit == 'Light Snow' else 0]
    data['weather_Mist'] = [1 if weather_sit == 'Mist' else 0]
    
    return pd.DataFrame(data)

# Load columns from processed data to ensure exact match
try:
    processed_cols = pd.read_csv(data_path, nrows=0).columns.tolist()
    processed_cols = [c for c in processed_cols if c not in ['cnt', 'casual', 'registered']]
except:
    processed_cols = []

input_df = user_input_features()

# Ensure column order matches training data
if processed_cols:
    for col in processed_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[processed_cols]

st.subheader("Selected Parameters Summary")
st.write(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"### Predicted Total Bike Rentals: {int(prediction[0])}")

st.markdown("---")
st.markdown("Developed for Bike-Sharing Rental Demand Analysis.")
