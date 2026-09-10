import streamlit as st
import requests

st.title("🌧️ Monsoon Precipitation Predictor")
st.write("Enter current weather conditions to forecast next-day rainfall.")

city = st.selectbox("City", ["Mumbai", "Delhi", "Kerala_Kochi", "Kolkata", "Chennai"])
temp_max = st.slider("Max Temperature (°C)", 15.0, 45.0, 32.0)
temp_min = st.slider("Min Temperature (°C)", 10.0, 35.0, 25.0)
humidity_mean = st.slider("Humidity (%)", 20.0, 100.0, 70.0)
wind_speed_max = st.slider("Wind Speed (km/h)", 0.0, 60.0, 15.0)
pressure_mean = st.slider("Pressure (hPa)", 990.0, 1020.0, 1006.0)
cloud_cover_mean = st.slider("Cloud Cover Mean (%)", 0.0, 100.0, 50.0)
cloud_cover_max = st.slider("Cloud Cover Max (%)", 0.0, 100.0, 70.0)
is_monsoon_season = st.checkbox("Is Monsoon Season (Jun-Sep)?", value=True)
day_of_year = st.slider("Day of Year", 1, 365, 180)

st.subheader("Recent conditions (lag/rolling features)")
precip_lag1 = st.number_input("Yesterday's rainfall (mm)", 0.0, 300.0, 5.0)
precip_lag3 = st.number_input("Rainfall 3 days ago (mm)", 0.0, 300.0, 5.0)
precip_roll7 = st.number_input("7-day avg rainfall (mm)", 0.0, 100.0, 5.0)
humidity_roll7 = st.number_input("7-day avg humidity (%)", 0.0, 100.0, 70.0)
pressure_roll7 = st.number_input("7-day avg pressure (hPa)", 990.0, 1020.0, 1006.0)

if st.button("Predict Rainfall"):
	payload = {
		"temp_max": temp_max, "temp_min": temp_min, "humidity_mean": humidity_mean,
		"wind_speed_max": wind_speed_max, "pressure_mean": pressure_mean,
		"cloud_cover_mean": cloud_cover_mean, "cloud_cover_max": cloud_cover_max,
		"precipitation_sum_lag1": precip_lag1, "precipitation_sum_lag3": precip_lag3,
		"temp_max_lag1": temp_max, "temp_max_lag3": temp_max,
		"humidity_mean_lag1": humidity_mean, "humidity_mean_lag3": humidity_mean,
		"pressure_mean_lag1": pressure_mean, "pressure_mean_lag3": pressure_mean,
		"precipitation_sum_roll7": precip_roll7, "humidity_mean_roll7": humidity_roll7,
		"pressure_mean_roll7": pressure_roll7, "is_monsoon_season": int(is_monsoon_season),
		"day_of_year": day_of_year, "city": city
	}
	res = requests.post("http://localhost:8000/predict", json=payload)
	result = res.json()
	st.metric("Predicted Rainfall", f"{result['predicted_precipitation_mm']} mm")
	st.info(result["risk_level"])