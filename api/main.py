from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Monsoon Weather Intelligence API")

# Load model (local path, since large file not in repo)
model = joblib.load("models/rf_tuned_model.pkl")

FEATURE_ORDER = [
	'temp_max', 'temp_min', 'humidity_mean', 'wind_speed_max', 'pressure_mean',
	'cloud_cover_mean', 'cloud_cover_max', 'precipitation_sum_lag1', 'precipitation_sum_lag3',
	'temp_max_lag1', 'temp_max_lag3', 'humidity_mean_lag1', 'humidity_mean_lag3',
	'pressure_mean_lag1', 'pressure_mean_lag3', 'precipitation_sum_roll7',
	'humidity_mean_roll7', 'pressure_mean_roll7', 'is_monsoon_season', 'day_of_year',
	'city_Delhi', 'city_Kerala_Kochi', 'city_Kolkata', 'city_Mumbai'
]

class WeatherInput(BaseModel):
	temp_max: float
	temp_min: float
	humidity_mean: float
	wind_speed_max: float
	pressure_mean: float
	cloud_cover_mean: float
	cloud_cover_max: float
	precipitation_sum_lag1: float
	precipitation_sum_lag3: float
	temp_max_lag1: float
	temp_max_lag3: float
	humidity_mean_lag1: float
	humidity_mean_lag3: float
	pressure_mean_lag1: float
	pressure_mean_lag3: float
	precipitation_sum_roll7: float
	humidity_mean_roll7: float
	pressure_mean_roll7: float
	is_monsoon_season: int
	day_of_year: int
	city: str  # "Mumbai", "Delhi", "Kerala_Kochi", "Kolkata", "Chennai"

@app.get("/")
def root():
	return {"status": "Monsoon Weather Intelligence API running"}

@app.post("/predict")
def predict(data: WeatherInput):
	row = data.dict()
	city = row.pop("city")
	for c in ["Delhi", "Kerala_Kochi", "Kolkata", "Mumbai"]:
		row[f"city_{c}"] = 1 if city == c else 0

	X = pd.DataFrame([row])[FEATURE_ORDER]
	pred = model.predict(X)[0]

	if pred < 2.5:
		risk = "Low / Light rain"
	elif pred < 15:
		risk = "Moderate rain"
	else:
		risk = "Heavy rain — flood risk"

	return {"predicted_precipitation_mm": round(float(pred), 2), "risk_level": risk}