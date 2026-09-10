from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
import chromadb
from sentence_transformers import SentenceTransformer
import joblib
import pandas as pd
import requests

# --- Load resources ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("monsoon_knowledge")
model = joblib.load("models/rf_tuned_model.pkl")

FEATURE_ORDER = [
	'temp_max', 'temp_min', 'humidity_mean', 'wind_speed_max', 'pressure_mean',
	'cloud_cover_mean', 'cloud_cover_max', 'precipitation_sum_lag1', 'precipitation_sum_lag3',
	'temp_max_lag1', 'temp_max_lag3', 'humidity_mean_lag1', 'humidity_mean_lag3',
	'pressure_mean_lag1', 'pressure_mean_lag3', 'precipitation_sum_roll7',
	'humidity_mean_roll7', 'pressure_mean_roll7', 'is_monsoon_season', 'day_of_year',
	'city_Delhi', 'city_Kerala_Kochi', 'city_Kolkata', 'city_Mumbai'
]

# --- Tools ---
@tool
def search_knowledge_base(query: str) -> str:
	"""Search the monsoon/climate knowledge base for relevant scientific information."""
	emb = embedder.encode(query).tolist()
	results = collection.query(query_embeddings=[emb], n_results=3)
	docs = results["documents"][0]
	return "\n\n".join(docs) if docs else "No relevant information found."

@tool
def predict_precipitation(city: str, temp_max: float, humidity_mean: float,
						   pressure_mean: float, precipitation_sum_lag1: float,
						   is_monsoon_season: int, day_of_year: int) -> str:
	"""Predict next-day precipitation (mm) given current weather conditions.
	city must be one of: Mumbai, Delhi, Kerala_Kochi, Kolkata, Chennai."""
	row = {f: 0.0 for f in FEATURE_ORDER}
	row["temp_max"] = temp_max
	row["humidity_mean"] = humidity_mean
	row["pressure_mean"] = pressure_mean
	row["precipitation_sum_lag1"] = precipitation_sum_lag1
	row["precipitation_sum_roll7"] = precipitation_sum_lag1
	row["humidity_mean_roll7"] = humidity_mean
	row["pressure_mean_roll7"] = pressure_mean
	row["is_monsoon_season"] = is_monsoon_season
	row["day_of_year"] = day_of_year
	for c in ["Delhi", "Kerala_Kochi", "Kolkata", "Mumbai"]:
		row[f"city_{c}"] = 1 if city == c else 0

	X = pd.DataFrame([row])[FEATURE_ORDER]
	pred = model.predict(X)[0]
	return f"Predicted precipitation for {city}: {pred:.2f} mm"

@tool
def get_live_weather(city: str) -> str:
	"""Fetch current live weather for a city using Open-Meteo (no API key needed)."""
	geo = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}).json()
	if not geo.get("results"):
		return f"Could not find location: {city}"
	lat, lon = geo["results"][0]["latitude"], geo["results"][0]["longitude"]
	weather = requests.get("https://api.open-meteo.com/v1/forecast", params={
		"latitude": lat, "longitude": lon, "current": "temperature_2m,relative_humidity_2m,precipitation"
	}).json()
	cur = weather["current"]
	return f"{city} now: {cur['temperature_2m']}°C, humidity {cur['relative_humidity_2m']}%, precipitation {cur['precipitation']}mm"

# --- Agent ---
llm = ChatOllama(model="qwen2.5:3b", temperature=0)
tools = [search_knowledge_base, predict_precipitation, get_live_weather]
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory)

if __name__ == "__main__":
	config = {"configurable": {"thread_id": "test1"}}
	print("Monsoon AI Agent ready. Type 'exit' to quit.\n")
	while True:
		q = input("You: ")
		if q.lower() == "exit":
			break
		result = agent.invoke({"messages": [("user", q)]}, config)
		print("Agent:", result["messages"][-1].content, "\n")