# Weather Intelligence Platform 🌧️

An AI-powered Weather Intelligence and Climate Decision Support Platform for
predicting the Indian Southwest Monsoon, built for the **Team GenAI Research
2026 — Phase 02** internship project at **MacroEdtech**, under the guidance
of **Sagar Sakalley**.

The project integrates Machine Learning, Explainable AI, Computer
Vision/Remote Sensing, and a Generative AI (RAG + Agent) layer, using only
free and open-source tools, models, and publicly available data.

---

## Project Structure

```
weather_intelligence/
├── api/
│   └── main.py                       # FastAPI backend serving the precipitation model
├── app/
│   ├── streamlit_app.py              # Streamlit UI: slider-based rainfall predictor
│   └── chat_app.py                   # Streamlit UI: RAG + Agent chat interface
├── chroma_db/                        # Persisted ChromaDB vector store (auto-generated)
├── data/
│   └── weather_features_5cities_2015_2024.csv
├── docs/
│   ├── baseline_model_results.csv
│   ├── cloud_coverage_estimates.csv
│   └── knowledge_base/               # Markdown knowledge base for RAG (5 docs)
│       ├── monsoon_basics.md
│       ├── monsoon_mumbai.md
│       ├── elnino_lanina.md
│       ├── rainfall_measurement.md
│       └── climate_change_monsoon.md
├── models/
│   ├── gb_baseline_model.pkl
│   ├── xgb_baseline_model.pkl
│   ├── rf_baseline_model.pkl         # kept local only, see Known Issues
│   ├── rf_tuned_model.pkl            # kept local only, see Known Issues
│   └── temp_model.pkl
├── notebooks/
│   └── Final_notebook.ipynb          # Full pipeline: data → EDA → models → SHAP → CV
├── src/
│   ├── build_vectorstore.py          # Builds the ChromaDB knowledge base
│   └── agent.py                      # LangGraph agent (RAG + tools)
├── monthly_rainfall_by_city.png      # Figure used in the research paper
├── satellite_samples.png             # Figure used in the research paper
├── shap_summary.png                  # Figure used in the research paper
├── MacroEdtech_Phase02_Tamanna.pdf   # Compiled research paper (Overleaf export)
├── requirements.txt
└── .gitignore
```

---

## Part 01 — Weather Prediction Engine

**Data**: 10 years (2015–2024) of daily weather data for 5 Indian cities
(Mumbai, Kochi, Delhi, Kolkata, Chennai) — 17,050 records, zero missing
values — collected via the [Open-Meteo Archive API](https://open-meteo.com/).

**Feature engineering**: lag features (1-day, 3-day), 7-day rolling means,
monsoon-season flag, day-of-year, one-hot encoded city.

**Models compared** (precipitation forecasting, chronological 80/20 split):

| Model | MAE (mm) | RMSE (mm) | R² |
|---|---|---|---|
| Linear Regression | 5.27 | 10.84 | 0.571 |
| Random Forest (baseline) | 3.08 | 9.40 | 0.677 |
| Gradient Boosting | 3.41 | 9.52 | 0.669 |
| XGBoost | 3.42 | 9.73 | 0.654 |
| **Random Forest (tuned)** | **3.13** | **9.33** | **0.682** |

A companion **temperature model** (next-day max temp, Random Forest) achieved
MAE = 0.82°C, R² = 0.812.

**Explainability**: SHAP analysis identified `humidity_mean` as the dominant
predictor, followed by 7-day rolling precipitation and max wind speed.

**Computer Vision**: A proof-of-concept satellite cloud-coverage estimator
(OpenCV, HSV thresholding) was applied to MODIS true-color imagery (NASA
GIBS) across 5 seasonal dates, showing directionally correct results (0%
cloud cover in dry months vs. 16–36% in monsoon months).

**Deployment**: FastAPI backend + Streamlit frontend, allowing users to input
current weather conditions and receive a rainfall prediction with a
risk-tier message.

---

## Part 02 — Generative AI Layer

- **Knowledge Base**: 5 markdown documents on monsoon science, embedded with
  `sentence-transformers/all-MiniLM-L6-v2`, stored in **ChromaDB** (20
  chunks).
- **LLM**: locally-hosted, open-weight **`qwen2.5:3b`** via **Ollama** — no
  paid API used anywhere in this project.
- **Agent**: a **LangGraph** ReAct agent with 3 tools:
  - `search_knowledge_base` — semantic search over the RAG knowledge base
  - `predict_precipitation` — wraps the trained Random Forest model
  - `get_live_weather` — live weather lookup via Open-Meteo
- **Interface**: Streamlit chat app (`app/chat_app.py`) with conversation
  memory.

---

## Setup & Running Locally

```bash
# 1. Clone and set up environment
git clone https://github.com/tamanna26072/weather_intelligence.git
cd weather_intelligence
python -m venv venv
venv\Scripts\activate.bat        # Windows
pip install -r requirements.txt

# 2. Install Ollama and pull the model (for Part 02)
#    Download from https://ollama.com, then:
ollama pull qwen2.5:3b

# 3. Build the RAG vector store (for Part 02)
python src/build_vectorstore.py

# 4. Run the prediction API (Terminal 1)
uvicorn api.main:app --reload

# 5. Run the rainfall predictor UI (Terminal 2)
streamlit run app/streamlit_app.py

# 6. Or run the RAG + Agent chat assistant (Terminal 2)
streamlit run app/chat_app.py
```

> **Note:** `rf_tuned_model.pkl` (the best-performing model, used by the API)
> is not included in this repository — see Known Issues below. To run the
> API/app, regenerate it by running the modeling section of
> `notebooks/Final_notebook.ipynb`, or use `models/gb_baseline_model.pkl` as
> a substitute by adjusting `api/main.py`.

---

## Known Issues / Limitations

- **`rf_tuned_model.pkl` (154.6 MB) and `rf_baseline_model.pkl` exceed
  GitHub's 100 MB file limit** and are kept local-only (not pushed to this
  repo). Gradient Boosting, XGBoost, and the temperature model are tracked
  normally, along with the full results CSV.
- **Development environment switch**: kernel instability in the local
  VS Code/Jupyter setup required moving primary development to Google Colab
  partway through the project, with periodic re-sync back to this repo.
- **Small-LLM tool-calling reliability**: `qwen2.5:3b` reliably parses
  structured (`key=value`) input to the `predict_precipitation` tool, but
  occasionally fails to correctly map free-form natural-language weather
  descriptions — reported here rather than hidden.
- **Remote sensing**: the free NASA GIBS on-demand snapshot service returned
  a partial no-data region for one of five requested satellite images,
  likely causing an underestimate of cloud coverage for that date.
- **Docker**: not implemented within the project timeline.
- **No k-fold cross-validation**: model comparison uses a single
  chronological train/test split.

---

## Data Sources

- [Open-Meteo Archive API](https://open-meteo.com/) — historical weather data
- [NASA GIBS](https://wiki.earthdata.nasa.gov/display/GIBS) — satellite imagery

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · OpenCV · FastAPI ·
Streamlit · LangChain · LangGraph · ChromaDB · sentence-transformers ·
Ollama (qwen2.5:3b)

---

*Part of the Team GenAI Research 2026 internship program at MacroEdtech.*