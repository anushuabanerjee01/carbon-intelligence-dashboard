# 🌿 Carbon Intelligence Dashboard

> **Interactive ML-powered carbon emission prediction and sustainability analytics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

---

## 🧭 Problem Statement

Organizations and individuals increasingly need to understand and reduce their carbon footprint — but most tools only provide estimates without explaining *why* emissions are high or *what* changes will have the greatest impact.

**Carbon Intelligence Dashboard** closes that gap by combining machine learning prediction with an interactive scenario simulator and data-driven recommendations.

---

## 💡 Key Features

| Feature | Description |
|---|---|
| 🎯 **Emission Prediction** | Predict carbon output from 8 IoT-sourced inputs using ML |
| 🔬 **Scenario Simulator** | Model the impact of reducing energy, transport, or increasing renewables |
| 📊 **Sustainability Insights** | Feature importance, correlation analysis, emission distributions |
| 💡 **Recommendations** | Contextual, actionable reduction strategies based on user inputs |
| 📈 **Model Comparison** | Side-by-side evaluation of Linear Regression vs Random Forest |

---

## 🗃️ Dataset

**10,000 synthetic IoT observations** covering:

| Variable | Type | Range |
|---|---|---|
| Energy Usage (kWh) | Continuous | 50–500 |
| Transportation Distance (km) | Continuous | 5–200 |
| Smart Appliance Usage (hrs) | Continuous | 0–12 |
| Renewable Energy Usage (%) | Continuous | 0–100 |
| Temperature (°C) | Continuous | -5–40 |
| Humidity (%) | Continuous | 20–90 |
| Vehicle Type | Categorical | Electric, Hybrid, Petrol, Diesel, Motorcycle, Public Transport |
| Building Type | Categorical | Apartment, House, Office, Commercial |

**Target:** `Carbon_Emission_kgCO2`

---

## 🔬 Methodology

### Feature Engineering
- **Effective Energy Usage** = `Energy_kWh × (1 − Renewable% / 100)`
- **Distance × Vehicle Interaction** = `Transport_km × Vehicle_Type_Encoded`

### Models
- **Multiple Linear Regression** — interpretable baseline
- **Random Forest Regressor** — 100 trees, depth-optimized via GridSearchCV

### Validation
- 80/20 train-test split (random_state=42)
- 5-fold cross-validation

---

## 📊 Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.854 | — | — |
| Random Forest | 0.838 | — | — |

**Top predictors:** Energy Usage, Transportation Distance, Effective Energy Usage

---

## 🚀 Quick Start

### Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/carbon-intelligence-dashboard.git
cd carbon-intelligence-dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — live in ~2 minutes

---

## 📁 Repository Structure

```
carbon-intelligence-dashboard/
│
├── app.py                  ← Main Streamlit application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── notebooks/
│   └── Data_Analytics_Project_Code.ipynb   ← Original analysis notebook
│
├── data/
│   └── README.md           ← Dataset documentation
│
├── src/
│   └── utils.py            ← Helper functions (optional)
│
├── models/
│   └── README.md           ← Model artifacts documentation
│
├── screenshots/
│   ├── overview.png
│   ├── prediction.png
│   ├── simulator.png
│   └── insights.png
│
└── docs/
    └── methodology.md      ← Detailed methodology notes
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit, Plotly |
| ML/Data | Scikit-learn, Pandas, NumPy |
| Visualization | Plotly Express, Plotly Graph Objects |
| Deployment | Streamlit Cloud |

---

## 🗺️ Future Roadmap

- [ ] Real IoT dataset integration (EPA / smart home APIs)
- [ ] LLM-based sustainability advisor (OpenAI / Claude)
- [ ] Regional carbon intensity overlays
- [ ] Carbon reduction forecasting over 30/60/90 days
- [ ] User profiles and historical tracking

---

## 👤 Author

**Anushua Banerjee**
MS Business Analytics | California State University, East Bay
Senior Analyst & Azure Data Engineer (EY)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/anushuabanerjee)

---

*Built as part of MS Business Analytics portfolio — CSUEB, 2025*
