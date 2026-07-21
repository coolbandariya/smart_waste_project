# 🗑️ Smart Waste Collection & Route Optimization System

An end-to-end, AI-powered waste management pipeline designed to monitor IoT-enabled bins, predict fill levels, score collection priorities using continuous logic, and optimize collection truck routes using Google OR-Tools.

---

## 🌟 Key Features

* **Data Preprocessing & Feature Engineering**: Generates and processes IoT time-series data, engineering key lag features and rolling averages.
* **ML Fill Level Prediction**: Predicts future bin capacity levels using a Random Forest Regressor model.
* **Dynamic Priority Scoring**: Calculates a normalised 0–100 priority score using mathematical membership curves taking fill %, time elapsed, and bin criticality into account.
* **Route Optimisation (CVRP)**: Solves the Capacitated Vehicle Routing Problem using Google OR-Tools to minimise travel distances while honouring truck capacity limits.
* **Interactive Dashboard**: Features a live Streamlit interface with Folium map rendering, dynamic priority thresholds, and route visualisation.

---

## 🛠️ Project Architecture & Modules

```text
smart_waste_project/
│
├── person1_preprocessing.py   # IoT synthetic data generator & feature engineering
├── person2_ml_prediction.py   # Random Forest model training & persistence
├── person3_fuzzy_logic.py     # Priority engine (Fuzzy logic logic/membership math)
├── person4_optimization.py    # Google OR-Tools Vehicle Routing Problem (VRP) solver
├── app.py                     # Streamlit frontend & geospatial Folium map dashboard
├── cleaned_waste_data.csv     # Preprocessed dataset
├── waste_model.pkl            # Trained ML model artefact
└── requirements.txt           # Project dependencies
