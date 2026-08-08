"""
PIDS Weather-Based Sensor Calibration Suggestion System
Backend: FastAPI + Open-Meteo (free, no API key needed)
Recommendation engine: trained Random Forest classifier (see train_model.py)
"""

import sqlite3
import requests
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PIDS Weather Calibration System")

# Allow frontend (index.html) to call this API directly from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "calibration_log.db"

# Load the trained Random Forest model (run train_model.py first to generate this)
try:
    ML_MODEL = joblib.load("model.joblib")
except FileNotFoundError:
    ML_MODEL = None
    print("WARNING: model.joblib not found. Run `python train_model.py` first.")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lat REAL,
            lon REAL,
            wind_speed REAL,
            rainfall REAL,
            temperature REAL,
            humidity REAL,
            storm_flag INTEGER,
            sensitivity TEXT,
            confidence REAL,
            reason TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def fetch_weather(lat: float, lon: float):
    """Fetch live weather from Open-Meteo (no API key required)."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,"
        "wind_speed_10m,weather_code"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()["current"]

    # weather_code >= 95 means thunderstorm (WMO weather codes)
    storm_flag = 1 if data.get("weather_code", 0) >= 95 else 0

    return {
        "temperature": data.get("temperature_2m", 0),
        "humidity": data.get("relative_humidity_2m", 0),
        "rainfall": data.get("precipitation", 0),
        "wind_speed": data.get("wind_speed_10m", 0),
        "storm_flag": storm_flag,
    }


def explain_conditions(weather: dict):
    """
    Human-readable explanation of which conditions are notable.
    Used only for the dashboard's 'Reason' text -- the actual sensitivity
    decision comes from the trained ML model below, not from this list.
    """
    wind = weather["wind_speed"]
    rain = weather["rainfall"]
    humidity = weather["humidity"]
    storm = weather["storm_flag"]

    notes = []
    if storm:
        notes.append("Storm/thunderstorm detected")
    if wind > 40:
        notes.append(f"High wind ({wind} km/h)")
    elif wind > 20:
        notes.append(f"Moderate wind ({wind} km/h)")
    if rain > 10:
        notes.append(f"Heavy rainfall ({rain} mm)")
    elif rain > 2:
        notes.append(f"Light rainfall ({rain} mm)")
    if humidity > 85:
        notes.append(f"High humidity ({humidity}%)")

    if not notes:
        notes.append("Normal weather conditions")

    return "; ".join(notes)


def recommend_sensitivity(weather: dict):
    """
    ML-based recommendation: feeds live weather features into the trained
    Random Forest classifier (train_model.py) and returns its predicted
    class + probability as the confidence score.
    """
    if ML_MODEL is None:
        raise RuntimeError("Model not loaded. Run `python train_model.py` first.")

    features = pd.DataFrame([{
        "wind_speed": weather["wind_speed"],
        "rainfall": weather["rainfall"],
        "humidity": weather["humidity"],
        "temperature": weather["temperature"],
        "storm_flag": weather["storm_flag"],
    }])

    sensitivity = ML_MODEL.predict(features)[0]
    proba = ML_MODEL.predict_proba(features)[0]
    class_index = list(ML_MODEL.classes_).index(sensitivity)
    confidence = round(float(proba[class_index]), 2)

    reason_text = explain_conditions(weather)
    return sensitivity, confidence, reason_text


def log_recommendation(lat, lon, weather, sensitivity, confidence, reason):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO calibration_log
        (timestamp, lat, lon, wind_speed, rainfall, temperature, humidity,
         storm_flag, sensitivity, confidence, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            lat, lon,
            weather["wind_speed"], weather["rainfall"],
            weather["temperature"], weather["humidity"],
            weather["storm_flag"], sensitivity, confidence, reason,
        ),
    )
    conn.commit()
    conn.close()


@app.get("/")
def root():
    return {"status": "PIDS Weather Calibration API running"}


@app.get("/recommend")
def recommend(lat: float = Query(19.076), lon: float = Query(72.877)):
    """
    Main endpoint: fetch weather for given lat/lon and return
    sensor sensitivity recommendation.
    Default coords = Mumbai (change as needed for demo).
    """
    weather = fetch_weather(lat, lon)
    sensitivity, confidence, reason = recommend_sensitivity(weather)
    log_recommendation(lat, lon, weather, sensitivity, confidence, reason)

    return {
        "location": {"lat": lat, "lon": lon},
        "weather": weather,
        "recommendation": {
            "sensitivity": sensitivity,
            "confidence": confidence,
            "reason": reason,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/history")
def history(limit: int = 20):
    """Return last N recommendation logs for analytics/dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM calibration_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/model-info")
def model_info():
    """
    Expose real feature importances from the trained Random Forest,
    so the dashboard can show WHY the model weighs each factor the way
    it does -- genuine model transparency, not a hardcoded explanation.
    """
    if ML_MODEL is None:
        return {"error": "Model not loaded"}

    feature_names = ["wind_speed", "rainfall", "humidity", "temperature", "storm_flag"]
    importances = ML_MODEL.feature_importances_.tolist()

    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": ML_MODEL.n_estimators,
        "classes": list(ML_MODEL.classes_),
        "feature_importance": [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in sorted(
                zip(feature_names, importances), key=lambda x: -x[1]
            )
        ],
    }