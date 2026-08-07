"""
PIDS Weather-Based Sensor Calibration Suggestion System
Backend: FastAPI + Open-Meteo (free, no API key needed)
"""

import sqlite3
import requests
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


def recommend_sensitivity(weather: dict):
    """
    Rule + weighted-scoring engine.
    Higher risk_score = more false-alarm risk => recommend LOWER sensitivity.
    """
    wind = weather["wind_speed"]        # km/h
    rain = weather["rainfall"]          # mm
    humidity = weather["humidity"]      # %
    storm = weather["storm_flag"]

    risk_score = 0.0
    reasons = []

    if wind > 40:
        risk_score += 3
        reasons.append(f"High wind ({wind} km/h)")
    elif wind > 20:
        risk_score += 1.5
        reasons.append(f"Moderate wind ({wind} km/h)")

    if rain > 10:
        risk_score += 2.5
        reasons.append(f"Heavy rainfall ({rain} mm)")
    elif rain > 2:
        risk_score += 1
        reasons.append(f"Light rainfall ({rain} mm)")

    if humidity > 85:
        risk_score += 1
        reasons.append(f"High humidity ({humidity}%)")

    if storm:
        risk_score += 4
        reasons.append("Storm/thunderstorm detected")

    # Decide sensitivity band
    if risk_score >= 5:
        sensitivity = "Low"
    elif risk_score >= 2:
        sensitivity = "Medium"
    else:
        sensitivity = "High"
        reasons.append("Normal weather conditions")

    # Confidence: how far the score is from the nearest threshold (0-1 scale)
    confidence = min(1.0, round(risk_score / 8, 2)) if risk_score > 0 else 0.95

    reason_text = "; ".join(reasons)
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