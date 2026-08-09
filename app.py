"""
VIGIL PIDS
Weather-Based Sensor Calibration Suggestion System

FastAPI backend
Open-Meteo weather
Random Forest recommendation engine
SQLite audit logging
"""

import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
from fastapi.responses import FileResponse

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="VIGIL PIDS Weather Calibration API"
)


app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"]

)


# ============================================================
# FILES
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "calibration_log.db"
MODEL_PATH = BASE_DIR / "model.joblib"

# ============================================================
# LOAD MODEL
# ============================================================

try:

    ML_MODEL = joblib.load(
        MODEL_PATH
    )

    print("✓ Random Forest model loaded")

except Exception as e:

    # Catches FileNotFoundError AND version-mismatch / corrupt-pickle
    # errors from joblib/scikit-learn, so a bad model file can never
    # crash the whole app at import time — it just disables /recommend.

    ML_MODEL = None

    print(f"⚠ Failed to load model.joblib: {type(e).__name__}: {e}")
    print("Run: python train_model.py")


# ============================================================
# DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_log (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT,

            lat REAL,

            lon REAL,

            wind_speed REAL,

            wind_gusts REAL,

            rainfall REAL,

            precipitation REAL,

            rain REAL,

            showers REAL,

            precipitation_probability REAL,

            temperature REAL,

            humidity REAL,

            storm_flag INTEGER,

            weather_code INTEGER,

            sensitivity TEXT,

            confidence REAL,

            reason TEXT

        )
        """
    )

    conn.commit()

    conn.close()


init_db()


# ============================================================
# WEATHER
# ============================================================

def fetch_weather(
    lat: float,
    lon: float
):

    """
    Fetch current weather from Open-Meteo.

    Current conditions include:

    - temperature
    - humidity
    - precipitation
    - rain
    - showers
    - wind speed
    - wind gusts
    - weather code

    We also request hourly precipitation probability.
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )


    params = {

        "latitude": lat,

        "longitude": lon,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "showers,"
            "wind_speed_10m,"
            "wind_gusts_10m,"
            "weather_code"
        ),

        "hourly": (
            "precipitation_probability"
        ),

        "forecast_days": 1,

        "timezone": "auto"

    }


    print("\n" + "=" * 60)

    print("WEATHER REQUEST")

    print(
        f"Latitude : {lat}"
    )

    print(
        f"Longitude: {lon}"
    )

    print("=" * 60)


    try:

        response = requests.get(

            url,

            params=params,

            timeout=15

        )

        response.raise_for_status()


    except requests.RequestException as e:

        print(
            f"Weather API ERROR: {e}"
        )

        raise RuntimeError(
            "Unable to fetch weather from Open-Meteo"
        )


    payload = response.json()


    current = payload.get(
        "current",
        {}
    )


    # ========================================================
    # CURRENT VALUES
    # ========================================================

    temperature = float(
        current.get(
            "temperature_2m"
        ) or 0
    )


    humidity = float(
        current.get(
            "relative_humidity_2m"
        ) or 0
    )


    precipitation = float(
        current.get(
            "precipitation"
        ) or 0
    )


    rain = float(
        current.get(
            "rain"
        ) or 0
    )


    showers = float(
        current.get(
            "showers"
        ) or 0
    )


    wind_speed = float(
        current.get(
            "wind_speed_10m"
        ) or 0
    )


    wind_gusts = float(
        current.get(
            "wind_gusts_10m"
        ) or 0
    )


    weather_code = int(
        current.get(
            "weather_code"
        ) or 0
    )


    # ========================================================
    # EFFECTIVE RAINFALL
    # ========================================================

    rainfall = max(
        precipitation,
        rain,
        showers
    )


    # ========================================================
    # PRECIPITATION PROBABILITY
    # ========================================================

    hourly = payload.get(
        "hourly",
        {}
    )


    probabilities = hourly.get(
        "precipitation_probability",
        []
    )


    precipitation_probability = 0.0


    if probabilities:

        precipitation_probability = float(
            probabilities[0] or 0
        )


    # ========================================================
    # STORM DETECTION
    # ========================================================

    storm_flag = int(
        weather_code >= 95
    )


    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print("\nRAW OPEN-METEO VALUES")

    print(
        f"Temperature               : "
        f"{temperature} °C"
    )

    print(
        f"Humidity                  : "
        f"{humidity} %"
    )

    print(
        f"Precipitation             : "
        f"{precipitation} mm"
    )

    print(
        f"Rain                      : "
        f"{rain} mm"
    )

    print(
        f"Showers                   : "
        f"{showers} mm"
    )

    print(
        f"Effective Rainfall        : "
        f"{rainfall} mm"
    )

    print(
        f"Precipitation Probability : "
        f"{precipitation_probability} %"
    )

    print(
        f"Wind                      : "
        f"{wind_speed} km/h"
    )

    print(
        f"Wind Gusts                : "
        f"{wind_gusts} km/h"
    )

    print(
        f"Weather Code              : "
        f"{weather_code}"
    )

    print(
        f"Storm Flag                : "
        f"{storm_flag}"
    )


    return {

        "temperature":
            round(temperature, 2),

        "humidity":
            round(humidity, 2),

        "rainfall":
            round(rainfall, 2),

        "precipitation":
            round(precipitation, 2),

        "rain":
            round(rain, 2),

        "showers":
            round(showers, 2),

        "precipitation_probability":
            round(
                precipitation_probability,
                1
            ),

        "wind_speed":
            round(wind_speed, 2),

        "wind_gusts":
            round(wind_gusts, 2),

        "weather_code":
            weather_code,

        "storm_flag":
            storm_flag

    }


# ============================================================
# WEATHER EXPLANATION
# ============================================================

def explain_conditions(
    weather
):

    notes = []


    # --------------------------------------------------------
    # STORM
    # --------------------------------------------------------

    if weather["storm_flag"]:

        notes.append(
            "Thunderstorm detected"
        )


    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    if weather["wind_gusts"] >= 60:

        notes.append(
            f"Very high wind gusts "
            f"({weather['wind_gusts']} km/h)"
        )

    elif weather["wind_gusts"] >= 45:

        notes.append(
            f"Strong wind gusts "
            f"({weather['wind_gusts']} km/h)"
        )

    elif weather["wind_speed"] >= 30:

        notes.append(
            f"High wind "
            f"({weather['wind_speed']} km/h)"
        )

    elif weather["wind_speed"] >= 20:

        notes.append(
            f"Moderate wind "
            f"({weather['wind_speed']} km/h)"
        )


    # --------------------------------------------------------
    # RAIN
    # --------------------------------------------------------

    if weather["rainfall"] >= 20:

        notes.append(
            f"Heavy rainfall "
            f"({weather['rainfall']} mm)"
        )

    elif weather["rainfall"] >= 10:

        notes.append(
            f"Moderate-heavy rainfall "
            f"({weather['rainfall']} mm)"
        )

    elif weather["rainfall"] >= 5:

        notes.append(
            f"Moderate rainfall "
            f"({weather['rainfall']} mm)"
        )

    elif weather["rainfall"] >= 1:

        notes.append(
            f"Light rainfall "
            f"({weather['rainfall']} mm)"
        )

    elif weather["rainfall"] >= 0.2:

        notes.append(
            f"Drizzle/light precipitation "
            f"({weather['rainfall']} mm)"
        )


    # --------------------------------------------------------
    # HUMIDITY
    # --------------------------------------------------------

    if weather["humidity"] >= 90:

        notes.append(
            f"Very high humidity "
            f"({weather['humidity']}%)"
        )

    elif weather["humidity"] >= 85:

        notes.append(
            f"High humidity "
            f"({weather['humidity']}%)"
        )


    # --------------------------------------------------------
    # PRECIPITATION PROBABILITY
    # --------------------------------------------------------

    if (
        weather["precipitation_probability"]
        >= 80
    ):

        notes.append(
            "High precipitation probability"
        )


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    if not notes:

        notes.append(
            "Normal weather conditions"
        )


    return "; ".join(notes)


# ============================================================
# ML RECOMMENDATION
# ============================================================

def recommend_sensitivity(
    weather
):

    if ML_MODEL is None:

        raise RuntimeError(
            "ML model is not loaded."
        )


    # ========================================================
    # FEATURES
    # ========================================================

    features = pd.DataFrame([{

        "wind_speed":
            weather["wind_speed"],

        "wind_gusts":
            weather["wind_gusts"],

        "rainfall":
            weather["rainfall"],

        "humidity":
            weather["humidity"],

        "temperature":
            weather["temperature"],

        "storm_flag":
            weather["storm_flag"],

        "precipitation_probability":
            weather["precipitation_probability"]

    }])


    # ========================================================
    # PREDICTION
    # ========================================================

    sensitivity = ML_MODEL.predict(
        features
    )[0]


    probabilities = (
        ML_MODEL.predict_proba(
            features
        )[0]
    )


    class_names = list(
        ML_MODEL.classes_
    )


    class_index = class_names.index(
        sensitivity
    )


    confidence = float(
        probabilities[class_index]
    )


    reason = explain_conditions(
        weather
    )


    # ========================================================
    # DEBUG
    # ========================================================

    print("\nMODEL INPUT")

    print(
        features.to_string(
            index=False
        )
    )


    print(
        f"\nMODEL OUTPUT: "
        f"{sensitivity}"
    )


    print(
        f"CONFIDENCE: "
        f"{confidence * 100:.2f}%"
    )


    return (

        sensitivity,

        round(
            confidence,
            3
        ),

        reason

    )


# ============================================================
# DATABASE LOGGING
# ============================================================

def log_recommendation(

    lat,

    lon,

    weather,

    sensitivity,

    confidence,

    reason

):

    conn = sqlite3.connect(
        DB_PATH
    )


    conn.execute(

        """
        INSERT INTO calibration_log (

            timestamp,

            lat,

            lon,

            wind_speed,

            wind_gusts,

            rainfall,

            precipitation,

            rain,

            showers,

            precipitation_probability,

            temperature,

            humidity,

            storm_flag,

            weather_code,

            sensitivity,

            confidence,

            reason

        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?
        )

        """,

        (

            datetime.now(
                timezone.utc
            ).isoformat(),

            lat,

            lon,

            weather["wind_speed"],

            weather["wind_gusts"],

            weather["rainfall"],

            weather["precipitation"],

            weather["rain"],

            weather["showers"],

            weather[
                "precipitation_probability"
            ],

            weather["temperature"],

            weather["humidity"],

            weather["storm_flag"],

            weather["weather_code"],

            sensitivity,

            confidence,

            reason

        )

    )


    conn.commit()

    conn.close()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return FileResponse(BASE_DIR / "index.html")


# ============================================================
# HEALTH CHECK
# ============================================================
# Hit this directly in a browser to instantly see whether the
# backend is awake and whether the model loaded — no weather
# API call involved, so it isolates backend-vs-weather issues.
# Also a good target for an uptime pinger (e.g. UptimeRobot /
# cron-job.org) to reduce Render free-tier cold starts.

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ML_MODEL is not None
    }


# ============================================================
# RECOMMEND
# ============================================================

@app.get("/recommend")
def recommend(

    lat: float = Query(
        19.076,
        ge=-90,
        le=90
    ),

    lon: float = Query(
        72.877,
        ge=-180,
        le=180
    )

):

    # ========================================================
    # WEATHER FETCH — Open-Meteo can be slow/unreachable, and
    # this should never crash the process, only this request.
    # ========================================================

    try:

        weather = fetch_weather(
            lat,
            lon
        )

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=f"Weather fetch failed: {e}"
        )


    # ========================================================
    # MODEL PREDICTION — surface a clear 503 (not a bare 500)
    # if the model failed to load, so the frontend/dashboard
    # can show something more useful than a generic error.
    # ========================================================

    try:

        sensitivity, confidence, reason = (
            recommend_sensitivity(
                weather
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=f"Model prediction failed: {e}"
        )


    try:

        log_recommendation(

            lat,

            lon,

            weather,

            sensitivity,

            confidence,

            reason

        )

    except Exception as e:

        # Logging to SQLite should never take the whole
        # recommendation down — just report it and continue.
        print(f"⚠ Failed to log recommendation: {e}")


    return {

        "location": {

            "lat": lat,

            "lon": lon

        },

        "weather": weather,

        "recommendation": {

            "sensitivity":
                sensitivity,

            "confidence":
                confidence,

            "reason":
                reason

        },

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat()

    }


# ============================================================
# HISTORY
# ============================================================

@app.get("/history")
def history(

    limit: int = Query(
        20,
        ge=1,
        le=100
    )

):

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row


    rows = conn.execute(

        """
        SELECT *

        FROM calibration_log

        ORDER BY id DESC

        LIMIT ?
        """,

        (limit,)

    ).fetchall()


    conn.close()


    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    if ML_MODEL is None:

        return {
            "error":
                "Model not loaded"
        }


    feature_names = [

        "wind_speed",

        "wind_gusts",

        "rainfall",

        "humidity",

        "temperature",

        "storm_flag",

        "precipitation_probability"

    ]


    importances = (
        ML_MODEL
        .feature_importances_
        .tolist()
    )


    return {

        "model_type":
            "RandomForestClassifier",

        "n_estimators":
            ML_MODEL.n_estimators,

        "classes":
            list(
                ML_MODEL.classes_
            ),

        "feature_importance": [

            {

                "feature":
                    name,

                "importance":
                    round(
                        float(importance),
                        4
                    )

            }

            for name, importance
            in sorted(

                zip(
                    feature_names,
                    importances
                ),

                key=lambda x:
                    -x[1]

            )

        ]

    }