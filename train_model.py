"""
train_model.py
Trains a real ML classifier (Random Forest) for sensor sensitivity recommendation.

WHY SYNTHETIC DATA:
No real historical false-alarm dataset exists for A-1's PIDS deployment (that data
lives inside their operations, not publicly available). So we generate a large,
randomized synthetic dataset spanning realistic weather conditions, and label each
sample using domain-expert rules (meteorological thresholds for wind/rain/humidity/
storm impact on infrared & vibration sensors, based on standard PIDS engineering
guidance). This is a standard "weak supervision" approach: use domain rules to
bootstrap labels, then train a real supervised model on top of them, so the model
learns the *interaction* between features (not just simple thresholds) and produces
genuine probability-based confidence scores instead of a hand-coded formula.

Run this once to produce model.joblib, which app.py loads at startup.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

np.random.seed(42)
N_SAMPLES = 8000


def domain_label(wind, rain, humidity, storm):
    """
    Domain-expert rule used ONLY to generate training labels (weak supervision).
    Same logic as the original engineering thresholds, kept here purely as the
    label source for the dataset -- NOT used at inference time anymore.
    """
    risk = 0.0
    if wind > 40:
        risk += 3
    elif wind > 20:
        risk += 1.5

    if rain > 10:
        risk += 2.5
    elif rain > 2:
        risk += 1

    if humidity > 85:
        risk += 1

    if storm:
        risk += 4

    if risk >= 5:
        return "Low"
    elif risk >= 2:
        return "Medium"
    return "High"


def generate_dataset(n):
    wind = np.random.gamma(shape=2.0, scale=10, size=n).clip(0, 90)
    rain = np.random.exponential(scale=4, size=n).clip(0, 60)
    humidity = np.random.normal(loc=65, scale=18, size=n).clip(10, 100)
    temperature = np.random.normal(loc=27, scale=6, size=n).clip(5, 45)
    storm = (np.random.rand(n) < 0.08).astype(int)  # ~8% storm events

    labels = [
        domain_label(w, r, h, s)
        for w, r, h, s in zip(wind, rain, humidity, storm)
    ]

    df = pd.DataFrame({
        "wind_speed": wind,
        "rainfall": rain,
        "humidity": humidity,
        "temperature": temperature,
        "storm_flag": storm,
        "sensitivity": labels,
    })
    return df


def main():
    print(f"Generating {N_SAMPLES} synthetic training samples...")
    df = generate_dataset(N_SAMPLES)
    print(df["sensitivity"].value_counts())

    X = df[["wind_speed", "rainfall", "humidity", "temperature", "storm_flag"]]
    y = df["sensitivity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds))

    print("Feature importances:")
    for feat, imp in zip(X.columns, model.feature_importances_):
        print(f"  {feat}: {imp:.3f}")

    joblib.dump(model, "model.joblib")
    print("\nSaved trained model to model.joblib")


if __name__ == "__main__":
    main()