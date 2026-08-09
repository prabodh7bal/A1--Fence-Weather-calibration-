"""
VIGIL PIDS
Weather-Based Sensor Calibration ML Model

Random Forest classifier trained on synthetic weather scenarios.

IMPORTANT:
The synthetic labels are engineering/domain-rule labels used to bootstrap
the ML model. The model learns interactions between weather features and
produces probability-based confidence scores.

This is NOT claimed as real-world sensor accuracy.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import joblib


# ============================================================
# CONFIGURATION
# ============================================================

np.random.seed(42)

N_SAMPLES = 12000


# ============================================================
# DOMAIN LABELING
# ============================================================

def domain_label(
    wind_speed,
    wind_gusts,
    rainfall,
    humidity,
    storm_flag,
    precipitation_probability
):
    """
    Generate engineering-inspired sensitivity labels.

    High:
        Normal / relatively stable weather

    Medium:
        Weather beginning to increase false-alarm/environmental risk

    Low:
        Severe weather where sensor sensitivity should be reduced
    """

    risk = 0.0

    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    if wind_speed >= 45:
        risk += 3.0

    elif wind_speed >= 30:
        risk += 2.0

    elif wind_speed >= 20:
        risk += 1.0


    # --------------------------------------------------------
    # WIND GUSTS
    # --------------------------------------------------------

    if wind_gusts >= 60:
        risk += 3.0

    elif wind_gusts >= 45:
        risk += 2.0

    elif wind_gusts >= 30:
        risk += 1.0


    # --------------------------------------------------------
    # RAINFALL
    # --------------------------------------------------------

    if rainfall >= 20:
        risk += 3.5

    elif rainfall >= 10:
        risk += 2.5

    elif rainfall >= 5:
        risk += 1.8

    elif rainfall >= 1:
        risk += 1.0

    elif rainfall >= 0.2:
        risk += 0.5


    # --------------------------------------------------------
    # HUMIDITY
    # --------------------------------------------------------

    if humidity >= 90:
        risk += 1.5

    elif humidity >= 85:
        risk += 0.8


    # --------------------------------------------------------
    # PRECIPITATION PROBABILITY
    # --------------------------------------------------------

    if precipitation_probability >= 80:
        risk += 1.0

    elif precipitation_probability >= 60:
        risk += 0.5


    # --------------------------------------------------------
    # THUNDERSTORM
    # --------------------------------------------------------

    if storm_flag:
        risk += 5.0


    # --------------------------------------------------------
    # FINAL CLASS
    # --------------------------------------------------------

    if risk >= 6:
        return "Low"

    elif risk >= 2:
        return "Medium"

    else:
        return "High"


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(n):

    # Wind speed
    wind_speed = np.random.gamma(
        shape=2.0,
        scale=10,
        size=n
    ).clip(0, 90)


    # Wind gusts are generally greater than wind speed
    gust_extra = np.random.gamma(
        shape=2.0,
        scale=6,
        size=n
    )

    wind_gusts = (
        wind_speed + gust_extra
    ).clip(0, 120)


    # Rainfall
    rainfall = np.random.exponential(
        scale=4,
        size=n
    ).clip(0, 60)


    # Humidity
    humidity = np.random.normal(
        loc=68,
        scale=18,
        size=n
    ).clip(10, 100)


    # Temperature
    temperature = np.random.normal(
        loc=27,
        scale=6,
        size=n
    ).clip(5, 45)


    # Storm probability
    storm_flag = (
        np.random.rand(n) < 0.08
    ).astype(int)


    # Precipitation probability
    precipitation_probability = (
        np.random.beta(2, 3, size=n) * 100
    ).clip(0, 100)


    # Make storms more likely to have precipitation
    precipitation_probability = np.where(
        storm_flag == 1,
        np.maximum(
            precipitation_probability,
            np.random.uniform(70, 100, n)
        ),
        precipitation_probability
    )


    # Generate labels
    labels = [
        domain_label(
            w,
            g,
            r,
            h,
            s,
            p
        )

        for w, g, r, h, s, p in zip(
            wind_speed,
            wind_gusts,
            rainfall,
            humidity,
            storm_flag,
            precipitation_probability
        )
    ]


    df = pd.DataFrame({

        "wind_speed": wind_speed,

        "wind_gusts": wind_gusts,

        "rainfall": rainfall,

        "humidity": humidity,

        "temperature": temperature,

        "storm_flag": storm_flag,

        "precipitation_probability":
            precipitation_probability,

        "sensitivity": labels

    })


    return df


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 60)
    print("VIGIL PIDS — MODEL TRAINING")
    print("=" * 60)

    print(
        f"\nGenerating {N_SAMPLES:,} synthetic weather samples..."
    )


    df = generate_dataset(
        N_SAMPLES
    )


    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nClass distribution:")

    print(
        df["sensitivity"]
        .value_counts()
    )


    print("\nClass percentages:")

    print(
        (
            df["sensitivity"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    feature_columns = [

        "wind_speed",

        "wind_gusts",

        "rainfall",

        "humidity",

        "temperature",

        "storm_flag",

        "precipitation_probability"

    ]


    X = df[
        feature_columns
    ]

    y = df[
        "sensitivity"
    ]


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("\nTraining Random Forest...")


    model = RandomForestClassifier(

        n_estimators=250,

        max_depth=10,

        min_samples_leaf=3,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1

    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        f"\nTest Accuracy: {accuracy:.4f}"
    )


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nFeature Importance:")

    for feature, importance in zip(
        feature_columns,
        model.feature_importances_
    ):

        print(
            f"  {feature:<28}"
            f"{importance:.4f}"
        )


    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)


    test_cases = [

        {
            "name": "Normal weather",

            "wind_speed": 10,
            "wind_gusts": 15,
            "rainfall": 0,
            "humidity": 60,
            "temperature": 28,
            "storm_flag": 0,
            "precipitation_probability": 10
        },

        {
            "name": "Light drizzle",

            "wind_speed": 16,
            "wind_gusts": 25,
            "rainfall": 0.5,
            "humidity": 78,
            "temperature": 28,
            "storm_flag": 0,
            "precipitation_probability": 60
        },

        {
            "name": "Moderate rain",

            "wind_speed": 25,
            "wind_gusts": 40,
            "rainfall": 5,
            "humidity": 85,
            "temperature": 26,
            "storm_flag": 0,
            "precipitation_probability": 85
        },

        {
            "name": "Heavy rain",

            "wind_speed": 35,
            "wind_gusts": 55,
            "rainfall": 15,
            "humidity": 92,
            "temperature": 25,
            "storm_flag": 0,
            "precipitation_probability": 95
        },

        {
            "name": "Thunderstorm",

            "wind_speed": 30,
            "wind_gusts": 65,
            "rainfall": 12,
            "humidity": 95,
            "temperature": 25,
            "storm_flag": 1,
            "precipitation_probability": 100
        },

        {
            "name": "Extreme storm",

            "wind_speed": 55,
            "wind_gusts": 90,
            "rainfall": 30,
            "humidity": 98,
            "temperature": 24,
            "storm_flag": 1,
            "precipitation_probability": 100
        }

    ]


    sanity_df = pd.DataFrame(
        test_cases
    )


    sanity_features = sanity_df[
        feature_columns
    ]


    sanity_predictions = model.predict(
        sanity_features
    )


    sanity_probabilities = (
        model.predict_proba(
            sanity_features
        )
    )


    for i, case in enumerate(
        test_cases
    ):

        prediction = sanity_predictions[i]

        class_index = list(
            model.classes_
        ).index(
            prediction
        )

        confidence = (
            sanity_probabilities[i][class_index]
        )


        print(

            f"{case['name']:<20}"
            f"→ {prediction:<7}"
            f"({confidence * 100:.1f}%)"

        )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    joblib.dump(
        model,
        "model.joblib"
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL SAVED → model.joblib"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()