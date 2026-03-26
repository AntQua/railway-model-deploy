import json
import os
import pickle

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)


# =========================
# Load serialized artifacts
# =========================
with open("columns.json", "r") as fh:
    columns = json.load(fh)

with open("dtypes.pickle", "rb") as fh:
    dtypes = pickle.load(fh)

with open("pipeline.pickle", "rb") as fh:
    pipeline = joblib.load(fh)


# =========================
# Expected schema
# =========================
REQUIRED_FIELDS = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

VALID_CATEGORIES = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "race": [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ],
    "sex": [
        "Female",
        "Male",
    ],
}


# =========================
# Helpers
# =========================
def error_response(observation_id, message):
    return jsonify(
        {
            "observation_id": observation_id,
            "error": message,
        }
    ), 200


def validate_payload(payload):
    if not isinstance(payload, dict):
        return None, "Invalid request format"

    if "observation_id" not in payload:
        return None, "Missing observation_id"

    observation_id = payload["observation_id"]

    if "data" not in payload:
        return observation_id, "Missing data"

    data = payload["data"]

    if not isinstance(data, dict):
        return observation_id, "Invalid data format"

    for field in REQUIRED_FIELDS:
        if field not in data:
            return observation_id, f"Missing field: {field}"

    for field in data:
        if field not in REQUIRED_FIELDS:
            return observation_id, f"Unexpected field: {field}"

    for col, valid_values in VALID_CATEGORIES.items():
        if data[col] not in valid_values:
            return observation_id, f"Invalid value for {col}: {data[col]}"

    try:
        age = float(data["age"])
        capital_gain = float(data["capital-gain"])
        capital_loss = float(data["capital-loss"])
        hours_per_week = float(data["hours-per-week"])
    except Exception:
        return observation_id, "Invalid numeric values"

    if not (0 <= age <= 100):
        return observation_id, f"Invalid value for age: {data['age']}"

    if capital_gain < 0:
        return observation_id, f"Invalid value for capital-gain: {data['capital-gain']}"

    if capital_loss < 0:
        return observation_id, f"Invalid value for capital-loss: {data['capital-loss']}"

    if not (0 <= hours_per_week <= 168):
        return observation_id, f"Invalid value for hours-per-week: {data['hours-per-week']}"

    return observation_id, None


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    observation_id, validation_error = validate_payload(payload)
    if validation_error is not None:
        return error_response(observation_id, validation_error)

    data = payload["data"]

    try:
        X = pd.DataFrame([[data[col] for col in columns]], columns=columns).astype(dtypes)
        pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0, 1]
    except Exception as e:
        return error_response(observation_id, f"Prediction failed: {str(e)}")

    return jsonify(
        {
            "observation_id": observation_id,
            "prediction": bool(pred),
            "probability": float(proba),
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))