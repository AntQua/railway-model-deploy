import json
import os
import pickle

import pandas as pd
from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


app = Flask(__name__)


# =========================
# Database
# =========================
DB = SqliteDatabase("predictions.db")


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.connect(reuse_if_open=True)
DB.create_tables([Prediction], safe=True)


# =========================
# Load serialized artifacts
# =========================
with open("columns.json", "r") as fh:
    columns = json.load(fh)

with open("dtypes.pickle", "rb") as fh:
    dtypes = pickle.load(fh)

with open("pipeline.pickle", "rb") as fh:
    pipeline = pickle.load(fh)


# =========================
# Helpers
# =========================
def is_valid_observation(observation):
    if not isinstance(observation, dict):
        return False

    expected_keys = {"age", "education", "hours-per-week", "native-country"}
    if set(observation.keys()) != expected_keys:
        return False

    if not isinstance(observation["age"], int):
        return False

    if not isinstance(observation["education"], str):
        return False

    if not isinstance(observation["hours-per-week"], int):
        return False

    if not isinstance(observation["native-country"], str):
        return False

    return True


# =========================
# /predict
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid request payload!"}), 400

    if "id" not in payload or "observation" not in payload:
        return jsonify({"error": "Invalid request payload!"}), 400

    _id = payload["id"]
    observation = payload["observation"]

    if not is_valid_observation(observation):
        return jsonify({"error": "Observation is invalid!"}), 400

    try:
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
        proba = float(pipeline.predict_proba(obs)[0, 1])
    except Exception:
        return jsonify({"error": "Observation is invalid!"}), 400

    response = {"proba": proba}

    p = Prediction(
        observation_id=_id,
        observation=json.dumps(observation),
        proba=proba,
        true_class=None
    )

    try:
        p.save()
    except IntegrityError:
        existing = Prediction.get(Prediction.observation_id == _id)
        response["error"] = f'Observation ID: "{_id}" already exists'
        response["proba"] = existing.proba

    return jsonify(response)


# =========================
# /update
# =========================
@app.route("/update", methods=["POST"])
def update():
    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid request payload!"}), 400

    if "id" not in payload or "true_class" not in payload:
        return jsonify({"error": "Invalid request payload!"}), 400

    _id = payload["id"]
    true_class = payload["true_class"]

    if true_class not in [0, 1]:
        return jsonify({"error": "true_class is invalid!"}), 400

    try:
        p = Prediction.get(Prediction.observation_id == _id)
        p.true_class = true_class
        p.save()

        result = model_to_dict(p)
        result["id"] = result["id"]
        result["observation_id"] = int(result["observation_id"])
        result["true_class"] = int(result["true_class"]) if result["true_class"] is not None else None

        return jsonify(result)

    except Prediction.DoesNotExist:
        return jsonify({"error": f'Observation ID: "{_id}" does not exist'}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)