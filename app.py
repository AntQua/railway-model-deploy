import json
import os
import pickle
import sqlite3

import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

DB_PATH = os.path.join(BASE_DIR, "database.db")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.json")
DTYPES_PATH = os.path.join(BASE_DIR, "dtypes.pickle")
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pickle")


def load_columns():
    with open(COLUMNS_PATH, "r") as f:
        return json.load(f)


def load_dtypes():
    with open(DTYPES_PATH, "rb") as f:
        return pickle.load(f)


def load_pipeline():
    with open(PIPELINE_PATH, "rb") as f:
        return pickle.load(f)


COLUMNS = load_columns()
DTYPES = load_dtypes()
PIPELINE = load_pipeline()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            observation TEXT NOT NULL,
            proba REAL NOT NULL,
            true_class TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def observation_is_valid(observation):
    if not isinstance(observation, dict):
        return False

    required_fields = ["age", "education", "hours-per-week", "native-country"]

    if set(observation.keys()) != set(required_fields):
        return False

    # age must be integer-like
    if not isinstance(observation["age"], int):
        return False

    # education must be string
    if not isinstance(observation["education"], str):
        return False

    # hours-per-week must be integer-like
    if not isinstance(observation["hours-per-week"], int):
        return False

    # native-country must be string
    if not isinstance(observation["native-country"], str):
        return False

    return True


def make_dataframe_from_observation(observation):
    df = pd.DataFrame([observation], columns=COLUMNS)

    # enforce training dtypes
    for col in COLUMNS:
        df[col] = df[col].astype(DTYPES[col])

    return df


def get_existing_prediction(obs_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, observation, proba, true_class FROM predictions WHERE id = ?",
        (str(obs_id),)
    )
    row = cursor.fetchone()
    conn.close()

    return row


def insert_prediction(obs_id, observation, proba):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (id, observation, proba, true_class)
        VALUES (?, ?, ?, ?)
        """,
        (str(obs_id), json.dumps(observation), float(proba), None)
    )

    conn.commit()
    conn.close()


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid request payload!"}), 400

    if "id" not in payload or "observation" not in payload:
        return jsonify({"error": "Invalid request payload!"}), 400

    obs_id = payload["id"]
    observation = payload["observation"]

    if not observation_is_valid(observation):
        return jsonify({"error": "Observation is invalid!"}), 400

    try:
        observation_df = make_dataframe_from_observation(observation)
        proba = PIPELINE.predict_proba(observation_df)[0, 1]
    except Exception:
        return jsonify({"error": "Observation is invalid!"}), 400

    existing_row = get_existing_prediction(obs_id)

    if existing_row is not None:
        return jsonify({
            "error": f'Observation ID: "{obs_id}" already exists',
            "proba": existing_row["proba"]
        }), 200

    insert_prediction(obs_id, observation, proba)

    return jsonify({"proba": proba}), 200

@app.route("/update", methods=["POST"])
def update():
    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid request payload!"}), 400

    if "id" not in payload or "true_class" not in payload:
        return jsonify({"error": "Invalid request payload!"}), 400

    obs_id = payload["id"]
    true_class = payload["true_class"]

    if true_class not in [0, 1]:
        return jsonify({"error": "true_class is invalid!"}), 400

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, observation, proba, true_class FROM predictions WHERE id = ?",
        (str(obs_id),)
    )
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return jsonify({
            "error": f'Observation ID: "{obs_id}" does not exist'
        }), 404

    cursor.execute(
        "UPDATE predictions SET true_class = ? WHERE id = ?",
        (true_class, str(obs_id))
    )
    conn.commit()

    cursor.execute(
        "SELECT id, observation, proba, true_class FROM predictions WHERE id = ?",
        (str(obs_id),)
    )
    updated_row = cursor.fetchone()
    conn.close()

    return jsonify({
        "id": int(updated_row["id"]),
        "observation": updated_row["observation"],
        "proba": updated_row["proba"],
        "true_class": int(updated_row["true_class"]) if updated_row["true_class"] is not None else None
    }), 200

if __name__ == "__main__":
    initialize_db()
    app.run(host="0.0.0.0", port=5000, debug=True)