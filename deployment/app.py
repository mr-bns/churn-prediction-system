from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

# Package-relative imports (correct architecture)
from .model_loader import load_model
from .input_processor import (
    validate_input,
    consistency_check,
    encode_input
)
from .csv_ingestion import (
    validate_csv_schema,
    clean_dataframe,
    convert_to_json
)

app = Flask(__name__)

# ------------------------------
# Load ML Model
# ------------------------------
model = load_model()

# ------------------------------
# Root Landing Endpoint
# ------------------------------
@app.route("/")
def home():
    return jsonify({
        "service": "Churn Prediction API",
        "status": "running",
        "message": "Production ML API is live",
        "endpoints": {
            "/health": "Health check endpoint",
            "/predict": "POST single prediction",
            "/predict-batch": "POST batch prediction (JSON dataset)",
            "/convert-csv": "POST CSV ingestion (CSV → JSON converter)"
        }
    })

# ------------------------------
# Health Check Endpoint
# ------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "API running",
        "model_loaded": True
    })

# ------------------------------
# Single Prediction Endpoint
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Schema validation
    valid, error = validate_input(data)
    if not valid:
        return jsonify({"error": error}), 400

    # Consistency validation
    consistent, error = consistency_check(data)
    if not consistent:
        return jsonify({"error": error}), 400

    # Encoding
    try:
        features = encode_input(data)
    except Exception as e:
        return jsonify({"error": "Encoding failed", "details": str(e)}), 400

    # Prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4)),
        "status": "success",
        "input_verified": True
    })

# ------------------------------
# JSON Batch Prediction Endpoint
# ------------------------------
@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    payload = request.json

    if not payload or "dataset" not in payload:
        return jsonify({
            "status": "failed",
            "error": "Missing 'dataset' field"
        }), 400

    dataset = payload["dataset"]

    if not isinstance(dataset, list) or len(dataset) == 0:
        return jsonify({
            "status": "failed",
            "error": "Dataset must be a non-empty list"
        }), 400

    # -------- Global Validation Phase --------
    for idx, record in enumerate(dataset):
        valid, error = validate_input(record)
        if not valid:
            return jsonify({
                "status": "failed",
                "validated": False,
                "error": f"Schema error in record {idx}: {error}"
            }), 400

        consistent, error = consistency_check(record)
        if not consistent:
            return jsonify({
                "status": "failed",
                "validated": False,
                "error": f"Consistency error in record {idx}: {error}"
            }), 400

    # -------- Encoding Phase --------
    try:
        feature_matrix = []
        for record in dataset:
            encoded = encode_input(record)
            feature_matrix.append(encoded[0])

        X = np.array(feature_matrix)
    except Exception as e:
        return jsonify({
            "status": "failed",
            "validated": False,
            "error": "Encoding failed",
            "details": str(e)
        }), 400

    # -------- Prediction Phase --------
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # -------- Output Phase --------
    results = []
    for i in range(len(preds)):
        results.append({
            "index": i,
            "churn_prediction": int(preds[i]),
            "churn_probability": round(float(probs[i]), 4)
        })

    return jsonify({
        "status": "success",
        "validated": True,
        "total_records": len(dataset),
        "predictions": results
    })

# ------------------------------
# CSV Ingestion Endpoint
# ------------------------------
@app.route("/convert-csv", methods=["POST"])
def convert_csv():
    if "file" not in request.files:
        return jsonify({
            "status": "failed",
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({
            "status": "failed",
            "error": "Only CSV files are supported"
        }), 400

    # Read CSV
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": "Invalid CSV file",
            "details": str(e)
        }), 400

    # Schema validation
    valid, error = validate_csv_schema(df)
    if not valid:
        return jsonify({
            "status": "failed",
            "validated": False,
            "error": error
        }), 400

    # Cleaning
    df_clean = clean_dataframe(df)

    # Conversion
    json_dataset = convert_to_json(df_clean)

    return jsonify({
        "status": "success",
        "validated": True,
        "total_records": len(json_dataset),
        "converted_dataset": json_dataset
    })

# ------------------------------
# App Runner
# ------------------------------
if __name__ == "__main__":
    app.run()
