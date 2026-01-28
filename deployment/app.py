from flask import Flask, request, jsonify
import numpy as np
from deployment.model_loader import load_model
from deployment.input_processor import validate_input, consistency_check, encode_input

app = Flask(__name__)

# Load ML model
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
            "/predict": "POST endpoint for churn prediction"
        }
    })

# ------------------------------
# Health Check
# ------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "API running",
        "model_loaded": True
    })

# ------------------------------
# Prediction Endpoint
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
# App Runner
# ------------------------------
if __name__ == "__main__":
    app.run()
