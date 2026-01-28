from flask import Flask, request, jsonify
import numpy as np
from deployment.model_loader import load_model
from deployment.config import Config

app = Flask(__name__)

# Load trained model
model = load_model()

# ------------------------------
# Home Route (Landing Endpoint)
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

    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request"}), 400

    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "churn_prediction": int(prediction),
        "churn_probability": float(round(probability, 4))
    })

# ------------------------------
# App Runner
# ------------------------------
if __name__ == "__main__":
    app.run()
