# Churn Prediction API

## Endpoints

### Health Check
GET /health

### Prediction
POST /predict

Request JSON:
{
  "features": [f1, f2, f3, ..., fn]
}

Response:
{
  "churn_prediction": 1,
  "churn_probability": 0.8423
}
