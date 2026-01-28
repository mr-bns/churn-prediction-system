import numpy as np

# -------------------------------
# Required Input Schema
# -------------------------------
REQUIRED_FIELDS = [
    "gender","senior_citizen","partner","dependents","tenure",
    "phone_service","multiple_lines","internet_service",
    "online_security","online_backup","device_protection","tech_support",
    "streaming_tv","streaming_movies","contract","paperless_billing",
    "payment_method","monthly_charges","total_charges"
]

# -------------------------------
# Field Mappings (Encoding Layer)
# -------------------------------
MAPPINGS = {
    "gender": {"Male": 1, "Female": 0},
    "partner": {"Yes": 1, "No": 0},
    "dependents": {"Yes": 1, "No": 0},
    "phone_service": {"Yes": 1, "No": 0},
    "multiple_lines": {"Yes": 1, "No": 0},
    "internet_service": {"Fiber optic": 2, "DSL": 1, "No": 0},
    "online_security": {"Yes": 1, "No": 0},
    "online_backup": {"Yes": 1, "No": 0},
    "device_protection": {"Yes": 1, "No": 0},
    "tech_support": {"Yes": 1, "No": 0},
    "streaming_tv": {"Yes": 1, "No": 0},
    "streaming_movies": {"Yes": 1, "No": 0},
    "contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "paperless_billing": {"Yes": 1, "No": 0},
    "payment_method": {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }
}

# -------------------------------
# Validation Layer
# -------------------------------
def validate_input(data):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing required fields: {missing}"
    return True, None

# -------------------------------
# Consistency Check
# -------------------------------
def consistency_check(data):
    if data["tenure"] < 0:
        return False, "Tenure cannot be negative"
    if data["monthly_charges"] < 0:
        return False, "Monthly charges cannot be negative"
    if data["total_charges"] < 0:
        return False, "Total charges cannot be negative"
    return True, None

# -------------------------------
# Encoding + Feature Builder
# -------------------------------
def encode_input(data):
    encoded = []

    encoded.append(MAPPINGS["gender"][data["gender"]])
    encoded.append(int(data["senior_citizen"]))
    encoded.append(MAPPINGS["partner"][data["partner"]])
    encoded.append(MAPPINGS["dependents"][data["dependents"]])
    encoded.append(int(data["tenure"]))
    encoded.append(MAPPINGS["phone_service"][data["phone_service"]])
    encoded.append(MAPPINGS["multiple_lines"][data["multiple_lines"]])
    encoded.append(MAPPINGS["internet_service"][data["internet_service"]])
    encoded.append(MAPPINGS["online_security"][data["online_security"]])
    encoded.append(MAPPINGS["online_backup"][data["online_backup"]])
    encoded.append(MAPPINGS["device_protection"][data["device_protection"]])
    encoded.append(MAPPINGS["tech_support"][data["tech_support"]])
    encoded.append(MAPPINGS["streaming_tv"][data["streaming_tv"]])
    encoded.append(MAPPINGS["streaming_movies"][data["streaming_movies"]])
    encoded.append(MAPPINGS["contract"][data["contract"]])
    encoded.append(MAPPINGS["paperless_billing"][data["paperless_billing"]])
    encoded.append(MAPPINGS["payment_method"][data["payment_method"]])
    encoded.append(float(data["monthly_charges"]))
    encoded.append(float(data["total_charges"]))

    return np.array(encoded).reshape(1, -1)
