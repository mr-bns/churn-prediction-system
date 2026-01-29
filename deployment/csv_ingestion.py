import pandas as pd
from deployment.input_processor import REQUIRED_FIELDS

def validate_csv_schema(df):
    missing = [col for col in REQUIRED_FIELDS if col not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, None

def clean_dataframe(df):
    # Standard cleaning rules
    df = df.copy()
    df = df.dropna()
    return df

def convert_to_json(df):
    records = df.to_dict(orient="records")
    return records
