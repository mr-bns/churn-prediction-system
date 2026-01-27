import pickle

MODEL_PATH = "D:\\churn-prediction\\model\\churn_model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model
