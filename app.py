from flask import Flask, request, jsonify
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
model_name = bundle["model_name"]
features = bundle["features"]

labels = {0: "Real", 1: "Fake"}


@app.get("/")
def home():
    return jsonify({"message": "Fake News Detection API is running"})


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": model_name
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a valid JSON object"}), 400

    missing = [feature for feature in features if feature not in data]

    if missing:
        return jsonify({"error": "Missing features", "features": missing}), 400

    try:
        values = {feature: float(data[feature]) for feature in features}
    except (TypeError, ValueError):
        return jsonify({"error": "All features must be numeric"}), 400

    input_data = pd.DataFrame([values], columns=features)
    prediction = int(model.predict(input_data)[0])

    return jsonify({
        "prediction": labels[prediction],
        "label": prediction,
        "model": model_name
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)