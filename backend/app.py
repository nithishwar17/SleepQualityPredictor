# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import os
from model_utils import preprocess_input_for_model

MODEL_PATH = "model.joblib"
app = Flask(__name__)
CORS(app)  # Enable CORS for your app

# Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the training script first.")
model = joblib.load(MODEL_PATH)

# Define labels and suggestions
LABEL_MAP = {2: "Good", 1: "Average", 0: "Poor"}
SUGGESTIONS = {
    0: [
        "Try to establish a consistent sleep schedule, even on weekends.",
        "Avoid alcohol and heavy meals close to bedtime.",
        "Consider reducing caffeine intake, especially in the afternoon and evening.",
        "Create a relaxing bedtime routine to help you wind down."
    ],
    1: [
        "Aim for consistency in your bedtime and wake-up time.",
        "Ensure your sleep environment is dark, quiet, and cool.",
        "Regular exercise can improve sleep quality, but avoid intense workouts close to bed."
    ],
    2: [
        "Excellent! Keep maintaining your healthy habits and consistent sleep routine."
    ]
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Use our updated preprocessing function
        X = preprocess_input_for_model(data)
        
        # Make prediction
        prediction_class = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]

        # Format the response
        label = LABEL_MAP.get(prediction_class, "Unknown")
        tips = SUGGESTIONS.get(prediction_class, [])
        
        response = {
            "prediction_label": label,
            "prediction_class": prediction_class,
            "probabilities": {
                "Poor": float(probabilities[0]),
                "Average": float(probabilities[1]),
                "Good": float(probabilities[2])
            },
            "tips": tips
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)