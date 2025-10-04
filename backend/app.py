# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from model_utils import preprocess_input_for_model
# We are NOT including the database parts in this version
# as per your previous request.

MODEL_PATH = "model.joblib"
app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_PATH)

LABEL_MAP = {2: "Good", 1: "Average", 0: "Poor"}

# --- ADDING THIS SUGGESTIONS DICTIONARY BACK ---
SUGGESTIONS = {
    0: [ # Poor
        "Try to establish a consistent sleep schedule, even on weekends.",
        "Avoid alcohol and heavy meals 2-3 hours before bedtime.",
        "Consider reducing caffeine intake, especially after 2 PM."
    ],
    1: [ # Average
        "Ensure your sleep environment is dark, quiet, and cool.",
        "Regular exercise can improve sleep quality, but avoid intense workouts close to bed.",
        "Try a relaxing activity like reading or meditation before sleep."
    ],
    2: [ # Good
        "Excellent! Keep maintaining your healthy habits and consistent sleep routine.",
        "Your routine is working well. Continue to prioritize your sleep."
    ]
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Add a print statement here to see the data the backend receives
        print("Received data for prediction:", data)
        
        X = preprocess_input_for_model(data)
        
        prediction_class = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        label = LABEL_MAP.get(prediction_class, "Unknown")
        tips = SUGGESTIONS.get(prediction_class, []) # Get tips based on prediction
        
        response = {
            "prediction_label": label,
            "probabilities": {
                "Poor": float(probabilities[0]),
                "Average": float(probabilities[1]),
                "Good": float(probabilities[2])
            },
            "tips": tips # --- ADDING TIPS TO THE RESPONSE ---
        }
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
