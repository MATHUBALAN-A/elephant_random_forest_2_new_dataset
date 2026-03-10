import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
try:
    rf_model = joblib.load("thermal_random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")
except Exception as e:
    print("Error loading model or scaler:", e)
    rf_model = None
    scaler = None


# Home route
@app.route("/")
def home():
    return "Thermal Image Classifier API is running!"


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    if rf_model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        data = request.get_json(force=True)

        # Check JSON structure
        if "pixels" not in data:
            return jsonify({"error": "Missing 'pixels' field"}), 400

        pixel_data = data["pixels"]

        # Validate pixel length
        if not isinstance(pixel_data, list) or len(pixel_data) != 768:
            return jsonify({
                "error": "Invalid input. Expected a list of 768 pixel values."
            }), 400

        # Convert to numpy array
        input_array = np.array(pixel_data).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Prediction
        prediction = rf_model.predict(scaled_input)[0]

        # Confidence score
        probabilities = rf_model.predict_proba(scaled_input)
        confidence = float(np.max(probabilities))

        # Confidence threshold
        threshold = 0.85
        if confidence < threshold:
            return jsonify({
                "prediction": -1,
                "confidence": confidence,
                "message": "Low confidence prediction"
            })

        return jsonify({
            "prediction": int(prediction),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
