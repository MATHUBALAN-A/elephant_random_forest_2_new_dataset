import joblib
import numpy as np
import traceback
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model ONLY (no scaler needed)
try:
    rf_model = joblib.load("thermal_random_forest_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    rf_model = None


# Home route
@app.route("/")
def home():
    return "Thermal Image Classifier API is running!"


# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    if rf_model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)

        # Validate JSON
        if "pixels" not in data:
            return jsonify({"error": "Missing 'pixels' field"}), 400

        pixel_data = data["pixels"]

        # Validate length
        if not isinstance(pixel_data, list) or len(pixel_data) != 768:
            return jsonify({
                "error": "Invalid input. Expected 768 pixel values."
            }), 400

        # Convert to numpy float array
        input_array = np.array(pixel_data, dtype=np.float32).reshape(1, -1)

        # Prediction
        prediction = rf_model.predict(input_array)[0]

        # Confidence
        probabilities = rf_model.predict_proba(input_array)
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
        print("====== ERROR OCCURRED ======")
        traceback.print_exc()
        print("============================")

        return jsonify({"error": str(e)}), 500


# Run server (Render uses this)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
