
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
try:
    rf_model = joblib.load('thermal_random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    rf_model = None
    scaler = None

@app.route('/')
def home():
    return "Thermal Image Classifier API is running!"

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if rf_model is None or scaler is None:
        return jsonify({"error": "Model or scaler not loaded."}), 500

    try:
        data = request.get_json(force=True)
        pixel_data = data['pixels']

        # Ensure pixel_data is a list of 768 values
        if not isinstance(pixel_data, list) or len(pixel_data) != 768:
            return jsonify({"error": "Invalid pixel data format. Expected a list of 768 pixel values."}), 400

        # Convert to numpy array and reshape for the scaler and model
        # Assuming X originally had columns 'px_0' to 'px_767'
        input_data = pd.DataFrame([pixel_data], columns=[f'px_{i}' for i in range(768)])

        # Preprocess the data using the loaded scaler
        scaled_input = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = rf_model.predict(scaled_input)
        predicted_label = int(prediction[0]) # Convert numpy int to Python int

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
