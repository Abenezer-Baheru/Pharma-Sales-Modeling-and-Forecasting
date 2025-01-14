from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Load the scaler
scaler = joblib.load('model.pkl')

# Define the API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1)
    
    # Preprocess the input data
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled.reshape((1, input_data_scaled.shape[1], 1))
    
    # Make predictions
    prediction = model.predict(input_data_scaled)
    
    # Inverse transform the prediction
    prediction_inv = scaler.inverse_transform(prediction)
    
    # Ensure the prediction is non-negative
    prediction_inv[prediction_inv < 0] = 0
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction_inv.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
