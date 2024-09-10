import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import os

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="CNN_best_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label encoder to map the predictions to class names
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['car_horn', 'other', 'other', 'siren', 'trafficNoise'])

# Function to extract MFCC features from audio data
def extract_features_from_audio(audio_data, sample_rate=44100):
    # Normalize the audio data
    audio_data = audio_data / np.max(np.abs(audio_data))
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to prepare input for the TFLite model
def prepare_input_for_model(audio_data):
    # Extract MFCC features
    features = extract_features_from_audio(audio_data)
    # Prepare features for the model
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)  # Add channel dimension for CNN
    return features.astype(np.float32)

# Function to make predictions with the TFLite model
def predict_with_tflite_model(audio_data):
    input_data = prepare_input_for_model(audio_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Flask route to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read audio file
    audio_data, _ = sf.read(file)
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0]  # Use only the first channel if stereo

    # Make prediction
    prediction = predict_with_tflite_model(audio_data)
    predicted_class_index = np.argmax(prediction, axis=-1)
    predicted_class = label_encoder.inverse_transform(predicted_class_index)

    return jsonify({'prediction': predicted_class[0], 'confidence': float(np.max(prediction))})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
