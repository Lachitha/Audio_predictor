from flask import Flask, jsonify
from flask_sockets import Sockets
import numpy as np
import tensorflow as tf
import librosa
import queue

app = Flask(__name__)
sockets = Sockets(app)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="CNN_best_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Label encoder to map predictions to class names
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

@sockets.route('/audio_stream')
def audio_stream(ws):
    """Handle incoming WebSocket connections for audio streaming."""
    audio_buffer = bytearray()

    while not ws.closed:
        message = ws.receive()
        if message:
            # Append received audio data to the buffer
            audio_buffer.extend(message)

            # If enough data is collected (e.g., 5 seconds worth), make a prediction
            if len(audio_buffer) >= 44100 * 5 * 2:  # 5 seconds of audio (2 bytes per sample for float32)
                audio_data = np.frombuffer(audio_buffer[:44100 * 5 * 2], dtype=np.float32)

                # Make a prediction using the TFLite model
                prediction = predict_with_tflite_model(audio_data)
                predicted_class_index = np.argmax(prediction, axis=-1)
                predicted_class = label_encoder.inverse_transform(predicted_class_index)

                # Send the prediction back to the client
                ws.send(f'Prediction: {predicted_class[0]}, Confidence: {np.max(prediction)}')

                # Clear the buffer for the next batch
                audio_buffer = audio_buffer[44100 * 5 * 2:]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
