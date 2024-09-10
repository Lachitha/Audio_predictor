import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import queue
import threading
from flask import Flask, jsonify
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

# Global variables for managing real-time audio
duration = 5 # Duration for prediction intervals (adjustable)
fs = 44100  # Sampling rate
audio_queue = queue.Queue()  # Queue to store recorded audio chunks

# Directory to save audio files
save_directory = 'recorded_audio'
os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist
saved_files_count = 0  # Counter for saved audio files
max_saved_files = 3  # Maximum number of files to save

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

# Function to record audio continuously in 2-second chunks
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Function to save audio data to a file
def save_audio_file(audio_data, file_name):
    global saved_files_count
    file_path = os.path.join(save_directory, file_name)
    sf.write(file_path, audio_data, fs)  # Save audio data in .wav format
    saved_files_count += 1
    print(f'Saved file {file_name}.')

# Function to start recording from the microphone and predict every 2 seconds
def start_recording():
    global saved_files_count
    with sd.InputStream(samplerate=fs, channels=1, callback=audio_callback):
        while True:
            audio_data = []

            # Accumulate chunks of audio data until we have enough for the desired duration (2 seconds)
            while len(audio_data) < int(fs * duration):
                chunk = audio_queue.get().flatten()  # Get a chunk of audio from the queue
                audio_data.extend(chunk)  # Append the chunk to the audio_data list

            # Ensure the audio_data has exactly fs * duration samples
            audio_data = np.array(audio_data[:int(fs * duration)])

            # Save the first three audio files
            if saved_files_count < max_saved_files:
                file_name = f'audio_file_{saved_files_count + 1}.wav'
                save_audio_file(audio_data, file_name)

            # Make prediction using the TFLite model
            prediction = predict_with_tflite_model(audio_data)
            predicted_class_index = np.argmax(prediction, axis=-1)
            predicted_class = label_encoder.inverse_transform(predicted_class_index)

            # Print the prediction
            print(f'Prediction: {predicted_class[0]}, Confidence: {np.max(prediction)}')

# Flask route to start the real-time prediction process
@app.route('/start', methods=['GET'])
def start():
    # Start the real-time audio recording in a separate thread
    threading.Thread(target=start_recording, daemon=True).start()
    return jsonify({'message': 'Real-time audio prediction started, providing predictions every 2 seconds'})

# Flask route to stop the real-time prediction (optional)
@app.route('/stop', methods=['GET'])
def stop():
    # Logic to stop the real-time audio prediction (if needed)
    return jsonify({'message': 'Real-time audio prediction stopped'})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
