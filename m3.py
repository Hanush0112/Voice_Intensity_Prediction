import os
import opensmile
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the saved model
model = load_model(r"nervous_intensity_modelINTE.h5")

# Load the dataset to fit the scaler and encoder (as you did during training)
nervous_df = pd.read_csv(r"merged_with_emotion_intensity3.csv")

# Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(nervous_df.drop(columns=['emotion_label', 'intensity_label']))

# Fit the label encoder on the intensity labels
label_encoder = LabelEncoder()
label_encoder.fit(nervous_df['intensity_label'])

# Initialize the opensmile feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,  # Use 'emobase' for emotion-related features
    feature_level=opensmile.FeatureLevel.Functionals  # Functionals gives summarized features
)

# Function to list all files in the local directory
def list_wav_files_in_local_directory(directory):
    """
    List all WAV files in a local directory.
    """
    try:
        # Filter the list to only include WAV files
        wav_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]
        return wav_files
    except Exception as e:
        print(f"Error accessing local directory: {e}")
        return []

# Function to extract features from audio using OpenSMILE
def extract_opensmile_features(audio_path):
    # Extract features using opensmile
    features = smile.process_file(audio_path)
    
    # Convert the DataFrame to a numpy array
    features_values = features.values.flatten()
    
    return features_values

# Function to preprocess the features for the model
def preprocess_audio_features(audio_features, scaler):
    # Scale the features using the same scaler as in training
    audio_features_scaled = scaler.transform([audio_features])
    
    # Reshape to match the input format for LSTM (samples, timesteps, features)
    audio_features_reshaped = np.reshape(audio_features_scaled, (audio_features_scaled.shape[0], 1, audio_features_scaled.shape[1]))
    
    return audio_features_reshaped

# Function to predict nervousness intensity and emotion from the audio
def predict_emotion_intensity(audio_path, model, scaler, label_encoder):
    # Extract features from the audio file
    audio_features = extract_opensmile_features(audio_path)
    
    # Preprocess the features (scaling and reshaping)
    audio_features_reshaped = preprocess_audio_features(audio_features, scaler)
    
    # Make a prediction using the loaded model
    prediction = model.predict(audio_features_reshaped)
    
    # Get the predicted class for intensity
    intensity_label = np.argmax(prediction)
    
    # Convert the predicted intensity to its original label
    intensity_label_decoded = label_encoder.inverse_transform([intensity_label])
    
    return intensity_label_decoded

# Function to process all WAV files in a local directory
def process_all_wav_files_in_local_directory(directory):
    # List all wav files in the directory
    wav_files = list_wav_files_in_local_directory(directory)
    
    if not wav_files:
        print(f"No WAV files found in the directory: {directory}")
        return
    
    # Iterate over each file, and predict
    results = []
    for file_path in wav_files:
        print(f"Processing file: {file_path}")
        predicted_intensity_label = predict_emotion_intensity(file_path, model, scaler, label_encoder)
        results.append((file_path, predicted_intensity_label))
    
    # Print the results for all files
    for file_path, intensity_label in results:
        return str(intensity_label)

# Example usage: process all WAV files in the local 'output_wav' directory
output_wav_directory = "output"  # Local path to your output directory
a=process_all_wav_files_in_local_directory(output_wav_directory)
 