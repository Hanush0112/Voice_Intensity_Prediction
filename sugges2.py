import os
import opensmile
import torch
import torchaudio
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import tempfile

# Load the Keras and Wav2Vec2 models
keras_model = load_model("nervous_intensity_modelEMO.h5")  # Keras model for emotions
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained('finetuned_wav2vec2')  # Wav2Vec2 for emotions
wav2vec_processor = Wav2Vec2Processor.from_pretrained('finetuned_wav2vec2')
wav2vec_model.eval()

# Load Model 3 for intensity prediction
model_3 = load_model("nervous_intensity_modelINTE.h5")  # Model 3 for intensity

# Load the dataset for fitting the scaler and label encoder for emotions (Keras)
nervous_df = pd.read_csv("merged_with_emotion_intensity3.csv")
scaler = StandardScaler().fit(nervous_df.drop(columns=['emotion_label', 'intensity_label']))
label_encoder = LabelEncoder().fit(nervous_df['emotion_label'])

# Initialize the OpenSMILE feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals
)

# OpenSMILE feature extraction
def extract_opensmile_features(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav_file:
        tmp_wav_file.write(audio_bytes)
        tmp_wav_file_path = tmp_wav_file.name
    features = smile.process_file(tmp_wav_file_path)
    os.remove(tmp_wav_file_path)
    return features.values.flatten()

# Keras model prediction (emotion)
def predict_keras_emotion(audio_bytes):
    audio_features = extract_opensmile_features(audio_bytes)
    scaled_features = scaler.transform([audio_features])
    reshaped_features = scaled_features.reshape(1, 1, -1)
    predicted_probabilities = keras_model.predict(reshaped_features)
    predicted_label = label_encoder.inverse_transform([np.argmax(predicted_probabilities)])[0]
    return predicted_label

# Wav2Vec2 model prediction (emotion)
def predict_wav2vec_emotion(audio_bytes):
    waveform, sample_rate = torchaudio.load(BytesIO(audio_bytes))
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    inputs = wav2vec_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec_model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return "nervous" if predicted_label >= 4 else "not nervous"

# Model 3 prediction (intensity)
def predict_intensity(audio_bytes):
    audio_features = extract_opensmile_features(audio_bytes)
    scaled_features = scaler.transform([audio_features])
    reshaped_features = scaled_features.reshape(1, 1, -1)
    intensity_prediction = model_3.predict(reshaped_features)
    return np.argmax(intensity_prediction)

# Function to process all WAV files in the local directory
def process_all_wav_files(directory):
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    keras_predictions = []
    wav2vec_predictions = []
    intensity_predictions = []
    
    for file_path in wav_files:
        print(f"Processing {file_path}")
        with open(os.path.join(directory, file_path), 'rb') as f:
            audio_bytes = f.read()

        # Get predictions from both emotion models
        keras_prediction = predict_keras_emotion(audio_bytes)
        wav2vec_prediction = predict_wav2vec_emotion(audio_bytes)

        # Get intensity prediction from Model 3
        intensity_prediction = predict_intensity(audio_bytes)

        # Append to respective lists
        keras_predictions.append(keras_prediction)
        wav2vec_predictions.append(wav2vec_prediction)
        intensity_predictions.append(intensity_prediction)
    
    # Return combined outputs: 2D array for emotions, 1D array for intensities
    return [keras_predictions, wav2vec_predictions], intensity_predictions

# Function to process model outputs and find timestamps for the target emotion
def process_model_outputs(input_array, target_element):
    df = pd.DataFrame(input_array)
    timestamps = []
    for index, row in df.iterrows():
        unique_elements, counts = np.unique(row, return_counts=True)
        most_common_element = unique_elements[counts.argmax()]
        if most_common_element == target_element:
            start_time = index * 4
            end_time = start_time + 3
            timestamps.append(f"{start_time}-{end_time}")
    return {target_element: timestamps}

# Example usage
if __name__ == "__main__":
    # Directory containing the WAV files
    input_directory = "upload"  # Replace with your local input folder path

    # Process all files in the input_wav directory
    emotions_2d, intensities_1d = process_all_wav_files(input_directory)
    
    # Print the 2D emotions array and the 1D intensity array
    print("Emotions (2D list):", emotions_2d)
    print("Intensities (1D list):", intensities_1d)
    
    # Find and print the timestamps for 'nervous' emotion
    nervous_timestamps = process_model_outputs(emotions_2d, "nervous")
    print("Nervous timestamps:", nervous_timestamps)
