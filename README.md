# 🎙️ Voice Emotion and Intensity Detection System

This project is an advanced audio emotion recognition pipeline that uses **Swarm Intelligence-based optimization**, **OpenSMILE audio feature extraction**, and **Wav2Vec2 Transformer-based deep learning** to detect emotions (like "nervous") and their intensity from audio `.wav` files.

---

## 🔍 Overview

The system performs the following tasks:

1. **Emotion Detection using Keras Model** trained on OpenSMILE features.
2. **Emotion Detection using Fine-tuned Wav2Vec2 Transformer model**.
3. **Intensity Prediction using a dedicated Keras model**.
4. **Timestamp Analysis** to identify the time intervals where a specific emotion (e.g., *nervous*) is detected.
5. **Swarm Intelligence Techniques** were used in the model training phase for feature selection and optimization.

---

## 🧠 Technologies Used

- 🐍 **Python 3**
- 🎯 **Swarm Intelligence** (for training optimization)
- 🤖 **Wav2Vec2** (via HuggingFace Transformers)
- 🔊 **OpenSMILE** (for acoustic feature extraction)
- 🔬 **TensorFlow/Keras** (for deep learning models)
- 🔥 **PyTorch & torchaudio**
- 📊 **Pandas, NumPy, Scikit-learn**

---

## 🧪 Models Used

| Model Type         | Description                                 |
|--------------------|---------------------------------------------|
| `nervous_intensity_modelEMO.h5` | Keras model for emotion classification using OpenSMILE features |
| `finetuned_wav2vec2`            | Wav2Vec2 transformer fine-tuned for emotion recognition          |
| `nervous_intensity_modelINTE.h5`| Keras model for emotion intensity prediction                     |

---

## 📁 Directory Structure

project_root/
├── upload/ # Folder containing .wav files to process
├── nervous_intensity_modelEMO.h5 # Trained Keras emotion model
├── nervous_intensity_modelINTE.h5 # Trained Keras intensity model
├── merged_with_emotion_intensity3.csv # Dataset used for scaling and encoding
├── main.py # Main script to run predictions
└── README.md # You're reading it :)

## 📌 Use Cases
This system can be applied in real-world scenarios including:

🔍 1. Interrogation Rooms (Covert Monitoring)
Emotions like nervousness or anxiety can be detected without alerting the suspect.

Helps law enforcement or psychologists assess stress levels in real-time.

Can be used as a supportive tool during interviews, especially in cases of lie detection or behavioral analysis.

🎤 2. Public Speaking & Stage Practice
Speakers and presenters can analyze recordings of their voice to identify moments of nervousness or loss of confidence.

Helps in improving performance, vocal stability, and stage presence.

Useful for trainers, students, corporate speakers, and motivational coaches.


## 🧠 Swarm Intelligence in Training
Swarm Intelligence (e.g., Particle Swarm Optimization - PSO) was used during the training phase to:

Select the most relevant acoustic features from OpenSMILE output

Optimize hyperparameters of the Keras models for better accuracy and generalization

⚠️ This optimization is not performed during inference but plays a crucial role in how well the models perform on unseen data.



## 📌 Notes
All .wav files should be in mono or stereo format and preferably 16kHz sampling rate.

If not, the script automatically resamples audio for Wav2Vec2 compatibility.

Make sure the pretrained models are present in the project root.

## 🚀 Future Work
Add real-time audio capture support

Support for more emotions and multilingual models

Integrate a GUI using Streamlit or Flask
