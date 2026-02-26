import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
from scipy.ndimage import zoom
import requests
import soundfile as sf
import io
import os
import gdown

# ==========================================================
# 1. MODEL ARCHITECTURE (Must match training exactly)
# ==========================================================
class MegaClassifier(nn.Module):
    def __init__(self, num_classes=286):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.3),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.4),
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================================
# 2. CACHED MODEL LOADING (Downloads from Google Drive)
# ==========================================================
@st.cache_resource
def load_model():
    model_path = 'single_model_best.pth'
    
    # Download the model if it doesn't exist locally
    if not os.path.exists(model_path):
        st.info("Downloading AI model from Google Drive... This may take a minute but only happens once!")
        file_id = '1Jky0WHb7G0ni9jJHG5zYYrxgQTQSHtux' 
        url = 'https://drive.google.com/uc?id=1Jky0WHb7G0ni9jJHG5zYYrxgQTQSHtux'
        gdown.download(url, model_path, quiet=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = MegaClassifier(num_classes=286)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# ==========================================================
# 3. PREPROCESSING (Handles In-Memory Audio)
# ==========================================================
def preprocess_audio(audio_bytes, target_length=256):
    # Load audio directly from bytes
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

    # Generate Mel Spectrogram
    melspec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # Resize (Zoom)
    current_len = log_melspec.shape[1]
    if current_len != target_length:
        zoom_factor = target_length / current_len
        log_melspec = zoom(log_melspec, (1, zoom_factor), order=1)

    # Normalize
    spec_linear = 10 ** (log_melspec / 20)
    spec_linear = np.clip(spec_linear, 0, 1)
    spec_mean = spec_linear.mean()
    spec_std = spec_linear.std()

    if spec_std > 1e-6:
        spec_norm = (spec_linear - spec_mean) / spec_std
    else:
        spec_norm = spec_linear

    # Add dimensions (1, 1, 128, 256)
    tensor = torch.FloatTensor(spec_norm).unsqueeze(0).unsqueeze(0)
    return tensor

# ==========================================================
# 4. STREAMLIT UI & MAIN PIPELINE
# ==========================================================
st.title("🎙️ Quran Recitation Translator")
st.write("Recite an Ayat, and the AI will identify and translate it.")

# Load Model
model, device = load_model()

# Audio Input Widget
audio_value = st.audio_input("Record a recitation")

# If an audio file is recorded or uploaded, run immediately
if audio_value is not None:
    st.audio(audio_value) # Playback option for the user
    
    if model is None:
        st.error("Model failed to load. Please check the Google Drive link permissions.")
    else:
        with st.spinner("Analyzing recitation..."):
            try:
                # Read bytes from the uploaded/recorded file
                audio_bytes = audio_value.read()
                
                # Preprocess
                input_tensor = preprocess_audio(audio_bytes).to(device)

                # Predict
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_idx = torch.topk(probs, 1)
                
                ayah_num = top_idx.item() + 1
                confidence = top_prob.item() * 100

                # Display Result
                st.success(f"Prediction: **Ayat {ayah_num}**")
                st.info(f"Confidence: {confidence:.2f}%")

                # Fetch Translation (Assuming Surah 2 / Al-Baqarah)
                surah_number = 2 
                
                api_url = f"https://api.alquran.cloud/v1/ayah/{surah_number}:{ayah_num}/en.asad"
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Translation:")
                    st.markdown(f"> {data['data']['text']}")
                    st.caption(f"Surah {data['data']['surah']['englishName']}, Ayat {ayah_num}")
                else:
                    st.warning("Could not fetch translation from API.")

            except Exception as e:
                st.error(f"Error processing audio: {e}")