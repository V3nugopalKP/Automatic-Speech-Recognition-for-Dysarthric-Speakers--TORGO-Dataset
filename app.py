import streamlit as st
from speech_to_text import transcribe_audio
import soundfile as sf
import librosa
import tempfile
import os

st.title("🎤 Speech Recognition using VOSK")

uploaded_file = st.file_uploader("Upload WAV file (16kHz mono)", type=["wav"])

if uploaded_file is not None:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name

    st.audio(uploaded_file, format='audio/wav')

    # Ensure audio is 16kHz mono (Vosk requirement)
    y, sr = librosa.load(temp_file_path, sr=16000, mono=True)
    sf.write(temp_file_path, y, 16000)

    with st.spinner("Transcribing..."):
        text = transcribe_audio(temp_file_path)

    st.success("Transcription complete!")
    st.write(f"### 📝 Recognized Text:\n{text}")

    os.remove(temp_file_path)
