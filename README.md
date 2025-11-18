# 🔊 Dysarthric Speech Recognition using the TORGO Dataset  
### Baseline Automatic Speech Recognition (ASR) System

This project implements a **baseline speech-to-text system** for **dysarthric / impaired speech**, using the **TORGO dataset** and an offline ASR engine.  
The goal is to evaluate the performance of general-purpose ASR models on dysarthric speech and build a foundation for future improvements.

## 📌 Project Overview

Dysarthric speech is significantly challenging for Automatic Speech Recognition (ASR) systems due to:

- Slurred articulation  
- Inconsistent phoneme production  
- Lower intelligibility  
- Variability between speakers  

This project builds a complete pipeline to evaluate ASR performance on dysarthric speakers and establish a **baseline system** before moving to advanced models next semester.

## 🎯 Objectives

- Preprocess TORGO dataset (audio + transcripts)  
- Create speaker-wise train/validation/test split  
- Extract log-Mel spectrogram features  
- Run baseline transcription using a classical ASR model  
- Evaluate performance using **WER** and **CER**  
- Provide a simple web interface for testing  

## 🗂 Dataset: TORGO Dysarthric Speech Corpus

The project uses the **TORGO** dataset, which contains:

- Dysarthric and control speakers  
- Male and female speakers  
- Multiple sessions & repetitions  
- Paired audio (`.wav`) and text (`.txt`) files  

Since TORGO is not publicly distributed, the dataset is **not included** in this repository.

## 🏗 Project Architecture

Dataset → Metadata Builder → Train/Val/Test Split → Audio Preprocessing → Log-Mel Spectrograms → Baseline ASR → Evaluation

## 🤖 Baseline Speech Recognition Model

A baseline system was built using the **VOSK (Voice Open Source Toolkit)** model, a lightweight hybrid **DNN + HMM** architecture based on Kaldi.

### Why VOSK?
- Works offline  
- Low resource requirements  
- Easy to integrate with Python  
- Suitable for baseline testing  

## 📊 Evaluation Results (TORGO)

Performance on dysarthric speech:

- **Word Error Rate (WER): 61%**  
- **Character Error Rate (CER): 35%**

These results highlight the difficulty of recognizing dysarthric speech using a general ASR model.  
Models like CNN and SVM were also tested on extracted features, but **performed worse than the VOSK baseline**.

## 🧪 Features Extracted

Each audio file is converted into:

- 16 kHz mono waveform  
- 64 Mel-band log-scaled spectrogram  
- Padded/cropped to fixed length  
- Saved as NumPy arrays (`.npy`)  

## 📦 Repository Structure

- vosk-model-small-en-us-0.15/   → Pretrained ASR model  
- .gitignore  
- app.py                          → Streamlit app  
- build_text_dict_and_metadata.py → Scan TORGO dataset & build metadata  
- create_split.py                 → Speaker-wise train/val/test split  
- extract_spectrograms.py         → Log-Mel feature extraction  
- evaluate_vosk.py                → WER/CER evaluation  
- speech_to_text.py               → VOSK transcription wrapper  

## 🛠 Installation

`git clone <your-repo-link>`  
`pip install -r requirements.txt`

## ▶️ Running the Baseline Transcription

`python evaluate_vosk.py`

Generates:

- vosk_detailed_results.csv  
- vosk_summary.json  

## 🎛 Streamlit Web Demo

`streamlit run app.py`

Upload a 16 kHz mono WAV file to test live transcription.

## 🚀 Future Work (Next Semester)

- Fine-tuning **wav2vec2** or **Whisper** on dysarthric speech  
- Using phoneme-level supervision  
- Domain-specific language models  
- Speaker adaptation  
- Training a full end-to-end ASR  
- Building a mobile/edge deployment version  

## 🙏 Acknowledgements

- TORGO Dataset  
- VOSK / Kaldi Team  
- Python Libraries: librosa, soundfile, pandas, jiwer, streamlit
