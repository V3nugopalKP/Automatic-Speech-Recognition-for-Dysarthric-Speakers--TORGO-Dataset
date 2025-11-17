import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import dump

metadata = pd.read_csv("./out/metadata_with_split.csv")
metadata = metadata[metadata["wav_missing"] == False]

spectrograms = []
labels = []

def load_audio(path):
    try:
        y, sr = sf.read(path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        return y, sr
    except:
        y, sr = librosa.load(path, sr=16000)
        return y, sr

def create_log_mel_spectrogram(path):
    y, sr = load_audio(path)

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return log_mel.astype(np.float32)

print("Extracting Spectrograms...\n")

for idx, row in metadata.iterrows():
    wav_path = row["wav_path"]
    text_code = row["text_code"]

    try:
        spec = create_log_mel_spectrogram(wav_path)

        # Pad / crop spectrograms to same size
        target_length = 300  
        current_length = spec.shape[1]

        if current_length < target_length:
            pad_width = target_length - current_length
            spec = np.pad(spec, ((0,0),(0,pad_width)), mode='constant')
        else:
            spec = spec[:, :target_length]

        spectrograms.append(spec)
        labels.append(text_code)

    except:
        print("❌ Error reading:", wav_path)
        continue

# Convert to arrays
X = np.array(spectrograms)
y = np.array(labels)

np.save("./out/spectrograms.npy", X)
np.save("./out/labels.npy", y)

print("DONE! Saved spectrograms and labels.")
