import pandas as pd
import soundfile as sf
import librosa
from speech_to_text import transcribe_audio
from jiwer import wer, cer
import time
import json
import os

# Track total runtime
start_time = time.time()

# Load metadata
metadata = pd.read_csv("./out/metadata_with_split.csv")

# Only evaluate files that exist
metadata = metadata[metadata["wav_missing"] == False]

true_texts = []
pred_texts = []
wav_files = []
text_files = []

print("Evaluating Vosk on your dataset...")

for idx, row in metadata.iterrows():

    wav_path = row["wav_path"]
    txt_path = row["txt_path"]

    try:
        # ---- Load reference text ----
        with open(txt_path, "r", encoding="utf-8") as f:
            true_text = f.read().strip().lower()

        # ---- Load audio and resample to 16kHz mono ----
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        sf.write("temp.wav", y, 16000)

        # ---- Predict text using Vosk ----
        predicted_text = transcribe_audio("temp.wav")

        true_texts.append(true_text)
        pred_texts.append(predicted_text)
        wav_files.append(wav_path)
        text_files.append(txt_path)

        print(f"[OK] {wav_path}")

    except Exception as e:
        print(f"[SKIP] {wav_path} — {e}")

# ---- Compute metrics ----
word_error_rate = wer(true_texts, pred_texts)
char_error_rate = cer(true_texts, pred_texts)
accuracy = (1 - word_error_rate)

# ---- Save detailed results ----
results_df = pd.DataFrame({
    "wav_path": wav_files,
    "txt_path": text_files,
    "true_text": true_texts,
    "predicted_text": pred_texts
})

os.makedirs("./out/results", exist_ok=True)

results_df.to_csv("./out/results/vosk_detailed_results.csv", index=False)

# ---- Save summary results ----
summary = {
    "WER": float(word_error_rate),
    "CER": float(char_error_rate),
    "Accuracy": float(accuracy),
    "Total_files_evaluated": len(true_texts),
    "Total_runtime_seconds": round(time.time() - start_time, 2),
    "Total_runtime_minutes": round((time.time() - start_time) / 60, 2)
}

with open("./out/results/vosk_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

# ---- Print summary ----
print("\n========= RESULTS =========")
print(f"WER  (word error rate):     {word_error_rate:.4f}")
print(f"CER  (char error rate):     {char_error_rate:.4f}")
print(f"Approx accuracy:            {accuracy:.4f}")
print(f"Files evaluated:            {len(true_texts)}")
print(f"Total runtime (seconds):    {summary['Total_runtime_seconds']}")
print(f"Total runtime (minutes):    {summary['Total_runtime_minutes']}")
print("================================\n")

print("Saved detailed results to: ./out/results/vosk_detailed_results.csv")
print("Saved summary to:          ./out/results/vosk_summary.json")
