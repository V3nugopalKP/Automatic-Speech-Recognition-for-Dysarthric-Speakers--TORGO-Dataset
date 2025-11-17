from vosk import Model, KaldiRecognizer
import soundfile as sf
import json

# Load Vosk Model (only once)
model = Model("vosk-model-small-en-us-0.15")

def transcribe_audio(path):
    data, sr = sf.read(path)

    if sr != 16000:
        raise ValueError("Vosk requires 16kHz sample rate. Please resample your audio.")

    rec = KaldiRecognizer(model, sr)
    
    # Convert to 16-bit PCM required by Vosk
    pcm_data = (data * 32767).astype("int16").tobytes()

    rec.AcceptWaveform(pcm_data)
    result = rec.FinalResult()
    text = json.loads(result)["text"]

    return text
