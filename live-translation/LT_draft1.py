""" Dependencies
brew install portaudio
pip install faster-whisper sounddevice numpy pyttsx3 langcodes requests
brew install ollama
ollama pull mistral

"""

""" Run in CLI
ollama serve
"""

import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3
import langcodes
import requests
import threading

# ------------------------
# CONFIG
# ------------------------

MODEL_SIZE = "base"  # try "small" if you want better accuracy
OLLAMA_MODEL = "qwen2.5"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.2

# ------------------------
# INIT
# ------------------------

model = WhisperModel(MODEL_SIZE, compute_type="int8")
engine = pyttsx3.init()
audio_queue = queue.Queue()

# ------------------------
# LANGUAGE INPUT
# ------------------------

VALID_LANGS = {
    "english": "en",
    "en": "en",
    "spanish": "es",
    "es": "es",
    "french": "fr",
    "fr": "fr",
    "german": "de",
    "de": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
}

def is_silence(audio_chunk, threshold=0.01):
    return np.abs(audio_chunk).mean() < threshold

while True:
    user_input = input("Enter your native language (e.g. English or en): ").strip().lower()

    if user_input in VALID_LANGS:
        USER_LANG = VALID_LANGS[user_input]
        break

    print("Invalid language. Try: English, Spanish, French, etc.")

# ------------------------
# TRANSLATION (OLLAMA)
# ------------------------

def translate(text, target_lang):
    prompt = f"""
You are a translation engine.

RULES:
- Translate the input into {target_lang}
- Output ONLY the translation
- Do NOT explain
- Do NOT add punctuation unless present in original meaning
- Do NOT respond conversationally

TEXT:
{text}
"""

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0
            }
        )
        return r.json()["response"].strip()
    except Exception as e:
        print("Translation error:", e)
        return None

# ------------------------
# SPEECH OUTPUT
# ------------------------

def speak(text):
    engine.stop()  # prevent overlap
    engine.say(text)
    engine.runAndWait()

# ------------------------
# AUDIO CALLBACK
# ------------------------

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# ------------------------
# STREAM LOOP
# ------------------------

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=callback
)

print("Listening... (Ctrl+C to stop)")

with stream:
    buffer = np.zeros((0, 1), dtype=np.float32)

    try:
        while True:
            data = audio_queue.get()
            buffer = np.concatenate((buffer, data), axis=0)

            if len(buffer) >= SAMPLE_RATE * CHUNK_SECONDS:
                chunk = buffer.flatten()
                buffer = np.zeros((0, 1), dtype=np.float32)

                if is_silence(chunk):
                    continue

                segments, info = model.transcribe(chunk, beam_size=1)
                detected_lang = info.language

                text = " ".join([s.text for s in segments]).strip()

                if not text or len(text) < 3:
                    continue

                print(f"[{detected_lang}] {text}")

                if detected_lang != USER_LANG:
                    translated = translate(text, USER_LANG)

                    if translated:
                        print(f"→ {translated}")
                        speak(translated)

    except KeyboardInterrupt:
        print("\nStopped.")