"""
Dependencies
brew install portaudio
pip install faster-whisper sounddevice numpy langcodes requests
brew install ollama
ollama pull qwen2.5
"""

"""
Run Ollama server first:
ollama serve
"""

import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import langcodes
import requests
import threading
import subprocess

# ------------------------
# CONFIG
# ------------------------

MODEL_SIZE = "base"
OLLAMA_MODEL = "qwen2.5"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.0

# ------------------------
# INIT
# ------------------------

model = WhisperModel(MODEL_SIZE, compute_type="int8")
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

while True:
    user_input = input("Enter your native language (e.g. English or en): ").strip().lower()

    if user_input in VALID_LANGS:
        USER_LANG = VALID_LANGS[user_input]
        break

    print("Invalid language. Try again.")

# ------------------------
# SILENCE DETECTION
# ------------------------

def is_silence(audio_chunk, threshold=0.008):
    return np.abs(audio_chunk).mean() < threshold

# ------------------------
# TRANSLATION (OLLAMA)
# ------------------------

def translate(text, target_lang):
    prompt = f"""
You are a strict translation engine.

RULES:
- Translate into {target_lang}
- Output ONLY the translation
- No explanations
- No extra text

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
# FAST TTS
# ------------------------

tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        try:
            # macOS native TTS (FAST + RELIABLE)
            subprocess.run(["say", "-r", "190", text])
        except Exception as e:
            print("TTS error:", e)

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    tts_queue.put(text)

# ------------------------
# AUDIO CALLBACK
# ------------------------

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# ------------------------
# STREAM SETUP
# ------------------------

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    callback=callback
)

print("Listening... (Ctrl+C to stop)")

# ------------------------
# MAIN LOOP
# ------------------------

with stream:
    buffer = np.zeros((0, 1), dtype=np.float32)

    try:
        while True:
            data = audio_queue.get()
            buffer = np.concatenate((buffer, data), axis=0)

            if len(buffer) >= SAMPLE_RATE * CHUNK_SECONDS:

                chunk = buffer.flatten()
                buffer = np.zeros((0, 1), dtype=np.float32)

                # 🔥 silence filter
                if is_silence(chunk):
                    continue

                # ------------------------
                # SPEECH RECOGNITION
                # ------------------------

                segments, info = model.transcribe(
                    chunk,
                    beam_size=3,
                    temperature=0,
                    condition_on_previous_text=True
                )

                detected_lang = info.language
                text = " ".join([s.text for s in segments]).strip()

                if not text or len(text) < 2:
                    continue

                print(f"[{detected_lang}] {text}")

                # ------------------------
                # TRANSLATION + SPEECH
                # ------------------------

                if detected_lang != USER_LANG:
                    translated = translate(text, USER_LANG)

                    if translated:
                        print(f"→ {translated}")
                        speak(translated)
                    else:
                        print("→ translation failed")

    except KeyboardInterrupt:
        print("\nStopped.")