import sounddevice as sd
import numpy as np
import queue
import tempfile
import threading
import wave
import asyncio
import time
import subprocess

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from edge_tts import Communicate

# ======================
# USER INPUT
# ======================
source_lang = input("Native language (e.g. en, es, fr): ").strip()
target_lang = input("Translate to (e.g. es, en, ja): ").strip()

# ======================
# WHISPER MODEL
# ======================
model = WhisperModel("base", device="cpu", compute_type="int8")

# ======================
# AUDIO SETTINGS
# ======================
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000

audio_queue = queue.Queue()

# ======================
# DEVICE SELECTION (FIX)
# ======================
def get_input_device():
    devices = sd.query_devices()
    print("\nAvailable audio input devices:\n")

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"[{i}] {dev['name']}")

    default = sd.default.device[0]

    if default is not None and default >= 0:
        print(f"\nUsing default input device: {default}")
        return default

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"\nFallback input device: {i}")
            return i

    raise RuntimeError("No valid microphone found")

device_id = get_input_device()

# ======================
# AUDIO CALLBACK
# ======================
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    if indata is None or len(indata) == 0:
        return

    audio_queue.put(indata.copy())

# ======================
# SAVE WAV FOR WHISPER
# ======================
def save_wav(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        with wave.open(f.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio)
        return f.name

# ======================
# PLAY AUDIO (FIXED)
# ======================
def play_audio(path):
    def _play():
        subprocess.run(["afplay", path])  # macOS-native MP3 player

    threading.Thread(target=_play, daemon=True).start()

# ======================
# TEXT TO SPEECH
# ======================
async def speak(text, lang):
    voice_map = {
        "en": "en-US-GuyNeural",
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "ja": "ja-JP-KeitaNeural",
        "zh": "zh-CN-YunxiNeural",
        "it": "it-IT-DiegoNeural",
        "pt": "pt-BR-AntonioNeural",
        "ru": "ru-RU-DmitryNeural",
    }

    voice = voice_map.get(lang, "en-US-GuyNeural")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        comm = Communicate(text, voice)
        await comm.save(f.name)
        play_audio(f.name)

# ======================
# PROCESS AUDIO
# ======================
def process_audio(audio_bytes):
    wav_path = save_wav(audio_bytes)

    segments, info = model.transcribe(
        wav_path,
        language=source_lang,
        beam_size=1
    )

    text = " ".join([s.text for s in segments]).strip()

    if not text:
        return

    print(f"\nHeard: {text}")

    try:
        translated = GoogleTranslator(
            source=source_lang,
            target=target_lang
        ).translate(text)

        print(f"Translated: {translated}")

        asyncio.run(speak(translated, target_lang))

    except Exception as e:
        print("Translation error:", e)

# ======================
# MAIN LOOP
# ======================
def main():
    print("\nListening... Speak naturally.\n")

    buffer = []
    last_flush = time.time()

    with sd.InputStream(
        device=device_id,          # 🔥 FIXED: explicit device selection
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    ):
        while True:
            try:
                audio = audio_queue.get(timeout=1)
                buffer.append(audio)

            except queue.Empty:
                continue

            # simple sentence chunking (stable fallback)
            if time.time() - last_flush > 2.0:
                if buffer:
                    audio_data = np.concatenate(buffer, axis=0)
                    audio_bytes = audio_data.tobytes()

                    threading.Thread(
                        target=process_audio,
                        args=(audio_bytes,),
                        daemon=True
                    ).start()

                    buffer = []

                last_flush = time.time()

# ======================
# START
# ======================
if __name__ == "__main__":
    main()