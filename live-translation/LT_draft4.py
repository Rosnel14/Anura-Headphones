"""
Dependencies
brew install portaudio
pip install faster-whisper sounddevice numpy argostranslate
"""

import queue
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import argostranslate.package
import argostranslate.translate
import threading
import subprocess

# ------------------------
# CONFIG
# ------------------------

MODEL_SIZE = "base"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 2.0

# ------------------------
# INIT
# ------------------------

model = WhisperModel(MODEL_SIZE, compute_type="int8")
audio_queue = queue.Queue()

# ------------------------
# LANGUAGE SETUP
# ------------------------

VALID_LANGS = {
    "english": "en",   "en": "en",
    "spanish": "es",   "es": "es",
    "french":  "fr",   "fr": "fr",
    "german":  "de",   "de": "de",
    "italian": "it",   "it": "it",
    "portuguese": "pt","pt": "pt",
    "chinese": "zh",   "zh": "zh",
    "japanese": "ja",  "ja": "ja",
    "korean":  "ko",   "ko": "ko",
    "arabic":  "ar",   "ar": "ar",
    "russian": "ru",   "ru": "ru",
    "hindi":   "hi",   "hi": "hi",
}

LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German",  "it": "Italian", "pt": "Portuguese",
    "zh": "Chinese", "ja": "Japanese","ko": "Korean",
    "ar": "Arabic",  "ru": "Russian", "hi": "Hindi",
}

def prompt_language(prompt_text):
    supported = sorted(set(VALID_LANGS.values()))
    while True:
        user_input = input(prompt_text).strip().lower()
        if user_input in VALID_LANGS:
            return VALID_LANGS[user_input]
        print(f"  Supported: {', '.join(supported)}")

print("\n--- Anura Live Translation ---")
USER_LANG   = prompt_language("Your language (output) — e.g. English / en : ")
SOURCE_LANG = prompt_language("Speaker's language (input) — e.g. Spanish / es: ")

src_name = LANG_NAMES.get(SOURCE_LANG, SOURCE_LANG)
tgt_name = LANG_NAMES.get(USER_LANG,   USER_LANG)

# ------------------------
# ARGOSTRANSLATE SETUP
# ------------------------
# Downloads language packages on first run; cached locally after that.
# Falls back through English if no direct package exists (e.g. ko → fr).

def ensure_packages(from_code, to_code):
    print("Checking translation packages (may download on first run)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed = {(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()}

    def install_pair(fc, tc):
        if (fc, tc) in installed:
            return True
        pkg = next((p for p in available if p.from_code == fc and p.to_code == tc), None)
        if pkg:
            print(f"  Downloading {LANG_NAMES.get(fc, fc)} → {LANG_NAMES.get(tc, tc)}...")
            argostranslate.package.install_from_path(pkg.download())
            installed.add((fc, tc))
            return True
        return False

    if not install_pair(from_code, to_code):
        # No direct package — chain through English
        ok1 = install_pair(from_code, "en") if from_code != "en" else True
        ok2 = install_pair("en", to_code)   if to_code   != "en" else True
        if not (ok1 and ok2):
            raise RuntimeError(
                f"No argostranslate package found for {from_code} → {to_code} "
                f"(or via English). Check https://www.argosopentech.com/argospm/index/"
            )
    print("Translation packages ready.")

if SOURCE_LANG != USER_LANG:
    ensure_packages(SOURCE_LANG, USER_LANG)

print(f"\n{src_name} → {tgt_name}  |  Ctrl+C to stop\n")

# ------------------------
# TRANSLATION
# ------------------------

def translate(text):
    try:
        return argostranslate.translate.translate(text, SOURCE_LANG, USER_LANG)
    except Exception as e:
        print(f"  [translation error: {e}]")
        return None

# ------------------------
# TTS
# ------------------------

tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        try:
            subprocess.run(["say", "-r", "190", text])
        except Exception as e:
            print(f"  [TTS error: {e}]")

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    tts_queue.put(text)

# ------------------------
# AUDIO CALLBACK
# ------------------------

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# ------------------------
# MAIN LOOP
# ------------------------

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)

recent_transcript = ""

with stream:
    buffer = np.zeros((0, 1), dtype=np.float32)
    try:
        while True:
            data = audio_queue.get()
            buffer = np.concatenate((buffer, data), axis=0)

            if len(buffer) < SAMPLE_RATE * CHUNK_SECONDS:
                continue

            chunk = buffer.flatten()
            buffer = np.zeros((0, 1), dtype=np.float32)

            segments, _ = model.transcribe(
                chunk,
                language=SOURCE_LANG,
                beam_size=5,
                temperature=0,
                vad_filter=True,
                condition_on_previous_text=bool(recent_transcript),
                initial_prompt=recent_transcript or None,
            )

            text = " ".join(s.text for s in segments).strip()

            if not text or len(text) < 3:
                continue

            recent_transcript = text[-300:]
            print(f"[{SOURCE_LANG}] {text}")

            if SOURCE_LANG != USER_LANG:
                translated = translate(text)
                if translated:
                    print(f"  → {translated}")
                    speak(translated)
                else:
                    print("  → translation failed")

    except KeyboardInterrupt:
        print("\nStopped.")
