"""
Dependencies
brew install portaudio
pip install mlx-whisper sounddevice numpy argostranslate
"""

import os
import warnings
import queue
import threading
import subprocess

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", message=".*token.*", category=UserWarning)

import sounddevice as sd
import numpy as np
import mlx_whisper
import argostranslate.package
import argostranslate.translate

# ------------------------
# CONFIG
# ------------------------

# swap to mlx-community/whisper-small-mlx for better accuracy (still fast on Apple Silicon)
MLX_MODEL = "mlx-community/whisper-base-mlx"

SAMPLE_RATE = 16000
BLOCK_SIZE  = 1600            # 100ms per block

SILENCE_THRESHOLD   = 0.02   # RMS below this = silence
MIN_UTTERANCE_RMS   = 0.025  # discard whole utterance if avg RMS is below this
SILENCE_SECONDS     = 0.8    # pause duration that ends an utterance
MIN_SPEECH_SECONDS  = 0.4    # ignore blips shorter than this
MAX_SPEECH_SECONDS  = 12.0   # force-process very long utterances

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

LANG_VOICES = {
    "en": "Samantha", "zh": "Ting-Ting", "es": "Monica",
    "fr": "Thomas",   "de": "Anna",      "it": "Alice",
    "pt": "Joana",    "ja": "Kyoko",     "ko": "Yuna",
    "ar": "Tarik",    "ru": "Milena",    "hi": "Lekha",
}

def prompt_language(prompt_text):
    supported = sorted(set(VALID_LANGS.values()))
    while True:
        user_input = input(prompt_text).strip().lower()
        if user_input in VALID_LANGS:
            return VALID_LANGS[user_input]
        print(f"  Supported: {', '.join(supported)}")

print("\n--- Anura Live Translation ---")

# load model in background while user types
_model_ready = threading.Event()

def _load_model():
    # warm up the model with a silent chunk so first transcription isn't slow
    mlx_whisper.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        path_or_hf_repo=MLX_MODEL,
        verbose=False,
    )
    _model_ready.set()

threading.Thread(target=_load_model, daemon=True).start()

USER_LANG   = prompt_language("Your language (output) — e.g. English / en : ")
SOURCE_LANG = prompt_language("Speaker's language (input) — e.g. Spanish / es: ")

src_name = LANG_NAMES.get(SOURCE_LANG, SOURCE_LANG)
tgt_name = LANG_NAMES.get(USER_LANG,   USER_LANG)

# ------------------------
# ARGOSTRANSLATE SETUP
# ------------------------

def ensure_packages(from_code, to_code):
    installed = {(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()}
    if (from_code, to_code) in installed:
        return

    print(f"Downloading {LANG_NAMES.get(from_code, from_code)} → {LANG_NAMES.get(to_code, to_code)} package...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    def install_pair(fc, tc):
        if (fc, tc) in installed:
            return True
        pkg = next((p for p in available if p.from_code == fc and p.to_code == tc), None)
        if pkg:
            argostranslate.package.install_from_path(pkg.download())
            installed.add((fc, tc))
            return True
        return False

    if not install_pair(from_code, to_code):
        ok1 = install_pair(from_code, "en") if from_code != "en" else True
        ok2 = install_pair("en", to_code)   if to_code   != "en" else True
        if not (ok1 and ok2):
            raise RuntimeError(f"No package found for {from_code} → {to_code}")

if SOURCE_LANG != USER_LANG:
    ensure_packages(SOURCE_LANG, USER_LANG)
    ensure_packages(USER_LANG, SOURCE_LANG)
    argostranslate.translate.translate("hello", SOURCE_LANG, USER_LANG)
    argostranslate.translate.translate("hello", USER_LANG, SOURCE_LANG)

if not _model_ready.is_set():
    print("Loading model...", end="", flush=True)
    _model_ready.wait()
    print(" ready.")

print(f"\n{src_name} ↔ {tgt_name}  |  Ctrl+C to stop\n")

# ------------------------
# TRANSLATION
# ------------------------

def translate(text, from_code, to_code):
    try:
        return argostranslate.translate.translate(text, from_code, to_code)
    except Exception as e:
        print(f"  [translation error: {e}]")
        return None

# ------------------------
# TTS
# ------------------------

tts_queue = queue.Queue()
tts_proc  = None

def tts_worker():
    global tts_proc
    while True:
        text, voice = tts_queue.get()
        try:
            tts_proc = subprocess.Popen(["say", "-r", "190", "-v", voice, text])
            tts_proc.wait()
        except Exception as e:
            print(f"  [TTS error: {e}]")
        finally:
            tts_proc = None

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text, lang):
    tts_queue.put((text, LANG_VOICES.get(lang, "Samantha")))

def flush_tts():
    global tts_proc
    if tts_proc and tts_proc.poll() is None:
        tts_proc.terminate()
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except Exception:
            break

# ------------------------
# HELPERS
# ------------------------

def is_repetitive(text):
    tokens = text.replace(",", " ").split()
    if len(tokens) < 6:
        return False
    return len(set(tokens)) / len(tokens) < 0.25

def process_utterance(speech_buffer, recent_transcript):
    chunk = np.concatenate(speech_buffer).flatten().astype(np.float32)

    # reject utterances that are mostly silence/noise
    if np.abs(chunk).mean() < MIN_UTTERANCE_RMS:
        return recent_transcript

    result = mlx_whisper.transcribe(
        chunk,
        path_or_hf_repo=MLX_MODEL,
        temperature=0,
        compression_ratio_threshold=1.5,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        verbose=False,
    )

    detected_lang = result.get("language", "")
    segments      = result.get("segments", [])

    if detected_lang not in (SOURCE_LANG, USER_LANG):
        return recent_transcript

    text = " ".join(
        s["text"] for s in segments if s.get("no_speech_prob", 0) < 0.6
    ).strip()

    if not text or len(text) < 3:
        return recent_transcript

    if is_repetitive(text):
        return recent_transcript

    if text.strip().lower() == recent_transcript.strip().lower():
        return recent_transcript

    print(f"[{detected_lang}] {text}")

    if detected_lang == USER_LANG:
        translated = translate(text, USER_LANG, SOURCE_LANG)
        if translated:
            print(f"  → [{SOURCE_LANG}] {translated}")
            speak(translated, SOURCE_LANG)
        else:
            print("  → translation failed")
    else:
        translated = translate(text, SOURCE_LANG, USER_LANG)
        if translated:
            print(f"  → [{USER_LANG}] {translated}")
            speak(translated, USER_LANG)
        else:
            print("  → translation failed")

    return text

# ------------------------
# MAIN LOOP
# ------------------------

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    callback=callback,
)

SILENCE_BLOCKS    = int(SILENCE_SECONDS    * SAMPLE_RATE / BLOCK_SIZE)
MIN_SPEECH_BLOCKS = int(MIN_SPEECH_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
MAX_SPEECH_BLOCKS = int(MAX_SPEECH_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

recent_transcript = ""

with stream:
    speech_buffer = []
    silent_blocks  = 0
    in_speech      = False

    try:
        while True:
            data      = audio_queue.get()
            rms       = np.abs(data).mean()
            is_speech = rms > SILENCE_THRESHOLD

            if is_speech:
                if not in_speech:
                    flush_tts()
                speech_buffer.append(data.copy())
                silent_blocks = 0
                in_speech     = True

            elif in_speech:
                speech_buffer.append(data.copy())
                silent_blocks += 1

                if silent_blocks >= SILENCE_BLOCKS or len(speech_buffer) >= MAX_SPEECH_BLOCKS:
                    if len(speech_buffer) >= MIN_SPEECH_BLOCKS:
                        recent_transcript = process_utterance(speech_buffer, recent_transcript)
                    speech_buffer = []
                    silent_blocks  = 0
                    in_speech      = False

    except KeyboardInterrupt:
        print("\nStopped.")
