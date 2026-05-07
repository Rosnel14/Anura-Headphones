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
BLOCK_SIZE = 1600             # 100ms audio chunks fed into queue

SILENCE_THRESHOLD = 0.015    # RMS below this = silence
SILENCE_SECONDS = 0.8        # silence duration that ends an utterance
MIN_SPEECH_SECONDS = 0.4     # ignore blips shorter than this
MAX_SPEECH_SECONDS = 12.0    # force-process if someone talks very long

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
# TRANSCRIBE + TRANSLATE
# ------------------------

def is_repetitive(text):
    # catches hallucinations like 嗯,嗯,嗯,嗯... or um,um,um,um...
    tokens = text.replace(",", " ").split()
    if len(tokens) < 6:
        return False
    return len(set(tokens)) / len(tokens) < 0.25

def process_utterance(speech_buffer, recent_transcript):
    chunk = np.concatenate(speech_buffer).flatten()

    segments, info = model.transcribe(
        chunk,
        beam_size=5,
        temperature=0,
        vad_filter=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=1.5,
        no_speech_threshold=0.6,
    )

    detected_lang = info.language

    # ignore if Whisper detected an unexpected language — likely a misdetection
    if detected_lang not in (SOURCE_LANG, USER_LANG):
        return recent_transcript

    text = " ".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()

    if not text or len(text) < 3:
        return recent_transcript

    if is_repetitive(text):
        return recent_transcript

    if text.strip().lower() == recent_transcript.strip().lower():
        return recent_transcript

    print(f"[{detected_lang}] {text}")

    if detected_lang != USER_LANG:
        translated = translate(text)
        if translated:
            print(f"  → {translated}")
            speak(translated)
        else:
            print("  → translation failed")

    return text

# ------------------------
# MAIN LOOP
# ------------------------

stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=BLOCK_SIZE,
    callback=callback,
)

recent_transcript = ""

SILENCE_BLOCKS    = int(SILENCE_SECONDS    * SAMPLE_RATE / BLOCK_SIZE)
MIN_SPEECH_BLOCKS = int(MIN_SPEECH_SECONDS * SAMPLE_RATE / BLOCK_SIZE)
MAX_SPEECH_BLOCKS = int(MAX_SPEECH_SECONDS * SAMPLE_RATE / BLOCK_SIZE)

with stream:
    speech_buffer = []
    silent_blocks  = 0
    in_speech      = False

    try:
        while True:
            data = audio_queue.get()
            rms  = np.abs(data).mean()
            is_speech = rms > SILENCE_THRESHOLD

            if is_speech:
                speech_buffer.append(data.copy())
                silent_blocks = 0
                in_speech = True

            elif in_speech:
                # trailing silence — keep accumulating until the pause is long enough
                speech_buffer.append(data.copy())
                silent_blocks += 1

                end_of_utterance = silent_blocks >= SILENCE_BLOCKS
                too_long         = len(speech_buffer) >= MAX_SPEECH_BLOCKS

                if end_of_utterance or too_long:
                    if len(speech_buffer) >= MIN_SPEECH_BLOCKS:
                        recent_transcript = process_utterance(speech_buffer, recent_transcript)
                    speech_buffer = []
                    silent_blocks  = 0
                    in_speech      = False

    except KeyboardInterrupt:
        print("\nStopped.")
