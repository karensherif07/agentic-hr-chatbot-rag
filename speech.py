"""
speech.py — Text-to-speech and speech-to-text.

STT strategy:
- PRIMARY: Deepgram Nova-3 Arabic API
    - Purpose-built for Arabic dialects (Egyptian, MSA, Gulf, Levantine)
    - No local GPU needed, no ffmpeg required, ~40% lower WER than competitors
    - Requires DEEPGRAM_API_KEY in .env
    - Free tier: $200 credit, no credit card required
    - Sign up: https://console.deepgram.com/signup

- FALLBACK: Local Whisper (medium model)
    - Used automatically if DEEPGRAM_API_KEY is not set or API call fails
    - Requires ffmpeg on PATH and openai-whisper installed

Installation:
    pip install deepgram-sdk
    Add DEEPGRAM_API_KEY=your_key_here to your .env file
"""

import io
import os
import re
import shutil
import subprocess
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Deepgram config ──────────────────────────────────────────
_DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()

# Nova-3 Arabic model — purpose-built for Egyptian, MSA, Gulf, Levantine dialects
# Supports 17 Arabic dialect variants with best-in-class WER on conversational Arabic
_DEEPGRAM_MODEL   = "nova-3"
_DEEPGRAM_LANG_AR = "ar"   # Arabic (covers all dialects automatically)
_DEEPGRAM_LANG_EN = "en"

# ─── Whisper fallback config ──────────────────────────────────
_WHISPER_MODEL_SIZE = "medium"

_ARABIC_INITIAL_PROMPT = (
    "محادثة بالعامية المصرية عن سياسات الموارد البشرية. "
    "الموظف يسأل عن الإجازات والراتب والتقييم والترقية والتدريب. "
    "مش، عايز، عايزة، إيه، ليه، فين، إزاي، كده، دلوقتي، "
    "بتاعي، بتاعتي، عندي، عندك، ممكن، طيب، تمام، يعني."
)

# ─── ffmpeg (needed for Whisper fallback only) ────────────────
_FFMPEG_CANDIDATES = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/opt/local/bin/ffmpeg",
]


def _resolve_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for p in _FFMPEG_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


_FFMPEG_PATH = _resolve_ffmpeg()

if _FFMPEG_PATH:
    ffmpeg_dir = os.path.dirname(os.path.abspath(_FFMPEG_PATH))
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


# ─── Audio format detection ───────────────────────────────────
def _detect_suffix(audio_bytes: bytes) -> str:
    """Detect audio format from magic bytes."""
    if not audio_bytes or len(audio_bytes) < 12:
        return ".webm"
    h = audio_bytes[:12]
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return ".wav"
    if h[:4] == b"OggS":
        return ".ogg"
    if h[:3] == b"ID3" or (h[0] == 0xFF and h[1] in (0xFB, 0xFA, 0xF3, 0xF2)):
        return ".mp3"
    if h[4:8] in (b"ftyp", b"moov", b"mdat"):
        return ".mp4"
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    return ".webm"


# ═══════════════════════════════════════════════════════════════
# PRIMARY: Deepgram Nova-3 Arabic
# ═══════════════════════════════════════════════════════════════

def deepgram_available() -> bool:
    """True if DEEPGRAM_API_KEY is set and deepgram-sdk is installed."""
    if not _DEEPGRAM_API_KEY:
        return False
    try:
        from deepgram import DeepgramClient  # noqa
        return True
    except ImportError:
        return False


def _transcribe_deepgram(audio_bytes: bytes) -> str | None:
    """
    Transcribe audio using Deepgram Nova-3 Arabic.

    Strategy:
    - Always sends to Nova-3 with language="ar" first (covers Egyptian,
      MSA, Gulf, Levantine — Nova-3 handles dialect detection automatically)
    - Nova-3 multilingual also handles English and code-switching natively
    - Falls back to None on any API error so Whisper can take over
    """
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, BufferSource

        client = DeepgramClient(api_key=_DEEPGRAM_API_KEY)

        # Determine MIME type from magic bytes
        suffix   = _detect_suffix(audio_bytes)
        mime_map = {
            ".wav":  "audio/wav",
            ".mp3":  "audio/mpeg",
            ".ogg":  "audio/ogg",
            ".mp4":  "audio/mp4",
            ".webm": "audio/webm",
        }
        mimetype = mime_map.get(suffix, "audio/webm")

        payload: BufferSource = {"buffer": audio_bytes}

        options = PrerecordedOptions(
            model=_DEEPGRAM_MODEL,
            language=_DEEPGRAM_LANG_AR,   # Arabic — covers all dialects
            smart_format=True,             # auto punctuation + formatting
            punctuate=True,
            utterances=False,
            filler_words=False,
        )

        response = client.listen.rest.v("1").transcribe_file(
            payload, options, timeout=30
        )

        transcript = (
            response["results"]["channels"][0]
            ["alternatives"][0]["transcript"]
        )
        transcript = transcript.strip()

        # Clean up punctuation artifacts at boundaries
        transcript = re.sub(r"^[.,،\s]+", "", transcript)
        transcript = re.sub(r"[.,،\s]+$",  "", transcript)
        transcript = re.sub(r"\s+", " ", transcript).strip()

        print(f"[speech] Deepgram Nova-3 transcript: '{transcript}'")
        return transcript or None

    except Exception as e:
        print(f"[speech] Deepgram error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# FALLBACK: Local Whisper
# ═══════════════════════════════════════════════════════════════

def whisper_available() -> bool:
    """True if ffmpeg is on PATH and openai-whisper is installed."""
    if _FFMPEG_PATH is None:
        return False
    try:
        import whisper  # noqa
        return True
    except ImportError:
        return False


@st.cache_resource
def _load_whisper_model():
    import whisper
    model = whisper.load_model(_WHISPER_MODEL_SIZE)
    print(f"[speech] Loaded Whisper fallback model: {_WHISPER_MODEL_SIZE}")
    return model


def _detect_spoken_language_whisper(model, wav_path: str) -> str:
    """Whisper built-in language detector. Falls back to 'ar'."""
    try:
        import whisper
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected = max(probs, key=probs.get)
        print(f"[speech] Whisper detected language: {detected}")
        return detected
    except Exception as e:
        print(f"[speech] Whisper language detection failed: {e}, defaulting to 'ar'")
        return "ar"


def _transcribe_whisper(audio_bytes: bytes) -> str | None:
    """Transcribe using local Whisper with Egyptian Arabic prompt."""
    if not whisper_available() or not audio_bytes:
        return None

    suffix   = _detect_suffix(audio_bytes)
    in_path  = None
    out_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            in_path = tmp_in.name

        print(f"[speech] Whisper fallback: suffix={suffix}, size={len(audio_bytes)} bytes")

        out_path = in_path + ".wav"
        result   = subprocess.run(
            [_FFMPEG_PATH, "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1",
             "-af", "loudnorm",
             "-f", "wav", out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print(f"[speech] ffmpeg error: {result.stderr.decode(errors='replace')}")
            return None

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            print("[speech] ffmpeg produced empty output")
            return None

        model         = _load_whisper_model()
        detected_lang = _detect_spoken_language_whisper(model, out_path)
        is_arabic     = detected_lang != "en"

        kwargs = {
            "fp16":                     False,
            "task":                     "transcribe",
            "language":                 "ar" if is_arabic else "en",
            "beam_size":                5,
            "temperature":              (0.0, 0.2, 0.4),
            "condition_on_previous_text": True,
        }
        if is_arabic:
            kwargs["initial_prompt"] = _ARABIC_INITIAL_PROMPT

        res  = model.transcribe(out_path, **kwargs)
        text = res.get("text", "").strip()

        text = re.sub(r"^[.,،\s]+", "", text)
        text = re.sub(r"[.,،\s]+$",  "", text)
        text = re.sub(r"\s+", " ", text).strip()

        print(f"[speech] Whisper transcript (lang={detected_lang}): '{text}'")
        return text or None

    except Exception as e:
        print(f"[speech] Whisper error: {e}")
        return None
    finally:
        for p in [in_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════
# PUBLIC STT API — used by app.py
# ═══════════════════════════════════════════════════════════════

def stt_available() -> bool:
    """True if any STT backend is available."""
    return deepgram_available() or whisper_available()


# Keep this name so app.py doesn't need to change
def whisper_available() -> bool:  # type: ignore[no-redef]
    """
    Kept for app.py compatibility.
    Returns True if ANY STT backend is available (Deepgram or Whisper).
    """
    return stt_available()


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Transcribe audio bytes to text.

    Tries Deepgram Nova-3 Arabic first (if API key is set).
    Falls back to local Whisper if Deepgram is unavailable or fails.
    Returns None if both backends fail.
    """
    if not audio_bytes:
        return None

    # Try Deepgram first
    if deepgram_available():
        result = _transcribe_deepgram(audio_bytes)
        if result:
            return result
        print("[speech] Deepgram failed, falling back to Whisper")

    # Whisper fallback
    if whisper_available():
        return _transcribe_whisper(audio_bytes)

    print("[speech] No STT backend available")
    return None


# ═══════════════════════════════════════════════════════════════
# TTS — unchanged, uses gTTS
# ═══════════════════════════════════════════════════════════════

def text_to_speech(text: str, lang: str = "en", dialect: str = None) -> bytes | None:
    """
    Convert answer text to speech using gTTS.
    Uses lang="ar" for Arabic and Franco Arabic.
    """
    try:
        from gtts import gTTS
        cleaned = re.sub(r"\[Page\s*\d+[^\]]*\]", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[*_`#]", "", cleaned).strip()
        if not cleaned:
            return None
        buf       = io.BytesIO()
        gtts_lang = "ar" if lang in ("arabic", "franco") else "en"
        gTTS(text=cleaned, lang=gtts_lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def tts_available() -> bool:
    try:
        import gtts  # noqa
        return True
    except ImportError:
        return False


def tts_audio_format(lang: str) -> str:
    return "audio/mp3"