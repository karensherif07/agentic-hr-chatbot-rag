"""
speech.py — Text-to-speech and speech-to-text.
"""

import io
import os
import re
import shutil
import subprocess
import tempfile
import streamlit as st

# ─── Guarantee ffmpeg is on PATH ───────────────────────────────
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


def _detect_suffix(audio_bytes: bytes) -> str:
    """
    Detect audio format from magic bytes and return the correct file extension.
    st.audio_input records WebM/Opus in Chrome/Edge, WAV in some others.
    Without the right extension ffmpeg can't auto-detect the codec and fails.
    """
    if not audio_bytes or len(audio_bytes) < 12:
        return ".webm"
    h = audio_bytes[:12]
    # WAV: "RIFF....WAVE"
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return ".wav"
    # OGG / Opus
    if h[:4] == b"OggS":
        return ".ogg"
    # MP3: ID3 tag or sync bytes
    if h[:3] == b"ID3" or (h[0] == 0xFF and h[1] in (0xFB, 0xFA, 0xF3, 0xF2)):
        return ".mp3"
    # MP4 / M4A: ftyp box at offset 4
    if h[4:8] in (b"ftyp", b"moov", b"mdat"):
        return ".mp4"
    # WebM / MKV: EBML header
    if h[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm"
    # Default — browser MediaRecorder usually outputs WebM
    return ".webm"


# ─── STT API ───────────────────────────────────────────────────
def whisper_available() -> bool:
    if _FFMPEG_PATH is None:
        return False
    try:
        import whisper
        return True
    except ImportError:
        return False


@st.cache_resource
def _load_whisper_model():
    import whisper
    return whisper.load_model("tiny")


def transcribe_audio(audio_bytes: bytes) -> str:
    if not whisper_available() or not audio_bytes:
        return None

    suffix   = _detect_suffix(audio_bytes)
    in_path  = None
    out_path = None

    try:
        # 1. Save with the correct extension so ffmpeg knows the format
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            in_path = tmp_in.name

        print(f"[speech] input suffix={suffix}, size={len(audio_bytes)} bytes, path={in_path}")

        # 2. Convert to 16 kHz mono WAV that Whisper expects
        out_path = in_path + ".wav"
        result = subprocess.run(
            [_FFMPEG_PATH, "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print(f"[speech] ffmpeg error:\n{result.stderr.decode(errors='replace')}")
            return None

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            print("[speech] ffmpeg produced empty/missing output file")
            return None

        wav_size = os.path.getsize(out_path)
        print(f"[speech] wav output size={wav_size} bytes")

        # 3. Transcribe
        model = _load_whisper_model()
        res   = model.transcribe(out_path, fp16=False, task="transcribe")
        text  = res.get("text", "").strip()
        print(f"[speech] transcription: '{text}'")
        return text or None

    except Exception as e:
        print(f"[speech] transcribe_audio error: {e}")
        return None
    finally:
        for p in [in_path, out_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


# ─── TTS API ───────────────────────────────────────────────────
def text_to_speech(text: str, lang: str = "en", dialect: str = None) -> bytes:
    try:
        from gtts import gTTS
        cleaned = re.sub(r"\[Page\s*\d+[^\]]*\]", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[*_`#]", "", cleaned).strip()
        if not cleaned:
            return None
        buf      = io.BytesIO()
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