"""
speech.py — Text-to-speech and speech-to-text.

STT: openai-whisper (pip install openai-whisper) + ffmpeg
"""

import io
import os
import re
import shutil
import subprocess
import tempfile

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

def _resolve_ffmpeg() -> "str | None":
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
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + current_path

# ─── Helpers ───────────────────────────────────────────────────
def _clean_for_tts(text: str) -> str:
    text = re.sub(r"\[Page\s*\d+(?:\s*\|\s*(?:AR|EN))?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _franco_to_arabic_script(text: str) -> str:
    arabic_only = re.sub(r"[a-zA-Z]", " ", text)
    arabic_only = re.sub(r"\s+", " ", arabic_only).strip()
    return arabic_only if len(arabic_only) >= 8 else "النص غير مدعوم صوتياً"

def _gtts_speak(text: str, gtts_lang: str) -> "bytes | None":
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=gtts_lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

# ─── Public TTS API ────────────────────────────────────────────
def text_to_speech(text: str, lang: str = "en", dialect: str = None) -> "bytes | None":
    cleaned = _clean_for_tts(text)
    if not cleaned: return None
    gtts_lang = "ar" if lang in ("arabic", "franco") else "en"
    tts_text = _franco_to_arabic_script(cleaned) if lang == "franco" else cleaned
    return _gtts_speak(tts_text, gtts_lang)

def tts_available() -> bool:
    try:
        import gtts
        return True
    except ImportError:
        return False

def tts_audio_format(lang: str) -> str:
    return "audio/mp3"

def whisper_available() -> bool:
    if _FFMPEG_PATH is None: return False
    try:
        import whisper
        return True
    except ImportError:
        return False

_WHISPER_MODEL = None

def _load_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper
        _WHISPER_MODEL = whisper.load_model("small")
    return _WHISPER_MODEL

def transcribe_audio(audio_bytes: bytes) -> "str | None":
    """Transcribe raw audio bytes with Whisper."""
    if not whisper_available() or not audio_bytes:
        return None
    
    tmp_path = None
    try:
        # Whisper works best when it can read from a file with a proper extension
        # Streamlit's st.audio_input usually provides WAV or WebM
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        model  = _load_whisper_model()
        result = model.transcribe(tmp_path)
        return result.get("text", "").strip() or None
    except Exception as e:
        print(f"[speech] transcribe_audio error: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)