"""
speech.py — Text-to-speech and speech-to-text.

Primary TTS:  gTTS  (pip install gtts)   — online, needs internet
Fallback TTS: pyttsx3 (pip install pyttsx3) — fully offline

STT: openai-whisper (pip install openai-whisper) + ffmpeg required
     brew install ffmpeg  OR  apt install ffmpeg

For Franco Arabic: strips Latin letters, passes remaining Arabic script to
the Arabic TTS engine (gTTS 'ar' / pyttsx3 Arabic voice if available).
"""

import io
import re
import tempfile
import os
import struct


def _clean_for_tts(text: str) -> str:
    """Strip [Page N | AR/EN] citations and normalise whitespace."""
    text = re.sub(r"\[Page\s*\d+(?:\s*\|\s*(?:AR|EN))?\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _franco_to_arabic_script(text: str) -> str:
    """Keep only Arabic-script chars so TTS engine can read something."""
    arabic_only = re.sub(r"[a-zA-Z]", " ", text)
    arabic_only = re.sub(r"\s+", " ", arabic_only).strip()
    if len(arabic_only) < 8:
        return "النص مكتوب بالفرانكو ولا يدعم التحويل الصوتي الكامل."
    return arabic_only


def _gtts_speak(text: str, gtts_lang: str) -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=gtts_lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def _pyttsx3_speak(text: str, lang: str) -> bytes | None:
    """Offline TTS via pyttsx3 — saves to temp WAV then returns bytes."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        if lang == "ar":
            voices = engine.getProperty("voices")
            for v in voices:
                if "arabic" in v.name.lower() or "ar" in (v.id or "").lower():
                    engine.setProperty("voice", v.id)
                    break
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return data if len(data) > 100 else None
    except Exception:
        return None


def text_to_speech(text: str, lang: str = "en", dialect: str = None) -> bytes | None:
    """
    Convert answer text to audio bytes (MP3 from gTTS, WAV from pyttsx3).

    lang: "english" | "arabic" | "franco"
    Returns None if no TTS engine is available or synthesis fails.
    """
    cleaned = _clean_for_tts(text)
    if not cleaned:
        return None

    if lang == "english":
        gtts_lang, tts_text = "en", cleaned
    elif lang == "franco":
        gtts_lang, tts_text = "ar", _franco_to_arabic_script(cleaned)
    else:
        gtts_lang, tts_text = "ar", cleaned

    if not tts_text.strip():
        return None

    result = _gtts_speak(tts_text, gtts_lang)
    if result:
        return result

    return _pyttsx3_speak(tts_text, gtts_lang)


def tts_available() -> bool:
    """True if at least one TTS engine is installed."""
    try:
        import gtts  # noqa
        return True
    except ImportError:
        pass
    try:
        import pyttsx3  # noqa
        return True
    except ImportError:
        pass
    return False


def tts_audio_format(lang: str) -> str:
    """Returns the streamlit audio format string."""
    try:
        import gtts  # noqa
        return "audio/mp3"
    except ImportError:
        return "audio/wav"


def whisper_available() -> bool:
    """True if Whisper transcription support is installed."""
    try:
        import whisper  # noqa
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


def _detect_audio_suffix(audio_bytes: bytes) -> str:
    """
    Detect format from magic bytes.
    WAV:  starts with b'RIFF'
    OGG:  starts with b'OggS'
    MP3:  starts with 0xFF 0xFB or ID3 tag
    Everything else: assume .webm (browser MediaRecorder default)
    ffmpeg handles all these when installed.
    """
    if not audio_bytes or len(audio_bytes) < 4:
        return ".webm"
    hdr = audio_bytes[:4]
    if hdr == b"RIFF":
        return ".wav"
    if hdr == b"OggS":
        return ".ogg"
    if hdr[:3] == b"ID3" or (hdr[0] == 0xFF and hdr[1] in (0xFB, 0xFA, 0xF3)):
        return ".mp3"
    return ".webm"


def transcribe_audio(audio_bytes: bytes) -> str | None:
    """
    Transcribe recorded audio bytes using Whisper.
    Requires: pip install openai-whisper  AND  ffmpeg on PATH.
    """
    try:
        import whisper  # noqa
    except ImportError:
        return None

    suffix = _detect_audio_suffix(audio_bytes)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        model = _load_whisper_model()
        result = model.transcribe(tmp_path)
        text = result.get("text", "").strip()
        return text or None
    except Exception as e:
        print(f"[speech] transcribe_audio error: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass