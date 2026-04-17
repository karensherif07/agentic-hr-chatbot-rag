"""
speech.py — Text-to-speech and speech-to-text.

STT strategy:
1. Use whisper "medium" model — minimum viable for Arabic. Use "large-v3" for GPU.
2. First pass: auto-detect language using Whisper's built-in detector.
3. If detected as Arabic (or ambiguous): re-transcribe with language="ar" and
   an Egyptian Arabic initial_prompt to bias the decoder toward dialect vocab.
4. If detected as English: transcribe with language="en" — no Arabic prompt.
5. Franco Arabic is spoken as Arabic sounds, so Whisper detects it as "ar"
   and the NLP layer downstream handles the Latin-script normalization.
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

# ─── Whisper model config ──────────────────────────────────────
# "tiny"   → fast, ~terrible for Arabic (what you had before)
# "base"   → still unreliable for Egyptian
# "small"  → acceptable, misses dialectal words often
# "medium" → good quality, ~2.5 GB VRAM, recommended minimum for Arabic
# "large-v3" → best quality, ~6 GB VRAM, use if you have GPU
#
# Change this to "large-v3" for your final demo / submission if your machine
# has a decent GPU — it makes a very noticeable difference.
_WHISPER_MODEL_SIZE = "medium"

# Egyptian Arabic initial prompt — fed to Whisper's decoder as context.
# This steers token probabilities toward colloquial Egyptian vocabulary and
# keeps Whisper from switching to English or MSA mid-sentence.
_ARABIC_INITIAL_PROMPT = (
    "محادثة بالعامية المصرية عن سياسات الموارد البشرية. "
    "الموظف يسأل عن الإجازات والراتب والتقييم والترقية والتدريب. "
    "مش، عايز، عايزة، إيه، ليه، فين، إزاي، كده، دلوقتي، "
    "بتاعي، بتاعتي، عندي، عندك، ممكن، طيب، تمام، يعني."
)


def _detect_suffix(audio_bytes: bytes) -> str:
    """
    Detect audio format from magic bytes and return the correct file extension.
    st.audio_input records WebM/Opus in Chrome/Edge, WAV in some others.
    Without the right extension ffmpeg can't auto-detect the codec and fails.
    """
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


# ─── STT API ───────────────────────────────────────────────────
def whisper_available() -> bool:
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
    # suppress_tokens=[] prevents Whisper from suppressing Arabic punctuation tokens
    model = whisper.load_model(_WHISPER_MODEL_SIZE)
    print(f"[speech] Loaded Whisper model: {_WHISPER_MODEL_SIZE}")
    return model


def _detect_spoken_language(model, wav_path: str) -> str:
    """
    Use Whisper's built-in language detector on the first 30 seconds of audio.
    Returns an ISO-639-1 code e.g. "ar", "en".
    Falls back to "ar" on any error — safer for this app's audience.
    """
    try:
        import whisper
        import torch
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel   = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        detected = max(probs, key=probs.get)
        print(f"[speech] detected language: {detected} (confidence: {probs[detected]:.2f})")
        return detected
    except Exception as e:
        print(f"[speech] language detection failed: {e}, defaulting to 'ar'")
        return "ar"


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text with automatic language detection.

    Strategy:
    - Pass 1: detect spoken language using Whisper's built-in detector.
    - If Arabic (or any non-English): transcribe with language="ar" and the
      Egyptian Arabic initial_prompt to bias the decoder toward dialect vocab.
    - If English: transcribe with language="en" — no Arabic prompt injected.
    - Franco Arabic is spoken with Arabic phonetics so Whisper detects it as
      "ar"; the NLP layer downstream handles Latin-script normalization.
    """
    if not whisper_available() or not audio_bytes:
        return None

    suffix   = _detect_suffix(audio_bytes)
    in_path  = None
    out_path = None

    try:
        # 1. Write audio with correct extension so ffmpeg knows the codec
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            in_path = tmp_in.name

        print(f"[speech] input suffix={suffix}, size={len(audio_bytes)} bytes")

        # 2. Convert to 16 kHz mono WAV — required by Whisper
        out_path = in_path + ".wav"
        result = subprocess.run(
            [_FFMPEG_PATH, "-y", "-i", in_path,
             "-ar", "16000", "-ac", "1",
             "-af", "loudnorm",      # normalize quiet microphone input
             "-f", "wav", out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print(f"[speech] ffmpeg error:\n{result.stderr.decode(errors='replace')}")
            return None

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 100:
            print("[speech] ffmpeg produced empty/missing output file")
            return None

        model = _load_whisper_model()

        # 3. Auto-detect spoken language
        detected_lang = _detect_spoken_language(model, out_path)
        is_arabic = detected_lang != "en"

        # 4. Transcribe with the right language and optional Arabic prompt
        transcribe_kwargs = {
            "fp16": False,
            "task": "transcribe",
            "language": "ar" if is_arabic else "en",
            "beam_size": 5,
            "temperature": (0.0, 0.2, 0.4),
            "condition_on_previous_text": True,
        }
        if is_arabic:
            transcribe_kwargs["initial_prompt"] = _ARABIC_INITIAL_PROMPT

        res  = model.transcribe(out_path, **transcribe_kwargs)
        text = res.get("text", "").strip()

        # 5. Strip punctuation artifacts Whisper adds at boundaries
        text = re.sub(r"^[\.\,\،\s]+", "", text)
        text = re.sub(r"[\.\،\s]+$",   "", text)
        text = re.sub(r"\s+", " ", text).strip()

        print(f"[speech] transcription (lang={detected_lang}): '{text}'")
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
    """
    Convert answer text to speech.

    Uses gTTS with lang="ar" for Arabic and Egyptian (gTTS uses Google's
    neural Arabic TTS which sounds natural for both MSA and Egyptian).
    """
    try:
        from gtts import gTTS
        # Strip citation tags and markdown artifacts
        cleaned = re.sub(r"\[Page\s*\d+[^\]]*\]", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[*_`#]", "", cleaned).strip()
        if not cleaned:
            return None
        buf = io.BytesIO()
        gtts_lang = "ar" if lang in ("arabic", "franco") else "en"
        # slow=False is fine — Google's Arabic TTS is already clear at normal speed
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