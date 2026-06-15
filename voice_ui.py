"""
voice_ui.py — Voice input using plain st.button widgets.
No JS, no iframes. Two emoji buttons above chat input.
"""

import streamlit as st
from speech import whisper_available, transcribe_audio


def clean_query(text: str) -> str:
    return text.replace('"', "").replace("'", "").strip()


_VOICE_BTN_CSS = """
<style>

/* ── LOWER TOOLBAR MORE (IMPORTANT FIX) ───────── */
div[data-testid="stHorizontalBlock"]:has(button[title="Record voice"]) {
    margin-top: 18px !important;   /* ⬅️ increase this to lower more */
}

/* ── TIGHT GAP BETWEEN BUTTONS ───────────────── */
div[data-testid="stHorizontalBlock"]:has(button[title="Record voice"]) {
    gap: 2px !important;
}

/* ── COMPACT ICON BUTTON STYLE ───────────────── */
button[title="Record voice"],
button[title="Upload audio"] {
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    line-height: 1 !important;
    min-height: unset !important;

    border: 1px solid rgba(255,255,255,0.2) !important;
    background: rgba(40,40,60,0.9) !important;
    color: white !important;

    transition: all 0.15s ease-in-out;
}

/* ── HOVER ───────────────────────────────────── */
button[title="Record voice"]:hover,
button[title="Upload audio"]:hover {
    border-color: #4a7aff !important;
    background: rgba(74,122,255,0.25) !important;
}

/* ── PRESSED / ACTIVE STATE (NO RED ANYMORE) ─── */
button[title="Record voice"]:active,
button[title="Upload audio"]:active {
    background: rgba(74,122,255,0.35) !important;
    border-color: rgba(74,122,255,0.6) !important;
    transform: scale(0.96);
}

/* ── REMOVE STREAMLIT PRIMARY RED GLOW ──────── */
button[kind="primary"],
button[kind="primary"]:active,
button[kind="primary"]:focus {
    background: rgba(40,40,60,0.9) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    box-shadow: none !important;
    outline: none !important;
}
</style>
"""


def render_voice_panel():
    if not whisper_available():
        return

    st.markdown(_VOICE_BTN_CSS, unsafe_allow_html=True)

    mode = st.session_state.get("_voice_mode")

    # ── Toolbar layout ───────────────────────────────
    spacer, toolbar = st.columns([8, 2])

    with toolbar:
        b1, b2 = st.columns(2)

        with b1:
            mic_clicked = st.button(
                "🎙️",
                key="btn_mic",
                help="Record voice",
                type="primary" if mode == "mic" else "secondary",
                use_container_width=True,
            )

        with b2:
            upload_clicked = st.button(
                "📎",
                key="btn_upload",
                help="Upload audio",
                type="primary" if mode == "upload" else "secondary",
                use_container_width=True,
            )

    # ── Toggle logic ────────────────────────────────
    if mic_clicked:
        new_mode = None if mode == "mic" else "mic"
        st.session_state["_voice_mode"] = new_mode
        st.session_state.pop("_mic_transcript", None)
        st.session_state.pop("_cached_audio_bytes", None)
        st.rerun()

    if upload_clicked:
        new_mode = None if mode == "upload" else "upload"
        st.session_state["_voice_mode"] = new_mode
        st.session_state.pop("_mic_transcript", None)
        st.session_state.pop("_cached_audio_bytes", None)
        st.rerun()

    # ── Chat input spacing tweak ────────────────────
    st.markdown("""
    <style>
    div[data-testid="stChatInput"] {
        margin-top: -20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Re-read mode after rerun ────────────────────
    mode = st.session_state.get("_voice_mode")

    if mode == "mic":
        rec = st.audio_input(
            "Record your question:",
            key="mic_input",
            label_visibility="collapsed",
        )
        if rec:
            _handle_audio(rec)

    elif mode == "upload":
        up = st.file_uploader(
            "Upload audio",
            type=["wav", "mp3", "m4a"],
            key="file_input",
            label_visibility="collapsed",
        )
        if up:
            _handle_audio(up)


def _handle_audio(audio_source):
    raw_audio = audio_source.read()
    if not raw_audio:
        return

    if st.session_state.get("_cached_audio_bytes") != raw_audio:
        st.session_state["_cached_audio_bytes"] = raw_audio
        st.session_state.pop("_mic_transcript", None)

    if st.session_state.get("_mic_transcript") is None:
        with st.spinner("Transcribing…"):
            txt = transcribe_audio(st.session_state["_cached_audio_bytes"])

        if not txt:
            st.warning("Could not transcribe — please try again.")
            return

        st.session_state["_mic_transcript"] = txt

    transcript = st.session_state.get("_mic_transcript")
    if not transcript:
        return

    edited = st.text_area(
        "Confirm or edit:",
        value=transcript,
        height=72,
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 5])

    with c1:
        if st.button("🚀 Send", key="voice_send"):
            st.session_state.transcribed_voice_question = clean_query(edited)
            st.session_state.pop("_mic_transcript", None)
            st.session_state.pop("_cached_audio_bytes", None)
            st.session_state["_voice_mode"] = None
            st.rerun()

    with c2:
        if st.button("🗑️ Reset", key="voice_reset"):
            st.session_state.pop("_mic_transcript", None)
            st.session_state.pop("_cached_audio_bytes", None)
            st.rerun()