"""
voice_ui.py
Handles the voice input panel:
- Record tab (st.audio_input)
- Upload tab (file uploader)
- Transcription via Deepgram / Whisper
- Confirm / edit / send transcript UI
"""

import streamlit as st
from speech import whisper_available, transcribe_audio


def clean_query(text: str) -> str:
    return text.replace('"', "").replace("'", "").strip()


def render_voice_panel():
    """
    Render the full voice input panel.
    Sets st.session_state.transcribed_voice_question when the user hits Send,
    which app.py picks up on the next rerun as the question input.
    Only rendered if any STT backend is available.
    """
    if not whisper_available():
        return

    st.markdown('<div class="mic-container">', unsafe_allow_html=True)
    t_rec, t_up  = st.tabs(["🎙️ Record", "📁 Upload"])
    audio_source = None

    with t_rec:
        rec = st.audio_input("Record", key="mic_input", label_visibility="collapsed")
        if rec:
            audio_source = rec

    with t_up:
        up = st.file_uploader(
            "Upload", type=["wav", "mp3", "m4a"],
            key="file_input", label_visibility="collapsed",
        )
        if up:
            audio_source = up

    if audio_source:
        raw_audio = audio_source.read()
        if raw_audio:
            # Only re-transcribe if audio bytes actually changed
            if st.session_state.get("_cached_audio_bytes") != raw_audio:
                st.session_state["_cached_audio_bytes"] = raw_audio
                st.session_state.pop("_mic_transcript", None)

            if st.session_state.get("_mic_transcript") is None:
                with st.status("🎙️ Transcribing...", expanded=False) as status:
                    txt = transcribe_audio(st.session_state["_cached_audio_bytes"])
                    if txt:
                        st.session_state["_mic_transcript"] = txt
                        status.update(label="✅ Ready", state="complete")
                    else:
                        status.update(label="❌ Try again", state="error")

            transcript = st.session_state.get("_mic_transcript")
            if transcript:
                edited = st.text_area("Confirm query:", value=transcript, height=80)
                c1, c2 = st.columns([1, 4])
                if c1.button("🚀 Send"):
                    st.session_state.transcribed_voice_question = clean_query(edited)
                    st.session_state.pop("_mic_transcript", None)
                    st.session_state.pop("_cached_audio_bytes", None)
                    st.rerun()
                if c2.button("🗑️ Reset"):
                    st.session_state.pop("_mic_transcript", None)
                    st.session_state.pop("_cached_audio_bytes", None)
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)