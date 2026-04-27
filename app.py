"""
app.py — Main entry point for the HR Assistant chatbot.

Responsibilities:
- Page config, CSS injection, auth
- Sidebar (user info, sign out, clear chat)
- Session state initialisation + history restore
- Model loading via setup()
- Main chat loop: question → run_agent() → render
- Delegates rendering to chat_ui.py
- Delegates voice input to voice_ui.py
- Delegates escalation UI to escalation_ui.py
- Bottom action bar (translate + TTS)
"""

import streamlit as st

from auth import init_cookie_manager, require_login, logout, is_admin
from agent import run_agent
from setup import setup
from nlp_utils import detect_language_type, get_semantic_dialect
from utils import (
    build_history_str, is_no_info_answer,
    translate, strip_citations,
)
from speech import tts_available, tts_audio_format, text_to_speech
from sessions import load_session, clear_session

from chat_ui import render_answer, save_and_summarise, maybe_trigger_escalation, log_query
from voice_ui import render_voice_panel, clean_query
from escalation_ui import render_escalation_ui


# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="HR Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<style>[data-testid='stSidebarNav'] {display: none;}</style>",
    unsafe_allow_html=True,
)

# ─── Auth ─────────────────────────────────────────────────────
init_cookie_manager()
require_login()

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**{st.session_state.employee_name}**")
    role_label = f"{st.session_state.employee_grade} · {st.session_state.employee_dept}"
    if st.session_state.get("admin_role"):
        role_label += f" · 🔑 {st.session_state.admin_role}"
    st.caption(role_label)

    if st.button("Sign out"):
        logout()
    st.divider()

    if is_admin():
        st.page_link("pages/admin_portal.py", label="⚙️ Admin Portal", icon="⚙️")
        st.divider()

    if st.button("🗑 Clear chat history"):
        clear_session(st.session_state.employee_id)
        st.session_state.chat_history         = []
        st.session_state.conversation_summary = ""
        st.rerun()

# ─── Session state defaults ───────────────────────────────────
_DEFAULTS = {
    "chat_history":               [],
    "conversation_summary":       "",
    "history_loaded":             False,
    "translated_answer":          None,
    "last_answer":                None,
    "last_lang":                  None,
    "last_dialect":               None,
    "last_cited_docs":            [],
    "last_top_docs":              [],
    "last_cited_pages":           set(),
    "last_scores":                {},
    "transcribed_voice_question": None,
    "_mic_transcript":            None,
    "_cached_audio_bytes":        None,
    "_show_esc":                  False,
    "_esc_hr_email":              "",
    "_esc_question":              "",
    "_esc_name":                  "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Restore persisted history ────────────────────────────────
if not st.session_state.history_loaded:
    hist, summ = load_session(st.session_state.employee_id)
    if hist:
        st.session_state.chat_history         = hist
        st.session_state.conversation_summary = summ
    st.session_state.history_loaded = True

# ─── Global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.rtl-answer {
    direction: rtl; text-align: right;
    font-size: 1rem; line-height: 1.9; padding: .5rem 0;
}
.ltr-answer {
    direction: ltr; text-align: left;
    font-size: 1rem; line-height: 1.9; padding: .5rem 0;
}
.conf-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: .72rem; font-weight: 600; color: white;
    margin-left: 8px; vertical-align: middle;
}
.mic-container {
    background: #f8f9fa; border-radius: 20px; padding: 20px;
    border: 1px solid #dee2e6; margin-top: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,.05);
}
.tool-badge {
    display: inline-block; padding: 1px 7px; border-radius: 10px;
    font-size: .7rem; background: #e3f2fd; color: #1565c0;
    margin: 2px; border: 1px solid #bbdefb;
}
</style>
""", unsafe_allow_html=True)

# ─── Admin gate ───────────────────────────────────────────────
if is_admin():
    st.info(
        "👋 You are logged in as an HR administrator.\n\n"
        "Use the **⚙️ Admin Portal** button in the sidebar."
    )
    st.stop()

# ─── Title ────────────────────────────────────────────────────
st.title("💼 HR Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian), or Franco Arabic.")

# ─── Load models ──────────────────────────────────────────────
try:
    ar_index, en_index, ar_llm, en_llm, reranker, dialect_pipe, ara_tokenizer = setup()
    st.session_state.ar_llm = ar_llm
    st.session_state.en_llm = en_llm
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# ─── Render existing chat history ────────────────────────────
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            is_rtl = msg.get("is_arabic", False) and not msg.get("is_franco", False)
            css    = "rtl-answer" if is_rtl else "ltr-answer"
            st.markdown(
                f'<div class="{css}">'
                f'{msg["content"].replace(chr(10), "<br>")}'
                f"</div>",
                unsafe_allow_html=True,
            )

# ─── Question input ───────────────────────────────────────────
question = st.session_state.pop("transcribed_voice_question", None)
if question is None:
    question = st.chat_input("Ask your question…")

# ─── Main chat loop ───────────────────────────────────────────
if question:
    question = clean_query(question)
    st.session_state.translated_answer = None

    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    # Language + dialect detection
                    lang    = detect_language_type(question)
                    dialect = (get_semantic_dialect(question, dialect_pipe)
                               if lang == "arabic" else None)
                    llm     = en_llm if lang == "english" else ar_llm

                    history_str = build_history_str(
                        st.session_state.chat_history,
                        st.session_state.conversation_summary,
                    )

                    # Agentic call
                    result = run_agent(
                        question=question,
                        employee_id=st.session_state.employee_id,
                        lang=lang,
                        dialect=dialect,
                        history_str=history_str,
                        ar_index=ar_index,
                        en_index=en_index,
                        llm=llm,
                        reranker=reranker,
                        ara_tokenizer=ara_tokenizer,
                    )

                    answer            = result["answer"]
                    cited_docs        = result["cited_docs"]
                    scores_dict       = result["scores"]
                    intent            = result["intent"]
                    topic             = result["topic"]
                    tools_called      = result["tools_called"]
                    personal_data_str = result["personal_data"]
                    _no_info          = is_no_info_answer(answer)
                    is_franco         = (lang == "franco")
                    is_arabic_script  = (lang == "arabic")

                    # Render answer + all expanders
                    render_answer(
                        answer=answer,
                        intent=intent,
                        lang=lang,
                        dialect=dialect,
                        tools_called=tools_called,
                        cited_docs=cited_docs,
                        scores_dict=scores_dict,
                        personal_data_str=personal_data_str,
                        _no_info=_no_info,
                    )

                    # Save history + summarise
                    save_and_summarise(
                        question=question,
                        answer=answer,
                        lang=lang,
                        is_franco=is_franco,
                        is_arabic_script=is_arabic_script,
                        en_llm=en_llm,
                        employee_id=st.session_state.employee_id,
                    )

                    # Persist for bottom action bar
                    st.session_state.last_answer     = answer
                    st.session_state.last_lang       = lang
                    st.session_state.last_dialect    = dialect
                    st.session_state.last_scores     = scores_dict
                    st.session_state.last_cited_docs = cited_docs

                    # Analytics
                    log_query(
                        st.session_state.employee_id,
                        intent, topic, lang, dialect, _no_info, question,
                    )

                    # Escalation
                    maybe_trigger_escalation(
                        _no_info, question, st.session_state.employee_name
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ─── Escalation UI ────────────────────────────────────────────
render_escalation_ui()

# ─── Voice panel ──────────────────────────────────────────────
render_voice_panel()

# ─── Bottom action bar ────────────────────────────────────────
if st.session_state.get("last_answer"):
    ans = st.session_state.last_answer
    l   = st.session_state.last_lang
    st.divider()
    cols = st.columns([1, 1, 4])

    with cols[0]:
        if st.button("🔄 Translate"):
            target = "Arabic" if l == "english" else "English"
            llm2   = (st.session_state.ar_llm if l == "english"
                      else st.session_state.en_llm)
            st.session_state.translated_answer = translate(llm2, ans, target)

    with cols[1]:
        if tts_available() and st.button("🔊 Read"):
            audio = text_to_speech(
                strip_citations(ans),
                lang=l,
                dialect=st.session_state.last_dialect,
            )
            if audio:
                st.audio(audio, format=tts_audio_format(l))

    if st.session_state.get("translated_answer"):
        css = "ltr-answer" if l != "english" else "rtl-answer"
        st.markdown(
            f'<div class="{css}">'
            f'{st.session_state.translated_answer}'
            f"</div>",
            unsafe_allow_html=True,
        )