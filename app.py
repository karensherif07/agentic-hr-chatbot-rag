"""
app.py — Main entry point for the HR Assistant chatbot.
"""

import streamlit as st

from auth import init_cookie_manager, require_login, logout, is_admin
from agent import run_agent
from setup import setup
from nlp_utils import detect_language_type, get_semantic_dialect
from utils import build_history_str, is_no_info_answer, translate, strip_citations
from speech import tts_available, tts_audio_format, text_to_speech
from sessions import load_session, clear_session

from chat_ui import render_answer, save_and_summarise, maybe_trigger_escalation, log_query
from voice_ui import render_voice_panel, clean_query
from escalation_ui import render_escalation_ui, get_hr_email, send_escalation_email


st.set_page_config(page_title="HR Assistant", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>[data-testid='stSidebarNav'] {display: none;}</style>", unsafe_allow_html=True)

init_cookie_manager()
require_login()

def _send_contact_hr_email(
    employee_name: str,
    employee_email: str,
    hr_email: str,
    subject: str,
    body: str,
) -> bool:
    import os, smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_pass, hr_email]):
        return False

    try:
        msg = MIMEMultipart()
        msg["Subject"] = f"[Employee Message] {subject}"
        msg["From"]    = from_addr
        msg["To"]      = hr_email
        msg["Reply-To"] = employee_email
        full_body = (
            f"Message from: {employee_name} ({employee_email})\n"
            f"{'─' * 40}\n\n{body}"
        )
        msg.attach(MIMEText(full_body, "plain"))
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [hr_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[contact_hr] {e}")
        return False

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**{st.session_state.employee_name}**")
    role_label = f"{st.session_state.employee_grade} · {st.session_state.employee_dept}"
    if st.session_state.get("admin_role"):
        role_label += f" · 🔑 {st.session_state.admin_role}"
    st.caption(role_label)
    st.divider()

    if is_admin():
        st.page_link("pages/admin_portal.py", label="Admin Portal", icon="⚙️")
        st.divider()

    if not is_admin():
        with st.expander("📧 Contact HR"):
            with st.form("contact_hr_form", clear_on_submit=True):
                subject = st.text_input("Subject", placeholder="e.g. Leave request question")
                body = st.text_area("Message", placeholder="Write your message to HR here...", height=120)
                send_btn = st.form_submit_button("Send", use_container_width=True)
            if send_btn:
                if not subject.strip() or not body.strip():
                    st.warning("Please fill in both subject and message.")
                else:
                    hr_email = get_hr_email()
                    if not hr_email:
                        st.error("HR email not configured. Contact your IT team.")
                    else:
                        sent = _send_contact_hr_email(
                            employee_name=st.session_state.employee_name,
                            employee_email=st.session_state.employee_email,
                            hr_email=hr_email,
                            subject=subject.strip(),
                            body=body.strip(),
                        )
                        if sent:
                            st.success("✅ Your message has been sent to HR.")
                        else:
                            st.error("Failed to send. Check SMTP settings in .env.")
        st.divider()

    if st.button("🗑 Clear chat history"):
        clear_session(st.session_state.employee_id)
        st.session_state.chat_history         = []
        st.session_state.conversation_summary = ""
        st.rerun()

    # ── Sign out pinned to the very bottom of sidebar ─────────
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] > div:first-child {
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
    }
    .sidebar-spacer { flex: 1 1 auto; }
    </style>
    <div class="sidebar-spacer"></div>
    
    """, unsafe_allow_html=True)
    st.divider()
    if st.button("Sign out", use_container_width=True, key="signout_btn"):
        logout()


# ─── Session state defaults ───────────────────────────────────
_DEFAULTS = {
    "chat_history": [], "conversation_summary": "", "history_loaded": False,
    "translated_answer": None, "last_answer": None, "last_lang": None,
    "last_dialect": None, "last_cited_docs": [], "last_top_docs": [],
    "last_cited_pages": set(), "last_scores": {},
    "transcribed_voice_question": None, "_mic_transcript": None,
    "_cached_audio_bytes": None, "_show_esc": False,
    "_esc_hr_email": "", "_esc_question": "", "_esc_name": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.history_loaded:
    hist, summ = load_session(st.session_state.employee_id)
    if hist:
        st.session_state.chat_history         = hist
        st.session_state.conversation_summary = summ
    st.session_state.history_loaded = True

# ─── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Strip Streamlit's own chat bubble backgrounds so ours win ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 !important;
}

/* ── Assistant reply bubble: semi-transparent lift over navy bg ── */
.ltr-answer, .rtl-answer, .welcome-bubble {
    font-size: 1rem;
    line-height: 1.8;
    padding: .7rem 1rem;
    border-radius: 10px;
    margin: .15rem 0;
    color: inherit;
    background: rgba(255, 255, 255, 0.10);
}
.rtl-answer    { direction: rtl; text-align: right; }
.ltr-answer    { direction: ltr; text-align: left;  }
.welcome-bubble{ direction: ltr; text-align: left;  }

/* ── User message bubble: warmer tint to distinguish from assistant ── */
.user-bubble {
    font-size: 1rem;
    line-height: 1.8;
    padding: .7rem 1rem;
    border-radius: 10px;
    margin: .15rem 0;
    color: inherit;
    background: rgba(100, 160, 255, 0.13);
    direction: ltr;
    text-align: left;
}

/* ── Badges ── */
.conf-badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: .72rem; font-weight: 600; color: white;
    margin-left: 8px; vertical-align: middle;
}
.tool-badge {
    display: inline-block; padding: 1px 7px; border-radius: 10px;
    font-size: .7rem; background: #e3f2fd; color: #1565c0;
    margin: 2px; border: 1px solid #bbdefb;
}

/* ── Voice buttons: rendered via components.v1.html, no CSS needed ── */
/* Pad textarea right edge so typed text clears the button area       */
div[data-testid="stChatInputContainer"] textarea {
    padding-right: 100px !important;
}
div[data-testid="stChatInputContainer"] textarea {
    padding-right: 100px !important;
}
</style>
""", unsafe_allow_html=True)

if is_admin():
    st.info("👋 You are logged in as an HR administrator.\n\nUse the **⚙️ Admin Portal** button in the sidebar.")
    st.stop()

# st.title("💼 HR Assistant")

# ─── Load models ──────────────────────────────────────────────
try:
    (ar_index, en_index,
     routing_llm, en_llm, ar_llm, critique_llm,
     reranker, dialect_pipe, ara_tokenizer) = setup()
    st.session_state.en_llm       = en_llm
    st.session_state.ar_llm       = ar_llm
    st.session_state.dialect_pipe = dialect_pipe
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# ─── Render history ───────────────────────────────────────────
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                '<div class="welcome-bubble">'
                "👋 <strong>Welcome to the Employee Support Chatbot.</strong><br><br>"
                "I can help with HR policies and employee information.<br>"
                "You may ask questions in:<br>"
                "&nbsp;&nbsp;🌐 English<br>"
                "&nbsp;&nbsp;📜 Modern Standard Arabic<br>"
                "&nbsp;&nbsp;🗣️ Egyptian Arabic<br>"
                "&nbsp;&nbsp;💬 Franco-Arabic<br><br>"
                "How can I assist you today?"
                "</div>",
                unsafe_allow_html=True,
            )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{msg["content"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
            else:
                is_rtl = msg.get("is_arabic", False) and not msg.get("is_franco", False)
                css = "rtl-answer" if is_rtl else "ltr-answer"
                st.markdown(
                    f'<div class="{css}">{msg["content"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )

# ─── Input ────────────────────────────────────────────────────
render_voice_panel()

question = st.session_state.pop("transcribed_voice_question", None)
if question is None:
    question = st.chat_input("Ask your question…")

# ─── Main loop ────────────────────────────────────────────────
if question:
    question = clean_query(question)
    st.session_state.translated_answer = None

    with chat_container:
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-bubble">{question.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    lang    = detect_language_type(question)
                    dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else None

                    history_str = build_history_str(
                        st.session_state.chat_history,
                        st.session_state.conversation_summary,
                    )

                    result = run_agent(
                        question=question,
                        employee_id=st.session_state.employee_id,
                        lang=lang,
                        dialect=dialect,
                        history_str=history_str,
                        ar_index=ar_index,
                        en_index=en_index,
                        routing_llm=routing_llm,
                        en_llm=en_llm,
                        ar_llm=ar_llm,
                        critique_llm=critique_llm,
                        reranker=reranker,
                        ara_tokenizer=ara_tokenizer,
                        dialect_pipe=dialect_pipe,
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

                    render_answer(
                        answer=answer, intent=intent, lang=lang, dialect=dialect,
                        tools_called=tools_called, cited_docs=cited_docs,
                        scores_dict=scores_dict, personal_data_str=personal_data_str,
                        _no_info=_no_info,
                    )

                    summary_llm = en_llm if lang == "english" else ar_llm
                    save_and_summarise(
                        question=question, answer=answer, lang=lang,
                        is_franco=is_franco, is_arabic_script=is_arabic_script,
                        summary_llm=summary_llm,
                        employee_id=st.session_state.employee_id,
                    )

                    st.session_state.last_answer     = answer
                    st.session_state.last_lang       = lang
                    st.session_state.last_dialect    = dialect
                    st.session_state.last_scores     = scores_dict
                    st.session_state.last_cited_docs = cited_docs

                    log_query(st.session_state.employee_id, intent, topic,
                              lang, dialect, _no_info, question)
                    maybe_trigger_escalation(
                        _no_info, question, st.session_state.employee_name
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

render_escalation_ui()

# ─── Bottom action bar ────────────────────────────────────────
if st.session_state.get("last_answer"):
    ans = st.session_state.last_answer
    l   = st.session_state.last_lang
    st.divider()
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("🔄 Translate"):
            target = "Arabic" if l == "english" else "English"
            llm2   = st.session_state.ar_llm if l == "english" else st.session_state.en_llm
            st.session_state.translated_answer = translate(llm2, ans, target)
    with cols[1]:
        if tts_available() and st.button("🔊 Read"):
            audio = text_to_speech(strip_citations(ans), lang=l,
                                   dialect=st.session_state.last_dialect)
            if audio:
                st.audio(audio, format=tts_audio_format(l))
    if st.session_state.get("translated_answer"):
        css = "ltr-answer" if l != "english" else "rtl-answer"
        st.markdown(
            f'<div class="{css}">{st.session_state.translated_answer}</div>',
            unsafe_allow_html=True,
        )