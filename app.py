import os
import streamlit as st
from auth import init_cookie_manager, require_login, logout, is_admin
from agent import run_agent
from setup import setup, render_page_to_image
from nlp_utils import detect_language_type, get_semantic_dialect
from utils import (
    translate, build_history_str, is_no_info_answer,
    summarize_history, confidence_badge, strip_citations,
)
from speech import (
    text_to_speech, tts_available, tts_audio_format,
    whisper_available, transcribe_audio,
)
from sessions import save_session, load_session, clear_session

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"


# ─── Analytics logging ────────────────────────────────────────
def _log_query(employee_id, intent, topic, lang, dialect, is_no_info, question):
    if lang == "arabic":
        log_lang = "arabic_egyptian" if dialect == "egyptian" else "arabic_msa"
    elif lang == "franco":
        log_lang = "franco"
    else:
        log_lang = "english"
    try:
        from database import get_db
        from sqlalchemy import text
        with get_db() as db:
            db.execute(text("""
                INSERT INTO analytics_log
                    (employee_id, intent, topic, language, unanswered, question_text, asked_at)
                VALUES (:eid, :intent, :topic, :lang, :unans, :q, NOW())
            """), {
                "eid":    employee_id,
                "intent": intent,
                "topic":  topic or "",
                "lang":   log_lang,
                "unans":  is_no_info,
                "q":      question[:300],
            })
    except Exception as e:
        print(f"[analytics] {e}")


# ─── Escalation helpers ───────────────────────────────────────
def _get_hr_email() -> str:
    hr_email = os.environ.get("HR_EMAIL", "").strip()
    if hr_email:
        return hr_email
    try:
        from database import get_db
        from sqlalchemy import text
        with get_db() as db:
            row = db.execute(text("""
                SELECT email FROM employees
                WHERE admin_role IN ('hr_admin', 'super_admin') AND is_active = TRUE
                ORDER BY admin_role DESC, id ASC LIMIT 1
            """)).fetchone()
        if row:
            return row[0]
    except Exception:
        pass
    return ""


def _send_escalation_email(employee_name: str, hr_email: str, question: str) -> bool:
    import smtplib
    from email.mime.text import MIMEText
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("SMTP_FROM", smtp_user)
    if not all([smtp_host, smtp_user, smtp_pass, hr_email]):
        return False
    try:
        body = (
            f"Hi HR Team,\n\n"
            f"The HR chatbot could not answer the following question "
            f"from {employee_name}:\n\n"
            f'  "{question}"\n\n'
            f"Please follow up with them directly.\n\n"
            f"— HR Assistant (automated)"
        )
        msg             = MIMEText(body)
        msg["Subject"]  = f"HR Chatbot: Unanswered query from {employee_name}"
        msg["From"]     = from_addr
        msg["To"]       = hr_email
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [hr_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[escalation] {e}")
        return False


def clean_query(text: str) -> str:
    return text.replace('"', "").replace("'", "").strip()


# ─── Page config + login ──────────────────────────────────────
st.set_page_config(
    page_title="HR Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    "<style>[data-testid='stSidebarNav'] {display: none;}</style>",
    unsafe_allow_html=True,
)
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

# ─── Load persisted history ───────────────────────────────────
if not st.session_state.history_loaded:
    hist, summ = load_session(st.session_state.employee_id)
    if hist:
        st.session_state.chat_history         = hist
        st.session_state.conversation_summary = summ
    st.session_state.history_loaded = True

# ─── Styles ───────────────────────────────────────────────────
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

# ─── Render chat history ──────────────────────────────────────
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

# ─── Main input handling ──────────────────────────────────────
question = st.session_state.pop("transcribed_voice_question", None)
if question is None:
    question = st.chat_input("Ask your question…")

if question:
    question = clean_query(question)
    st.session_state.translated_answer = None

    with chat_container:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    # ── Language + dialect detection ──────────────────
                    lang    = detect_language_type(question)
                    dialect = (get_semantic_dialect(question, dialect_pipe)
                               if lang == "arabic" else None)
                    llm     = en_llm if lang == "english" else ar_llm

                    history_str = build_history_str(
                        st.session_state.chat_history,
                        st.session_state.conversation_summary,
                    )

                    # ── Agentic call ──────────────────────────────────
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

                    answer           = result["answer"]
                    top_docs         = result["docs"]
                    cited_docs       = result["cited_docs"]
                    scores_dict      = result["scores"]
                    intent           = result["intent"]
                    topic            = result["topic"]
                    tools_called     = result["tools_called"]
                    personal_data_str = result["personal_data"]
                    _no_info         = is_no_info_answer(answer)

                    # ── Render answer ─────────────────────────────────
                    is_franco        = (lang == "franco")
                    is_arabic_script = (lang == "arabic")
                    css = "rtl-answer" if is_arabic_script else "ltr-answer"
                    st.markdown(
                        f'<div class="{css}">'
                        f'{answer.replace(chr(10), "<br>")}'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # ── Confidence badge (policy + hybrid only) ───────
                    if scores_dict and intent in ("policy", "hybrid") and not _no_info:
                        peer_scores  = list(scores_dict.values())
                        raw_score    = max(scores_dict.values())
                        label, color = confidence_badge(raw_score, peer_scores)
                        st.markdown(
                            f'<span class="conf-badge" style="background:{color}">'
                            f"{label}</span>",
                            unsafe_allow_html=True,
                        )

                    # ── Query Info expander ───────────────────────────
                    with st.expander("🔍 Query Info"):
                        st.info(
                            f"**Intent:** {intent} | **Lang:** {lang}"
                            + (f" | **Dialect:** {dialect}" if dialect else "")
                        )
                        if tools_called:
                            badges = " ".join(
                                f'<span class="tool-badge">🔧 {t}</span>'
                                for t in tools_called
                            )
                            st.markdown(
                                f"**Tools used:** {badges}",
                                unsafe_allow_html=True,
                            )

                    # ── Your data used (personal + hybrid only) ───────
                    if personal_data_str and intent in ("personal", "hybrid") and not _no_info:
                        with st.expander("📋 Your data used to answer this"):
                            st.code(personal_data_str, language=None)

                    # ── Source evidence (policy + hybrid only) ────────
                    if cited_docs and intent in ("policy", "hybrid") and not _no_info:
                        with st.expander("📄 Source Evidence"):
                            unique_pages: dict = {}
                            for d in cited_docs:
                                p_no = d.metadata.get("page", 0) + 1
                                src  = (
                                    ARABIC_PDF_PATH
                                    if ARABIC_PDF_PATH in d.metadata.get("source", "")
                                    else ENGLISH_PDF_PATH
                                )
                                if (src, p_no) not in unique_pages:
                                    unique_pages[(src, p_no)] = d

                            for (pdf, p_no), d in unique_pages.items():
                                src_label = (
                                    "Arabic PDF" if pdf == ARABIC_PDF_PATH
                                    else "English PDF"
                                )
                                st.markdown(f"**📄 Page {p_no} — {src_label}**")
                                try:
                                    st.image(
                                        render_page_to_image(pdf, p_no),
                                        width=700,
                                    )
                                except Exception:
                                    direction = "rtl" if pdf == ARABIC_PDF_PATH else "ltr"
                                    align     = "right" if direction == "rtl" else "left"
                                    st.markdown(
                                        f'<div style="direction:{direction};'
                                        f"text-align:{align};"
                                        f"background:#f9f9f9;padding:12px;"
                                        f'border-left:3px solid #1976d2;">'
                                        f"{d.page_content}</div>",
                                        unsafe_allow_html=True,
                                    )

                    # ── Save history ──────────────────────────────────
                    st.session_state.chat_history.append({
                        "role":      "user",
                        "content":   question,
                        "is_arabic": False,
                        "is_franco": is_franco,
                    })
                    st.session_state.chat_history.append({
                        "role":      "assistant",
                        "content":   answer,
                        "is_arabic": is_arabic_script,
                        "is_franco": is_franco,
                    })

                    n_msgs = len(st.session_state.chat_history)
                    if n_msgs % 8 == 0 or n_msgs <= 4:
                        st.session_state.conversation_summary = summarize_history(
                            en_llm,
                            st.session_state.chat_history,
                            st.session_state.conversation_summary,
                        )
                    save_session(
                        st.session_state.employee_id,
                        st.session_state.chat_history,
                        st.session_state.conversation_summary,
                    )

                    # ── Persist for bottom action bar ─────────────────
                    st.session_state.last_answer      = answer
                    st.session_state.last_lang        = lang
                    st.session_state.last_dialect     = dialect
                    st.session_state.last_scores      = scores_dict
                    st.session_state.last_cited_docs  = cited_docs
                    st.session_state.last_top_docs    = top_docs

                    # ── Analytics ─────────────────────────────────────
                    _log_query(
                        st.session_state.employee_id,
                        intent, topic, lang, dialect, _no_info, question,
                    )

                    # ── Escalation trigger ────────────────────────────
                    if _no_info:
                        hr_email = _get_hr_email()
                        if hr_email:
                            st.session_state["_esc_hr_email"] = hr_email
                            st.session_state["_esc_question"] = question
                            st.session_state["_esc_name"]     = st.session_state.employee_name
                            st.session_state["_show_esc"]     = True

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# ─── Escalation UI ────────────────────────────────────────────
if st.session_state.get("_show_esc"):
    st.info(
        "I could not find an answer in the policy documents. "
        "Would you like me to notify the HR team?"
    )
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("📧 Notify HR", key="esc_btn"):
            sent = _send_escalation_email(
                st.session_state.get("_esc_name", ""),
                st.session_state.get("_esc_hr_email", ""),
                st.session_state.get("_esc_question", ""),
            )
            st.session_state["_show_esc"] = False
            if sent:
                st.success("✅ The HR team has been notified.")
            else:
                st.warning(
                    "Email not sent. Check SMTP settings in .env or set HR_EMAIL."
                )
    with col_b:
        if st.button("Dismiss", key="esc_dismiss"):
            st.session_state["_show_esc"] = False
            st.rerun()


# ─── Voice panel ──────────────────────────────────────────────
if whisper_available():
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