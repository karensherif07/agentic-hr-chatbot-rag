import streamlit as st
from auth import init_cookie_manager, require_login, logout
from intent import classify_intent
from personal_data import fetch_for_intent
from personal_prompts import get_personal_prompt, get_hybrid_prompt, format_personal_data
from setup import setup, render_page_to_image
from nlp_utils import (detect_language_type, get_semantic_dialect,
                       franco_to_arabic, normalize_arabic, normalize_english)
from retrieval import retrieve, rerank, rrf, build_retrieval_query
from utils import (
    translate, build_context, build_history_str, validate,
    get_cited_pages, strip_citations, filter_cited_chunks, is_no_info_answer,
    summarize_history, extract_snippet, confidence_badge,
    batch_rerank_query_excerpts,
)
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt
from speech import (
    text_to_speech, tts_available, tts_audio_format,
    whisper_available, transcribe_audio
)
from sessions import save_session, load_session, clear_session


# ─── Analytics logging ────────────────────────────────────────
def _log_query(employee_id: int, intent: str, topic: str | None,
               lang: str, is_no_info: bool, question: str) -> None:
    """Log every query to analytics_log table for the admin dashboard."""
    try:
        from database import get_db
        from sqlalchemy import text
        with get_db() as db:
            db.execute(text("""
                INSERT INTO analytics_log
                    (employee_id, intent, topic, language, unanswered, question_text, asked_at)
                VALUES
                    (:eid, :intent, :topic, :lang, :unans, :q, NOW())
            """), {"eid": employee_id, "intent": intent, "topic": topic or "",
                   "lang": lang, "unans": is_no_info, "q": question[:300]})
    except Exception as e:
        print(f"[analytics] log error (non-fatal): {e}")


# ─── Escalation helper ────────────────────────────────────────
def _send_escalation_email(employee_name: str, manager_email: str,
                            question: str) -> bool:
    """Email the employee's manager when the bot can't answer."""
    import smtplib, os
    from email.mime.text import MIMEText
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")
    from_addr = os.environ.get("SMTP_FROM", smtp_user)
    if not all([smtp_host, smtp_user, smtp_pass, manager_email]):
        return False
    try:
        body = (
            f"Hi,\n\n"
            f"The HR chatbot could not answer the following question from {employee_name}:\n\n"
            f"  \"{question}\"\n\n"
            f"Please follow up with them directly.\n\n"
            f"— HR Assistant (automated)"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"HR Chatbot: Unanswered query from {employee_name}"
        msg["From"]    = from_addr
        msg["To"]      = manager_email
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [manager_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[escalation] email error: {e}")
        import streamlit as _st
        _st.error(f"Email error: {e}")
        return False

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"

def clean_query(text: str) -> str:
    return text.replace('"', '').replace("'", "").strip()

# ─── Page config + login gate ─────────────────────────────────
st.set_page_config(page_title="HR Assistant", layout="wide")
init_cookie_manager()
require_login()

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**{st.session_state.employee_name}**")
    st.caption(f"{st.session_state.employee_grade} · {st.session_state.employee_dept}")
    if st.button("Sign out"):
        logout()
    st.divider()
    if st.button("🗑 Clear chat history"):
        clear_session(st.session_state.employee_id)
        st.session_state.chat_history         = []
        st.session_state.conversation_summary = ""
        st.rerun()

# ─── Session state defaults ───────────────────────────────────
_DEFAULTS = {
    "chat_history":              [],
    "conversation_summary":      "",
    "history_loaded":            False,
    "translated_answer":         None,
    "translation_for_question":  "",
    "last_answer":               None,
    "last_lang":                 None,
    "last_dialect":              None,
    "last_q_ar":                 None,
    "last_q_en":                 None,
    "last_cited_docs":           [],
    "last_top_docs":             [],
    "last_cited_pages":          set(),
    "last_scores":               {},
    "tts_audio":                 None,
    "tts_for_answer":            None,
    "transcribed_voice_question": None,
    "_mic_transcript":           None,
    "_cached_audio_bytes":       None,
    "escalation_sent":           None,
    "_show_esc":                 False,
    "_esc_email":                "",
    "_esc_question":             "",
    "_esc_name":                 "",
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Load persisted history ───────────────────────────────────
if not st.session_state.history_loaded:
    hist, summ = load_session(st.session_state.employee_id)
    if hist:
        st.session_state.chat_history        = hist
        st.session_state.conversation_summary = summ
    st.session_state.history_loaded = True

# ─── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
.rtl-answer { direction:rtl; text-align:right; font-size:1rem; line-height:1.9; padding:0.5rem 0; }
.ltr-answer { direction:ltr; text-align:left;  font-size:1rem; line-height:1.9; padding:0.5rem 0; }
.conf-badge {
    display:inline-block; padding:2px 8px; border-radius:4px;
    font-size:0.72rem; font-weight:600; color:white; margin-left:8px; vertical-align:middle;
}
.mic-container {
    background: #f8f9fa; border-radius: 20px; padding: 20px;
    border: 1px solid #dee2e6; margin-top: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("💼 HR Assistant")
st.caption("Ask in English, Arabic, or Franco Arabic.")

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
            css = "rtl-answer" if msg.get("is_arabic") else "ltr-answer"
            st.markdown(f'<div class="{css}">{msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


def _inject_adjacent_pages(docs_pool: list, retrieved: list) -> list:
    """
    For each retrieved chunk, inject chunks from the immediately FOLLOWING
    page only (page+1). This handles the cross-page list problem (content
    continues on next page) while keeping token count under control.
    Max 2 extra chunks per adjacent page to avoid bloat.
    """
    injected     = set()
    retrieved_ids = {id(d) for d in retrieved}
    extra        = []

    for d in retrieved:
        src  = d.metadata.get("source", "")
        page = d.metadata.get("page", -1)
        key  = (src, page + 1)
        if key in injected:
            continue
        injected.add(key)
        count = 0
        for c in docs_pool:
            if (c.metadata.get("source", "") == src and
                    c.metadata.get("page", -1) == page + 1 and
                    id(c) not in retrieved_ids):
                extra.append(c)
                retrieved_ids.add(id(c))
                count += 1
                if count >= 2:   # max 2 chunks per adjacent page
                    break

    return retrieved + extra


def _dedupe_docs(docs: list) -> list:
    """Remove duplicate chunks (same source + page + first 100 chars)."""
    seen = set()
    out  = []
    for d in docs:
        key = (d.metadata.get("source",""), d.metadata.get("page",-1), d.page_content[:100])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


# ─── Logic for processing input ──────────────────────────────
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
            with st.spinner("Searching…"):
                try:
                    lang    = detect_language_type(question)
                    dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else None

                    if lang == "english":
                        q_en, q_ar = question, translate(ar_llm, question, "Arabic")
                    elif lang == "arabic":
                        q_ar, q_en = question, translate(en_llm, question, "English")
                    else:
                        fa   = franco_to_arabic(question)
                        q_ar = translate(ar_llm, fa, "Modern Standard Arabic")
                        q_en = translate(en_llm, q_ar, "English")

                    history_str       = build_history_str(st.session_state.chat_history, st.session_state.conversation_summary)
                    intent, topic     = classify_intent(question)
                    personal_data_str = ""
                    top_docs          = []
                    scores_dict       = {}

                    # ── RAG/Policy Retrieval ──────────────────────────────
                    if intent in ("policy", "hybrid"):
                        ar_vs, ar_bm25, ar_docs = ar_index
                        en_vs, en_bm25, en_docs = en_index
                        ar_base  = (q_ar + " " + franco_to_arabic(question)).strip() if lang == "franco" else q_ar
                        q_ar_ret = build_retrieval_query(ar_base, st.session_state.chat_history).replace('"','').replace("'",'')
                        q_en_ret = build_retrieval_query(q_en,    st.session_state.chat_history).replace('"','').replace("'",'')

                        docs_ar  = retrieve(q_ar_ret, ar_vs, ar_bm25, ar_docs, lambda t: normalize_arabic(t, ara_tokenizer))
                        docs_en  = retrieve(q_en_ret, en_vs, en_bm25, en_docs, normalize_english)
                        combined = rrf(docs_ar, docs_en)
                        top_docs, scores_dict = rerank(q_en, combined, reranker, top_n=6)

                        # Inject adjacent pages so cross-page lists are never cut off
                        all_pool = ar_docs + en_docs
                        top_docs = _dedupe_docs(_inject_adjacent_pages(all_pool, top_docs))

                    # ── LLM Invocation ────────────────────────────────────
                    llm = en_llm if lang == "english" else ar_llm

                    if intent == "personal":
                        pd                = fetch_for_intent(st.session_state.employee_id, topic)
                        personal_data_str = format_personal_data(pd)
                        prompt            = get_personal_prompt(lang, dialect)
                        res               = llm.invoke(prompt.format(
                            personal_data=personal_data_str, question=question, history=history_str))

                    elif intent == "hybrid":
                        pd                = fetch_for_intent(st.session_state.employee_id, topic)
                        personal_data_str = format_personal_data(pd)
                        prompt            = get_hybrid_prompt(lang, dialect)
                        res               = llm.invoke(prompt.format(
                            personal_data=personal_data_str,
                            policy_context=build_context(top_docs),
                            question=question, history=history_str))

                    else:  # policy
                        if not top_docs:
                            res = type('obj', (object,), {'content': "No relevant policy documents found."})
                        else:
                            prompt = (english_prompt if lang == "english" else
                                      franco_prompt  if lang == "franco"  else
                                      egy_prompt     if dialect == "egyptian" else msa_prompt)
                            res = llm.invoke(prompt.format(
                                context=build_context(top_docs), question=question, history=history_str))

                    raw_answer  = res.content
                    cited_pages = get_cited_pages(raw_answer)
                    cited_docs  = filter_cited_chunks(top_docs, cited_pages)
                    clean_ans   = strip_citations(raw_answer)
                    answer      = (clean_ans if intent == "personal"
                                   else validate(clean_ans, lang, has_citations=bool(cited_pages)))

                    is_arabic = lang in ("arabic", "franco")
                    css       = "rtl-answer" if is_arabic else "ltr-answer"
                    st.markdown(f'<div class="{css}">{answer.replace(chr(10), "<br>")}</div>',
                                unsafe_allow_html=True)

                    with st.expander("🔍 Query Info"):
                        st.info(f"**Intent:** {intent} | **Lang:** {lang}" +
                                (f" | **Dialect:** {dialect}" if dialect else ""))
                        st.info(f"**AR:** {q_ar} | **EN:** {q_en}")

                    if personal_data_str:
                        with st.expander("📋 Your data used to answer this"):
                            st.code(personal_data_str, language=None)

                    # Show cited pages as full PDF page images
                    if cited_docs:
                        with st.expander("📄 Source Evidence"):
                            unique_pages = {}
                            for d in cited_docs:
                                p_no = d.metadata.get("page", 0) + 1
                                src  = (ARABIC_PDF_PATH
                                        if ARABIC_PDF_PATH in d.metadata.get("source", "")
                                        else ENGLISH_PDF_PATH)
                                if (src, p_no) not in unique_pages:
                                    unique_pages[(src, p_no)] = d
                            for (pdf, p_no), d in unique_pages.items():
                                src_label = "Arabic PDF" if pdf == ARABIC_PDF_PATH else "English PDF"
                                st.markdown(f"**📄 Page {p_no} — {src_label}**")
                                try:
                                    st.image(render_page_to_image(pdf, p_no), width=700)
                                except Exception:
                                    direction = "rtl" if pdf == ARABIC_PDF_PATH else "ltr"
                                    align     = "right" if direction == "rtl" else "left"
                                    st.markdown(
                                        f'<div style="direction:{direction};text-align:{align};'
                                        f'background:#f9f9f9;padding:12px;border-left:3px solid #1976d2;">'
                                        f'{d.page_content}</div>',
                                        unsafe_allow_html=True)

                    st.session_state.chat_history.append({"role": "user",      "content": question, "is_arabic": False})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer,   "is_arabic": is_arabic})
                    st.session_state.conversation_summary = summarize_history(
                        en_llm, st.session_state.chat_history, st.session_state.conversation_summary)
                    save_session(st.session_state.employee_id,
                                 st.session_state.chat_history,
                                 st.session_state.conversation_summary)
                    st.session_state.last_answer  = answer
                    st.session_state.last_lang    = lang
                    st.session_state.last_dialect = dialect

                    # ── Analytics logging ─────────────────────────────
                    _no_info = is_no_info_answer(answer)
                    _log_query(st.session_state.employee_id, intent, topic,
                               lang, _no_info, question)

                    # ── Store escalation data if bot cannot answer ─────
                    if _no_info:
                        from personal_data import get_employee_profile
                        _prof      = get_employee_profile(st.session_state.employee_id)
                        _mgr_email = _prof.get("manager_email", "")
                        if _mgr_email:
                            st.session_state["_esc_email"]    = _mgr_email
                            st.session_state["_esc_question"] = question
                            st.session_state["_esc_name"]     = st.session_state.employee_name
                            st.session_state["_show_esc"]     = True

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ─── Escalation UI (always rendered at top level) ────────────
if st.session_state.get("_show_esc"):
    st.info(
        "I could not find an answer in the policy documents. "
        "Would you like me to notify your HR manager?"
    )
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("📧 Notify HR manager", key="esc_btn"):
            sent = _send_escalation_email(
                st.session_state.get("_esc_name", ""),
                st.session_state.get("_esc_email", ""),
                st.session_state.get("_esc_question", "")
            )
            st.session_state["_show_esc"] = False
            if sent:
                st.success("✅ Your manager has been notified.")
            else:
                st.warning("Email not sent — check your SMTP settings in .env.")
    with col_b:
        if st.button("Dismiss", key="esc_dismiss"):
            st.session_state["_show_esc"] = False
            st.rerun()

# ─── Voice panel ──────────────────────────────────────────────
if whisper_available():
    st.markdown('<div class="mic-container">', unsafe_allow_html=True)
    t_rec, t_up = st.tabs(["🎙️ Record", "📁 Upload"])
    audio_source = None
    with t_rec:
        rec = st.audio_input("Record", key="mic_input", label_visibility="collapsed")
        if rec:
            audio_source = rec
    with t_up:
        up = st.file_uploader("Upload", type=["wav", "mp3", "m4a"], key="file_input", label_visibility="collapsed")
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
                    text = transcribe_audio(st.session_state["_cached_audio_bytes"])
                    if text:
                        st.session_state["_mic_transcript"] = text
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
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Bottom action bar ────────────────────────────────────────
if st.session_state.get("last_answer"):
    ans, l = st.session_state.last_answer, st.session_state.last_lang
    st.divider()
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("🔄 Translate"):
            target = "Arabic" if l == "english" else "English"
            llm    = st.session_state.ar_llm if l == "english" else st.session_state.en_llm
            st.session_state.translated_answer = translate(llm, ans, target)
    with cols[1]:
        if tts_available() and st.button("🔊 Read"):
            audio = text_to_speech(strip_citations(ans), lang=l,
                                   dialect=st.session_state.last_dialect)
            if audio:
                st.audio(audio, format=tts_audio_format(l))
    if st.session_state.get("translated_answer"):
        css = "ltr-answer" if l != "english" else "rtl-answer"
        st.markdown(f'<div class="{css}">{st.session_state.translated_answer}</div>',
                    unsafe_allow_html=True)