import os
import streamlit as st
from auth import init_cookie_manager, require_login, logout, is_admin
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
    summarize_history, confidence_badge,
)
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt
from speech import (
    text_to_speech, tts_available, tts_audio_format,
    whisper_available, transcribe_audio
)
from sessions import save_session, load_session, clear_session

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"

# Confidence threshold below which we regenerate the answer once
_LOW_CONFIDENCE_THRESHOLD = -9.0   # raw reranker score


# ─── Analytics logging ────────────────────────────────────────
def _log_query(employee_id, intent, topic, lang, dialect, is_no_info, question):
    """4-way language logging: english | franco | arabic_msa | arabic_egyptian."""
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
            """), {"eid": employee_id, "intent": intent, "topic": topic or "",
                   "lang": log_lang, "unans": is_no_info, "q": question[:300]})
    except Exception as e:
        print(f"[analytics] {e}")


# ─── Escalation — sends to HR, not manager ────────────────────
def _get_hr_email() -> str:
    """
    Returns the HR contact email for escalation.
    Priority: HR_EMAIL env var → first hr_admin in DB → empty string.
    """
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
            f"The HR chatbot could not answer the following question from {employee_name}:\n\n"
            f"  \"{question}\"\n\n"
            f"Please follow up with them directly.\n\n"
            f"— HR Assistant (automated)"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"HR Chatbot: Unanswered query from {employee_name}"
        msg["From"]    = from_addr
        msg["To"]      = hr_email
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_addr, [hr_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[escalation] {e}")
        return False


def clean_query(text: str) -> str:
    return text.replace('"', '').replace("'", "").strip()


# ─── Page config + login ──────────────────────────────────────
st.set_page_config(page_title="HR Assistant", layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit default page navigation
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)
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

    # Admin portal link — shown ONLY to admins
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
    "last_q_ar":                  None,
    "last_q_en":                  None,
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
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Load persisted history (once per session) ────────────────
if not st.session_state.history_loaded:
    hist, summ = load_session(st.session_state.employee_id)
    if hist:
        st.session_state.chat_history        = hist
        st.session_state.conversation_summary = summ
    st.session_state.history_loaded = True

# ─── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
.rtl-answer{direction:rtl;text-align:right;font-size:1rem;line-height:1.9;padding:.5rem 0}
.ltr-answer{direction:ltr;text-align:left;font-size:1rem;line-height:1.9;padding:.5rem 0}
.conf-badge{display:inline-block;padding:2px 8px;border-radius:4px;
            font-size:.72rem;font-weight:600;color:white;margin-left:8px;vertical-align:middle}
.mic-container{background:#f8f9fa;border-radius:20px;padding:20px;
               border:1px solid #dee2e6;margin-top:20px;box-shadow:0 4px 6px rgba(0,0,0,.05)}
</style>
""", unsafe_allow_html=True)

# ─── Admin gate: admins use admin portal, not chatbot ─────
if is_admin():
    st.info("👋 You are logged in as an HR administrator.\n\nUse the **⚙️ Admin Portal** button in the sidebar to access analytics, escalations, and system configuration.")
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
            css = "rtl-answer" if is_rtl else "ltr-answer"
            st.markdown(
                f'<div class="{css}">{msg["content"].replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )


# ─── Helpers ──────────────────────────────────────────────────
def _inject_adjacent_pages(docs_pool, retrieved):
    """Inject page+1 chunks so cross-page lists are never cut off.
    
    FIX #1: Increased from 2 to 4 chunks per adjacent page
    to ensure multi-page answers (like promotion criteria on p.5-6) 
    are fully retrieved.
    """
    injected      = set()
    retrieved_ids = {id(d) for d in retrieved}
    extra         = []
    for d in retrieved:
        src, page = d.metadata.get("source",""), d.metadata.get("page",-1)
        key = (src, page + 1)
        if key in injected:
            continue
        injected.add(key)
        count = 0
        for c in docs_pool:
            if (c.metadata.get("source","") == src and
                    c.metadata.get("page",-1) == page + 1 and
                    id(c) not in retrieved_ids):
                extra.append(c)
                retrieved_ids.add(id(c))
                count += 1
                if count >= 4:  # WAS: 2 → NOW: 4
                    break
    return retrieved + extra


def _dedupe_docs(docs):
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("source",""), d.metadata.get("page",-1), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _get_top_raw_score(scores_dict, cited_docs, top_docs):
    """Return the best raw reranker score for the current answer."""
    if not scores_dict:
        return None
    badge_doc = cited_docs[0] if cited_docs else (top_docs[0] if top_docs else None)
    if badge_doc is None:
        return None
    score = scores_dict.get(id(badge_doc))
    if score is None:
        score = max(scores_dict.values())
    return score


def _invoke_llm(llm, prompt_template, **kwargs):
    """Single LLM call with formatted prompt."""
    return llm.invoke(prompt_template.format(**kwargs))


def _run_rag(question, q_ar, q_en, lang, fa):
    """Run retrieval + rerank and return (top_docs, scores_dict).
    
    FIX #2: For Arabic (including Egyptian) questions, translate to MSA first
    before retrieval to improve recall. Hajj leave and other Egyptian-dialect
    questions are answered from MSA policy text.
    """
    ar_vs, ar_bm25, ar_docs = ar_index
    en_vs, en_bm25, en_docs = en_index
    
    # FIX #2: If Arabic (Egyptian), translate to MSA for better retrieval
    if lang == "arabic":
        q_ar_for_retrieval = translate(st.session_state.ar_llm, q_ar, "Modern Standard Arabic (formal)")
    else:
        ar_base = (fa + " " + q_ar).strip() if lang == "franco" and fa else q_ar
        q_ar_for_retrieval = ar_base
    
    history_msgs = st.session_state.chat_history
    q_ar_ret = build_retrieval_query(q_ar_for_retrieval, history_msgs).replace('"','').replace("'",'')
    q_en_ret = build_retrieval_query(q_en,    history_msgs).replace('"','').replace("'",'')
    docs_ar  = retrieve(q_ar_ret, ar_vs, ar_bm25, ar_docs, lambda t: normalize_arabic(t, ara_tokenizer))
    docs_en  = retrieve(q_en_ret, en_vs, en_bm25, en_docs, normalize_english)
    combined = rrf(docs_ar, docs_en)
    top_docs, scores_dict = rerank(q_en, combined, reranker, top_n=5)
    all_pool = ar_docs + en_docs
    top_docs = _dedupe_docs(_inject_adjacent_pages(all_pool, top_docs))
    return top_docs, scores_dict


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
            with st.spinner("Searching…"):
                try:
                    # ── Language + dialect ────────────────────────────────
                    lang    = detect_language_type(question)
                    dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else None

                    # ── Build queries ─────────────────────────────────────
                    if lang == "english":
                        q_en, q_ar, fa = question, translate(ar_llm, question, "Arabic"), ""
                    elif lang == "arabic":
                        q_ar, q_en, fa = question, translate(en_llm, question, "English"), ""
                    else:  # franco
                        fa   = franco_to_arabic(question)
                        q_ar = translate(ar_llm, fa, "Modern Standard Arabic")
                        q_en = translate(en_llm, q_ar, "English")

                    history_str   = build_history_str(
                        st.session_state.chat_history,
                        st.session_state.conversation_summary
                    )
                    # LLM-based intent classification — understands all 4 languages
                    # natively without hardcoded patterns
                    intent, topic = classify_intent(question, llm=en_llm)
                    personal_data_str = ""
                    top_docs, scores_dict = [], {}

                    # ── RAG retrieval ─────────────────────────────────────
                    if intent in ("policy", "hybrid"):
                        top_docs, scores_dict = _run_rag(question, q_ar, q_en, lang, fa)

                    # ── LLM ───────────────────────────────────────────────
                    llm = en_llm if lang == "english" else ar_llm

                    if intent == "personal":
                        pd                = fetch_for_intent(st.session_state.employee_id, topic)
                        personal_data_str = format_personal_data(pd)
                        prompt            = get_personal_prompt(lang, dialect)
                        res               = _invoke_llm(llm, prompt,
                                               personal_data=personal_data_str,
                                               question=question, history=history_str)

                    elif intent == "hybrid":
                        pd                = fetch_for_intent(st.session_state.employee_id, topic)
                        personal_data_str = format_personal_data(pd)
                        prompt            = get_hybrid_prompt(lang, dialect)
                        res               = _invoke_llm(llm, prompt,
                                               personal_data=personal_data_str,
                                               policy_context=build_context(top_docs),
                                               question=question, history=history_str)

                    else:  # policy
                        if not top_docs:
                            res = type('obj', (object,), {
                                'content': "This information is not available in the policy documents."
                            })
                        else:
                            prompt = (english_prompt if lang == "english" else
                                      franco_prompt  if lang == "franco"  else
                                      egy_prompt     if dialect == "egyptian" else msa_prompt)
                            res = _invoke_llm(llm, prompt,
                                              context=build_context(top_docs),
                                              question=question, history=history_str)

                    # ── Post-process ──────────────────────────────────────
                    raw_answer  = res.content
                    cited_pages = get_cited_pages(raw_answer)
                    cited_docs  = filter_cited_chunks(top_docs, cited_pages)
                    clean_ans   = strip_citations(raw_answer)
                    answer      = (clean_ans if intent == "personal"
                                   else validate(clean_ans, lang, has_citations=bool(cited_pages)))
                    _no_info    = is_no_info_answer(answer)

                    # ── Low-confidence regeneration ───────────────────────
                    if (intent in ("policy", "hybrid") and scores_dict and not _no_info
                            and top_docs):
                        raw_score = _get_top_raw_score(scores_dict, cited_docs, top_docs)
                        if raw_score is not None and raw_score < _LOW_CONFIDENCE_THRESHOLD:
                            ar_vs, ar_bm25, ar_docs = ar_index
                            en_vs, en_bm25, en_docs = en_index
                            ar_base = (fa + " " + q_ar).strip() if lang == "franco" and fa else q_ar
                            docs_ar  = retrieve(q_ar, ar_vs, ar_bm25, ar_docs, lambda t: normalize_arabic(t, ara_tokenizer))
                            docs_en  = retrieve(q_en, en_vs, en_bm25, en_docs, normalize_english)
                            combined2 = rrf(docs_ar, docs_en)
                            top_docs2, scores_dict2 = rerank(q_en, combined2, reranker, top_n=8)
                            all_pool2 = ar_docs + en_docs
                            top_docs2 = _dedupe_docs(_inject_adjacent_pages(all_pool2, top_docs2))
                            if top_docs2:
                                if intent == "hybrid":
                                    res2 = _invoke_llm(llm, prompt,
                                                       personal_data=personal_data_str,
                                                       policy_context=build_context(top_docs2),
                                                       question=question, history=history_str)
                                else:
                                    res2 = _invoke_llm(llm, prompt,
                                                       context=build_context(top_docs2),
                                                       question=question, history=history_str)
                                raw2        = res2.content
                                cited2      = get_cited_pages(raw2)
                                cdocs2      = filter_cited_chunks(top_docs2, cited2)
                                clean2      = strip_citations(raw2)
                                answer2     = (clean2 if intent == "personal"
                                               else validate(clean2, lang, has_citations=bool(cited2)))
                                if not is_no_info_answer(answer2):
                                    answer      = answer2
                                    top_docs    = top_docs2
                                    scores_dict = scores_dict2
                                    cited_pages = cited2
                                    cited_docs  = cdocs2
                                    _no_info    = False

                    _no_info = is_no_info_answer(answer)

                    # ── Render ────────────────────────────────────────────
                    is_franco        = (lang == "franco")
                    is_arabic_script = (lang == "arabic")
                    css = "rtl-answer" if is_arabic_script else "ltr-answer"
                    st.markdown(
                        f'<div class="{css}">{answer.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )

                    # ── Confidence badge ──────────────────────────────────
                    if scores_dict and intent in ("policy", "hybrid") and not _no_info:
                        peer_scores = list(scores_dict.values())
                        raw_score   = _get_top_raw_score(scores_dict, cited_docs, top_docs)
                        if raw_score is not None:
                            label, color = confidence_badge(raw_score, peer_scores)
                            st.markdown(
                                f'<span class="conf-badge" style="background:{color}">{label}</span>',
                                unsafe_allow_html=True
                            )

                    # ── Query info ────────────────────────────────────────
                    with st.expander("🔍 Query Info"):
                        st.info(f"**Intent:** {intent} | **Lang:** {lang}" +
                                (f" | **Dialect:** {dialect}" if dialect else ""))
                        st.info(f"**AR:** {q_ar} | **EN:** {q_en}")

                    # ── Personal data used ────────────────────────────────
                    if personal_data_str:
                        with st.expander("📋 Your data used to answer this"):
                            st.code(personal_data_str, language=None)

                    # ── Source evidence (hidden when "not in policy") ─────
                    if cited_docs and not _no_info:
                        with st.expander("📄 Source Evidence"):
                            unique_pages = {}
                            for d in cited_docs:
                                p_no = d.metadata.get("page", 0) + 1
                                src  = (ARABIC_PDF_PATH
                                        if ARABIC_PDF_PATH in d.metadata.get("source","")
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

                    # ── Save history ──────────────────────────────────────
                    st.session_state.chat_history.append({
                        "role": "user", "content": question,
                        "is_arabic": False, "is_franco": is_franco,
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer,
                        "is_arabic": is_arabic_script, "is_franco": is_franco,
                    })
                    st.session_state.conversation_summary = summarize_history(
                        en_llm, st.session_state.chat_history,
                        st.session_state.conversation_summary
                    )
                    save_session(st.session_state.employee_id,
                                 st.session_state.chat_history,
                                 st.session_state.conversation_summary)

                    # ── Persist state ─────────────────────────────────────
                    st.session_state.last_answer      = answer
                    st.session_state.last_lang        = lang
                    st.session_state.last_dialect     = dialect
                    st.session_state.last_scores      = scores_dict
                    st.session_state.last_cited_docs  = cited_docs
                    st.session_state.last_top_docs    = top_docs
                    st.session_state.last_cited_pages = cited_pages
                    st.session_state.last_q_ar        = q_ar
                    st.session_state.last_q_en        = q_en

                    # ── Analytics ─────────────────────────────────────────
                    _log_query(st.session_state.employee_id, intent, topic,
                               lang, dialect, _no_info, question)

                    # ── Escalation to HR ──────────────────────────────────
                    # FIX #3: Show escalation for ALL intents when no answer found
                    # (was only showing for policy/hybrid, not personal)
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
    st.info("I couldn't find an answer in the policy documents. Would you like me to notify the HR team?")
    col_a, col_b = st.columns([1, 5])
    with col_a:
        if st.button("📧 Notify HR", key="esc_btn"):
            sent = _send_escalation_email(
                st.session_state.get("_esc_name", ""),
                st.session_state.get("_esc_hr_email", ""),
                st.session_state.get("_esc_question", "")
            )
            st.session_state["_show_esc"] = False
            if sent:
                st.success("✅ The HR team has been notified.")
            else:
                st.warning("Email not sent — check SMTP settings in .env or set HR_EMAIL.")
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
        up = st.file_uploader("Upload", type=["wav","mp3","m4a"],
                               key="file_input", label_visibility="collapsed")
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
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Bottom action bar ────────────────────────────────────────
if st.session_state.get("last_answer"):
    ans, l = st.session_state.last_answer, st.session_state.last_lang
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
            unsafe_allow_html=True
        )