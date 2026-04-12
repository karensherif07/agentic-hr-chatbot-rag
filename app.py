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

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"

def clean_query(text: str) -> str:
    return text.replace('"', '').replace("'", "").strip()

# ─── Page config + login gate ─────────────────────────────────
st.set_page_config(page_title="HR Assistant", layout="wide")
init_cookie_manager()   # no-op now, kept for compatibility
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
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Load persisted history once per login session ────────────
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
    background: #f8f9fa;
    border-radius: 20px;
    padding: 20px;
    border: 1px solid #dee2e6;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.stAudioInput { margin-bottom: 0px !important; }
</style>
""", unsafe_allow_html=True)

st.title("💼 HR Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian dialect), or Franco Arabic.")

# ─── Load models ──────────────────────────────────────────────
try:
    ar_index, en_index, ar_llm, en_llm, reranker, dialect_pipe, ara_tokenizer = setup()
    st.session_state.ar_llm = ar_llm
    st.session_state.en_llm = en_llm
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

# ─── Render chat history ──────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        css = "rtl-answer" if msg.get("is_arabic") else "ltr-answer"
        st.markdown(
            f'<div class="{css}">{msg["content"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

# ─── Voice panel + chat input ────────────────────────────────
question = st.session_state.pop("transcribed_voice_question", None)

if not whisper_available():
    st.caption("🎤 Voice input disabled: Ensure whisper and ffmpeg are installed.")
else:
    with st.container():
        st.markdown('<div class="mic-container">', unsafe_allow_html=True)
        
        # Dual Audio Input: Native Recording + File Upload
        tab_rec, tab_up = st.tabs(["🎙️ Record", "📁 Upload Sound"])
        
        audio_source = None
        with tab_rec:
            recorded = st.audio_input("Record your question", key="mic_input", label_visibility="collapsed")
            if recorded: audio_source = recorded
        with tab_up:
            uploaded = st.file_uploader("Upload audio (WAV/MP3/M4A/WebM)", type=["wav", "mp3", "m4a", "webm"], key="file_input", label_visibility="collapsed")
            if uploaded: audio_source = uploaded
        
        if audio_source:
            audio_bytes = audio_source.read()
            if st.session_state.get("_mic_transcript") is None:
                with st.status("🎙️ Processing audio...", expanded=False) as status:
                    text = transcribe_audio(audio_bytes)
                    if text:
                        st.session_state["_mic_transcript"] = text
                        status.update(label="✅ Transcription complete", state="complete")
                    else:
                        status.update(label="❌ Could not understand audio", state="error")

            transcript = st.session_state.get("_mic_transcript")
            if transcript:
                edited_text = st.text_area("Edit or confirm your question:", value=transcript, height=100)
                b1, b2, _ = st.columns([1, 1, 3])
                if b1.button("🚀 Send", use_container_width=True):
                    question = clean_query(edited_text)
                    st.session_state.pop("_mic_transcript", None)
                if b2.button("🗑️ Reset", use_container_width=True):
                    st.session_state.pop("_mic_transcript", None)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if question is None:
    question = st.chat_input("Ask your question…")

if question:
    question = clean_query(question)
    st.session_state.translated_answer        = None
    st.session_state.translation_for_question = ""
    st.session_state.tts_audio               = None
    st.session_state.tts_for_answer          = None

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                # 1. Language detection
                lang    = detect_language_type(question)
                dialect = None
                if lang == "arabic":
                    dialect = get_semantic_dialect(question, dialect_pipe)
                elif lang == "franco":
                    dialect = "franco"

                # 2. Translate for retrieval
                if lang == "english":
                    q_en = question
                    q_ar = translate(ar_llm, question, "Arabic")
                elif lang == "arabic":
                    q_ar = question
                    q_en = translate(en_llm, question, "English")
                else:
                    franco_arabic = franco_to_arabic(question)
                    q_ar = translate(ar_llm, franco_arabic, "Modern Standard Arabic")
                    q_en = translate(en_llm, q_ar, "English")

                # 3. History
                history_str = build_history_str(
                    st.session_state.chat_history,
                    st.session_state.conversation_summary
                )
                employee_id = st.session_state.employee_id

                # 4. Intent
                intent, topic = classify_intent(question)

                top_docs, cited_docs, cited_pages = [], [], set()
                personal_data_str, answer, scores_dict = "", "", {}

                # ── PERSONAL ──────────────────────────────────
                if intent == "personal":
                    personal_data     = fetch_for_intent(employee_id, topic)
                    personal_data_str = format_personal_data(personal_data)
                    prompt = get_personal_prompt(lang, dialect)
                    llm    = en_llm if lang == "english" else ar_llm
                    res    = llm.invoke(prompt.format(
                        personal_data=personal_data_str,
                        question=question, history=history_str
                    ))
                    answer = res.content

                # ── HYBRID / POLICY ────────────────────────────
                else:
                    ar_vs, ar_bm25, ar_docs = ar_index
                    en_vs, en_bm25, en_docs = en_index

                    ar_base  = (q_ar + " " + franco_to_arabic(question)).strip() if lang == "franco" else q_ar
                    q_ar_ret = build_retrieval_query(ar_base, st.session_state.chat_history).replace('"', '').replace("'", '')
                    q_en_ret = build_retrieval_query(q_en,    st.session_state.chat_history).replace('"', '').replace("'", '')

                    docs_ar  = retrieve(q_ar_ret, ar_vs, ar_bm25, ar_docs, lambda t: normalize_arabic(t, ara_tokenizer))
                    docs_en  = retrieve(q_en_ret, en_vs, en_bm25, en_docs, normalize_english)
                    combined = rrf(docs_ar, docs_en)
                    top_docs, scores_dict = rerank(q_en, combined, reranker, top_n=8)

                    if not top_docs and intent == "policy":
                        answer = "No relevant policy documents found."
                    else:
                        context = build_context(top_docs) if top_docs else ""
                        llm     = en_llm if lang == "english" else ar_llm

                        if intent == "hybrid":
                            personal_data     = fetch_for_intent(employee_id, topic)
                            personal_data_str = format_personal_data(personal_data)
                            prompt = get_hybrid_prompt(lang, dialect)
                            res = llm.invoke(prompt.format(
                                personal_data=personal_data_str,
                                policy_context=context,
                                question=question, history=history_str
                            ))
                        else:
                            if lang == "english":  prompt = english_prompt
                            elif lang == "franco": prompt = franco_prompt
                            else: prompt = egy_prompt if dialect == "egyptian" else msa_prompt
                            res = llm.invoke(prompt.format(
                                context=context, question=question, history=history_str
                            ))

                        raw_answer   = res.content
                        cited_pages  = get_cited_pages(raw_answer)
                        cited_docs   = filter_cited_chunks(top_docs, cited_pages)
                        clean_answer = strip_citations(raw_answer)
                        answer       = validate(clean_answer, lang, has_citations=bool(cited_pages))

                # 5. Render answer
                is_arabic = lang in ("arabic", "franco")
                css       = "rtl-answer" if is_arabic else "ltr-answer"
                st.markdown(
                    f'<div class="{css}">{answer.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )

                if intent in ("personal", "hybrid") and personal_data_str:
                    with st.expander("📋 Your data used to answer this"):
                        st.code(personal_data_str, language=None)

                if cited_docs:
                    pages_sorted = sorted(cited_pages)
                    with st.expander(f"📄 Source Evidence — pages: {', '.join(str(p) for p in pages_sorted)}"):
                        unique_pages: dict = {}
                        for d in cited_docs:
                            page_no = d.metadata.get("page", 0) + 1
                            src     = (ARABIC_PDF_PATH
                                       if ARABIC_PDF_PATH in d.metadata.get("source", "")
                                       else ENGLISH_PDF_PATH)
                            if (src, page_no) not in unique_pages:
                                unique_pages[(src, page_no)] = d

                        cited_for_display = list(unique_pages.values())
                        rerank_cited = {}
                        if cited_for_display and reranker:
                            pr = batch_rerank_query_excerpts(reranker, q_en, cited_for_display, window=720)
                            if pr:
                                rerank_cited = {id(x): s for x, s in zip(cited_for_display, pr)}
                        peer_scores = list(rerank_cited.values()) if rerank_cited else []

                        for (pdf_path, page_no), d in unique_pages.items():
                            src_name  = "Arabic PDF" if ARABIC_PDF_PATH in pdf_path else "English PDF"
                            raw_score = rerank_cited.get(id(d), scores_dict.get(id(d)))

                            if raw_score is not None:
                                conf_label, conf_color = confidence_badge(raw_score, peer_scores)
                                badge = (f'<span class="conf-badge" style="background:{conf_color};">'
                                         f'{conf_label} ({raw_score:.1f})</span>')
                                st.markdown(f"**📄 Page {page_no} — {src_name}**{badge}",
                                            unsafe_allow_html=True)
                            else:
                                st.markdown(f"**📄 Page {page_no} — {src_name}**")

                            img_bytes = render_page_to_image(pdf_path, page_no)
                            st.image(img_bytes, width=700)

                # 9. Save to history
                st.session_state.chat_history.append({"role": "user",      "content": question, "is_arabic": False})
                st.session_state.chat_history.append({"role": "assistant", "content": answer,   "is_arabic": is_arabic})

                st.session_state.conversation_summary = summarize_history(
                    en_llm, st.session_state.chat_history,
                    st.session_state.conversation_summary
                )

                save_session(
                    employee_id,
                    st.session_state.chat_history,
                    st.session_state.conversation_summary
                )

                st.session_state.last_answer      = answer
                st.session_state.last_lang        = lang
                st.session_state.last_dialect     = dialect
                st.session_state.last_q_ar        = q_ar
                st.session_state.last_q_en        = q_en
                st.session_state.last_cited_docs  = cited_docs
                st.session_state.last_top_docs    = top_docs
                st.session_state.last_cited_pages = cited_pages
                st.session_state.last_scores      = scores_dict

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ─── Bottom action bar ────────────────────────────────────────
if st.session_state.last_answer:
    answer, lang = st.session_state.last_answer, st.session_state.last_lang
    st.divider()
    cols = st.columns([1, 1, 4])
    with cols[0]:
        if st.button("🔄 Translate"):
            target = "Arabic" if lang == "english" else "English"
            llm    = st.session_state.ar_llm if lang == "english" else st.session_state.en_llm
            st.session_state.translated_answer        = translate(llm, answer, target)
            st.session_state.translation_for_question = answer
    with cols[1]:
        if tts_available():
            if st.button("🔊 Read aloud"):
                audio = text_to_speech(strip_citations(answer), lang=lang, dialect=st.session_state.last_dialect)
                if audio:
                    st.session_state.tts_audio, st.session_state.tts_for_answer = audio, answer
    if st.session_state.tts_audio and st.session_state.tts_for_answer == answer:
        st.audio(st.session_state.tts_audio, format=tts_audio_format(lang))
    if st.session_state.translated_answer and st.session_state.translation_for_question == answer:
        is_arabic = lang in ("arabic", "franco")
        css       = "ltr-answer" if is_arabic else "rtl-answer"
        st.markdown(f'<div class="{css}">{st.session_state.translated_answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)