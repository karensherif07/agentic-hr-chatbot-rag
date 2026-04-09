import streamlit as st
from auth import require_login, logout
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

ARABIC_PDF_PATH = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"


def clean_query(text: str) -> str:
    return text.replace('"', '').replace("'", "").strip()


# ─── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="HR Policy Assistant", layout="wide")
require_login()

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"**{st.session_state.employee_name}**")
    st.caption(f"{st.session_state.employee_grade} · {st.session_state.employee_dept}")
    if st.button("Sign out"):
        logout()

# ─── Session State ────────────────────────────────────────────
_DEFAULTS = {
    "chat_history": [],
    "conversation_summary": "",
    "translated_answer": None,
    "translation_for_question": "",
    "last_answer": None,
    "last_lang": None,
    "last_dialect": None,
    "last_q_ar": None,
    "last_q_en": None,
    "last_cited_docs": [],
    "last_top_docs": [],
    "last_cited_pages": set(),
    "last_scores": {},
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Styles ───────────────────────────────────────────────────
st.markdown("""
<style>
.rtl-answer {
    direction: rtl; text-align: right;
    font-size: 1rem; line-height: 1.9; padding: 0.5rem 0;
}
.ltr-answer {
    direction: ltr; text-align: left;
    font-size: 1rem; line-height: 1.9; padding: 0.5rem 0;
}
.conf-badge {
    display: inline-block;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 600;
    color: white; margin-left: 8px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 HR Policy Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian dialect), or Franco Arabic.")

# ─── Load models (cached) ─────────────────────────────────────
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
        css_class = "rtl-answer" if msg.get("is_arabic") else "ltr-answer"
        st.markdown(
            f'<div class="{css_class}">{msg["content"].replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )

# ─── Chat input ───────────────────────────────────────────────
question = st.chat_input("Ask your question...")

if question:
    question = clean_query(question)
    st.session_state.translated_answer = None
    st.session_state.translation_for_question = ""

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                # ── 1. Language + dialect detection ───────────────────
                lang = detect_language_type(question)
                dialect = None
                if lang == "arabic":
                    dialect = get_semantic_dialect(question, dialect_pipe)
                elif lang == "franco":
                    dialect = "franco"

                # ── 2. Translate to AR + EN for retrieval ─────────────
                if lang == "english":
                    q_en = question
                    q_ar = translate(ar_llm, question, "Arabic")
                elif lang == "arabic":
                    q_ar = question
                    q_en = translate(en_llm, question, "English")
                else:  # franco
                    # Convert franco tokens → rough Arabic, then translate to proper MSA
                    franco_arabic = franco_to_arabic(question)
                    q_ar = translate(ar_llm, franco_arabic, "Modern Standard Arabic")
                    q_en = translate(en_llm, q_ar, "English")

                # ── 3. Build history string ───────────────────────────
                history_str = build_history_str(
                    st.session_state.chat_history,
                    st.session_state.conversation_summary
                )
                employee_id = st.session_state.employee_id

                # ── 4. Intent classification ──────────────────────────
                intent, topic = classify_intent(question)

                # ── 5. Initialise outputs ─────────────────────────────
                top_docs        = []
                cited_docs      = []
                cited_pages     = set()
                personal_data_str = ""
                answer          = ""
                scores_dict     = {}

                # ══════════════════════════════════════════════════════
                # PERSONAL — answered from DB only, no RAG
                # ══════════════════════════════════════════════════════
                if intent == "personal":
                    personal_data     = fetch_for_intent(employee_id, topic)
                    personal_data_str = format_personal_data(personal_data)
                    prompt = get_personal_prompt(lang, dialect)
                    llm    = en_llm if lang == "english" else ar_llm
                    res    = llm.invoke(prompt.format(
                        personal_data=personal_data_str,
                        question=question,
                        history=history_str
                    ))
                    answer = res.content

                # ══════════════════════════════════════════════════════
                # HYBRID / POLICY — RAG retrieval
                # ══════════════════════════════════════════════════════
                else:
                    ar_vs, ar_bm25, ar_docs = ar_index
                    en_vs, en_bm25, en_docs = en_index

                    # For franco: enrich AR retrieval query with raw franco→arabic
                    # so BM25 catches keywords the LLM translation might miss
                    if lang == "franco":
                        ar_retrieval_base = (q_ar + " " + franco_to_arabic(question)).strip()
                    else:
                        ar_retrieval_base = q_ar

                    # Build context-aware queries (appends recent user turns)
                    q_ar_ret = build_retrieval_query(
                        ar_retrieval_base, st.session_state.chat_history
                    ).replace('"', '').replace("'", "")
                    q_en_ret = build_retrieval_query(
                        q_en, st.session_state.chat_history
                    ).replace('"', '').replace("'", "")

                    # Retrieve from both indexes, fuse with RRF
                    docs_ar  = retrieve(q_ar_ret, ar_vs, ar_bm25, ar_docs,
                                        lambda t: normalize_arabic(t, ara_tokenizer))
                    docs_en  = retrieve(q_en_ret, en_vs, en_bm25, en_docs, normalize_english)
                    combined = rrf(docs_ar, docs_en)

                    # ── ALWAYS rerank with English query ──────────────
                    # The mmarco cross-encoder is most reliable with English
                    # queries regardless of the document language. Using dialect
                    # Arabic or Franco as the rerank query produces artificially
                    # low scores because the model was not trained on that style.
                    top_docs, scores_dict = rerank(q_en, combined, reranker, top_n=8)

                    if not top_docs and intent == "policy":
                        answer = "No relevant policy documents found."
                    else:
                        context = build_context(top_docs) if top_docs else ""
                        llm = en_llm if lang == "english" else ar_llm

                        if intent == "hybrid":
                            personal_data     = fetch_for_intent(employee_id, topic)
                            personal_data_str = format_personal_data(personal_data)
                            prompt = get_hybrid_prompt(lang, dialect)
                            res = llm.invoke(prompt.format(
                                personal_data=personal_data_str,
                                policy_context=context,
                                question=question,
                                history=history_str
                            ))
                        else:
                            if lang == "english":
                                prompt = english_prompt
                            elif lang == "franco":
                                prompt = franco_prompt
                            else:
                                prompt = egy_prompt if dialect == "egyptian" else msa_prompt

                            res = llm.invoke(prompt.format(
                                context=context,
                                question=question,
                                history=history_str
                            ))

                        raw_answer  = res.content
                        cited_pages = get_cited_pages(raw_answer)
                        cited_docs  = filter_cited_chunks(top_docs, cited_pages)
                        clean_answer = strip_citations(raw_answer)
                        answer = validate(clean_answer, lang, has_citations=bool(cited_pages))

                # ── 6. Render answer ──────────────────────────────────
                is_arabic = lang in ("arabic", "franco")
                css_class = "rtl-answer" if is_arabic else "ltr-answer"
                st.markdown(
                    f'<div class="{css_class}">{answer.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )

                # ── 7. Personal data expander ─────────────────────────
                if intent in ("personal", "hybrid") and personal_data_str:
                    with st.expander("📋 Your data used to answer this"):
                        st.code(personal_data_str, language=None)

                # ── 8. Query info expander ────────────────────────────
                with st.expander("🔍 Query Info"):
                    st.info(
                        f"**Intent:** {intent} | **Language:** {lang}"
                        + (f" | **Dialect:** {dialect}" if dialect else "")
                    )
                    st.info(f"**AR query:** {q_ar}")
                    st.info(f"**EN query:** {q_en}")

                # ── 9. Source evidence — ONLY when pages were cited ───
                # If the LLM answered from its absolute rules without citing
                # a page, we do not show source evidence (no valid citations).
                if cited_docs:
                    pages_sorted = sorted(cited_pages)
                    chunk_label = (
                        f"📄 Source Evidence — pages cited: "
                        f"{', '.join(str(p) for p in pages_sorted)}"
                    )
                    with st.expander(chunk_label):
                        # Deduplicate: one entry per (pdf, page)
                        unique_pages: dict = {}
                        for d in cited_docs:
                            page_no = d.metadata.get("page", 0) + 1
                            src = (ARABIC_PDF_PATH
                                   if ARABIC_PDF_PATH in d.metadata.get("source", "")
                                   else ENGLISH_PDF_PATH)
                            key = (src, page_no)
                            if key not in unique_pages:
                                unique_pages[key] = d

                        cited_for_display = list(unique_pages.values())
                        rerank_cited = {}
                        if cited_for_display and reranker is not None:
                            pr = batch_rerank_query_excerpts(
                                reranker, q_en, cited_for_display, window=720
                            )
                            if pr:
                                rerank_cited = {
                                    id(x): s for x, s in zip(cited_for_display, pr)
                                }

                        peer_scores = list(rerank_cited.values()) if rerank_cited else []
                        ans_for_snippet = strip_citations(answer)[:1200]

                        for (pdf_path, page_no), d in unique_pages.items():
                            src_name = "Arabic PDF" if ARABIC_PDF_PATH in pdf_path else "English PDF"

                            raw_score = rerank_cited.get(id(d), scores_dict.get(id(d)))
                            if raw_score is not None:
                                conf_label, conf_color = confidence_badge(raw_score, peer_scores)
                                badge = (
                                    f'<span class="conf-badge" style="background:{conf_color};">'
                                    f'{conf_label} (score {raw_score:.1f})</span>'
                                )
                                st.markdown(
                                    f"**📄 Page {page_no} — {src_name}**{badge}",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(f"**📄 Page {page_no} — {src_name}**")

                            # Full page image (avoids bad crops; cross-encoder score is excerpt-based)
                            try:
                                img_bytes = render_page_to_image(
                                    pdf_path, page_no, zoom=1.75
                                )
                                st.image(
                                    img_bytes,
                                    caption=f"Full page {page_no} — {src_name}",
                                    width=700,
                                )
                            except Exception:
                                # Text fallback
                                direction = "rtl" if "ar_policy" in pdf_path else "ltr"
                                align = "right" if direction == "rtl" else "left"
                                snippet = extract_snippet(
                                    d.page_content,
                                    f"{q_en} {q_ar} {ans_for_snippet}",
                                    window=700,
                                )
                                st.markdown(
                                    f'<div style="direction:{direction};text-align:{align};'
                                    f'background:#f9f9f9;padding:10px;border-radius:5px;'
                                    f'font-size:0.9rem;">{snippet}</div>',
                                    unsafe_allow_html=True
                                )

                # ── 10. Update history + running summary ──────────────
                st.session_state.chat_history.append(
                    {"role": "user",      "content": question, "is_arabic": False}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer,   "is_arabic": is_arabic}
                )

                st.session_state.conversation_summary = summarize_history(
                    en_llm,
                    st.session_state.chat_history,
                    st.session_state.conversation_summary
                )

                # Persist last-answer state for translate button
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
                import traceback
                st.code(traceback.format_exc())

# ─── Translate last answer ────────────────────────────────────
if st.session_state.last_answer:
    answer = st.session_state.last_answer
    lang   = st.session_state.last_lang
    st.divider()

    if st.button("🔄 Translate last answer"):
        with st.spinner("Translating..."):
            target = "Arabic" if lang == "english" else "English"
            llm    = st.session_state.ar_llm if lang == "english" else st.session_state.en_llm
            st.session_state.translated_answer          = translate(llm, answer, target)
            st.session_state.translation_for_question   = answer

    if (st.session_state.translated_answer
            and st.session_state.translation_for_question == answer):
        is_arabic = lang in ("arabic", "franco")
        trans_css = "ltr-answer" if is_arabic else "rtl-answer"
        st.markdown(
            f'<div class="{trans_css}">'
            f'{st.session_state.translated_answer.replace(chr(10), "<br>")}'
            f'</div>',
            unsafe_allow_html=True
        )