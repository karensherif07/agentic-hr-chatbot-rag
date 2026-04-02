
import streamlit as st
from setup import setup
from nlp_utils import detect_language_type, get_semantic_dialect, franco_to_arabic, normalize_arabic, normalize_english
from retrieval import retrieve, rerank, rrf
from utils import translate, build_context, validate, get_cited_pages, strip_citations, filter_cited_chunks
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt

# ─── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="HR Policy Assistant", layout="wide")

if "translated_answer" not in st.session_state:
    st.session_state.translated_answer = None
if "translation_for_question" not in st.session_state:
    st.session_state.translation_for_question = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_lang" not in st.session_state:
    st.session_state.last_lang = None
if "last_dialect" not in st.session_state:
    st.session_state.last_dialect = None
if "last_q_ar" not in st.session_state:
    st.session_state.last_q_ar = None
if "last_q_en" not in st.session_state:
    st.session_state.last_q_en = None
if "last_cited_docs" not in st.session_state:
    st.session_state.last_cited_docs = []
if "last_top_docs" not in st.session_state:
    st.session_state.last_top_docs = []

st.markdown("""
<style>
.rtl-answer {
    direction: rtl;
    text-align: right;
    font-size: 1rem;
    line-height: 1.9;
    padding: 0.5rem 0;
}
.ltr-answer {
    direction: ltr;
    text-align: left;
    font-size: 1rem;
    line-height: 1.9;
    padding: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("💼 HR Policy Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian dialect), or Franco Arabic.")

try:
    ar_index, en_index, ar_llm, en_llm, reranker, dialect_pipe, ara_tokenizer = setup()
    st.session_state.ar_llm = ar_llm
    st.session_state.en_llm = en_llm
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

with st.form("question_form"):
    question = st.text_input("Ask your question:")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    st.session_state.translated_answer = None
    st.session_state.translation_for_question = ""

    with st.spinner("Searching policy documents..."):
        try:
            lang = detect_language_type(question)

            if lang == "english":
                q_en = question
                q_ar = translate(ar_llm, question, "Arabic")
            elif lang == "arabic":
                q_ar = question
                q_en = translate(en_llm, question, "English")
            elif lang == "franco":
                franco_arabic = franco_to_arabic(question)
                q_ar = translate(ar_llm, franco_arabic, "Modern Standard Arabic")
                q_en = translate(en_llm, q_ar, "English")

            dialect = None
            if lang == "arabic":
                dialect = get_semantic_dialect(question, dialect_pipe)
            elif lang == "franco":
                dialect = "franco"

            ar_vs, ar_bm25, ar_docs = ar_index
            en_vs, en_bm25, en_docs = en_index

            docs_ar = retrieve(
                q_ar, ar_vs, ar_bm25, ar_docs,
                lambda text: normalize_arabic(text, ara_tokenizer)
            )
            docs_en = retrieve(q_en, en_vs, en_bm25, en_docs, normalize_english)

            combined = rrf(docs_ar, docs_en)

            rerank_query = q_ar if lang in ("arabic", "franco") else q_en
            top_docs = rerank(rerank_query, combined, reranker, top_n=5)

            if not top_docs:
                st.warning("No relevant documents found.")
                st.stop()

            context = build_context(top_docs)

            if lang == "english":
                prompt = english_prompt
                llm = en_llm
            elif lang == "franco":
                prompt = franco_prompt
                llm = ar_llm
            else:
                prompt = egy_prompt if dialect == "egyptian" else msa_prompt
                llm = ar_llm

            final_prompt = prompt.format(context=context, question=question)
            res = llm.invoke(final_prompt)
            raw_answer = res.content

            cited_pages = get_cited_pages(raw_answer)
            cited_docs = filter_cited_chunks(top_docs, cited_pages)
            clean_answer = strip_citations(raw_answer)
            answer = validate(clean_answer, lang, has_citations=bool(cited_pages))

            st.session_state.last_answer = answer
            st.session_state.last_lang = lang
            st.session_state.last_dialect = dialect
            st.session_state.last_q_ar = q_ar
            st.session_state.last_q_en = q_en
            st.session_state.last_cited_docs = cited_docs
            st.session_state.last_top_docs = top_docs

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ── Render answer (outside form, persists across reruns) ──────
if st.session_state.last_answer:
    answer = st.session_state.last_answer
    lang = st.session_state.last_lang
    dialect = st.session_state.last_dialect
    q_ar = st.session_state.last_q_ar
    q_en = st.session_state.last_q_en
    cited_docs = st.session_state.last_cited_docs
    top_docs = st.session_state.last_top_docs

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        css_class = "rtl-answer" if lang == "arabic" else "ltr-answer"
        st.markdown(
            f'<div class="{css_class}">{answer.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )
        st.divider()

        if st.button("🔄 Translate answer"):
            with st.spinner("Translating..."):
                if lang == "english":
                    st.session_state.translated_answer = translate(
                        st.session_state.ar_llm, answer, "Arabic"
                    )
                else:
                    st.session_state.translated_answer = translate(
                        st.session_state.en_llm, answer, "English"
                    )
                st.session_state.translation_for_question = answer

        if (st.session_state.translated_answer
                and st.session_state.translation_for_question == answer):
            trans_css = "rtl-answer" if lang != "english" else "ltr-answer"
            st.markdown(
                f'<div class="{trans_css}">{st.session_state.translated_answer.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.subheader("Query Info")
        st.info(f"**Detected language:** {lang}")
        if dialect:
            st.info(f"**Dialect:** {dialect}")
        st.info(f"**Arabic query:** {q_ar}")
        st.info(f"**English query:** {q_en}")

    display_docs = cited_docs if cited_docs else top_docs
    chunk_label = (
        f"📄 Source Chunks ({len(cited_docs)} cited)"
        if cited_docs
        else "📄 Source Chunks (retrieved — no citations parsed)"
    )
    with st.expander(chunk_label):
        for i, d in enumerate(display_docs, 1):
            source = (
                "Arabic PDF"
                if ARABIC_PDF_PATH in d.metadata.get("source", "")
                else "English PDF"
            )
            page_no = d.metadata.get("page", 0) + 1
            st.markdown(f"**Chunk {i} — Page {page_no} ({source})**")
            st.write(d.page_content)