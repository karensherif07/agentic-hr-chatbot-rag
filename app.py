
import streamlit as st
from setup import setup, render_page_to_image
from nlp_utils import detect_language_type, get_semantic_dialect, franco_to_arabic, normalize_arabic, normalize_english
from retrieval import retrieve, rerank, rrf
from utils import translate, build_context, validate, get_cited_pages, strip_citations, filter_cited_chunks, is_no_info_answer
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt


ARABIC_PDF_PATH = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"

# ─── Streamlit UI ─────────────────────────────────────────────

def clean_query(text):
    return text.replace('"', '').replace("'", "").strip()

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
if "last_cited_pages" not in st.session_state:
    st.session_state.last_cited_pages = set()

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
    question = clean_query(question)
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

            q_ar_clean = q_ar.replace('"', '').replace("'", "")
            q_en_clean = q_en.replace('"', '').replace("'", "")

            docs_ar = retrieve(
                q_ar_clean, ar_vs, ar_bm25, ar_docs,
                lambda text: normalize_arabic(text, ara_tokenizer)
            )
            docs_en = retrieve(q_en_clean, en_vs, en_bm25, en_docs, normalize_english)

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
            st.session_state.last_cited_pages = cited_pages

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

    is_no_info = is_no_info_answer(answer)

    if not is_no_info:
        display_docs = cited_docs if cited_docs else top_docs
        cited_pages = sorted(getattr(st.session_state, 'last_cited_pages', set()))
        if cited_docs:
            page_list = ", ".join(str(p) for p in cited_pages)
            chunk_label = f"📄 Source Evidence {page_list}"
        else:
            chunk_label = "📄 Source Evidence (retrieved — no citations parsed)"

        with st.expander(chunk_label):
            if cited_docs:
                st.markdown(f"**Cited pages:** {page_list}")

            # Collect unique pages
            unique_pages = {}
            for d in display_docs:
                page_no = d.metadata.get("page", 0) + 1
                source_path = ARABIC_PDF_PATH if ARABIC_PDF_PATH in d.metadata.get("source", "") else ENGLISH_PDF_PATH
                key = (source_path, page_no)
                if key not in unique_pages:
                    unique_pages[key] = d

            for (pdf_path, page_no), d in unique_pages.items():
                source = "Arabic PDF" if ARABIC_PDF_PATH in pdf_path else "English PDF"
                st.markdown(f"**📄 Page {page_no} — {source}**")
                try:
                    img_bytes = render_page_to_image(pdf_path, page_no, clip_text=d.page_content)
                    st.image(img_bytes, caption=f"Relevant section from Page {page_no}", width=800)
                except Exception as e:
                    st.error(f"Failed to render page {page_no}: {e}")
                    # Fallback to text
                    st.write(d.page_content)
                if len(unique_pages) > 1:
                    st.markdown("---")