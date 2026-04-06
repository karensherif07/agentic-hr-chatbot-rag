import streamlit as st
# --- NEW IMPORTS ---
from auth import require_login, logout
from intent import classify_intent
from personal_data import fetch_for_intent
from personal_prompts import get_personal_prompt, get_hybrid_prompt, format_personal_data
# --- YOUR ORIGINAL IMPORTS ---
from setup import setup, render_page_to_image
from nlp_utils import detect_language_type, get_semantic_dialect, franco_to_arabic, normalize_arabic, normalize_english
from retrieval import retrieve, rerank, rrf, build_retrieval_query
from utils import translate, build_context, validate, get_cited_pages, strip_citations, filter_cited_chunks, is_no_info_answer
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt

ARABIC_PDF_PATH = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"

def clean_query(text):
    return text.replace('"', '').replace("'", "").strip()

def build_history_str(chat_history: list, max_turns: int = 5) -> str:
    if not chat_history:
        return "No previous conversation."
    recent = chat_history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# ─── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="HR Policy Assistant", layout="wide")

# --- INTEGRATED LOGIN ---
require_login()

with st.sidebar:
    st.markdown(f"**{st.session_state.employee_name}**")
    st.caption(f"{st.session_state.employee_grade} · {st.session_state.employee_dept}")
    if st.button("Sign out"):
        logout()

# ─── Session State ────────────────────────────────────────────
for key, default in {
    "chat_history": [],
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
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<style>
.rtl-answer { direction: rtl; text-align: right; font-size: 1rem; line-height: 1.9; padding: 0.5rem 0; }
.ltr-answer { direction: ltr; text-align: left; font-size: 1rem; line-height: 1.9; padding: 0.5rem 0; }
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

# ─── Render chat history ──────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        is_arabic = msg.get("is_arabic", False)
        css_class = "rtl-answer" if is_arabic else "ltr-answer"
        st.markdown(f'<div class="{css_class}">{msg["content"].replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

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
                # --- PRE-PROCESSING ---
                lang = detect_language_type(question)
                dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else ("franco" if lang == "franco" else None)
                
                if lang == "english":
                    q_en, q_ar = question, translate(ar_llm, question, "Arabic")
                elif lang == "arabic":
                    q_ar, q_en = question, translate(en_llm, question, "English")
                elif lang == "franco":
                    franco_arabic = franco_to_arabic(question)
                    q_ar = translate(ar_llm, franco_arabic, "Modern Standard Arabic")
                    q_en = translate(en_llm, q_ar, "English")

                history_str = build_history_str(st.session_state.chat_history)
                employee_id = st.session_state.employee_id
                
                # --- NEW: CLASSIFY INTENT ---
                intent, topic = classify_intent(question)
                
                # Setup local variables to satisfy your existing UI expanders
                top_docs = []
                cited_docs = []
                cited_pages = set()
                personal_data_str = ""

                # --- BRANCH 1: PERSONAL ---
                if intent == "personal":
                    personal_data = fetch_for_intent(employee_id, topic)
                    personal_data_str = format_personal_data(personal_data)
                    prompt = get_personal_prompt(lang, dialect)
                    llm = en_llm if lang == "english" else ar_llm
                    res = llm.invoke(prompt.format(personal_data=personal_data_str, question=question, history=history_str))
                    answer = res.content

                # --- BRANCH 2: HYBRID OR POLICY ---
                else:
                    # Both Hybrid and Policy need RAG Retrieval
                    ar_vs, ar_bm25, ar_docs = ar_index
                    en_vs, en_bm25, en_docs = en_index
                    
                    q_ar_clean = build_retrieval_query(q_ar, st.session_state.chat_history).replace('"', '').replace("'", "")
                    q_en_clean = build_retrieval_query(q_en, st.session_state.chat_history).replace('"', '').replace("'", "")

                    docs_ar = retrieve(q_ar_clean, ar_vs, ar_bm25, ar_docs, lambda text: normalize_arabic(text, ara_tokenizer))
                    docs_en = retrieve(q_en_clean, en_vs, en_bm25, en_docs, normalize_english)
                    combined = rrf(docs_ar, docs_en)
                    top_docs = rerank(q_ar if lang in ("arabic", "franco") else q_en, combined, reranker, top_n=5)

                    if not top_docs and intent == "policy":
                        answer = "No relevant policy documents found."
                    else:
                        if intent == "hybrid":
                            personal_data = fetch_for_intent(employee_id, topic)
                            personal_data_str = format_personal_data(personal_data)
                            prompt = get_hybrid_prompt(lang, dialect)
                            context = build_context(top_docs)
                            llm = en_llm if lang == "english" else ar_llm
                            res = llm.invoke(prompt.format(personal_data=personal_data_str, policy_context=context, question=question, history=history_str))
                        else:
                            # Standard Policy Logic
                            if lang == "english": prompt = english_prompt
                            elif lang == "franco": prompt = franco_prompt
                            else: prompt = egy_prompt if dialect == "egyptian" else msa_prompt
                            
                            context = build_context(top_docs)
                            llm = en_llm if lang == "english" else ar_llm
                            res = llm.invoke(prompt.format(context=context, question=question, history=history_str))
                        
                        # Apply your existing Citation & Validation logic
                        raw_answer = res.content
                        cited_pages = get_cited_pages(raw_answer)
                        cited_docs = filter_cited_chunks(top_docs, cited_pages)
                        clean_answer = strip_citations(raw_answer)
                        answer = validate(clean_answer, lang, has_citations=bool(cited_pages))

                # --- YOUR ORIGINAL UI RENDERING ---
                is_arabic = lang in ("arabic", "franco")
                css_class = "rtl-answer" if is_arabic else "ltr-answer"
                st.markdown(f'<div class="{css_class}">{answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

                # NEW: Personal Data Expander
                if intent in ["personal", "hybrid"]:
                    with st.expander("📋 Your data used to answer this"):
                        st.code(personal_data_str, language=None)

                # Query Info Expander
                with st.expander("🔍 Query Info"):
                    st.info(f"**Intent:** {intent} | **Language:** {lang}")
                    if dialect: st.info(f"**Dialect:** {dialect}")
                    st.info(f"**Arabic query:** {q_ar}")
                    st.info(f"**English query:** {q_en}")

                # YOUR ORIGINAL Source chunks expander
                if not is_no_info_answer(answer) and top_docs:
                    display_docs = cited_docs if cited_docs else top_docs
                    cited_pages_sorted = sorted(cited_pages)
                    chunk_label = f"📄 Source Evidence (pages: {', '.join(str(p) for p in cited_pages_sorted)})" if cited_docs else "📄 Source Evidence"

                    with st.expander(chunk_label):
                        unique_pages = {}
                        for d in display_docs:
                            page_no = d.metadata.get("page", 0) + 1
                            source_path = ARABIC_PDF_PATH if ARABIC_PDF_PATH in d.metadata.get("source", "") else ENGLISH_PDF_PATH
                            if (source_path, page_no) not in unique_pages: unique_pages[(source_path, page_no)] = d

                        for (pdf_path, page_no), d in unique_pages.items():
                            source_name = "Arabic PDF" if ARABIC_PDF_PATH in pdf_path else "English PDF"
                            st.markdown(f"**📄 Page {page_no} — {source_name}**")
                            try:
                                img_bytes = render_page_to_image(pdf_path, page_no, clip_text=d.page_content)
                                st.image(img_bytes, caption=f"Page {page_no}", width=800)
                            except:
                                direction = "rtl" if "ar_policy" in pdf_path else "ltr"
                                st.markdown(f'<div style="direction:{direction}; text-align:{"right" if direction=="rtl" else "left"}; background:#f9f9f9; padding:10px; border-radius:5px;">{d.page_content}</div>', unsafe_allow_html=True)

                # --- SAVE TO HISTORY ---
                st.session_state.chat_history.append({"role": "user", "content": question, "is_arabic": False})
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "is_arabic": is_arabic})

                # --- SAVE STATE FOR TRANSLATION BUTTON ---
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

# ─── Translate last answer (below chat) ──────────────────────
if st.session_state.last_answer:
    answer = st.session_state.last_answer
    lang = st.session_state.last_lang

    st.divider()
    if st.button("🔄 Translate last answer"):
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
        is_arabic = lang in ("arabic", "franco")
        trans_css = "rtl-answer" if not is_arabic else "ltr-answer"
        st.markdown(
            f'<div class="{trans_css}">{st.session_state.translated_answer.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True
        )