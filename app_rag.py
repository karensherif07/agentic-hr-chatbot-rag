import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ===== FILES =====
ARABIC_PDF_PATH = "ar_policy.pdf"
ENGLISH_PDF_PATH = "eng_policy.pdf"

# ===== LANGUAGE DETECTION =====

# High-confidence Franco-Arabic words that never appear in plain English text.
# Covers pronouns, question words, negations, particles, and common expressions.
FRANCO_TIER1 = {
    # pronouns
    "ana", "enta", "enti", "ehna", "entom", "howa", "hya", "homma",
    # negation
    "msh", "mesh", "mish", "mafish",
    # question words
    "leh", "leih", "fein", "fen", "emta", "ezay", "meen", "eih", "eh",
    # common expressions / verbs
    "ya3ni", "3ashan", "momken", "3ayz", "3ayza", "yenfa3", "ynfa3",
    "tayeb", "tamam", "keda", "kidda", "bas",
    # nouns / particles
    "7aga", "haga", "feeh", "fieh", "la2", "aywa", "aiwa",
    "wenta", "wenti", "bs", "delwa2ty", "badein", "b3dein",
    # determiners / demonstratives (very common in Franco)
    "el", "di", "da", "dol", "aho", "ahi",
}


def has_arabic_digit_combo(text: str) -> bool:
    """
    Detect Franco digit-letter substitutions: 2=ء, 3=ع, 5=خ, 7=ح, 8=غ, 9=ق.
    A token qualifies if it mixes letters with one of these digits.
    Examples: 3ashan, 7aga, t2oly, m3ak, b3dein, a5od, w2fa
    This pattern is essentially impossible in normal English text.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    for tok in tokens:
        if len(tok) >= 2 and re.search(r"[a-z]", tok) and re.search(r"[23578]", tok):
            return True
    return False


def detect_language_type(text: str) -> str:
    """
    Returns 'arabic', 'franco', or 'english'.

    Logic:
    - Arabic Unicode script → 'arabic'  (checked first; mixed Arabic+Franco → 'arabic', handled fine by ar_llm)
    - Any FRANCO_TIER1 word  → 'franco'  (one strong signal is enough)
    - Any digit-letter combo → 'franco'
    - Otherwise             → 'english'
    """
    # 1. Arabic script characters present → Arabic
    if re.search(r"[\u0600-\u06FF]", text):
        return "arabic"

    tokens = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))

    # 2. Franco: one clear signal is sufficient
    if tokens & FRANCO_TIER1:
        return "franco"
    if has_arabic_digit_combo(text):
        return "franco"

    return "english"


def get_semantic_dialect(llm, arabic_text: str) -> str:
    """
    Classify Egyptian dialect vs MSA.
    MUST only be called when lang == 'arabic' (input is Arabic script).
    """
    try:
        prompt = (
            "You are an Arabic linguist. Classify the following Arabic text as either "
            "'egyptian' (Egyptian colloquial / عامية مصرية) or 'msa' (Modern Standard Arabic / فصحى). "
            "Reply with ONE word only: egyptian or msa.\n\nText: " + arabic_text
        )
        res = llm.invoke(prompt)
        return "egyptian" if "egyptian" in res.content.strip().lower() else "msa"
    except Exception:
        return "msa"


# ===== CLEANING & NORMALIZATION =====

def clean_pdf(text: str) -> str:
    # Remove zero-width and invisible Unicode characters common in Arabic PDFs
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u200e\u200f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text so BM25 queries match corpus tokens consistently.
    Applied at BOTH index-build time and query time.
    """
    text = re.sub(r"[أإآ]", "ا", text)          # alef variants → bare alef
    text = text.replace("ة", "ه")                # ta marbuta → ha
    text = text.replace("ى", "ي")                # alef maqsura → ya
    text = re.sub(r"[\u064B-\u065F]", "", text)  # strip tashkeel (diacritics)
    return text.lower()


def normalize_english(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


# ===== RRF =====

def rrf(docs1: list, docs2: list, k: int = 60) -> list:
    scores = {}

    def add(docs):
        for rank, d in enumerate(docs):
            key = d.page_content[:120]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    add(docs1)
    add(docs2)

    doc_map = {d.page_content[:120]: d for d in docs1 + docs2}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


# ===== RETRIEVAL =====

def retrieve(query: str, vs, bm25, docs: list, normalize_fn, k: int = 15) -> list:
    """
    Hybrid FAISS + BM25 retrieval for one index.

    FAISS: receives raw query — multilingual-e5 embeddings already handle
           cross-lingual matching, normalizing the query would hurt them.

    BM25:  receives normalized query — must match how the corpus was indexed,
           otherwise Arabic alef/ta marbuta variants cause zero lexical overlap.
    """
    # Semantic search on raw query
    faiss_docs = vs.similarity_search(query, k=k)

    # Lexical search on normalized query
    normalized_q = normalize_fn(query)
    scores = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]

    return rrf(faiss_docs, bm25_docs)


# ===== RERANK =====

def rerank(query: str, docs: list, reranker, top_n: int = 5) -> list:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


# ===== TRANSLATION =====

def translate(llm, text: str, target_language: str) -> str:
    try:
        prompt = (
            f"Translate the following text to {target_language}. "
            f"Return ONLY the translation, nothing else.\n\nText: {text}"
        )
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception:
        return text


# ===== CONTEXT BUILDER =====

def build_context(docs: list) -> str:
    """
    Build context string. Each chunk is labeled [Page N] so the LLM
    can cite it directly using the exact format instructed in the prompt.
    """
    out = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        source = d.metadata.get("source", "")
        lang_tag = "AR" if ARABIC_PDF_PATH in source else "EN"
        # Label format matches what the prompt instructs the LLM to cite
        out.append(f"[Page {page_num} | {lang_tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)


def validate(ans: str) -> str:
    """
    Post-generation check: warn if the LLM produced no page citation at all.
    Accepts: [Page 5], [Page 5 | AR], (Page 5), page 5, Page 5
    """
    has_citation = bool(re.search(r"(\[Page\s*\d+|\bpage\s*\d+|\(Page\s*\d+)", ans, re.IGNORECASE))
    if not has_citation:
        ans += "\n\n⚠️ Warning: The model did not produce page citations for this answer."
    return ans


# ===== PROMPTS =====
# Key improvements over original:
# 1. Citation format [Page N] matches EXACTLY what build_context() labels chunks with
# 2. A worked example is shown so the LLM knows the expected output format
# 3. "Every sentence" requirement makes citations granular, not just one at the end
# 4. Arabic prompts mirror the same structure in Arabic for consistency

CITATION_EXAMPLE_EN = """
EXAMPLE of a correct cited answer:
Q: How many annual leave days do employees get?
A: Employees are entitled to 21 working days of annual leave per year [Page 5].
   This increases to 30 days after 10 years of continuous service [Page 5].
"""

CITATION_EXAMPLE_AR = """
مثال على إجابة صحيحة مع اقتباسات:
س: كم يوم إجازة سنوية يحق للموظف؟
ج: يحق للموظف 21 يوم عمل إجازة سنوية [Page 5]. وترتفع إلى 30 يوماً بعد 10 سنوات [Page 5].
"""

BASE_EN = (
    "You are an HR policy assistant.\n\n"
    "RULES — follow exactly:\n"
    "1. Answer ONLY from the provided context. Never use outside knowledge.\n"
    "2. Every sentence in your answer MUST end with a citation in the format [Page N].\n"
    "   Use the page number from the context label, e.g. [Page 5] or [Page 5 | AR].\n"
    "3. If the same fact appears on multiple pages, cite all of them: [Page 3] [Page 7].\n"
    "4. If the answer is not in the context, respond with exactly:\n"
    "   \"This information is not available in the policy documents.\"\n"
    "5. Do not add information not present in the context.\n"
    + CITATION_EXAMPLE_EN
)

BASE_AR = (
    "أنت مساعد سياسة الموارد البشرية.\n\n"
    "القواعد — اتبعها بدقة:\n"
    "1. أجب فقط من السياق المقدم. لا تستخدم أي معلومات خارجية.\n"
    "2. كل جملة في إجابتك يجب أن تنتهي باقتباس بالتنسيق [Page N].\n"
    "   استخدم رقم الصفحة من تسمية السياق، مثلاً [Page 5] أو [Page 5 | AR].\n"
    "3. إذا لم تكن المعلومات في السياق، قل بالضبط:\n"
    "   \"هذه المعلومات غير متوفرة في وثائق السياسة.\"\n"
    "4. لا تضف معلومات غير موجودة في السياق.\n"
    + CITATION_EXAMPLE_AR
)

english_prompt = PromptTemplate(
    template=BASE_EN + "\nRespond in English.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"]
)

msa_prompt = PromptTemplate(
    template=BASE_AR + "\nأجب باللغة العربية الفصحى.\n\nالسياق:\n{context}\n\nالسؤال: {question}\nالإجابة:",
    input_variables=["context", "question"]
)

egy_prompt = PromptTemplate(
    template=BASE_AR + "\nأجب باللهجة المصرية العامية.\n\nالسياق:\n{context}\n\nالسؤال: {question}\nالإجابة:",
    input_variables=["context", "question"]
)

franco_prompt = PromptTemplate(
    template=(
        BASE_EN
        + "\nThe user wrote in Franco Arabic (Arabic written with Latin letters and numbers, "
        "e.g. '3' for ع, '7' for ح, '2' for ء, '5' for خ). "
        "Respond in Franco Arabic using the same digit-letter conventions.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["context", "question"]
)


# ===== SETUP (cached) =====

@st.cache_resource
def setup():
    emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    def build_index(path: str, normalize_fn):
        pages = PyMuPDFLoader(path).load()
        for d in pages:
            d.page_content = clean_pdf(d.page_content)

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = splitter.split_documents(pages)

        # Build BM25 corpus with the SAME normalization used at query time
        bm25_corpus = [tokenize(normalize_fn(d.page_content)) for d in docs]
        vs = FAISS.from_documents(docs, emb)
        bm25 = BM25Okapi(bm25_corpus)
        return vs, bm25, docs

    ar_index = build_index(ARABIC_PDF_PATH, normalize_arabic)
    en_index = build_index(ENGLISH_PDF_PATH, normalize_english)

    ar_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    en_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="openai/gpt-oss-120b",  # confirmed valid on Groq
        temperature=0
    )

    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    return ar_index, en_index, ar_llm, en_llm, reranker


# ===== UI =====

st.set_page_config(page_title="HR Policy Assistant", layout="wide")
st.title("💼 HR Policy Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian dialect), or Franco Arabic.")

question = st.text_input("Ask your question:",)

ar_index, en_index, ar_llm, en_llm, reranker = setup()

if question:
    with st.spinner("Searching policy documents..."):
        try:
            # ── 1. Detect language ──────────────────────────────────────────
            lang = detect_language_type(question)

            # ── 2. Translate query to BOTH languages ────────────────────────
            # Always query BOTH PDFs regardless of input language.
            # An English question can match Arabic PDF content and vice versa.
            if lang == "english":
                q_en = question
                q_ar = translate(ar_llm, question, "Arabic")
            elif lang == "arabic":
                q_ar = question
                q_en = translate(en_llm, question, "English")
            else:  # franco
                # Franco → Arabic first (needed for Arabic index retrieval)
                q_ar = translate(ar_llm, question, "Modern Standard Arabic")
                q_en = translate(en_llm, q_ar, "English")

            # ── 3. Dialect detection (Arabic-script input only) ─────────────
            dialect = None
            if lang == "arabic":
                # question IS Arabic script here — safe to classify
                dialect = get_semantic_dialect(ar_llm, question)
            elif lang == "franco":
                dialect = "egyptian"  # Franco Arabic is always Egyptian dialect

            # ── 4. Retrieve from BOTH indexes ───────────────────────────────
            ar_vs, ar_bm25, ar_docs = ar_index
            en_vs, en_bm25, en_docs = en_index

            docs_ar = retrieve(normalize_arabic(q_ar), ar_vs, ar_bm25, ar_docs, normalize_arabic)
            docs_en = retrieve(normalize_english(q_en), en_vs, en_bm25, en_docs, normalize_english)

            # ── 5. Merge + rerank ───────────────────────────────────────────
            combined = rrf(docs_ar, docs_en)
            rerank_query = q_ar if lang in ("arabic", "franco") else q_en
            top_docs = rerank(rerank_query, combined, reranker, top_n=5)

            if not top_docs:
                st.warning("No relevant documents found. Please rephrase your question.")
                st.stop()

            # ── 6. Build context ────────────────────────────────────────────
            context = build_context(top_docs)

            # ── 7. Select prompt + LLM ──────────────────────────────────────
            if lang == "english":
                prompt = english_prompt
                llm = en_llm
            elif lang == "franco":
                prompt = franco_prompt
                llm = ar_llm
            else:  # arabic
                llm = ar_llm
                prompt = egy_prompt if dialect == "egyptian" else msa_prompt

            # ── 8. Generate answer ──────────────────────────────────────────
            final_prompt = prompt.format(context=context, question=question)
            res = llm.invoke(final_prompt)
            answer = validate(res.content)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    # ── OUTPUT ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Answer")
        st.success(answer)

        st.divider()
        if lang == "english":
            if st.button("🔄 Translate answer to Arabic"):
                with st.spinner("Translating..."):
                    st.info(translate(ar_llm, answer, "Arabic"))
        elif lang in ("arabic", "franco"):
            if st.button("🔄 Translate answer to English"):
                with st.spinner("Translating..."):
                    st.info(translate(en_llm, answer, "English"))

    with col2:
        st.subheader("Query Info")
        lang_labels = {
            "english": "🇬🇧 English",
            "arabic":  "🇪🇬 Arabic",
            "franco":  "💬 Franco Arabic",
        }
        st.info(f"**Detected language:** {lang_labels.get(lang, lang)}")
        if dialect:
            st.info(f"**Dialect:** {'🗣️ Egyptian' if dialect == 'egyptian' else '📖 MSA'}")
        st.info(f"**Arabic query:** {q_ar}")
        st.info(f"**English query:** {q_en}")
        st.info(f"**Top chunks:** {len(top_docs)}")

    with st.expander("📄 Source Chunks"):
        for i, d in enumerate(top_docs, 1):
            page_num = d.metadata.get("page", 0) + 1
            source = d.metadata.get("source", "")
            lang_tag = "Arabic PDF" if ARABIC_PDF_PATH in source else "English PDF"
            st.markdown(f"**Chunk {i} — Page {page_num} ({lang_tag})**")
            st.write(d.page_content[:500])
            st.divider()