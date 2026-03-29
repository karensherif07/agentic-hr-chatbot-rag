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

ARABIC_PDF_PATH = "arabic_policy.pdf"
ENGLISH_PDF_PATH = "eng_policy.pdf"

FRANCO_TIER1 = {
    "ana", "enta", "enti", "ehna", "entom", "howa", "hya", "homma",
    "msh", "mesh", "mish", "mafish",
    "leh", "leih", "fein", "fen", "emta", "ezay", "meen", "eih", "eh",
    "ya3ni", "3ashan", "momken", "3ayz", "3ayza", "yenfa3", "ynfa3",
    "tayeb", "tamam", "keda", "kidda", "bas",
    "7aga", "haga", "feeh", "fieh", "la2", "aywa", "aiwa",
    "wenta", "wenti", "bs", "delwa2ty", "badein", "b3dein",
    "el", "di", "da", "dol", "aho", "ahi",
}


def has_arabic_digit_combo(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    for tok in tokens:
        if len(tok) >= 2 and re.search(r"[a-z]", tok) and re.search(r"[23578]", tok):
            return True
    return False


def detect_language_type(text: str) -> str:
    if re.search(r"[\u0600-\u06FF]", text):
        return "arabic"
    tokens = set(re.findall(r"[a-zA-Z0-9']+", text.lower()))
    if tokens & FRANCO_TIER1:
        return "franco"
    if has_arabic_digit_combo(text):
        return "franco"
    return "english"


def get_semantic_dialect(llm, arabic_text: str) -> str:
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


def clean_pdf(text: str) -> str:
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u200e\u200f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"[\u064B-\u065F]", "", text)
    return text.lower()


def normalize_english(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> list:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


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


def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 15) -> list:
    faiss_docs = vs.similarity_search(raw_query, k=k)
    normalized_q = normalize_fn(raw_query)
    scores = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]
    return rrf(faiss_docs, bm25_docs)


def rerank(query: str, docs: list, reranker, top_n: int = 5) -> list:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


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


def build_context(docs: list) -> str:
    out = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        source = d.metadata.get("source", "")
        lang_tag = "AR" if ARABIC_PDF_PATH in source else "EN"
        out.append(f"[Page {page_num} | {lang_tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)


# ─────────────────────────────────────────────────────────────
# FIX 1: Citation validation — only flag truly missing citations.
#         Do NOT append extra text to every valid answer.
# ─────────────────────────────────────────────────────────────
def validate(ans: str) -> str:
    # Strip any stray trailing whitespace
    ans = ans.strip()

    # Check for at least one page citation anywhere in the answer
    has_citation = bool(
        re.search(r"\[Page\s*\d+", ans, re.IGNORECASE)
    )

    # Only add the warning when the answer is substantive but has no citations at all
    not_found_phrases = [
        "not available in the policy",
        "غير متوفرة في وثائق",
        "not found in",
    ]
    is_not_found = any(p.lower() in ans.lower() for p in not_found_phrases)

    if not has_citation and not is_not_found:
        ans += "\n\n⚠️ لم يتم ذكر أرقام الصفحات في هذه الإجابة." if re.search(
            r"[\u0600-\u06FF]", ans
        ) else "\n\n⚠️ Page citations were not produced for this answer."

    return ans


# ─────────────────────────────────────────────────────────────
# FIX 2: "Used chunks" extraction — parse cited page numbers from
#         the answer and only show chunks whose page matches.
# ─────────────────────────────────────────────────────────────
def get_cited_pages(answer: str) -> set:
    """Return a set of 1-based page numbers mentioned in citations."""
    return {int(n) for n in re.findall(r"\[Page\s*(\d+)", answer, re.IGNORECASE)}


def filter_cited_chunks(docs: list, cited_pages: set) -> list:
    """Return only the chunks whose page number appears in the answer."""
    if not cited_pages:
        return docs  # fallback: show all if no citations parsed
    return [
        d for d in docs
        if (d.metadata.get("page", 0) + 1) in cited_pages
    ]


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

# ─────────────────────────────────────────────────────────────
# FIX 3: Arabic prompts rewritten in 3rd-person ("الموظف يستحق...")
#         instead of 1st-person ("يحق لك..."), since the bot is
#         answering factual HR policy questions, not addressing the
#         user directly.
# ─────────────────────────────────────────────────────────────
BASE_AR = (
    "أنت مساعد سياسة الموارد البشرية.\n\n"
    "القواعد — اتبعها بدقة:\n"
    "1. أجب فقط من السياق المقدم. لا تستخدم أي معلومات خارجية.\n"
    "2. كل جملة في إجابتك يجب أن تنتهي باقتباس بالتنسيق [Page N].\n"
    "   استخدم رقم الصفحة من تسمية السياق، مثلاً [Page 5] أو [Page 5 | AR].\n"
    "3. إذا لم تكن المعلومات في السياق، قل بالضبط:\n"
    "   \"هذه المعلومات غير متوفرة في وثائق السياسة.\"\n"
    "4. لا تضف معلومات غير موجودة في السياق.\n"
    "5. أجب بصيغة الغائب (الشخص الثالث)، مثل:\n"
    "   'يستحق الموظف...' و'تنص السياسة على...' — وليس 'يحق لك...' أو 'أنت تستحق...'.\n"
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
    template=BASE_AR + "\nأجب باللهجة المصرية العامية (صيغة الغائب أيضاً، مثل: 'الموظف بيستحق...').\n\nالسياق:\n{context}\n\nالسؤال: {question}\nالإجابة:",
    input_variables=["context", "question"]
)

franco_prompt = PromptTemplate(
    template=(
        BASE_EN
        + "\nThe user wrote in Franco Arabic (Arabic written with Latin letters and numbers, "
        "e.g. '3' for ع, '7' for ح, '2' for ء, '5' for خ). "
        "Respond in Franco Arabic using the same digit-letter conventions. "
        "Use third-person phrasing (e.g. 'el mowazaf bystahel...' not 'enta btstahel...').\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["context", "question"]
)


@st.cache_resource
def setup():
    emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    def build_index(path: str, normalize_fn):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        pages = PyMuPDFLoader(path).load()
        for d in pages:
            d.page_content = clean_pdf(d.page_content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = splitter.split_documents(pages)
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
        model_name="openai/gpt-oss-120b",
        temperature=0
    )

    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return ar_index, en_index, ar_llm, en_llm, reranker


# ─── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="HR Policy Assistant", layout="wide")
st.title("💼 HR Policy Assistant")
st.caption("Ask in English, Arabic (MSA or Egyptian dialect), or Franco Arabic.")

question = st.text_input("Ask your question:")

try:
    ar_index, en_index, ar_llm, en_llm, reranker = setup()
except Exception as e:
    st.error(f"Setup Error: {e}")
    st.stop()

if question:
    with st.spinner("Searching policy documents..."):
        try:
            lang = detect_language_type(question)

            if lang == "english":
                q_en = question
                q_ar = translate(ar_llm, question, "Arabic")
            elif lang == "arabic":
                q_ar = question
                q_en = translate(en_llm, question, "English")
            else:  # franco
                q_ar = translate(ar_llm, question, "Modern Standard Arabic")
                q_en = translate(en_llm, q_ar, "English")

            dialect = None
            if lang == "arabic":
                dialect = get_semantic_dialect(ar_llm, question)
            elif lang == "franco":
                dialect = "franco"

            ar_vs, ar_bm25, ar_docs = ar_index
            en_vs, en_bm25, en_docs = en_index

            docs_ar = retrieve(q_ar, ar_vs, ar_bm25, ar_docs, normalize_arabic)
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
            answer = validate(res.content)

            # ── FIX 2: Identify which chunks were actually cited ──
            cited_pages = get_cited_pages(answer)
            cited_docs = filter_cited_chunks(top_docs, cited_pages)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Answer")
                st.success(answer)
                st.divider()
                if st.button("🔄 Translate answer"):
                    if lang == "english":
                        st.info(translate(ar_llm, answer, "Arabic"))
                    else:
                        st.info(translate(en_llm, answer, "English"))

            with col2:
                st.subheader("Query Info")
                st.info(f"**Detected language:** {lang}")
                if dialect:
                    st.info(f"**Dialect:** {dialect}")
                st.info(f"**Arabic query:** {q_ar}")
                st.info(f"**English query:** {q_en}")

            # ── FIX 2: Show ONLY cited chunks ──
            if cited_docs:
                with st.expander(f"📄 Source Chunks ({len(cited_docs)} cited)"):
                    for i, d in enumerate(cited_docs, 1):
                        source = (
                            "Arabic PDF"
                            if ARABIC_PDF_PATH in d.metadata.get("source", "")
                            else "English PDF"
                        )
                        page_no = d.metadata.get("page", 0) + 1
                        st.markdown(f"**Chunk {i} — Page {page_no} ({source})**")
                        st.write(d.page_content)
            else:
                with st.expander("📄 Source Chunks (retrieved)"):
                    for i, d in enumerate(top_docs, 1):
                        source = (
                            "Arabic PDF"
                            if ARABIC_PDF_PATH in d.metadata.get("source", "")
                            else "English PDF"
                        )
                        page_no = d.metadata.get("page", 0) + 1
                        st.markdown(f"**Chunk {i} — Page {page_no} ({source})**")
                        st.write(d.page_content)

        except Exception as e:
            st.error(f"Error: {str(e)}")