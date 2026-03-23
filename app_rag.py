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

# ---------------- SETUP ----------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------- LANGUAGE DETECTION ----------------

# Franco Arabic uses Latin letters to write Arabic sounds.
# We detect it BEFORE checking for Arabic script so a mixed message
# (some Arabic letters + some Latin) still routes correctly.
FRANCO_PATTERN = re.compile(
    r'\b(ana|enta|enti|howa|hiya|mesh|msh|wala|bas|kda|keda|tayeb|tb|'
    r'leh|leih|izzay|ezay|fein|fen|meen|mn|3shan|3alshan|bs|'
    r'yalla|ya|ya3ni|ya3ny|asl|asln|mthlan|akid|tab3an|'
    r'momken|m3lsh|ma3lesh|khalas|xlas|la2|ah|aiwa|'
    r'kteer|ktir|shwaya|shwy|7aga|7aja|3ayz|3ayza|'
    r'aho|di|dah|deh|elly|illi|law|lw|kan|knt)\b',
    re.IGNORECASE
)

def detect_language_type(text: str) -> str:
    """
    Returns 'franco', 'arabic', or 'english'.
    Franco must be checked first because Franco text uses Latin script
    and would otherwise fall through to 'english'.
    """
    # Franco: Latin-script Arabic slang + common substitutions (3, 7, 2, etc.)
    has_franco_words = bool(FRANCO_PATTERN.search(text))
    # Also treat heavy use of Arabic numerals-as-letters as a Franco signal
    has_letter_digits = len(re.findall(r'[3792]', text)) >= 2

    if has_franco_words or has_letter_digits:
        return "franco"

    if re.search(r'[\u0600-\u06FF]', text):
        return "arabic"

    return "english"


def detect_arabic_dialect(text: str) -> str:
    """
    Among Arabic-script messages, distinguish Egyptian colloquial from MSA.
    Returns 'egyptian' or 'msa'.
    """
    egyptian_markers = re.compile(
        r'\b(إيه|ايه|ازاي|إزاي|فين|مين|عايز|عايزة|مش|بتاع|بتاعة|'
        r'دلوقتي|دلوقت|كده|كدا|طيب|اللي|علشان|ليه|إيه ده|ايه ده)\b'
    )
    if egyptian_markers.search(text):
        return "egyptian"
    return "msa"

# ---------------- TEXT CLEANING ----------------
# IMPORTANT: Only clean text for display/LLM input.
# Do NOT aggressively normalize document chunks — it destroys recall.

def clean_pdf_artifacts(text: str) -> str:
    """Remove common PDF extraction noise without mangling real content."""
    if not text:
        return ""
    # Remove sequences of dots (table of contents leaders, etc.)
    text = re.sub(r'\.{3,}', ' ', text)
    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_for_search(text: str) -> str:
    """
    Normalize Arabic text specifically for BM25/vector search queries.
    Apply ONLY to search queries, NEVER to stored document text.
    """
    if not text:
        return ""
    text = text.lower()

    # Normalize Alef variants
    text = re.sub(r'[أإآ]', 'ا', text)
    # Normalize Teh Marbuta and Alef Maqsura
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    # Normalize Hamza variants
    text = text.replace('ؤ', 'و').replace('ئ', 'ي')
    # Strip diacritics (tashkeel)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ---------------- PROMPTS ----------------

BASE_RULES = """
STRICT RULES:
- Answer ONLY from the provided context. Do not use outside knowledge.
- Every factual statement MUST include a page citation in the format [Page X].
- If the answer is not in the context, say clearly: "This information is not available in the policy document."
- Do NOT fabricate or guess.
- Rewrite any garbled or unclear extracted text into clean, readable language before using it.
"""

english_prompt = PromptTemplate(
    template=BASE_RULES + """
Answer in clear English.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

msa_prompt = PromptTemplate(
    template=BASE_RULES + """
أجب باللغة العربية الفصحى الواضحة.

السياق:
{context}

السؤال:
{question}

الإجابة:
""",
    input_variables=["context", "question"]
)

egyptian_prompt = PromptTemplate(
    template=BASE_RULES + """
أجب بالعربية المصرية العامية البسيطة (زي ما بنتكلم في مصر).

السياق:
{context}

السؤال:
{question}

الإجابة:
""",
    input_variables=["context", "question"]
)

franco_prompt = PromptTemplate(
    template=BASE_RULES + """
Answer in Franco Arabic (Arabizi) — the same mix of Latin letters and numbers
that the user wrote in. Example style: "el policy bte2ool en..." or "lazem t3mel...".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# ---------------- QUERY REWRITE ----------------

rewrite_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=(
        "أنت مساعد لتحويل الأسئلة إلى العربية الفصحى الواضحة لأغراض البحث فقط.\n"
        "حوّل السؤال التالي إلى عبارة بحث بالعربية الفصحى. أعد العبارة فقط بدون شرح.\n\n"
        "السؤال: {question}\n"
        "عبارة البحث:"
    )
)

def rewrite_query(llm, question: str) -> str:
    """Rewrite colloquial/Franco question to MSA for better retrieval."""
    try:
        res = llm.invoke(rewrite_prompt_template.format(question=question))
        rewritten = res.content.strip()
        # Basic sanity check: if rewrite looks empty or too short, fall back
        if len(rewritten) < 3:
            return question
        return rewritten
    except Exception:
        return question

# ---------------- TOKENIZER ----------------

def simple_tokenize(text: str) -> list:
    """Tokenize for BM25: splits on non-word boundaries, lowercases."""
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())

# ---------------- RETRIEVAL ----------------

def hybrid_retrieve(query: str, vectorstore, bm25, docs: list, k: int = 20) -> list:
    """Combine FAISS semantic search with BM25 keyword search (Reciprocal Rank Fusion)."""
    # Semantic search
    faiss_docs = vectorstore.similarity_search(query, k=k)

    # BM25 keyword search
    scores = bm25.get_scores(simple_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_docs = [docs[i] for i in ranked_indices[:k]]

    # Merge with deduplication (FAISS results first = higher implicit rank)
    combined, seen = [], set()
    for d in faiss_docs + bm25_docs:
        key = d.page_content[:120]  # Use prefix as dedup key
        if key not in seen:
            combined.append(d)
            seen.add(key)

    return combined


def rerank(query: str, docs: list, reranker, top_k: int = 5) -> list:
    """Re-rank retrieved docs using a cross-encoder model."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

# ---------------- CONTEXT BUILDER ----------------

def build_context(docs: list, llm=None) -> str:
    """
    Build a clean context string.
    If llm is provided, broken Arabic chunks are reconstructed before being
    passed to the answer LLM — improves both answer quality and citations.
    Uses [Page X] format consistently so the LLM produces matching citations.
    """
    blocks = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        content = doc.page_content
        if llm and looks_broken(content):
            content = reconstruct_arabic(llm, content)
        blocks.append(f"[Page {page_num}]\n{content}")
    return "\n\n---\n\n".join(blocks)

# ---------------- CITATION VALIDATION ----------------

def validate_citations(answer: str) -> str:
    """Warn if the model forgot to include any citations."""
    if "[Page" not in answer:
        return answer + "\n\n⚠️ Note: The model did not include page citations in this answer."
    return answer

# ---------------- TEXT RECONSTRUCTION ----------------

def looks_broken(text: str) -> bool:
    """
    Detect PDF-extracted Arabic text that has spurious spaces inserted between
    every letter/word — a common artifact of RTL font encoding in PyMuPDF.
    Heuristic: if the average Arabic 'word' length is <= 2 characters, the text
    is almost certainly character-fragmented.
    """
    arabic_tokens = re.findall(r'[\u0600-\u06FF]+', text)
    if not arabic_tokens:
        return False
    avg_len = sum(len(t) for t in arabic_tokens) / len(arabic_tokens)
    return avg_len <= 2.5

@st.cache_data(show_spinner=False)
def reconstruct_arabic(_llm, broken_text: str) -> str:
    """
    Ask the LLM to restore broken/fragmented Arabic text to readable form.
    Uses st.cache_data so the same chunk is only reconstructed once per session.
    """
    prompt = (
        "النص التالي مستخرج من ملف PDF وفيه مسافات زائدة بين الحروف والكلمات بسبب مشكلة في الترميز.\n"
        "أعد كتابة النص بشكل صحيح وواضح بدون أي تعليق أو إضافة. أعد النص فقط.\n\n"
        f"النص المكسور:\n{broken_text}\n\n"
        "النص الصحيح:"
    )
    try:
        res = _llm.invoke(prompt)
        result = res.content.strip()
        return result if len(result) > 10 else broken_text
    except Exception:
        return broken_text

# ---------------- SYSTEM SETUP ----------------

@st.cache_resource
def setup():
    loader = PyMuPDFLoader("arabic_policy_2.pdf")
    pages = loader.load()

    # Only clean PDF artifacts — do NOT normalize Arabic letters here.
    # Normalization is applied only to search queries, not stored content.
    for d in pages:
        d.page_content = clean_pdf_artifacts(d.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "،", " ", ""]
    )
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # BM25 over raw (lightly cleaned) text — no aggressive normalization
    bm25 = BM25Okapi([simple_tokenize(d.page_content) for d in docs])

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    return vectorstore, bm25, docs, reranker, llm

# ---------------- UI ----------------

st.set_page_config(page_title="HR Chatbot", layout="wide")

st.title("💼 HR Policy Assistant")
st.caption("Arabic (MSA & Egyptian) • Franco • English — With Verified Citations")

question = st.text_input("💬 Ask your question:")

vectorstore, bm25, docs, reranker, llm = setup()

if question:
    with st.spinner("🔍 Searching policy..."):

        # Step 1: Detect language
        lang = detect_language_type(question)
        dialect = detect_arabic_dialect(question) if lang == "arabic" else None

        # Step 2: Build search query
        # For non-English: rewrite to MSA first, then normalize for search
        if lang in ("arabic", "franco"):
            msa_query = rewrite_query(llm, question)
            search_query = normalize_for_search(msa_query)
        else:
            search_query = question

        # Step 3: Retrieve and rerank
        retrieved = hybrid_retrieve(search_query, vectorstore, bm25, docs)
        top_docs = rerank(search_query, retrieved, reranker, top_k=5)

        # Step 4: Build context (reconstruct broken Arabic before sending to LLM)
        context = build_context(top_docs, llm=llm)

        # Step 5: Select prompt based on language + dialect
        if lang == "english":
            prompt = english_prompt
        elif lang == "franco":
            prompt = franco_prompt
        elif lang == "arabic" and dialect == "egyptian":
            prompt = egyptian_prompt
        else:
            prompt = msa_prompt  # Default Arabic → MSA

        # Step 6: Generate answer
        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = validate_citations(answer)

    # -------- OUTPUT --------

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🧠 Answer")
        st.success(answer)

    with col2:
        st.markdown("### 🌐 Detected Language")
        display_lang = lang.upper()
        if dialect:
            display_lang += f" ({dialect.upper()})"
        st.info(display_lang)

        st.markdown("### 🔍 Search Query Used")
        st.code(search_query, language=None)

    # -------- SOURCES --------

    with st.expander("📚 Sources & Evidence"):
        for i, d in enumerate(top_docs):
            page = d.metadata.get("page", 0) + 1
            st.markdown(f"**📄 Source {i+1} — Page {page}**")

            snippet = d.page_content[:600]

            # If the extracted text is letter-fragmented, reconstruct it with the LLM
            if looks_broken(snippet):
                with st.spinner(f"🔧 Reconstructing source {i+1} text..."):
                    snippet = reconstruct_arabic(llm, snippet)

            st.write(snippet + ("..." if len(d.page_content) > 600 else ""))
            st.divider()