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
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if arabic_chars > latin_chars:
        return "arabic"
    
    english_starters = re.compile(r'^(what|how|is|are|can|where|who|the|give|show|tell|if|when|why)\b', re.I)
    
    has_franco_slang = bool(FRANCO_PATTERN.search(text))
    has_franco_numerals = bool(re.search(r'[23578]', text)) > 2
    
    if english_starters.match(text) and not has_franco_numerals:
        return "english"
        
    if has_franco_slang or has_franco_numerals:
        return "franco"

    return "english"

def get_semantic_dialect(llm, text: str) -> str:
    if len(text.split()) < 2:
        return "msa"

    prompt = (
        "Classify the following Arabic text into 'egyptian' or 'msa' (Modern Standard Arabic).\n"
        "Egyptian features: b- prefix in verbs (بياخد), words like 'عايز', 'ده', 'إيه', 'اللي', or Egyptian sentence structure.\n"
        "Text: {text}\n"
        "Classify as 'egyptian' or 'msa' only. Return one word."
    ).format(text=text)
    
    try:
        res = llm.invoke(prompt)
        label = res.content.strip().lower()
        return "egyptian" if "egyptian" in label else "msa"
    except:
        return "msa"

def clean_pdf_artifacts(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_for_search(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[أإآ]', 'ا', text)
    text = text.replace('ة', 'ه').replace('ى', 'ي')
    text = text.replace('ؤ', 'و').replace('ئ', 'ي')
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

BASE_RULES = """
STRICT RULES:
- Answer ONLY from the provided context. Do not use outside knowledge.
- Every factual statement MUST include a page citation in the format [Page X].
- If the answer is not in the context, say clearly: "This information is not available in the policy document."
- Do NOT fabricate or guess.
- Rewrite any garbled or unclear extracted text into clean, readable language before using it.
"""

english_prompt = PromptTemplate(
    template=BASE_RULES + "Answer in clear English.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n",
    input_variables=["context", "question"]
)

msa_prompt = PromptTemplate(
    template=BASE_RULES + "أجب باللغة العربية الفصحى الواضحة.\n\nالسياق:\n{context}\n\nالسؤال:\n{question}\n\nالإجابة:\n",
    input_variables=["context", "question"]
)

egyptian_prompt = PromptTemplate(
    template=BASE_RULES + "أجب بالعربية المصرية العامية البسيطة (زي ما بنتكلم في مصر).\n\nالسياق:\n{context}\n\nالسؤال:\n{question}\n\nالإجابة:\n",
    input_variables=["context", "question"]
)

franco_prompt = PromptTemplate(
    template=BASE_RULES + "Answer ONLY in Franco Arabic (Arabizi) using Latin letters and numbers. Do NOT use Arabic script. Do NOT use any other language.\n\nالسياق:\n{context}\n\nالسؤال:\n{question}\n\nالإجابة:\n",
    input_variables=["context", "question"]
)

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
    try:
        res = llm.invoke(rewrite_prompt_template.format(question=question))
        rewritten = res.content.strip()
        if len(rewritten) < 3:
            return question
        return rewritten
    except Exception:
        return question

def simple_tokenize(text: str) -> list:
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())

def hybrid_retrieve(query: str, vectorstore, bm25, docs: list, k: int = 20) -> list:
    faiss_docs = vectorstore.similarity_search(query, k=k)
    scores = bm25.get_scores(simple_tokenize(query))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_docs = [docs[i] for i in ranked_indices[:k]]
    combined, seen = [], set()
    for d in faiss_docs + bm25_docs:
        key = d.page_content[:120]
        if key not in seen:
            combined.append(d)
            seen.add(key)
    return combined

def rerank(query: str, docs: list, reranker, top_k: int = 5) -> list:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_k]]

def build_context(docs: list, llm=None) -> str:
    blocks = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        content = doc.page_content
        if llm and looks_broken(content):
            content = reconstruct_arabic(llm, content)
        blocks.append(f"[Page {page_num}]\n{content}")
    return "\n\n---\n\n".join(blocks)

def validate_citations(answer: str) -> str:
    if "[Page" not in answer:
        return answer + "\n\n⚠️ Note: The model did not include page citations in this answer."
    return answer

def looks_broken(text: str) -> bool:
    tokens = re.findall(r'[\u0600-\u06FF]', text)
    if not tokens:
        return False
    fragmentation_matches = re.findall(r'[\u0600-\u06FF]\s[\u0600-\u06FF]', text)
    if len(tokens) > 5 and (len(fragmentation_matches) / (len(tokens)/2)) > 0.5:
        return True
    return False

@st.cache_data(show_spinner=False)
def reconstruct_arabic(_llm, broken_text: str) -> str:
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

@st.cache_resource
def setup():
    loader = PyMuPDFLoader("english_policy.pdf")
    pages = loader.load()
    for d in pages:
        d.page_content = clean_pdf_artifacts(d.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "،", " ", ""]
    )
    docs = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vectorstore = FAISS.from_documents(docs, embeddings)
    bm25 = BM25Okapi([simple_tokenize(d.page_content) for d in docs])
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return vectorstore, bm25, docs, reranker, llm

st.set_page_config(page_title="HR Chatbot", layout="wide")
st.title("💼 HR Policy Assistant")
st.caption("Arabic (MSA & Egyptian) • Franco • English — With Verified Citations")

question = st.text_input("💬 Ask your question:")
vectorstore, bm25, docs, reranker, llm = setup()

if question:
    with st.spinner("🔍 Analyzing and Searching..."):
        lang = detect_language_type(question)
        dialect = get_semantic_dialect(llm, question) if lang == "arabic" else None

        if lang in ("arabic", "franco"):
            msa_query = rewrite_query(llm, question)
            search_query = normalize_for_search(msa_query)
        else:
            if lang == "english":
                search_query = question.lower().strip()

        retrieved = hybrid_retrieve(search_query, vectorstore, bm25, docs)
        top_docs = rerank(search_query, retrieved, reranker, top_k=5)
        context = build_context(top_docs, llm=llm)

        if lang == "english":
            prompt_template = english_prompt
        elif lang == "franco":
            prompt_template = franco_prompt
        elif lang == "arabic" and dialect == "egyptian":
            prompt_template = egyptian_prompt
        else:
            prompt_template = msa_prompt

        final_prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(final_prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = validate_citations(answer)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🧠 Answer")
        st.success(answer)
    with col2:
        st.markdown("### 🌐 Detected Language")
        display_lang = lang.upper()
        if dialect: display_lang += f" ({dialect.upper()})"
        st.info(display_lang)
        st.markdown("### 🔍 Search Query Used")
        st.code(search_query, language=None)

    with st.expander("📚 Sources & Evidence"):
        for i, d in enumerate(top_docs):
            page = d.metadata.get("page", 0) + 1
            st.markdown(f"**📄 Source {i+1} — Page {page}**")
            snippet = d.page_content[:600]
            if looks_broken(snippet):
                with st.spinner(f"🔧 Reconstructing source {i+1} text..."):
                    snippet = reconstruct_arabic(llm, snippet)
            st.write(snippet + ("..." if len(d.page_content) > 600 else ""))
            st.divider()