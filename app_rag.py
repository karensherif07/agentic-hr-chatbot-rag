import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS      
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------- LANGUAGE DETECTION ----------------

def is_arabic(text: str) -> bool:
    arabic_range = re.compile(r'[\u0600-\u06FF]')
    return bool(arabic_range.search(text))


# ---------------- NORMALIZATION ----------------

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower()

    arabic_map = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
    }

    for k, v in arabic_map.items():
        text = text.replace(k, v)

    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_arabic_text(text):
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------- PROMPTS ----------------

# English prompt (kept)
english_prompt_template = """Use the following pieces of context to answer the user's question. 

If the question is about eligibility for a specific employee group (like Faculty or exempt Admin), 
look carefully for explicit exclusions or negative constraints, such as sentences starting 
with "do not receive," "are not eligible," "is not obligated," or "program does not apply." 

Always prioritize specific eligibility criteria (FTE requirements, length of service) 
and check the FAQ sections provided in the context for clarifications.

If the context explicitly states that a group does NOT receive a benefit or is NOT eligible, 
your answer must reflect that.

 If the answer is not contained within the provided
 context, state clearly that you do not have this information in the current policy 
 and advise the user to contact the HR department directly. Do not use your internal knowledge.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

# Arabic prompt (your original)
arabic_prompt_template = """استخدم المعلومات التالية للإجابة على سؤال المستخدم.

إذا كان السؤال يتعلق بأهلية فئة معينة من الموظفين (مثل أعضاء هيئة التدريس أو الإداريين)،
ابحث بعناية عن أي استثناءات أو قيود واضحة مثل:
"لا يحق لهم"، "غير مؤهل"، "لا ينطبق عليهم".

يجب إعطاء الأولوية لشروط الأهلية المحددة مثل نسبة الدوام (FTE) أو مدة الخدمة.

إذا كان السياق يذكر بوضوح أن فئة معينة غير مؤهلة أو لا تحصل على ميزة،
فيجب أن تعكس إجابتك ذلك بوضوح.

تجاهل أرقام الصفحات الموجودة في الفهارس أو جداول المحتويات.

أجب بنفس لغة السؤال الأصلي.

السياق:
{context}

السؤال:
{question}

الإجابة:
"""

english_prompt = PromptTemplate(
    template=english_prompt_template,
    input_variables=["context", "question"]
)

arabic_prompt = PromptTemplate(
    template=arabic_prompt_template,
    input_variables=["context", "question"]
)


rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
قم بإعادة صياغة السؤال التالي إلى اللغة العربية الفصحى (MSA).
إذا كان السؤال باللهجة المصرية أو فرانكو، قم بتحويله إلى فصحى واضحة.

السؤال:
{question}

السؤال المعاد صياغته:
"""
)

# ---------------- FUNCTIONS ----------------

def rewrite_query(llm, question):
    try:
        prompt = rewrite_prompt.format(question=question)
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content.strip()
        return str(response).strip()
    except:
        return question


def simple_tokenize(text):
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())


# ---------------- RETRIEVAL ----------------

def bm25_retrieve(query, bm25, texts, k=10):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [texts[i] for i in ranked_indices[:k]]


def hybrid_retrieve(query, vectorstore, bm25, all_docs, k=20):
    # FAISS returns Document objects
    faiss_results = vectorstore.similarity_search(query, k=k)

    # BM25 retrieval
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    bm25_results = [all_docs[i] for i in ranked_indices[:k]]

    combined = []
    seen_content = set()

    for doc in faiss_results + bm25_results:
        if doc.page_content not in seen_content:
            combined.append(doc)
            seen_content.add(doc.page_content)

    return combined[:k]

def rerank(query, docs, reranker, top_k=5):
    # Pass page_content to the reranker
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ---------------- SETUP ----------------

@st.cache_resource
def setup_rag_system():
    loader = PyMuPDFLoader("english_policy.pdf")
    pages = loader.load()

    for doc in pages:
        doc.page_content = normalize_text(clean_arabic_text(doc.page_content))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # Split documents to keep metadata (like page numbers) attached to chunks
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # For BM25, we need the text and a way to reference the original doc
    tokenized_corpus = [simple_tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    return vectorstore, bm25, docs, reranker, llm


# ---------------- STREAMLIT UI ----------------

st.title("RAG HR Chatbot (Hybrid + Rerank + Arabic + English)")

user_question = st.text_input("Ask a question about the policy:")

vectorstore, bm25, texts, reranker, llm = setup_rag_system()

if user_question:
    with st.spinner("Processing..."):
        arabic_input = is_arabic(user_question)

        if arabic_input:
            rewritten_question = rewrite_query(llm, user_question)
            rewritten_question = normalize_text(rewritten_question)
            st.write("🔄 Rewritten Question (MSA):")
            st.write(rewritten_question)
            search_query = rewritten_question
        else:
            search_query = user_question

        # Retrieval (returns Document objects)
        retrieved_docs = hybrid_retrieve(
            search_query, vectorstore, bm25, texts, k=20
        )

        # Reranking (returns Document objects)
        top_docs = rerank(search_query, retrieved_docs, reranker, top_k=4)

        # Format context with Page Numbers for the LLM
        context_list = []
        for doc in top_docs:
            page_num = doc.metadata.get("page", 0) + 1 # PyMuPDF is 0-indexed
            context_list.append(f"[Source: Page {page_num}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_list)

        final_prompt_template = arabic_prompt if arabic_input else english_prompt
        final_prompt = final_prompt_template.format(
            context=context,
            question=user_question
        )

        response = llm.invoke(final_prompt)

        st.write("### Answer:")
        st.write(response.content if hasattr(response, "content") else str(response))

        with st.expander("View Retrieved Snippets & Sources"):
            for i, doc in enumerate(top_docs):
                page_label = doc.metadata.get("page", 0) + 1
                st.markdown(f"**Snippet {i+1} — Page {page_label}**")
                st.info(doc.page_content)