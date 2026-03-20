
import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader , PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS      
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------- PROMPTS ----------------

# Custom prompt template
# custom_prompt_template = """Use the following pieces of context to answer the user's question. 
# If the question is about eligibility for a specific employee group (like Faculty or exempt Admin), 
# look carefully for explicit exclusions or negative constraints, such as sentences starting 
# with "do not receive," "are not eligible," "is not obligated," or "program does not apply." 

# Always prioritize specific eligibility criteria (FTE requirements, length of service) 
# and check the FAQ sections provided in the context for clarifications.

# If the context explicitly states that a group does NOT receive a benefit or is NOT eligible, 
# your answer must reflect that.

# Context: {context}
# Question: {question}

# Helpful Answer:"""

custom_prompt_template = """استخدم المعلومات التالية للإجابة على سؤال المستخدم.

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

QA_CHAIN_PROMPT = PromptTemplate(
    template=custom_prompt_template,
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
    prompt = rewrite_prompt.format(question=question)
    response = llm.invoke(prompt)

    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()


def clean_arabic_text(text):
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------- RETRIEVAL ----------------

def faiss_retrieve(query, vectorstore, k=10):
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def bm25_retrieve(query, bm25, texts, k=10):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [texts[i] for i in ranked_indices[:k]]


def hybrid_retrieve(query, vectorstore, bm25, texts, k=20):
    faiss_results = vectorstore.similarity_search(query, k=k)
    faiss_docs = [doc.page_content for doc in faiss_results]
    bm25_docs = bm25_retrieve(query, bm25, texts, k)
    
    # Simple deduplication while preserving order
    combined = []
    seen = set()
    for doc in faiss_docs + bm25_docs:
        if doc not in seen:
            combined.append(doc)
            seen.add(doc)
    return combined[:k]


def rerank(query, docs, reranker, top_k=5):
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# ---------------- SETUP ----------------

@st.cache_resource
def setup_rag_system():
    # loader = PyPDFLoader("english_policy.pdf")
    loader = PyMuPDFLoader("arabic_policy.pdf")
    pages = loader.load()

    for doc in pages:
        doc.page_content = clean_arabic_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    texts = [doc.page_content for doc in docs]

    # Embeddings + FAISS
        
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # BM25
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # LLM
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # Reranker
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    return vectorstore, bm25, texts, reranker, llm


# ---------------- STREAMLIT UI ----------------

st.title("RAG HR Chatbot (Hybrid + Rerank + Arabic)")

user_question = st.text_input("Ask a question about the policy:")

vectorstore, bm25, texts, reranker, llm = setup_rag_system()

if user_question:
    with st.spinner("Processing..."):

        # Step 1: Rewrite query
        rewritten_question = rewrite_query(llm, user_question)

        st.write("🔄 Rewritten Question (MSA):")
        st.write(rewritten_question)

        # Step 2: Hybrid retrieval
        retrieved_docs = hybrid_retrieve(
            rewritten_question, vectorstore, bm25, texts, k=20
        )

        # Step 3: Reranking
        top_docs = rerank(rewritten_question, retrieved_docs, reranker, top_k=4)

        # Step 4: Build context
        context = "\n\n".join(top_docs)

        # Step 5: Generate answer
        final_prompt = QA_CHAIN_PROMPT.format(
            context=context,
            question=rewritten_question
        )

        response = llm.invoke(final_prompt)

        st.write("### Answer:")
        if hasattr(response, "content"):
            st.write(response.content)
        else:
            st.write(str(response))

        # Step 6: Show retrieved snippets
        with st.expander("View Retrieved Snippets"):
            for i, doc in enumerate(top_docs):
                st.markdown(f"**Snippet {i+1}:**")
                st.info(doc)