import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, pipeline

from nlp_utils import clean_pdf, normalize_arabic, normalize_english, tokenize


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

ARABIC_PDF_PATH = "ar_policy.pdf"
ENGLISH_PDF_PATH = "eng_policy.pdf"



# ─── Arabic NLP Stack (MARBERT + AraBERT) ─────────────────────

@st.cache_resource
def load_nlp_stack():
    dialect_pipe = pipeline("text-classification", model="UBC-NLP/MARBERT")
    ara_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    return dialect_pipe, ara_tokenizer


# ─── Setup (cached) ───────────────────────────────────────────

@st.cache_resource
def setup():
    dialect_pipe, ara_tokenizer = load_nlp_stack()

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    def build_index(path: str, normalize_fn):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        pages = PyMuPDFLoader(path).load()
        for d in pages:
            d.page_content = clean_pdf(d.page_content)
            d.metadata["doc_type"] = "policy"
            d.metadata["lang"] = "arabic" if ARABIC_PDF_PATH in path else "english"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separators=["\n\n", "\n", ".", " "]
        )
        docs = splitter.split_documents(pages)
        bm25_corpus = [tokenize(normalize_fn(d.page_content)) for d in docs]
        vs = FAISS.from_documents(docs, emb)
        bm25 = BM25Okapi(bm25_corpus)
        return vs, bm25, docs

    ar_index = build_index(
        ARABIC_PDF_PATH,
        lambda text: normalize_arabic(text, ara_tokenizer)
    )
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
    return ar_index, en_index, ar_llm, en_llm, reranker, dialect_pipe, ara_tokenizer