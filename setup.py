import os
from dotenv import load_dotenv
import streamlit as st
import fitz  # PyMuPDF

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

ARABIC_PDF_PATH = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"


# ─── PDF Page Rendering ────────────────────────────────────────

@st.cache_data
def render_page_to_image(pdf_path: str, page_num: int, zoom: float = 2.0, clip_text: str = None) -> bytes:
    """Render a full PDF page to PNG image bytes."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


@st.cache_data
def render_page_snippet(pdf_path: str, page_num: int, clip_text: str, zoom: float = 2.0,
                        padding: int = 20) -> bytes:
    """
    Renders only the region of the page containing clip_text, with padding.
    Falls back to the full page if the text is not found.
    Highlights the matched region in yellow.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    matrix = fitz.Matrix(zoom, zoom)

    # Try to find the text on the page
    # Use the first ~80 chars of chunk text to search (avoid too-long strings)
    search_text = clip_text[:80].strip() if clip_text else ""
    instances = page.search_for(search_text) if search_text else []

    if instances:
        # Union bounding box of all matches
        rect = instances[0]
        for inst in instances[1:]:
            rect = rect | inst

        # Add padding (in PDF units before zoom)
        pad = padding / zoom
        clip_rect = fitz.Rect(
            max(0, rect.x0 - pad),
            max(0, rect.y0 - pad),
            min(page.rect.width, rect.x1 + pad),
            min(page.rect.height, rect.y1 + pad),
        )

        # Highlight the matched text in yellow
        for inst in instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()

        pix = page.get_pixmap(matrix=matrix, clip=clip_rect)
    else:
        # Fall back to full page
        pix = page.get_pixmap(matrix=matrix)

    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


# ─── Arabic NLP Stack (MARBERT + AraBERT) ─────────────────────

@st.cache_resource
def load_nlp_stack():
    dialect_pipe = pipeline("text-classification", model="IbrahimAmin/marbertv2-arabic-written-dialect-classifier")
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
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", "---", "|"]
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
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    return ar_index, en_index, ar_llm, en_llm, reranker, dialect_pipe, ara_tokenizer