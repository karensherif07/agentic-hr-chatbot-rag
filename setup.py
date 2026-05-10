import os
import re
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

# ── PDF catalogue ─────────────────────────────────────────────────────────────
# Each entry is (file_path, short_doc_name).
# short_doc_name MUST match the "ground_truth_doc" field in your eval JSON
# so the evaluator can compare retrieved source against the gold standard.

ARABIC_PDF_ENTRIES = [
    ("policies/ar_policy.pdf",          "ar_policy.pdf"),
    ("policies/ar_recruitment.pdf",     "ar_recruitment.pdf"),
    ("policies/ar_payroll_finance.pdf", "ar_payroll_finance.pdf"),
]

ENGLISH_PDF_ENTRIES = [
    ("policies/eng_policy.pdf",              "eng_policy.pdf"),
    ("policies/eng_wellness_benefits.pdf",   "eng_wellness_benefits.pdf"),
    ("policies/eng_training_development.pdf","eng_training_development.pdf"),
    ("policies/eng_workplace_conduct.pdf",   "eng_workplace_conduct.pdf"),
]

# Convenience: plain path lists (used by retrieval / utils for lang detection)
ARABIC_PDF_PATHS  = [p for p, _ in ARABIC_PDF_ENTRIES]
ENGLISH_PDF_PATHS = [p for p, _ in ENGLISH_PDF_ENTRIES]

# Sets for fast membership checks
ARABIC_PDF_PATH_SET  = set(ARABIC_PDF_PATHS)
ENGLISH_PDF_PATH_SET = set(ENGLISH_PDF_PATHS)

# Backward-compat single-value aliases
ARABIC_PDF_PATH  = ARABIC_PDF_PATHS[0]
ENGLISH_PDF_PATH = ENGLISH_PDF_PATHS[0]

_HIGHLIGHT_COLOR   = (1.0, 0.95, 0.0)
_HIGHLIGHT_OPACITY = 0.35


# ── Highlight helpers ─────────────────────────────────────────────────────────

def _search_candidates(clip_text: str) -> list[str]:
    t = re.sub(r"\[Page\s*\d+[^\]]*\]", "", clip_text or "", flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    candidates = []
    seen = set()

    def _add(s: str):
        s = s.strip()
        if s and s not in seen and len(s) >= 25:
            seen.add(s)
            candidates.append(s)

    sentences = re.split(r'(?<=[.!?؟])\s+|\n', t)
    content_sentences = [
        s.strip() for s in sentences
        if len(s.strip()) >= 40
        and not re.match(r'^[A-Z\s\d\-:،.]+$', s.strip())
    ]
    for sent in content_sentences[:5]:
        s = sent.strip()
        if 40 <= len(s) <= 200:
            _add(s)
        elif len(s) > 200:
            _add(s[:180])
    for n in (180, 150, 120, 90, 65, 45):
        if len(t) >= n:
            prefix = t[:n].strip()
            if len(prefix.split()) >= 5:
                _add(prefix)
    if len(t) > 90:
        mid = len(t) // 3
        for n in (120, 80, 50):
            chunk = t[mid: mid + n]
            if len(chunk) >= 25:
                _add(chunk)
    return candidates


def _rect_too_small(rect: fitz.Rect, page: fitz.Page) -> bool:
    pr = page.rect
    h = rect.y1 - rect.y0
    w = rect.x1 - rect.x0
    if h < pr.height * 0.04 and w < pr.width * 0.25:
        return True
    return (w * h) < (pr.width * pr.height * 0.008)


@st.cache_data
def render_page_to_image(pdf_path: str, page_num: int, zoom: float = 1.75) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


@st.cache_data
def render_page_highlighted(
    pdf_path: str, page_num: int, clip_text: str, zoom: float = 1.75
) -> bytes:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    matrix = fitz.Matrix(zoom, zoom)
    highlight_rects: list[fitz.Rect] = []
    for phrase in _search_candidates(clip_text):
        found = page.search_for(phrase, quads=False)
        if not found:
            continue
        union = found[0]
        for r in found[1:]:
            union = union | r
        if _rect_too_small(union, page):
            continue
        highlight_rects = found
        break
    if highlight_rects:
        shape = page.new_shape()
        for rect in highlight_rects:
            padded = fitz.Rect(
                rect.x0, max(0, rect.y0 - 2),
                rect.x1, min(page.rect.height, rect.y1 + 2),
            )
            shape.draw_rect(padded)
        shape.finish(
            fill=_HIGHLIGHT_COLOR, color=None, fill_opacity=_HIGHLIGHT_OPACITY
        )
        shape.commit()
    pix = page.get_pixmap(matrix=matrix)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


@st.cache_resource
def load_nlp_stack():
    dialect_pipe = pipeline(
        "text-classification",
        model="IbrahimAmin/marbertv2-arabic-written-dialect-classifier",
    )
    ara_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    return dialect_pipe, ara_tokenizer


# ── Index builder ─────────────────────────────────────────────────────────────

def _build_index(
    pdf_entries: list[tuple[str, str]],
    normalize_fn,
    lang_tag: str,
    emb,
):
    """
    Load and index a list of (path, doc_name) PDF entries into a single
    merged FAISS + BM25 index.

    Each chunk's metadata carries:
      • "source"   – the full file path (set by PyMuPDFLoader)
      • "doc_name" – the short filename (e.g. "ar_policy.pdf").
                     This is what the evaluator compares against
                     ground_truth_doc in the eval JSON.
      • "lang"     – "arabic" or "english"
      • "page"     – 1-based page number (set by PyMuPDFLoader)

    WHY doc_name?
    PyMuPDFLoader sets metadata["source"] to the full path
    (e.g. "policies/ar_policy.pdf").  Your eval JSON stores only the
    filename ("ar_policy.pdf").  Storing doc_name gives retrieval and
    evaluation code a single reliable key to compare without fragile
    path-stripping logic.
    """
    all_docs = []

    for path, doc_name in pdf_entries:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing policy PDF: {path}")

        pages = PyMuPDFLoader(path).load()

        for d in pages:
            d.page_content = clean_pdf(d.page_content)
            d.metadata["doc_type"] = "policy"
            d.metadata["lang"]     = lang_tag
            # ↓ NEW: short doc identifier used by evaluation & citation UI
            d.metadata["doc_name"] = doc_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "?", "!", " ", "---", "|"],
        )
        all_docs.extend(splitter.split_documents(pages))

    # Prepend passage prefix required by multilingual-e5
    for d in all_docs:
        d.page_content = "passage: " + d.page_content

    bm25_corpus = [tokenize(normalize_fn(d.page_content)) for d in all_docs]
    vs   = FAISS.from_documents(all_docs, emb)
    bm25 = BM25Okapi(bm25_corpus)
    return vs, bm25, all_docs


# ── Main setup ────────────────────────────────────────────────────────────────

def setup():
    """
    Returns:
        ar_index  – (FAISS, BM25, docs) for Arabic PDFs
        en_index  – (FAISS, BM25, docs) for English PDFs
        routing_llm   – tool-calling orchestrator (llama-3.3-70b)
        en_llm        – English answer generation  (llama-3.3-70b)
        ar_llm        – Arabic / Franco generation  (qwen3-32b, no thinking)
        critique_llm  – self-critique               (llama-3.1-8b-instant)
        reranker, dialect_pipe, ara_tokenizer
    """
    dialect_pipe, ara_tokenizer = load_nlp_stack()

    emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    ar_index = _build_index(
        ARABIC_PDF_ENTRIES,
        lambda t: normalize_arabic(t, ara_tokenizer),
        lang_tag="arabic",
        emb=emb,
    )
    en_index = _build_index(
        ENGLISH_PDF_ENTRIES,
        normalize_english,
        lang_tag="english",
        emb=emb,
    )

    # ── LLM roles ─────────────────────────────────────────────────────────────
    routing_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
    )
    en_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )
    ar_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        temperature=0,
    )
    critique_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    return (
        ar_index, en_index,
        routing_llm, en_llm, ar_llm, critique_llm,
        reranker, dialect_pipe, ara_tokenizer,
    )


