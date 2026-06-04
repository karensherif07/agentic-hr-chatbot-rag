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
    ("policies/eng_policy.pdf",               "eng_policy.pdf"),
    ("policies/eng_wellness_benefits.pdf",    "eng_wellness_benefits.pdf"),
    ("policies/eng_training_development.pdf", "eng_training_development.pdf"),
    ("policies/eng_workplace_conduct.pdf",    "eng_workplace_conduct.pdf"),
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


# ── Safe cleaning helpers ─────────────────────────────────────────────────────
# Rules:
#   hardcore_clean  — encoding artifacts only (null bytes, bidi marks). Safe always.
#   safe_clean      — true boilerplate footers only. Never touches policy content.
#
# REMOVED from previous version (caused 52 false "not available" answers):
#   aggressive_clean   — deleted lines containing "policy", "Horizon", "HR" which
#                        are present in real policy sentences, destroying index quality.
#   clean_index_text   — "Page \d+" regex wiped page-number citations that are part
#                        of real table rows (e.g. "Page 9 — 21 days annual leave").
#   is_bad_chunk       — word threshold of 35 was too high, dropping valid short facts.
#   is_header_chunk    — word threshold of 30 and "page" in first 50 chars dropped
#                        real chunks whose first word happened to be a page reference.

def hardcore_clean(text: str) -> str:
    """Remove encoding artifacts only. Never removes policy words."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", text)  # bidi control chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_clean(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Strip leading punctuation that PDFs prepend (e.g. ". Confidential...")
        stripped_for_check = stripped.lstrip('. ')
        if re.fullmatch(
            r"(Confidential\s*[—\-]\s*Internal Use Only.*"
            r"|Version\s+\d+\.\d+.*"
            r"|Page\s*\d+"
            r"|سري\s*[—\-]\s*للاستخدام الداخلي فقط.*"
            r"|Horizon Tech\s*[—\-]\s*Human Resources.*)",
            stripped_for_check,
            flags=re.IGNORECASE,
        ):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"[-|_]{5,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_boilerplate_chunk(text: str) -> bool:
    """
    True only when the entire chunk is footer / header stamp with no policy content.

    Thresholds:
      < 12 words  → too short to contain a policy fact
      digit ratio > 0.45 → pure table of numbers with no prose (OCR artefact)

    NOT filtered:
      - chunks containing "policy", "Horizon", "HR" (real content words)
      - chunks with "page" anywhere except at the very start of a short line
      - chunks with word count 12-30 (short but may be a valid single-fact chunk)
    """
    t = text.strip()
    if "\x00" in t:
        return True
    words = t.split()
    if len(words) < 12:
        return True
    digit_ratio = sum(c.isdigit() for c in t) / max(len(t), 1)
    if digit_ratio > 0.45:
        return True
    # Strip leading ". " before checking
    t_check = t.lstrip('passage: ').lstrip('. ')
    if re.fullmatch(
        r"(Confidential\s*[—\-]\s*Internal Use Only.*"
        r"|Version\s+\d+\.\d+.*"
        r"|Page\s*\d+"
        r"|سري\s*[—\-]\s*للاستخدام الداخلي فقط.*)",
        t_check,
        flags=re.IGNORECASE,
    ):
        return True
    return False


# ── Highlight helpers (used by Streamlit UI) ──────────────────────────────────

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

    Cleaning pipeline (conservative by design):
      1. hardcore_clean  — encoding artifacts only
      2. clean_pdf       — from nlp_utils: normalises whitespace, line endings
      3. safe_clean      — removes footer-only lines, never touches policy words
      4. _is_boilerplate_chunk — drops chunks that are entirely footer/garbage
    """
    all_docs = []

    for path, doc_name in pdf_entries:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing policy PDF: {path}")

        raw_pages = PyMuPDFLoader(path).load()
        pages = []

        for d in raw_pages:
            text = d.page_content

            # Step 1 — encoding artifacts
            text = hardcore_clean(text)
            # Step 2 — whitespace / line-ending normalisation (from nlp_utils)
            text = clean_pdf(text)
            # Step 3 — footer-stamp lines only
            text = safe_clean(text)

            # Drop page if nothing meaningful survived
            if len(text.split()) < 10:
                continue

            d.page_content = text
            d.metadata["doc_type"] = "policy"
            d.metadata["lang"]     = lang_tag
            d.metadata["doc_name"] = doc_name

            pages.append(d)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "?", "!", " ", "---", "|"],
        )
        chunks = splitter.split_documents(pages)

        # Drop chunks that are entirely boilerplate / too short / OCR garbage
        chunks = [c for c in chunks if not _is_boilerplate_chunk(c.page_content)]

        all_docs.extend(chunks)

    # Prepend passage prefix required by multilingual-e5
    for d in all_docs:
        d.page_content = "passage: " + d.page_content

    bm25_corpus = [tokenize(normalize_fn(d.page_content)) for d in all_docs]
    vs   = FAISS.from_documents(all_docs, emb)
    bm25 = BM25Okapi(bm25_corpus)
    return vs, bm25, all_docs


# ── Main setup ────────────────────────────────────────────────────────────────
@st.cache_resource
def setup():
    """
    Returns:
        ar_index      – (FAISS, BM25, docs) for Arabic PDFs
        en_index      – (FAISS, BM25, docs) for English PDFs
        routing_llm   – intent router          (llama-4-scout-17b)
        en_llm        – English answer gen     (llama-3.3-70b-versatile)
        ar_llm        – Arabic / Franco gen    (qwen3-32b)
        critique_llm  – self-critique          (llama-3.1-8b-instant)
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