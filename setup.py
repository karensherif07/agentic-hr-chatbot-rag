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

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"

_HIGHLIGHT_COLOR   = (1.0, 0.95, 0.0)
_HIGHLIGHT_OPACITY = 0.35


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
def render_page_highlighted(pdf_path: str, page_num: int, clip_text: str, zoom: float = 1.75) -> bytes:
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
        shape.finish(fill=_HIGHLIGHT_COLOR, color=None, fill_opacity=_HIGHLIGHT_OPACITY)
        shape.commit()
    pix = page.get_pixmap(matrix=matrix)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


@st.cache_resource
def load_nlp_stack():
    dialect_pipe  = pipeline(
        "text-classification",
        model="IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
    )
    ara_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
    return dialect_pipe, ara_tokenizer


@st.cache_resource
def setup():
    """
    Returns:
        ar_index, en_index,
        routing_llm,   — tool-calling orchestrator (llama-3.3-70b, small prompt)
        en_llm,        — English answer generation (llama-3.3-70b)
        ar_llm,        — Arabic / Franco answer generation (qwen3-32b, no thinking)
        critique_llm,  — self-critique (llama-3.1-8b-instant, 131k ctx, cheap)
        reranker, dialect_pipe, ara_tokenizer
    """
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
            chunk_size=1000, chunk_overlap=300,
            separators=["\n\n", "\n", ".", "?", "!", " ", "---", "|"]
        )
        docs = splitter.split_documents(pages)
        bm25_corpus = [tokenize(normalize_fn(d.page_content)) for d in docs]
        vs   = FAISS.from_documents(docs, emb)
        bm25 = BM25Okapi(bm25_corpus)
        return vs, bm25, docs

    ar_index = build_index(ARABIC_PDF_PATH,  lambda t: normalize_arabic(t, ara_tokenizer))
    en_index = build_index(ENGLISH_PDF_PATH, normalize_english)

    # ── LLM roles ────────────────────────────────────────────────────────────
    # routing_llm: orchestrator tool-calling loop — prompt is small (no context),
    #              needs tool-calling support → llama-3.3-70b
    routing_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

    # en_llm: English policy + hybrid answer generation — large context needed
    en_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
    )

    # ar_llm: Arabic / Egyptian / Franco answer generation
    # qwen3-32b with thinking DISABLED via model_kwargs to prevent verbose output
    ar_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3-32b",
        temperature=0,
    )

    # critique_llm: self-critique — tiny prompt, fast, 131k token limit on free tier
    critique_llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

    reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    return (
        ar_index, en_index,
        routing_llm, en_llm, ar_llm, critique_llm,
        reranker, dialect_pipe, ara_tokenizer,
    )