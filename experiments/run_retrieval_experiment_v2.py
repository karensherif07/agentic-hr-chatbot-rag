# =============================================================================
# run_retrieval_experiment_v2.py
#
# Drop-in replacement for run_retrieval_experiment.py that adds four new
# hybrid configs ON TOP OF the original three, so you get a direct comparison
# in a single run.
#
# NEW CONFIGS
# ───────────
#   hybrid_w02   – weighted RRF, BM25 weight = 0.2  (conservative)
#   hybrid_w03   – weighted RRF, BM25 weight = 0.3  (recommended starting point)
#   hybrid_w05   – weighted RRF, BM25 weight = 0.5  (halfway)
#   hybrid_skip_ar – dense-only for Arabic/Egyptian/Franco queries,
#                    standard equal-weight hybrid for English only
#
# Everything else (setup, metrics, language dispatch, chunk filters) is
# identical to the original file — no imports from retrieval.py were changed.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit here if you want a subset
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS_TO_RUN = [
    "dense_only",
    "sparse_only",
    "hybrid",           # original equal-weight, kept as baseline
    "hybrid_w02",       # NEW: BM25 weight 0.2
    "hybrid_w03",       # NEW: BM25 weight 0.3
    "hybrid_w05",       # NEW: BM25 weight 0.5
    "hybrid_skip_ar",   # NEW: skip BM25 entirely for AR/EGY/franco
]

K_VALUES        = [1, 3, 5, 7, 10, 20, 40]
PRIMARY_K       = 20
OUTPUT_XLSX     = "retrieval_results_v2.xlsx"
EVAL_QUERY_FILE = "eval_queries.json"
_CANDIDATE_K    = 40

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
ARABIC_PDF_PATHS  = [p for p, _ in ARABIC_PDF_ENTRIES]
ENGLISH_PDF_PATHS = [p for p, _ in ENGLISH_PDF_ENTRIES]
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from setup     import setup
from retrieval import rrf
from nlp_utils import (
    normalize_arabic, normalize_english, tokenize,
    franco_to_arabic, egyptian_to_msa, get_semantic_dialect,
    detect_language_type,
)
import pandas as pd
from tqdm import tqdm


# =============================================================================
# CHUNK QUALITY FILTERS  (verbatim from original)
# =============================================================================

_BOILERPLATE_PATTERNS = [
    r"^Confidential\s*[—-]\s*Internal Use Only",
    r"^سري\s*[—-]\s*للاستخدام الداخلي فقط",
    r"^Mbps\s*\.",
    r"^Page\s*\d+\s*$",
    r"تاريخ النفاذ\s*سري",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.DOTALL)

_GARBAGE_PATTERNS = [
    r"[^\w\s\u0600-\u06FF]{15,}",
    r"(.)\1{10,}",
    r"\b[a-zA-Z]{1,2}\b(?:\s+\b[a-zA-Z]{1,2}\b){10,}",
]
_GARBAGE_RE = re.compile("|".join(_GARBAGE_PATTERNS), re.DOTALL)


def _is_boilerplate(text: str) -> bool:
    stripped = text.strip()
    stripped_check = re.sub(r'^[.\s]+', '', stripped)
    if len(stripped_check) < 50:
        return True
    if _BOILERPLATE_RE.search(stripped_check[:300]):
        return True
    _SIRI_ONLY = re.compile(
        r'^[\.\s]*سري\s*[—\-–]\s*للاستخدام الداخلي فقط',
        re.IGNORECASE
    )
    if _SIRI_ONLY.match(stripped):
        return True
    if 'سري' in stripped and 'للاستخدام الداخلي فقط' in stripped:
        _POLICY_WORDS = re.compile(
            r'(إجازة|راتب|موظف|بدل|تأمين|اشتراك|مكافأة|عمل إضافي|فترة|اختبار|تقييم|شهادة|تدريب|نفقة|مصروف)'
        )
        if not _POLICY_WORDS.search(stripped):
            return True
    _EN_STAMP = re.compile(
        r'^[\.\s]*(Confidential\s*[—\-–]\s*Internal Use Only|'
        r'Horizon Tech\s*[—\-–]\s*Human Resources)',
        re.IGNORECASE
    )
    if _EN_STAMP.match(stripped) and len(stripped) < 300:
        return True
    return False


def _looks_like_ocr_loop(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    repeated = sum(1 for i in range(len(lines) - 1) if lines[i] == lines[i + 1])
    if repeated >= 2:
        return True
    numeric_ratio = sum(
        1 for tok in text.split() if any(c.isdigit() for c in tok)
    ) / max(len(text.split()), 1)
    return numeric_ratio > 0.35


def _is_corrupted_chunk(text: str) -> bool:
    t = text.strip()
    if len(t) < 80:
        return True
    weird_ratio = sum(
        1 for c in t
        if not (c.isalnum() or c.isspace() or c in ".,:;!?-%()/")
    ) / max(len(t), 1)
    if weird_ratio > 0.30:
        return True
    if _GARBAGE_RE.search(t):
        return True
    words = t.split()
    if len(words) < 15 and len(t) > 300:
        return True
    return False


def _filter_bad_chunks(docs: list) -> list:
    return [
        d for d in docs
        if not (_is_boilerplate(d.page_content)
                or _is_corrupted_chunk(d.page_content)
                or _looks_like_ocr_loop(d.page_content))
    ]


# =============================================================================
# QUERY FILE
# =============================================================================

def load_queries() -> List[dict]:
    p = Path(EVAL_QUERY_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{EVAL_QUERY_FILE} not found.")
    return json.loads(p.read_text(encoding="utf-8"))


# =============================================================================
# RETRIEVAL PRIMITIVES
# =============================================================================

def dense_only(question, vs, bm25, docs, normalize_fn, k):
    """Pure semantic retrieval — BM25 never called."""
    normalized = normalize_fn(question)
    return vs.similarity_search("query: " + normalized, k=k)


def sparse_only(question, vs, bm25, docs, normalize_fn, k):
    """Pure BM25 retrieval — vector store never called."""
    normalized = normalize_fn(question)
    scores  = bm25.get_scores(tokenize(normalized))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_idx]


def hybrid(question, vs, bm25, docs, normalize_fn, k):
    """Original equal-weight hybrid (unchanged baseline)."""
    normalized = normalize_fn(question)
    dense_docs = vs.similarity_search("query: " + normalized, k=k)
    scores     = bm25.get_scores(tokenize(normalized))
    top_idx    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs  = [docs[i] for i in top_idx]
    return rrf(dense_docs, bm25_docs, k=20)


# ── NEW: weighted RRF helpers ─────────────────────────────────────────────────

def _rrf_weighted(docs1: list, docs2: list, rrf_k: int = 60,
                  w1: float = 1.0, w2: float = 1.0) -> list:
    """
    Reciprocal Rank Fusion with per-list weights.
    Identical dedup key to the original rrf() in retrieval.py.
    docs1 = dense (w1), docs2 = BM25 (w2).
    """
    scores  = {}
    doc_map = {}

    def add(docs, weight):
        for rank, d in enumerate(docs):
            source = d.metadata.get("source", "")
            page   = d.metadata.get("page", 0)
            key    = f"{source}:page_{page}:{d.page_content[:100]}"
            if key not in scores:
                scores[key]  = 0
                doc_map[key] = d
            scores[key] += weight / (rrf_k + rank + 1)

    add(docs1, w1)
    add(docs2, w2)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def _make_weighted_hybrid(bm25_weight: float):
    """Factory — returns a retrieval primitive with the given BM25 weight."""
    def _fn(question, vs, bm25, docs, normalize_fn, k):
        normalized = normalize_fn(question)
        dense_docs = vs.similarity_search("query: " + normalized, k=k)
        bm25_scores = bm25.get_scores(tokenize(normalized))
        top_idx     = sorted(range(len(bm25_scores)),
                             key=lambda i: bm25_scores[i], reverse=True)[:k]
        bm25_docs   = [docs[i] for i in top_idx]
        return _rrf_weighted(dense_docs, bm25_docs, rrf_k=20,
                             w1=1.0, w2=bm25_weight)
    _fn.__name__ = f"hybrid_w{int(bm25_weight*10):02d}"
    return _fn


# ── NEW: language-aware skip primitive ───────────────────────────────────────

def _is_arabic_script(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def hybrid_skip_ar(question, vs, bm25, docs, normalize_fn, k):
    """
    Dense-only for Arabic-script, Franco, and Egyptian queries.
    Standard equal-weight hybrid for English-only queries.
    Rationale: BM25 contributes near-zero signal on Arabic/Franco (see results),
    so skipping it avoids diluting the strong dense ranking.
    """
    from nlp_utils import detect_language_type as _detect
    q_lang = _detect(question)

    normalized = normalize_fn(question)
    dense_docs = vs.similarity_search("query: " + normalized, k=k)

    if q_lang in ("arabic", "egyptian", "franco") or _is_arabic_script(question):
        # BM25 is noise for Arabic — skip it entirely
        return dense_docs

    # English: standard hybrid
    bm25_scores = bm25.get_scores(tokenize(normalized))
    top_idx     = sorted(range(len(bm25_scores)),
                         key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs   = [docs[i] for i in top_idx]
    return rrf(dense_docs, bm25_docs, k=20)


# ── Register all primitives ───────────────────────────────────────────────────

RETRIEVAL_FNS = {
    "dense_only":    dense_only,
    "sparse_only":   sparse_only,
    "hybrid":        hybrid,
    "hybrid_w02":    _make_weighted_hybrid(0.2),
    "hybrid_w03":    _make_weighted_hybrid(0.3),
    "hybrid_w05":    _make_weighted_hybrid(0.5),
    "hybrid_skip_ar": hybrid_skip_ar,
}


# =============================================================================
# LANGUAGE-AWARE RETRIEVAL DISPATCH  (verbatim from original)
# =============================================================================

def retrieve_for_query(fn, question, ara_tokenizer, dialect_pipe, ar_index, en_index):
    from deep_translator import GoogleTranslator

    ar_vs, ar_bm25, ar_docs = ar_index
    en_vs, en_bm25, en_docs = en_index

    norm_ar = lambda t: normalize_arabic(t, ara_tokenizer)
    q_lang  = detect_language_type(question)

    if q_lang == "franco":
        franco_ar  = franco_to_arabic(question)
        msa_query  = egyptian_to_msa(franco_ar)
        docs_ar = rrf(
            fn(franco_ar, ar_vs, ar_bm25, ar_docs, norm_ar,          _CANDIDATE_K),
            fn(msa_query, ar_vs, ar_bm25, ar_docs, norm_ar,          _CANDIDATE_K),
        )
        try:
            en_query2 = GoogleTranslator(source='ar', target='en').translate(franco_ar)
        except Exception:
            en_query2 = question
        docs_en = rrf(
            fn(en_query2,  en_vs, en_bm25, en_docs, normalize_english, _CANDIDATE_K),
            fn(msa_query,  en_vs, en_bm25, en_docs, normalize_english, _CANDIDATE_K),
        )
        combined = rrf(rrf(docs_ar, docs_ar), docs_en)

    elif q_lang == "egyptian":
        ar_msa  = egyptian_to_msa(question)
        docs_ar = rrf(
            fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, _CANDIDATE_K),
            fn(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar, _CANDIDATE_K),
        )
        docs_en = fn(question, en_vs, en_bm25, en_docs, normalize_english, _CANDIDATE_K)
        combined = rrf(rrf(docs_ar, docs_ar), docs_en)

    elif q_lang == "arabic":
        if get_semantic_dialect(question, dialect_pipe) == "egyptian":
            ar_msa  = egyptian_to_msa(question)
            docs_ar = rrf(
                fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, _CANDIDATE_K),
                fn(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar, _CANDIDATE_K),
            )
        else:
            docs_ar = fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, _CANDIDATE_K)
        docs_en = fn(question, en_vs, en_bm25, en_docs, normalize_english, _CANDIDATE_K)
        combined = rrf(rrf(docs_ar, docs_ar), docs_en)

    elif q_lang == "english":
        docs_en  = fn(question, en_vs, en_bm25, en_docs, normalize_english, _CANDIDATE_K)
        docs_ar  = fn(question, ar_vs, ar_bm25, ar_docs, norm_ar,           _CANDIDATE_K)
        combined = rrf(rrf(docs_en, docs_en), docs_ar)

    else:
        combined = fn(question, en_vs, en_bm25, en_docs, normalize_english, _CANDIDATE_K)

    return _filter_bad_chunks(combined)


# =============================================================================
# RELEVANCE
# =============================================================================

def is_relevant(doc, ground_truth_doc: str, ground_truth_pages: List[int]) -> bool:
    doc_name = doc.metadata.get("doc_name", "")
    page_0   = doc.metadata.get("page", -1)
    page_1   = page_0 + 1
    return doc_name == ground_truth_doc and page_1 in ground_truth_pages


def get_relevance_list(retrieved, ground_truth_doc, ground_truth_pages) -> List[int]:
    return [
        1 if is_relevant(d, ground_truth_doc, ground_truth_pages) else 0
        for d in retrieved
    ]


# =============================================================================
# METRICS
# =============================================================================

def precision_at_k(rel: List[int], k: int) -> float:
    return sum(rel[:k]) / k if k else 0.0


def recall_at_k(rel: List[int], k: int, gt_pages: List[int]) -> float:
    if not gt_pages:
        return 1.0
    return min(sum(rel[:k]), len(gt_pages)) / len(gt_pages)


def hit_at_k(rel: List[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0


def mrr(rel: List[int]) -> float:
    for i, r in enumerate(rel):
        if r == 1:
            return 1 / (i + 1)
    return 0.0


def compute_all_metrics(retrieved, ground_truth_doc, ground_truth_pages) -> dict:
    rel = get_relevance_list(retrieved, ground_truth_doc, ground_truth_pages)
    out = {"mrr": round(mrr(rel), 4)}
    for k in K_VALUES:
        out[f"precision@{k}"] = round(precision_at_k(rel, k), 4)
        out[f"recall@{k}"]    = round(recall_at_k(rel, k, ground_truth_pages), 4)
        out[f"hit@{k}"]       = round(hit_at_k(rel, k), 4)
    return out


# =============================================================================
# MAIN
# =============================================================================

def run_experiment():
    print("\n🚀 Running Retrieval Experiment v2 (weighted RRF + skip-AR variants)...\n")
    print("Configs:", CONFIGS_TO_RUN, "\n")

    (ar_index, en_index, _routing_llm, _en_llm, _ar_llm, _critique_llm,
     _reranker, dialect_pipe, ara_tokenizer) = setup()

    queries = load_queries()
    rows    = []

    for cfg_name in CONFIGS_TO_RUN:
        fn = RETRIEVAL_FNS[cfg_name]
        print(f"\n── {cfg_name.upper()} ──")

        for q in tqdm(queries):
            gt_doc   = q.get("ground_truth_doc",  "")
            gt_pages = q.get("ground_truth_pages", [])

            if not gt_doc or not gt_pages:
                print(f"  [SKIP] {q['id']} — missing ground_truth_doc or ground_truth_pages")
                continue

            retrieved = retrieve_for_query(
                fn,
                q["question"],
                ara_tokenizer,
                dialect_pipe,
                ar_index,
                en_index,
            )

            metrics = compute_all_metrics(retrieved, gt_doc, gt_pages)

            rows.append({
                "config":        cfg_name,
                "query_id":      q["id"],
                "language":      q["language"],
                "complexity":    q.get("complexity", ""),
                "gt_doc":        gt_doc,
                "docs_returned": len(retrieved),
                **metrics,
            })

    df = pd.DataFrame(rows)

    primary_cols = [
        f"precision@{PRIMARY_K}",
        f"recall@{PRIMARY_K}",
        f"hit@{PRIMARY_K}",
        "mrr",
    ]
    all_metric_cols = ["mrr"] + [
        f"{m}@{k}"
        for k in K_VALUES
        for m in ["precision", "recall", "hit"]
    ]

    # ── OVERALL ───────────────────────────────────────────────────────────────
    pivot_overall = df.groupby("config")[primary_cols].mean().round(4)
    print(f"\n{'='*70}")
    print(f"RESULTS v2  ({len(rows)} rows total)")
    print(f"{'='*70}\n")
    print(f"── OVERALL (primary K={PRIMARY_K}) ─────────────────────────────")
    print(pivot_overall.to_string(), "\n")

    # ── FULL K SWEEP ──────────────────────────────────────────────────────────
    config_means = df.groupby("config")[all_metric_cols].mean().round(4)
    print("── FULL K SWEEP ─────────────────────────────────────────────────")
    for cfg in CONFIGS_TO_RUN:
        if cfg not in config_means.index:
            continue
        sub = config_means.loc[cfg]
        print(f"\n  {cfg.upper()}")
        for k in K_VALUES:
            print(
                f"    K={k:2d}  "
                f"Precision={sub.get(f'precision@{k}', 0):.4f}  "
                f"Recall={sub.get(f'recall@{k}', 0):.4f}  "
                f"Hit={sub.get(f'hit@{k}', 0):.4f}  "
                f"MRR={sub.get('mrr', 0):.4f}"
            )

    k7_cols = [c for c in ["precision@7", "recall@7", "hit@7", "mrr"] if c in df.columns]

    # ── BY LANGUAGE ───────────────────────────────────────────────────────────
    if "language" in df.columns:
        pivot_lang = df.groupby(["config", "language"])[primary_cols].mean().round(4)
        print(f"\n── BY LANGUAGE (K={PRIMARY_K}) ───────────────────────────────")
        print(pivot_lang.to_string(), "\n")

        pivot_lang_k7 = df.groupby(["config", "language"])[k7_cols].mean().round(4)
        print(f"── BY LANGUAGE (K=7, reranker-comparable) ────────────────────")
        print(pivot_lang_k7.to_string(), "\n")

    # ── BY DOCUMENT ───────────────────────────────────────────────────────────
    if "gt_doc" in df.columns:
        pivot_doc = df.groupby(["config", "gt_doc"])[primary_cols].mean().round(4)
        print(f"── BY DOCUMENT (K={PRIMARY_K}) ───────────────────────────────")
        print(pivot_doc.to_string(), "\n")

    # ── SAVE EXCEL ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw", index=False)
        pivot_overall.to_excel(writer, sheet_name="Overall")

        if "language" in df.columns:
            pivot_lang.to_excel(writer, sheet_name="By_Language")
            pivot_lang_k7.to_excel(writer, sheet_name="By_Language_K7")

        if "gt_doc" in df.columns:
            pivot_doc.to_excel(writer, sheet_name="By_Document")

        for k in K_VALUES:
            cols_k = [c for c in [f"precision@{k}", f"recall@{k}", f"hit@{k}", "mrr"]
                      if c in df.columns]
            df.groupby("config")[cols_k].mean().round(4).to_excel(
                writer, sheet_name=f"K={k}"
            )

    print(f"\n✓ Results saved to {OUTPUT_XLSX}")
    return df


if __name__ == "__main__":
    missing = [p for p in ARABIC_PDF_PATHS + ENGLISH_PDF_PATHS if not Path(p).exists()]
    if missing:
        print("[ERROR] Missing PDFs:", missing)
        exit(1)

    run_experiment()