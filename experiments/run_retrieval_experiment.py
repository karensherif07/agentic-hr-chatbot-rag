# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIGS_TO_RUN = ["dense_only", "sparse_only", "hybrid"]
K_VALUES       = [1, 3, 5, 10]
PRIMARY_K      = 10
OUTPUT_XLSX    = "retrieval_results.xlsx"
EVAL_QUERY_FILE = "eval_queries.json"

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

# Flat path lists kept for PDF existence check at startup
ARABIC_PDF_PATHS  = [p for p, _ in ARABIC_PDF_ENTRIES]
ENGLISH_PDF_PATHS = [p for p, _ in ENGLISH_PDF_ENTRIES]
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from setup     import setup
from retrieval import retrieve, rrf
from nlp_utils import (
    normalize_arabic, normalize_english, tokenize,
    franco_to_arabic, egyptian_to_msa, get_semantic_dialect,
)
import pandas as pd
from tqdm import tqdm


# =============================================================================
# QUERY FILE
# =============================================================================

def load_queries() -> List[dict]:
    p = Path(EVAL_QUERY_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{EVAL_QUERY_FILE} not found.")
    return json.loads(p.read_text(encoding="utf-8"))


# =============================================================================
# RETRIEVAL CONFIGS
# =============================================================================

def dense_only(question, vs, bm25, docs, normalize_fn, k):
    normalized = normalize_fn(question)
    return vs.similarity_search("query: " + normalized, k=k)


def sparse_only(question, vs, bm25, docs, normalize_fn, k):
    normalized = normalize_fn(question)
    scores  = bm25.get_scores(tokenize(normalized))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [docs[i] for i in top_idx]


def hybrid(question, vs, bm25, docs, normalize_fn, k):
    return retrieve(question, vs, bm25, docs, normalize_fn, k=k)


RETRIEVAL_FNS = {
    "dense_only":  dense_only,
    "sparse_only": sparse_only,
    "hybrid":      hybrid,
}


# =============================================================================
# RETRIEVAL DISPATCH  (unchanged from your original)
# =============================================================================

def retrieve_for_query(fn, question, lang, ara_tokenizer, ar_index, en_index):
    ar_vs, ar_bm25, ar_docs = ar_index
    en_vs, en_bm25, en_docs = en_index

    norm_ar = lambda t: normalize_arabic(t, ara_tokenizer)

    if lang == "franco":
        ar_raw = franco_to_arabic(question)
        ar_msa = egyptian_to_msa(ar_raw)
        docs_ar = rrf(
            fn(ar_raw, ar_vs, ar_bm25, ar_docs, norm_ar, 12),
            fn(ar_msa, ar_vs, ar_bm25, ar_docs, norm_ar, 12),
        )
        docs_en = fn(question, en_vs, en_bm25, en_docs, normalize_english, 12)
        return rrf(docs_ar, docs_en)[:10]

    elif lang == "egyptian":
        ar_msa = egyptian_to_msa(question)
        docs_ar = rrf(
            fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, 12),
            fn(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar, 12),
        )
        docs_en = fn(question, en_vs, en_bm25, en_docs, normalize_english, 12)
        return rrf(docs_ar, docs_en)[:10]

    elif lang == "arabic":
        if get_semantic_dialect(question, ara_tokenizer) == "egyptian":
            ar_msa = egyptian_to_msa(question)
            docs_ar = rrf(
                fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, 12),
                fn(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar, 12),
            )
        else:
            docs_ar = fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, 12)
        docs_en = fn(question, en_vs, en_bm25, en_docs, normalize_english, 12)
        return rrf(docs_ar, docs_en)[:10]

    elif lang == "english":
        docs_en = fn(question, en_vs, en_bm25, en_docs, normalize_english, 12)
        docs_ar = fn(question, ar_vs, ar_bm25, ar_docs, norm_ar, 12)
        return rrf(docs_en, docs_ar)[:10]

    else:
        return fn(question, en_vs, en_bm25, en_docs, normalize_english, 10)


# =============================================================================
# RELEVANCE  — doc_name + page only, no language check
# =============================================================================
# A retrieved chunk is relevant when:
#   1. Its metadata["doc_name"]  matches ground_truth_doc   (exact filename)
#   2. Its metadata["page"]  + 1 is in ground_truth_pages   (PyMuPDF stores
#      0-based page index, eval JSON stores 1-based page numbers)
#
# ground_truth_lang is intentionally ignored: we evaluate whether the
# retriever found the right chunk, not which language index it came from.
# =============================================================================

def is_relevant(doc, ground_truth_doc: str, ground_truth_pages: List[int]) -> bool:
    """Return True if this chunk is from the expected doc AND page."""
    doc_name = doc.metadata.get("doc_name", "")
    page_0   = doc.metadata.get("page", -1)          # 0-based from PyMuPDF
    page_1   = page_0 + 1                             # convert to 1-based

    return doc_name == ground_truth_doc and page_1 in ground_truth_pages


def get_relevance_list(
    retrieved: list,
    ground_truth_doc: str,
    ground_truth_pages: List[int],
) -> List[int]:
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
    """
    Recall = (# unique relevant pages hit in top-k) / (# ground-truth pages).
    Capped at 1.0 to handle queries whose answer spans multiple chunks on
    the same page.
    """
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


def compute_all_metrics(
    retrieved: list,
    ground_truth_doc: str,
    ground_truth_pages: List[int],
) -> dict:
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
    print("\n🚀 Running Retrieval Experiment...\n")

    # setup() returns:
    # ar_index, en_index, routing_llm, en_llm, ar_llm, critique_llm,
    # reranker, dialect_pipe, ara_tokenizer
    (ar_index, en_index, *_, ara_tokenizer) = setup()

    queries = load_queries()
    rows    = []

    for cfg_name in CONFIGS_TO_RUN:
        fn = RETRIEVAL_FNS[cfg_name]
        print(f"\n── {cfg_name.upper()} ──")

        for q in tqdm(queries):
            gt_doc   = q.get("ground_truth_doc",   "")
            gt_pages = q.get("ground_truth_pages",  [])

            # ── guard: skip queries without ground truth ──────────────────
            if not gt_doc or not gt_pages:
                print(f"  [SKIP] {q['id']} — missing ground_truth_doc or ground_truth_pages")
                continue

            retrieved = retrieve_for_query(
                fn,
                q["question"],
                q["language"],
                ara_tokenizer,
                ar_index,
                en_index,
            )

            metrics = compute_all_metrics(retrieved, gt_doc, gt_pages)

            rows.append({
                "config":     cfg_name,
                "query_id":   q["id"],
                "language":   q["language"],
                "complexity": q.get("complexity", ""),
                "gt_doc":     gt_doc,
                **metrics,
            })

    # ── BUILD DATAFRAME ───────────────────────────────────────────────────────
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

    # ── PRINT: OVERALL ────────────────────────────────────────────────────────
    pivot_overall = df.groupby("config")[primary_cols].mean().round(4)
    print(f"\n{'='*70}")
    print(f"RESULTS  ({len(rows)} rows total)")
    print(f"{'='*70}\n")
    print(f"── OVERALL (primary K={PRIMARY_K}) ─────────────────────────────")
    print(pivot_overall.to_string(), "\n")

    # ── PRINT: FULL K SWEEP ───────────────────────────────────────────────────
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

    # ── PRINT: BY COMPLEXITY ──────────────────────────────────────────────────
    if "complexity" in df.columns:
        pivot_complexity = (
            df.groupby(["config", "complexity"])[primary_cols].mean().round(4)
        )
        print(f"\n── BY COMPLEXITY (K={PRIMARY_K}) ─────────────────────────────")
        print(pivot_complexity.to_string(), "\n")

    # ── PRINT: BY LANGUAGE ────────────────────────────────────────────────────
    if "language" in df.columns:
        pivot_lang = (
            df.groupby(["config", "language"])[primary_cols].mean().round(4)
        )
        print(f"── BY LANGUAGE (K={PRIMARY_K}) ───────────────────────────────")
        print(pivot_lang.to_string(), "\n")

    # ── PRINT: BY DOCUMENT ───────────────────────────────────────────────────
    if "gt_doc" in df.columns:
        pivot_doc = (
            df.groupby(["config", "gt_doc"])[primary_cols].mean().round(4)
        )
        print(f"── BY DOCUMENT (K={PRIMARY_K}) ───────────────────────────────")
        print(pivot_doc.to_string(), "\n")

    # ── SAVE EXCEL ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw", index=False)
        pivot_overall.to_excel(writer, sheet_name="Overall")

        if "complexity" in df.columns:
            pivot_complexity.to_excel(writer, sheet_name="By_Complexity")

        if "language" in df.columns:
            pivot_lang.to_excel(writer, sheet_name="By_Language")

        if "gt_doc" in df.columns:
            pivot_doc.to_excel(writer, sheet_name="By_Document")

        for k in K_VALUES:
            cols_k = [c for c in [f"precision@{k}", f"recall@{k}", f"hit@{k}", "mrr"]
                      if c in df.columns]
            df.groupby("config")[cols_k].mean().round(4).to_excel(
                writer, sheet_name=f"K={k}"
            )

    print(f"✓ Results saved to {OUTPUT_XLSX}")
    return df


if __name__ == "__main__":
    missing = [p for p in ARABIC_PDF_PATHS + ENGLISH_PDF_PATHS if not Path(p).exists()]
    if missing:
        print("[ERROR] Missing PDFs:", missing)
        exit(1)

    run_experiment()