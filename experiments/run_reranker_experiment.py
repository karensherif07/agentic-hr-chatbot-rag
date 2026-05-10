import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv

from setup import setup
from retrieval import retrieve, rrf, rerank
from nlp_utils import (
    normalize_arabic, normalize_english,
    tokenize, franco_to_arabic,
    egyptian_to_msa, get_semantic_dialect,
)

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
K_VALUES          = [1, 3, 5, 10]
PRIMARY_K         = 10
CANDIDATE_K       = 20
RERANKED_TOP_N    = 5
OUTPUT_XLSX       = "reranker_results.xlsx"
EVAL_QUERY_FILE   = "eval_queries.json"

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

# baseline (exp 1, hybrid) — update these once you have new-system numbers
EXP1_HYBRID = {
    "precision@1":  0.4600,
    "recall@1":     0.4600,
    "hit@1":        0.4600,

    "precision@3":  0.3300,
    "recall@3":     0.6750,
    "hit@3":        0.6800,

    "precision@5":  0.2780,
    "recall@5":     0.8200,
    "hit@5":        0.8300,

    "precision@10": 0.2050,
    "recall@10":    0.9000,
    "hit@10":       0.9000,

    "mrr":          0.6094,
}

# ─────────────────────────────────────────────


def load_queries() -> List[dict]:
    p = Path(EVAL_QUERY_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{EVAL_QUERY_FILE} not found.")
    return json.loads(p.read_text(encoding="utf-8"))


# =============================================================================
# RELEVANCE  — doc_name + page, consistent with run_retrieval_experiment.py
# =============================================================================

def is_relevant(doc, ground_truth_doc: str, ground_truth_pages: List[int]) -> bool:
    """Return True if this chunk is from the expected doc AND page."""
    doc_name = doc.metadata.get("doc_name", "")
    page_1   = doc.metadata.get("page", -1) + 1   # PyMuPDF stores 0-based index
    return doc_name == ground_truth_doc and page_1 in ground_truth_pages


def compute_metrics(retrieved: list, gt_doc: str, gt_pages: List[int]) -> dict:
    relevance = [1 if is_relevant(d, gt_doc, gt_pages) else 0 for d in retrieved]

    res = {"mrr": 0.0}
    for i, r in enumerate(relevance):
        if r == 1:
            res["mrr"] = round(1 / (i + 1), 4)
            break

    for k in K_VALUES:
        rel_k = relevance[:k]
        res[f"precision@{k}"] = round(sum(rel_k) / k, 4)
        res[f"recall@{k}"]    = round(
            (min(sum(rel_k), len(gt_pages)) / len(gt_pages)) if gt_pages else 1.0, 4
        )
        res[f"hit@{k}"]       = round(1.0 if any(rel_k) else 0.0, 4)

    return res


# =============================================================================
# RETRIEVAL DISPATCH  — mirrors run_retrieval_experiment.py exactly
# =============================================================================

def retrieve_for_query(
    question: str,
    lang: str,
    ara_tokenizer,
    ar_index,
    en_index,
    candidate_k: int,
) -> list:
    ar_vs, ar_bm25, ar_docs = ar_index
    en_vs, en_bm25, en_docs = en_index

    norm_ar = lambda t: normalize_arabic(t, ara_tokenizer)

    if lang == "franco":
        ar_raw  = franco_to_arabic(question)
        ar_msa  = egyptian_to_msa(ar_raw)
        docs_ar = rrf(
            retrieve(ar_raw, ar_vs, ar_bm25, ar_docs, norm_ar,         k=candidate_k),
            retrieve(ar_msa, ar_vs, ar_bm25, ar_docs, norm_ar,         k=candidate_k),
        )
        docs_en = retrieve(question, en_vs, en_bm25, en_docs, normalize_english, k=candidate_k)
        return rrf(docs_ar, docs_en)

    elif lang == "egyptian":
        ar_msa  = egyptian_to_msa(question)
        docs_ar = rrf(
            retrieve(question, ar_vs, ar_bm25, ar_docs, norm_ar, k=candidate_k),
            retrieve(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar, k=candidate_k),
        )
        docs_en = retrieve(question, en_vs, en_bm25, en_docs, normalize_english, k=candidate_k)
        return rrf(docs_ar, docs_en)

    elif lang == "arabic":
        if get_semantic_dialect(question, ara_tokenizer) == "egyptian":
            ar_msa  = egyptian_to_msa(question)
            docs_ar = rrf(
                retrieve(question, ar_vs, ar_bm25, ar_docs, norm_ar, k=candidate_k),
                retrieve(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar, k=candidate_k),
            )
        else:
            docs_ar = retrieve(question, ar_vs, ar_bm25, ar_docs, norm_ar, k=candidate_k)
        docs_en = retrieve(question, en_vs, en_bm25, en_docs, normalize_english, k=candidate_k)
        return rrf(docs_ar, docs_en)

    elif lang == "english":
        docs_en = retrieve(question, en_vs, en_bm25, en_docs, normalize_english, k=candidate_k)
        docs_ar = retrieve(question, ar_vs, ar_bm25, ar_docs, norm_ar,          k=candidate_k)
        return rrf(docs_en, docs_ar)

    else:
        return retrieve(question, en_vs, en_bm25, en_docs, normalize_english, k=candidate_k)


# =============================================================================
# GROUPBY HELPER
# =============================================================================

def group_by(df: pd.DataFrame, key: str, k: int = 10) -> pd.DataFrame:
    cols = [f"precision@{k}", f"recall@{k}", f"hit@{k}", "mrr"]
    return df.groupby(key)[cols].mean().round(4)


# =============================================================================
# MAIN
# =============================================================================

def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 2: RERANKER IMPACT")
    print("=" * 80)

    (ar_index, en_index, _, _, _, _, reranker, _, ara_tokenizer) = setup()
    queries = load_queries()

    rows = []

    for q in tqdm(queries, desc="Reranking"):
        gt_doc   = q.get("ground_truth_doc",  "")
        gt_pages = q.get("ground_truth_pages", [])

        if not gt_doc or not gt_pages:
            print(f"  [SKIP] {q['id']} — missing ground_truth_doc or ground_truth_pages")
            continue

        # ── candidate pool ────────────────────────────────────────────────────
        candidates = retrieve_for_query(
            q["question"],
            q["language"],
            ara_tokenizer,
            ar_index,
            en_index,
            candidate_k=CANDIDATE_K,
        )

        # ── rerank ────────────────────────────────────────────────────────────
        reranked, _ = rerank(q["question"], candidates, reranker, top_n=RERANKED_TOP_N)

        m = compute_metrics(reranked, gt_doc, gt_pages)

        rows.append({
            "query_id":   q.get("id", ""),
            "language":   q["language"],
            "complexity": q.get("complexity", ""),
            "gt_doc":     gt_doc,
            **m,
        })

    df = pd.DataFrame(rows)

    # ─────────────────────────────────────────────
    # FULL K SWEEP
    # ─────────────────────────────────────────────
    print("\n── FULL K SWEEP — HYBRID WITH RERANKER ─────────────────────────────")

    sweep = df.mean(numeric_only=True)

    for k in K_VALUES:
        print(
            f"K={k:2d}  "
            f"Precision={sweep[f'precision@{k}']:.4f}  "
            f"Recall={sweep[f'recall@{k}']:.4f}  "
            f"Hit={sweep[f'hit@{k}']:.4f}  "
            f"MRR={sweep['mrr']:.4f}"
        )

    # ─────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────
    print("\n── COMPARISON: hybrid (no rerank) vs hybrid (reranked) ──────────────")
    print(f"{'Metric':<20}{'No Rerank':>12}{'Reranked':>12}{'Delta':>12}")
    print("-" * 60)

    for metric, baseline_val in EXP1_HYBRID.items():
        current_val = sweep.get(metric, 0.0)
        delta       = current_val - baseline_val
        arrow       = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        print(f"{metric:<20}{baseline_val:>12.4f}{current_val:>12.4f}{arrow}{abs(delta):>10.4f}")

    # ─────────────────────────────────────────────
    # BY COMPLEXITY
    # ─────────────────────────────────────────────
    print("\n── BY COMPLEXITY (K=10) ────────────────────────────────────────────")
    print(group_by(df, "complexity", 10).to_string())

    # ─────────────────────────────────────────────
    # BY LANGUAGE
    # ─────────────────────────────────────────────
    print("\n── BY LANGUAGE (K=10) ──────────────────────────────────────────────")
    print(group_by(df, "language", 10).to_string())

    # ─────────────────────────────────────────────
    # BY DOCUMENT
    # ─────────────────────────────────────────────
    print("\n── BY DOCUMENT (K=10) ──────────────────────────────────────────────")
    print(group_by(df, "gt_doc", 10).to_string())

    # ─────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Raw")
        group_by(df, "language",   10).to_excel(writer, sheet_name="By_Language")
        group_by(df, "complexity", 10).to_excel(writer, sheet_name="By_Complexity")
        group_by(df, "gt_doc",     10).to_excel(writer, sheet_name="By_Document")

        for k in K_VALUES:
            cols_k = [f"precision@{k}", f"recall@{k}", f"hit@{k}", "mrr"]
            df[cols_k].mean().to_frame(name="mean").round(4).to_excel(
                writer, sheet_name=f"K={k}"
            )

    print(f"\n✓ Saved to {OUTPUT_XLSX}")
    return df


if __name__ == "__main__":
    missing = [p for p in ARABIC_PDF_PATHS + ENGLISH_PDF_PATHS if not Path(p).exists()]
    if missing:
        print("[ERROR] Missing PDFs:", missing)
        exit(1)

    run_experiment()