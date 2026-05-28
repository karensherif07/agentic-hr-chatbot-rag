# =============================================================================
# run_reranker_experiment.py
#
# Experiment 2: reranker impact on top of the agent-parity retrieval pipeline.
# Retrieval layer uses hybrid_w02 (weighted RRF, BM25 weight=0.2) — the
# empirically best config from run_retrieval_experiment_v2.py.
# Adds reranking on top and compares against hybrid_w02 retrieval-only baseline.
#
# Changes vs original:
#   - retrieve_candidates() uses _retrieve_w02() (weighted hybrid, BM25 w=0.2)
#     instead of the old equal-weight hybrid retrieve() calls
#   - rrf_weighted imported from retrieval.py (no local duplicate)
#   - EXP1_HYBRID_BASELINE updated to HYBRID_W02 measured numbers
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
K_VALUES       = [1, 3, 5, 7, 10, 20, 40]
PRIMARY_K      = 7          # reranker output size — matches agent.py _RERANKED_TOP_N
_CANDIDATE_K   = 40         # retrieval pool — matches run_retrieval_experiment.py
_RERANKED_TOP_N = 7         # how many docs reranker keeps

OUTPUT_XLSX    = "reranker_results.xlsx"
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

ARABIC_PDF_PATHS  = [p for p, _ in ARABIC_PDF_ENTRIES]
ENGLISH_PDF_PATHS = [p for p, _ in ENGLISH_PDF_ENTRIES]

# Baseline from run_retrieval_experiment.py — HYBRID_W02 (BM25 weight=0.2).
# These are the measured numbers; update if you re-run experiment 1.
EXP1_HYBRID_BASELINE = {
    "precision@1":  0.2750,
    "recall@1":     0.2719,
    "hit@1":        0.2750,
    "precision@3":  0.2625,
    "recall@3":     0.6312,
    "hit@3":        0.6375,
    "precision@5":  0.2538,
    "recall@5":     0.8500,
    "hit@5":        0.8500,
    "precision@7":  0.2295,
    "recall@7":     0.8812,
    "hit@7":        0.8812,
    "precision@10": 0.1994,
    "recall@10":    0.9250,
    "hit@10":       0.9250,
    "mrr":          0.5146,
}
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from setup     import setup
from retrieval import rrf, rrf_weighted
from nlp_utils import (
    normalize_arabic, normalize_english, tokenize,
    franco_to_arabic, egyptian_to_msa, get_semantic_dialect,
    detect_language_type,
)
import pandas as pd
from tqdm import tqdm


# =============================================================================
# CHUNK QUALITY FILTERS  (copied from agent.py verbatim)
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
# WEIGHTED HYBRID RETRIEVAL PRIMITIVE
# Identical to hybrid_w02 in run_retrieval_experiment.py and agent.py.
# =============================================================================

def _retrieve_w02(text, vs, bm25, docs, normalize_fn, k=_CANDIDATE_K):
    """
    Single-leg weighted hybrid retrieval (BM25 weight = 0.2).
    Uses rrf_weighted from retrieval.py — no local duplicate.
    """
    normalized  = normalize_fn(text)
    dense_docs  = vs.similarity_search("query: " + normalized, k=k)
    bm25_scores = bm25.get_scores(tokenize(normalized))
    top_idx     = sorted(range(len(bm25_scores)),
                         key=lambda i: bm25_scores[i], reverse=True)[:k]
    bm25_docs   = [docs[i] for i in top_idx]
    return rrf_weighted(dense_docs, bm25_docs, w1=1.0, w2=0.2)


# =============================================================================
# RETRIEVAL DISPATCH — hybrid_w02 throughout, mirrors agent.py exactly.
# Returns filtered candidate pool (chunk quality applied, no reranking).
# =============================================================================

def retrieve_candidates(question, ara_tokenizer, dialect_pipe, ar_index, en_index):
    from deep_translator import GoogleTranslator

    ar_vs, ar_bm25, ar_docs = ar_index
    en_vs, en_bm25, en_docs = en_index

    norm_ar = lambda t: normalize_arabic(t, ara_tokenizer)
    q_lang  = detect_language_type(question)

    if q_lang == "franco":
        franco_ar = franco_to_arabic(question)
        msa_query = egyptian_to_msa(franco_ar)
        docs_ar = rrf(
            _retrieve_w02(franco_ar, ar_vs, ar_bm25, ar_docs, norm_ar),
            _retrieve_w02(msa_query, ar_vs, ar_bm25, ar_docs, norm_ar),
        )
        try:
            en_query2 = GoogleTranslator(source='ar', target='en').translate(franco_ar)
        except Exception:
            en_query2 = question
        docs_en = rrf(
            _retrieve_w02(en_query2,  en_vs, en_bm25, en_docs, normalize_english),
            _retrieve_w02(msa_query,  en_vs, en_bm25, en_docs, normalize_english),
        )
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "egyptian":
        ar_msa  = egyptian_to_msa(question)
        docs_ar = rrf(
            _retrieve_w02(question, ar_vs, ar_bm25, ar_docs, norm_ar),
            _retrieve_w02(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar),
        )
        docs_en = _retrieve_w02(question, en_vs, en_bm25, en_docs, normalize_english)
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "arabic":
        if get_semantic_dialect(question, dialect_pipe) == "egyptian":
            ar_msa  = egyptian_to_msa(question)
            docs_ar = rrf(
                _retrieve_w02(question, ar_vs, ar_bm25, ar_docs, norm_ar),
                _retrieve_w02(ar_msa,   ar_vs, ar_bm25, ar_docs, norm_ar),
            )
        else:
            docs_ar = _retrieve_w02(question, ar_vs, ar_bm25, ar_docs, norm_ar)
        docs_en  = _retrieve_w02(question, en_vs, en_bm25, en_docs, normalize_english)
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "english":
        docs_en  = _retrieve_w02(question, en_vs, en_bm25, en_docs, normalize_english)
        docs_ar  = _retrieve_w02(question, ar_vs, ar_bm25, ar_docs, norm_ar)
        combined = rrf(docs_en, docs_ar)

    else:
        combined = _retrieve_w02(question, en_vs, en_bm25, en_docs, normalize_english)

    return _filter_bad_chunks(combined)


# =============================================================================
# RERANKING  — mirrors agent.py exactly
# Franco uses max(franco_ar_score, msa_score) fusion before top-n slice.
# All other languages use retrieval.rerank() directly.
# =============================================================================

def rerank_candidates(question, candidates, reranker):
    """
    Rerank candidate pool exactly as agent.py _retrieve_policy() does.
    Returns (top_docs, scores_dict).
    """
    from retrieval import rerank

    if not candidates:
        return [], {}

    q_lang = detect_language_type(question)

    if q_lang == "franco":
        rerank_query     = franco_to_arabic(question)
        rerank_query_msa = egyptian_to_msa(rerank_query)
        pairs_ar  = [(rerank_query,     d.page_content) for d in candidates]
        pairs_msa = [(rerank_query_msa, d.page_content) for d in candidates]
        scores_ar  = reranker.predict(pairs_ar)
        scores_msa = reranker.predict(pairs_msa)
        scores     = [max(a, b) for a, b in zip(scores_ar, scores_msa)]
        ranked      = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_docs    = [d for d, _ in ranked[:_RERANKED_TOP_N]]
        scores_dict = {id(d): float(s) for d, s in ranked[:_RERANKED_TOP_N]}
        return top_docs, scores_dict
    else:
        return rerank(question, candidates, reranker, top_n=_RERANKED_TOP_N)


# =============================================================================
# METRICS
# =============================================================================

def is_relevant(doc, ground_truth_doc: str, ground_truth_pages: List[int]) -> bool:
    doc_name = doc.metadata.get("doc_name", "")
    page_1   = doc.metadata.get("page", -1) + 1   # 0-based → 1-based
    return doc_name == ground_truth_doc and page_1 in ground_truth_pages


def compute_metrics(retrieved, gt_doc, gt_pages) -> dict:
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
        res[f"hit@{k}"] = round(1.0 if any(rel_k) else 0.0, 4)

    return res


def group_by(df: pd.DataFrame, key: str, k: int = PRIMARY_K) -> pd.DataFrame:
    cols = [f"precision@{k}", f"recall@{k}", f"hit@{k}", "mrr"]
    cols = [c for c in cols if c in df.columns]
    return df.groupby(key)[cols].mean().round(4)


# =============================================================================
# MAIN
# =============================================================================

def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 2: RERANKER IMPACT  (baseline = HYBRID_W02)")
    print("=" * 70)

    # setup() returns:
    # ar_index, en_index, routing_llm, en_llm, ar_llm, critique_llm,
    # reranker, dialect_pipe, ara_tokenizer
    (ar_index, en_index, _routing_llm, _en_llm, _ar_llm, _critique_llm,
     reranker, dialect_pipe, ara_tokenizer) = setup()

    queries = load_queries()
    rows    = []

    for q in tqdm(queries, desc="Reranking"):
        gt_doc   = q.get("ground_truth_doc",  "")
        gt_pages = q.get("ground_truth_pages", [])

        if not gt_doc or not gt_pages:
            print(f"  [SKIP] {q['id']} — missing ground_truth_doc or ground_truth_pages")
            continue

        candidates = retrieve_candidates(
            q["question"], ara_tokenizer, dialect_pipe, ar_index, en_index
        )

        top_docs, _ = rerank_candidates(q["question"], candidates, reranker)

        m = compute_metrics(top_docs, gt_doc, gt_pages)

        rows.append({
            "query_id":        q.get("id", ""),
            "language":        q["language"],
            "complexity":      q.get("complexity", ""),
            "gt_doc":          gt_doc,
            "candidates_pool": len(candidates),
            "docs_reranked":   len(top_docs),
            **m,
        })

    df = pd.DataFrame(rows)

    primary_cols = [f"precision@{PRIMARY_K}", f"recall@{PRIMARY_K}",
                    f"hit@{PRIMARY_K}", "mrr"]

    # ── FULL K SWEEP ──────────────────────────────────────────────────────────
    sweep = df.mean(numeric_only=True)
    print(f"\n── FULL K SWEEP — HYBRID_W02 + RERANKER (top {_RERANKED_TOP_N}) ────")
    for k in K_VALUES:
        p = sweep.get(f"precision@{k}", 0)
        r = sweep.get(f"recall@{k}",    0)
        h = sweep.get(f"hit@{k}",       0)
        m = sweep.get("mrr",            0)
        print(f"  K={k:2d}  Precision={p:.4f}  Recall={r:.4f}  Hit={h:.4f}  MRR={m:.4f}")

    # ── COMPARISON vs hybrid_w02 retrieval-only baseline ─────────────────────
    print("\n── COMPARISON: hybrid_w02 (no rerank) vs hybrid_w02 (reranked) ──────")
    print(f"{'Metric':<20}{'No Rerank':>12}{'Reranked':>12}{'Delta':>12}")
    print("-" * 60)
    for metric, baseline_val in EXP1_HYBRID_BASELINE.items():
        current_val = float(sweep.get(metric, 0.0))
        delta       = current_val - baseline_val
        arrow       = "▲" if delta > 0.0001 else ("▼" if delta < -0.0001 else "─")
        print(f"{metric:<20}{baseline_val:>12.4f}{current_val:>12.4f}  {arrow}{abs(delta):.4f}")

    # ── BY COMPLEXITY ─────────────────────────────────────────────────────────
    print(f"\n── BY COMPLEXITY (K={PRIMARY_K}) ─────────────────────────────────")
    print(group_by(df, "complexity").to_string())
    print(f"\n── BY COMPLEXITY (K=20) ──────────────────────────────────────────")
    print(group_by(df, "complexity", k=20).to_string())
    print(f"\n── BY COMPLEXITY (K=40) ──────────────────────────────────────────")
    print(group_by(df, "complexity", k=40).to_string())

    # ── BY LANGUAGE ───────────────────────────────────────────────────────────
    print(f"\n── BY LANGUAGE (K={PRIMARY_K}) ───────────────────────────────────")
    print(group_by(df, "language").to_string())
    print(f"\n── BY LANGUAGE (K=20) ────────────────────────────────────────────")
    print(group_by(df, "language", k=20).to_string())
    print(f"\n── BY LANGUAGE (K=40) ────────────────────────────────────────────")
    print(group_by(df, "language", k=40).to_string())

    # ── BY DOCUMENT ───────────────────────────────────────────────────────────
    print(f"\n── BY DOCUMENT (K={PRIMARY_K}) ───────────────────────────────────")
    print(group_by(df, "gt_doc").to_string())
    print(f"\n── BY DOCUMENT (K=20) ────────────────────────────────────────────")
    print(group_by(df, "gt_doc", k=20).to_string())
    print(f"\n── BY DOCUMENT (K=40) ────────────────────────────────────────────")
    print(group_by(df, "gt_doc", k=40).to_string())

    # ── SAVE EXCEL ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Raw")
        group_by(df, "language").to_excel(writer,        sheet_name="By_Language_K7")
        group_by(df, "language",  k=20).to_excel(writer, sheet_name="By_Language_K20")
        group_by(df, "language",  k=40).to_excel(writer, sheet_name="By_Language_K40")
        group_by(df, "complexity").to_excel(writer,        sheet_name="By_Complexity_K7")
        group_by(df, "complexity", k=20).to_excel(writer,  sheet_name="By_Complexity_K20")
        group_by(df, "complexity", k=40).to_excel(writer,  sheet_name="By_Complexity_K40")
        group_by(df, "gt_doc").to_excel(writer,        sheet_name="By_Document_K7")
        group_by(df, "gt_doc",  k=20).to_excel(writer, sheet_name="By_Document_K20")
        group_by(df, "gt_doc",  k=40).to_excel(writer, sheet_name="By_Document_K40")

        for k in K_VALUES:
            cols_k = [c for c in [f"precision@{k}", f"recall@{k}", f"hit@{k}", "mrr"]
                      if c in df.columns]
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