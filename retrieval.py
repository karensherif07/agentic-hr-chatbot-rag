from nlp_utils import tokenize, normalize_english, normalize_arabic

_MAX_RETRIEVAL_QUERY_CHARS = 500

# ── Import path sets (multi-PDF aware) ───────────────────────────────────────
# These sets are the single source of truth for which files are "Arabic" vs
# "English".  Adding a new PDF to setup.py is all that's needed — retrieval
# and reranking adapt automatically.
try:
    from setup import ARABIC_PDF_PATH_SET, ENGLISH_PDF_PATH_SET
    # Legacy single-value names kept for any remaining direct references.
    from setup import ARABIC_PDF_PATH, ENGLISH_PDF_PATH
except ImportError:
    ARABIC_PDF_PATH      = "policies/ar_policy.pdf"
    ENGLISH_PDF_PATH     = "policies/eng_policy.pdf"
    ARABIC_PDF_PATH_SET  = {ARABIC_PDF_PATH}
    ENGLISH_PDF_PATH_SET = {ENGLISH_PDF_PATH}


def _lang_tag(source: str) -> str:
    """Return 'AR' if source matches any Arabic PDF path, else 'EN'."""
    if any(p in source for p in ARABIC_PDF_PATH_SET):
        return "AR"
    return "EN"


def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 40) -> list:
    """
    Hybrid retrieval: FAISS semantic + BM25 lexical with weighted RRF.

    BM25 weight = 0.2 (hybrid_w02) — empirically optimal; reduces BM25 noise
    while preserving lexical recall signal.
    Both retrievers operate on the normalized query to avoid vocabulary
    mismatch in RRF fusion.
    """
    normalized_q = normalize_fn(raw_query)

    faiss_docs = vs.similarity_search("query: " + normalized_q, k=k)

    scores  = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]

    return rrf_weighted(faiss_docs, bm25_docs, w1=1.0, w2=0.2)


def rerank(query: str, docs: list, reranker, top_n: int = 12) -> tuple:
    """
    Rerank with a guaranteed 50/50 minimum from Arabic vs English sources so
    neither language family crowds the other out entirely.

    Works with any number of PDF files — splitting is by language tag, not
    by individual file path.
    """
    if not docs:
        return [], {}

    en_docs = [d for d in docs if _lang_tag(d.metadata.get("source", "")) == "EN"]
    ar_docs = [d for d in docs if _lang_tag(d.metadata.get("source", "")) == "AR"]
    other   = [d for d in docs if d not in en_docs and d not in ar_docs]

    def _score(group):
        if not group:
            return []
        pairs      = [(query, d.page_content) for d in group]
        raw_scores = reranker.predict(pairs)
        return sorted(zip(group, raw_scores), key=lambda x: x[1], reverse=True)

    en_ranked = _score(en_docs)
    ar_ranked = _score(ar_docs)

    # 50/50 split; each side gets at least 2
    half    = max(top_n // 2, 2)
    en_take = min(half, len(en_ranked))
    ar_take = min(half, len(ar_ranked))

    # Give leftover slots to whichever side has more candidates
    if en_take < half and ar_ranked:
        ar_take = min(top_n - en_take, len(ar_ranked))
    elif ar_take < half and en_ranked:
        en_take = min(top_n - ar_take, len(en_ranked))

    top_pairs = en_ranked[:en_take] + ar_ranked[:ar_take]

    if other:
        o_ranked  = _score(other)
        remaining = max(0, top_n - len(top_pairs))
        top_pairs += o_ranked[:remaining]

    top_pairs   = sorted(top_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    scores_dict = {id(d): float(s) for d, s in top_pairs}
    return [d for d, _ in top_pairs], scores_dict


def rrf_weighted(docs1: list, docs2: list, rrf_k: int = 60,
                 w1: float = 1.0, w2: float = 1.0) -> list:
    """
    Reciprocal Rank Fusion with per-list weights.

    docs1 = dense results (w1 — default 1.0).
    docs2 = BM25 results  (w2 — set to 0.2 for hybrid_w02).
    Identical dedup key to rrf() so the two are interchangeable.
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


def rrf(docs1: list, docs2: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion combining FAISS and BM25 rankings.

    Dedup key: source + page + first 100 chars prevents near-duplicate
    chunks from the same page being treated as distinct documents.
    """
    scores  = {}
    doc_map = {}

    def add(docs):
        for rank, d in enumerate(docs):
            source = d.metadata.get("source", "")
            page   = d.metadata.get("page", 0)
            key    = f"{source}:page_{page}:{d.page_content[:100]}"

            if key not in scores:
                scores[key]  = 0
                doc_map[key] = d

            scores[key] += 1 / (k + rank + 1)

    add(docs1)
    add(docs2)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def build_retrieval_query(current_question: str, chat_history: list, max_turns: int = 2) -> str:
    if not chat_history:
        return current_question[:_MAX_RETRIEVAL_QUERY_CHARS]

    recent_user_turns = [
        msg["content"]
        for msg in chat_history[-(max_turns * 2):]
        if msg["role"] == "user"
    ][-max_turns:]

    if not recent_user_turns:
        return current_question[:_MAX_RETRIEVAL_QUERY_CHARS]

    combined = (" ".join(recent_user_turns) + " " + current_question).strip()
    return combined[:_MAX_RETRIEVAL_QUERY_CHARS]