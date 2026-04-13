from nlp_utils import tokenize

_MAX_RETRIEVAL_QUERY_CHARS = 500

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"


def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 15) -> list:
    faiss_docs    = vs.similarity_search(raw_query, k=k)
    normalized_q  = normalize_fn(raw_query)
    scores        = bm25.get_scores(tokenize(normalized_q))
    top_idx       = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs     = [docs[i] for i in top_idx]
    return rrf(faiss_docs, bm25_docs)


def rerank(query: str, docs: list, reranker, top_n: int = 12) -> tuple:
    """
    Rerank docs but GUARANTEE a minimum number of chunks from each source
    language so Arabic chunks never completely crowd out English ones or
    vice versa.

    Strategy:
      - Split docs into English-PDF and Arabic-PDF groups.
      - Rerank each group independently.
      - Take top ceil(top_n * 0.6) from the higher-scoring group and
        top ceil(top_n * 0.4) from the other, then merge.
      - This ensures cross-page English sections (like promotion criteria
        split across pages) always reach the LLM even when Arabic chunks
        score higher overall.
    """
    if not docs:
        return [], {}

    en_docs = [d for d in docs if ENGLISH_PDF_PATH in d.metadata.get("source", "")]
    ar_docs = [d for d in docs if ARABIC_PDF_PATH  in d.metadata.get("source", "")]
    other   = [d for d in docs if d not in en_docs and d not in ar_docs]

    def _score(group):
        if not group:
            return []
        pairs      = [(query, d.page_content) for d in group]
        raw_scores = reranker.predict(pairs)
        return sorted(zip(group, raw_scores), key=lambda x: x[1], reverse=True)

    en_ranked = _score(en_docs)
    ar_ranked = _score(ar_docs)

    # Allocate slots: 50/50 split, each side gets at least min(3, available)
    half     = max(top_n // 2, 3)
    en_take  = min(half, len(en_ranked))
    ar_take  = min(half, len(ar_ranked))

    # If one side has fewer than half, give the remainder to the other side
    if en_take < half:
        ar_take = min(top_n - en_take, len(ar_ranked))
    elif ar_take < half:
        en_take = min(top_n - ar_take, len(en_ranked))

    top_pairs  = en_ranked[:en_take] + ar_ranked[:ar_take]
    # Also include any "other" docs (shouldn't happen but be safe)
    if other:
        o_ranked  = _score(other)
        top_pairs += o_ranked[:max(1, top_n - en_take - ar_take)]

    # Re-sort the merged set by score for the LLM context order
    top_pairs  = sorted(top_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    scores_dict = {id(d): float(s) for d, s in top_pairs}
    return [d for d, _ in top_pairs], scores_dict


def rrf(docs1: list, docs2: list, k: int = 60) -> list:
    scores = {}

    def add(docs):
        for rank, d in enumerate(docs):
            key = d.page_content[:120]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    add(docs1)
    add(docs2)
    doc_map = {d.page_content[:120]: d for d in docs1 + docs2}
    ranked  = sorted(scores.items(), key=lambda x: x[1], reverse=True)
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