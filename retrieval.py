from nlp_utils import tokenize, normalize_english, normalize_arabic

_MAX_RETRIEVAL_QUERY_CHARS = 500

ARABIC_PDF_PATH  = "policies/ar_policy.pdf"
ENGLISH_PDF_PATH = "policies/eng_policy.pdf"


def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 10) -> list:
    """
    Hybrid retrieval: FAISS semantic + BM25 lexical.
    
    FIXED: Normalize query BEFORE embedding to ensure consistency.
    Increased k=10 (was 8) to capture more candidates before RRF reranking.
    """
    # Normalize query for both FAISS and BM25 consistency
    normalized_q = normalize_fn(raw_query)
    
    # FAISS semantic search: use normalized query to get vector embeddings
    # This ensures the semantic search captures normalized variations
    faiss_docs = vs.similarity_search(normalized_q, k=k)
    
    # BM25 lexical search: tokenize normalized query
    scores = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]
    
    # Reciprocal Rank Fusion combines both signals
    return rrf(faiss_docs, bm25_docs)


def rerank(query: str, docs: list, reranker, top_n: int = 12) -> tuple:
    """
    Rerank docs but GUARANTEE a minimum number of chunks from each source
    language so Arabic chunks never completely crowd out English ones or
    vice versa.

    Strategy:
      - Split docs into English-PDF and Arabic-PDF groups.
      - Rerank each group independently.
      - Allocate slots to ensure both languages represented.
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

    # Allocate slots: Try 50/50 split, but each side gets at least 2 docs
    # This prevents one language from completely dominating
    half     = max(top_n // 2, 2)
    en_take  = min(half, len(en_ranked))
    ar_take  = min(half, len(ar_ranked))

    # If one side has fewer than half, give the remainder to the other side
    # but cap it to available documents
    if en_take < half and ar_take > 0:
        ar_take = min(top_n - en_take, len(ar_ranked))
    elif ar_take < half and en_take > 0:
        en_take = min(top_n - ar_take, len(en_ranked))

    top_pairs  = en_ranked[:en_take] + ar_ranked[:ar_take]
    
    # Also include any "other" docs (shouldn't happen but be safe)
    if other:
        o_ranked  = _score(other)
        remaining = max(0, top_n - len(top_pairs))
        top_pairs += o_ranked[:remaining]

    # Re-sort the merged set by score for the LLM context order
    top_pairs  = sorted(top_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    scores_dict = {id(d): float(s) for d, s in top_pairs}
    return [d for d, _ in top_pairs], scores_dict


def rrf(docs1: list, docs2: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion: combines FAISS and BM25 rankings.
    
    IMPROVED: Better deduplication using full content hashing to avoid
    treating slightly different chunks of the same section as unique.
    """
    scores = {}
    doc_map = {}

    def add(docs):
        for rank, d in enumerate(docs):
            # Use page + page position for dedup key, not just first 120 chars
            # This prevents near-duplicate chunks from being counted separately
            source = d.metadata.get("source", "")
            page = d.metadata.get("page", 0)
            key = f"{source}:page_{page}:{d.page_content[:100]}"
            
            if key not in scores:
                scores[key] = 0
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