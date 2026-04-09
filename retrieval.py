from nlp_utils import tokenize


def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 15) -> list:
    faiss_docs = vs.similarity_search(raw_query, k=k)
    normalized_q = normalize_fn(raw_query)
    scores = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]
    return rrf(faiss_docs, bm25_docs)


def rerank(query: str, docs: list, reranker, top_n: int = 8) -> tuple:
    """
    Returns (ranked_docs, scores_dict).
    scores_dict = {doc_key: raw_score} where doc_key = page_content[:120].
    """
    if not docs:
        return [], {}
    pairs = [(query, d.page_content) for d in docs]
    raw_scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, raw_scores), key=lambda x: x[1], reverse=True)
    top = ranked[:top_n]
    scores_dict = {d.page_content[:120]: float(s) for d, s in top}
    return [d for d, _ in top], scores_dict


def rrf(docs1: list, docs2: list, k: int = 60) -> list:
    scores = {}

    def add(docs):
        for rank, d in enumerate(docs):
            key = d.page_content[:120]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

    add(docs1)
    add(docs2)
    doc_map = {d.page_content[:120]: d for d in docs1 + docs2}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]


def build_retrieval_query(current_question: str, chat_history: list, max_turns: int = 2) -> str:
    """
    Concatenates the last N user messages with the current question so that
    short follow-ups like 'what about paternity leave?' retrieve relevant chunks.
    """
    if not chat_history:
        return current_question
    recent_user_turns = [
        msg["content"]
        for msg in chat_history[-(max_turns * 2):]
        if msg["role"] == "user"
    ][-max_turns:]
    if not recent_user_turns:
        return current_question
    return (" ".join(recent_user_turns) + " " + current_question).strip()