from nlp_utils import tokenize

def retrieve(raw_query: str, vs, bm25, docs: list, normalize_fn, k: int = 15) -> list:
    faiss_docs = vs.similarity_search(raw_query, k=k)
    normalized_q = normalize_fn(raw_query)
    scores = bm25.get_scores(tokenize(normalized_q))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    bm25_docs = [docs[i] for i in top_idx]
    return rrf(faiss_docs, bm25_docs)


def rerank(query: str, docs: list, reranker, top_n: int = 5) -> list:
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


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