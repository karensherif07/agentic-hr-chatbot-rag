import re

ARABIC_PDF_PATH = "policies/ar_policy.pdf"


def translate(llm, text: str, target_language: str) -> str:
    try:
        res = llm.invoke(
            f"Translate to {target_language} (formal MSA if Arabic). "
            f"Return ONLY the translation.\n\nText: {text}"
        )
        return res.content.strip()
    except Exception:
        return text


def summarize_history(llm, chat_history: list, existing_summary: str = "") -> str:
    """
    Lazily summarize — only called every 4 turns, not after every message.
    Cap at 40 words to save tokens.
    """
    if not chat_history:
        return existing_summary or ""
    # Only last 4 messages (2 turns) to keep prompt tiny
    lines = []
    for msg in chat_history[-4:]:
        role = "U" if msg["role"] == "user" else "A"
        lines.append(f"{role}: {msg['content'][:120]}")
    transcript = "\n".join(lines)
    prompt = (
        "Summarize in max 40 words. Keep: topics, numbers, decisions.\n"
        f"Previous: {existing_summary or 'none'}\nNew:\n{transcript}\nSummary:"
    )
    try:
        res = llm.invoke(prompt)
        return res.content.strip()[:300]   # hard cap output
    except Exception:
        return existing_summary or ""


def build_history_str(chat_history: list, conversation_summary: str = "") -> str:
    """Token-optimized: summary only (40 words), or last 1 turn if no summary."""
    if conversation_summary:
        # Already capped at 300 chars in summarize_history
        return f"Context: {conversation_summary}"
    if not chat_history:
        return ""
    # No summary yet — send only the single most recent exchange
    recent = chat_history[-2:]
    lines = []
    for msg in recent:
        role = "U" if msg["role"] == "user" else "A"
        lines.append(f"{role}: {msg['content'][:150]}")
    return "\n".join(lines)


def build_context(docs: list) -> str:
    sorted_docs = sorted(
        docs,
        key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", 0))
    )
    out = []
    for d in sorted_docs:
        page_num = d.metadata.get("page", 0) + 1
        lang_tag = "AR" if ARABIC_PDF_PATH in d.metadata.get("source", "") else "EN"
        out.append(f"[Page {page_num} | {lang_tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)


NO_INFO_PATTERNS = [
    "this information is not available in the policy documents",
    "information is not available in the policy",
    "not available in the policy",
    "not found in",
    "هذه المعلومات غير متوفرة في وثائق السياسة",
    "هذه المعلومات غير متاحة في وثائق السياسة",
    "معلومات غير متوفرة في وثائق",
    "الموضوع ده مش موجود في السياسة",
    "mesh mawgoda f el policy",
    "ma3loma mesh mawgoda fel policy",
    "no relevant policy documents found",
]


def is_no_info_answer(ans: str) -> bool:
    normalized = ans.strip().lower()
    return any(p in normalized for p in NO_INFO_PATTERNS)


def validate(ans: str, lang: str, has_citations: bool = False) -> str:
    ans = ans.strip()
    if not has_citations and not is_no_info_answer(ans):
        if lang == "arabic":
            ans += "\n\n⚠️ لم يتم ذكر أرقام الصفحات في هذه الإجابة."
        elif lang == "franco":
            ans += "\n\n⚠️ Mafeesh arqam sa7fat fel egaba di."
        else:
            ans += "\n\n⚠️ Page citations were not produced for this answer."
    return ans


def get_cited_pages(answer: str) -> set:
    """
    Returns set of (page_num_int, lang_tag) tuples e.g. {(9, 'AR'), (5, 'EN')}.
    Using (page, lang) prevents AR p.9 from also displaying EN p.9 chunks.
    """
    matches = re.findall(r"\[Page\s*(\d+)\s*\|\s*(AR|EN)\]", answer, re.IGNORECASE)
    return {(int(n), tag.upper()) for n, tag in matches}


def strip_citations(answer: str) -> str:
    cleaned = re.sub(r"\s*\[Page\s*\d+(?:\s*\|\s*(?:AR|EN))?\]", "", answer, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", cleaned).strip()


def filter_cited_chunks(docs: list, cited_pages: set) -> list:
    """
    cited_pages is set of (page_num, lang_tag) tuples.
    Only include docs whose exact (page, AR/EN) pair was cited.
    """
    if not cited_pages:
        return docs
    result = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        lang_tag = "AR" if ARABIC_PDF_PATH in d.metadata.get("source", "") else "EN"
        if (page_num, lang_tag) in cited_pages:
            result.append(d)
    return result


_SNIPPET_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can",
    "what", "which", "who", "this", "that", "these", "those",
    "am", "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "their", "our", "its",
    "of", "and", "or", "as", "at", "by", "for", "in", "on", "to", "with", "from",
    "how", "when", "where", "why", "if", "about",
    "tell", "please", "policy", "document",
    "ما", "من", "في", "على", "هذا", "هذه", "هل", "و", "إلى", "عن",
})


def extract_snippet(page_text: str, query: str, window: int = 400) -> str:
    if not page_text:
        return ""
    n = len(page_text)
    if n <= window:
        return page_text.strip()
    raw = re.findall(r"[\w\u0600-\u06FF]{2,}", (query or "").lower())
    query_words = [w for w in raw if w not in _SNIPPET_STOPWORDS]
    if not query_words:
        return page_text[:window].strip()
    text_lower = page_text.lower()
    step = max(8, window // 16)
    best_start, best_score = 0, -1
    for s in range(0, n - window + 1, step):
        chunk = text_lower[s:min(n, s + window)]
        sc = sum(1 for w in query_words if w in chunk)
        if sc > best_score:
            best_score, best_start = sc, s
    end = min(n, best_start + window)
    snippet = page_text[best_start:end].strip()
    if best_start > 0:
        snippet = "…" + snippet
    if end < n:
        snippet += "…"
    return snippet


def anchor_for_pdf_search(chunk_text, query_en="", query_ar="", answer_excerpt="", max_len=200):
    if not chunk_text:
        return ""
    if answer_excerpt:
        clean = re.sub(r"\s+", " ", answer_excerpt.strip())
        for length in (140, 100, 70, 50, 35):
            if len(clean) >= length and clean[:length].lower() in chunk_text.lower():
                return clean[:length].strip()
    blob = " ".join(p for p in (query_en, query_ar, answer_excerpt) if p)
    snippet = extract_snippet(chunk_text, blob, window=max(500, max_len * 4))
    s = re.sub(r"^…|…$", "", snippet).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] if len(s) >= 35 else chunk_text[:max_len]


def get_doc_scores(query, docs, reranker):
    if not docs or reranker is None:
        return {}
    try:
        scores = reranker.predict([(query, d.page_content) for d in docs])
        return {id(d): float(s) for d, s in zip(docs, scores)}
    except Exception:
        return {}


def batch_rerank_query_excerpts(reranker, query, docs, window=480):
    if not docs or reranker is None:
        return []
    excerpts = [extract_snippet(d.page_content, query, window) or d.page_content[:window] for d in docs]
    try:
        return [float(s) for s in reranker.predict(list(zip([query]*len(docs), excerpts)))]
    except Exception:
        return []


def score_to_confidence(raw_score: float) -> tuple:
    if raw_score >= -4.0:
        return "High", "#2e7d32"
    if raw_score >= -9.0:
        return "Medium", "#f57c00"
    return "Low", "#c62828"


def confidence_badge(raw_score: float, peer_scores: list | None) -> tuple[str, str]:
    peers = [float(p) for p in (peer_scores or []) if p is not None]
    if len(peers) >= 2:
        lo, hi = min(peers), max(peers)
        norm = (float(raw_score) - lo) / max(hi - lo, 1e-6)
        if norm >= 0.5:
            return "Strong match", "#2e7d32"
        if norm >= 0.2:
            return "Good match", "#f57c00"
        return "Related", "#6d4c41"
    if float(raw_score) >= -4.0:
        return "High", "#2e7d32"
    if float(raw_score) >= -9.0:
        return "Medium", "#f57c00"
    return "Low", "#c62828"