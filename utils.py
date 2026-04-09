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


# ─── Conversation Memory ──────────────────────────────────────
def summarize_history(llm, chat_history: list, existing_summary: str = "") -> str:
    """
    Produces a compact running summary of the conversation so far.
    Called after each turn; replaces the raw message log.
    """
    if not chat_history:
        return existing_summary or ""
    lines = []
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:300]}")
    transcript = "\n".join(lines)

    prompt = (
        "You are a summarization assistant. "
        "Given the previous conversation summary and the latest exchange, "
        "produce a single concise paragraph (max 120 words) capturing: "
        "what topics were discussed, what personal data was mentioned (grades, balances, etc.), "
        "and any important conclusions. Preserve key facts (numbers, names, decisions).\n\n"
        f"Previous summary:\n{existing_summary or 'None'}\n\n"
        f"Latest exchange:\n{transcript}\n\n"
        "Updated summary (plain text, no bullet points):"
    )
    try:
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception:
        return existing_summary or transcript


def build_history_str(chat_history: list, conversation_summary: str = "") -> str:
    """
    Returns the history string for the prompt.
    Uses the running summary if available, otherwise falls back to last 2 raw turns.
    """
    if conversation_summary:
        return f"Conversation so far:\n{conversation_summary}"
    if not chat_history:
        return "No previous conversation."
    recent = chat_history[-4:]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content'][:300]}")
    return "\n".join(lines)


# ─── Context Building ─────────────────────────────────────────
def build_context(docs: list) -> str:
    out = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        lang_tag = "AR" if ARABIC_PDF_PATH in d.metadata.get("source", "") else "EN"
        out.append(f"[Page {page_num} | {lang_tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)


# ─── No-Info Detection ────────────────────────────────────────
NO_INFO_PATTERNS = [
    "this information is not available in the policy documents",
    "information is not available in the policy",
    "not available in the policy",
    "not found in",
    "هذه المعلومات غير متوفرة في وثائق السياسة",
    "هذه المعلومات غير متاحة في وثائق السياسة",
    "معلومات غير متوفرة",
    "mesh mawgoda f el policy",
    "el ma3loma di mesh mawgoda f el policy",
    "el ma3loma mesh mawgoda",
]


def is_no_info_answer(ans: str) -> bool:
    normalized = ans.strip().lower()
    return any(p in normalized for p in NO_INFO_PATTERNS)


def validate(ans: str, lang: str, has_citations: bool = False) -> str:
    ans = ans.strip()
    if not has_citations and not is_no_info_answer(ans):
        if lang == "arabic":
            ans += "\n\n⚠️ لم يتم ذكر أرقام الصفحات في هذه الإجابة."
        else:
            ans += "\n\n⚠️ Page citations were not produced for this answer."
    return ans


# ─── Citation Helpers ─────────────────────────────────────────
def get_cited_pages(answer: str) -> set:
    return {int(n) for n in re.findall(r"\[Page\s*(\d+)", answer, re.IGNORECASE)}


def strip_citations(answer: str) -> str:
    cleaned = re.sub(r"\s*\[Page\s*\d+(?:\s*\|\s*(?:AR|EN))?\]", "", answer, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", cleaned).strip()


def filter_cited_chunks(docs: list, cited_pages: set) -> list:
    if not cited_pages:
        return docs
    return [d for d in docs if (d.metadata.get("page", 0) + 1) in cited_pages]


# ─── Snippet Extraction ───────────────────────────────────────
# Words that should not anchor the snippet (they match everywhere / headers).
_SNIPPET_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "their", "our", "its", "me", "him", "them", "us",
    "of", "and", "or", "as", "at", "by", "for", "in", "on", "to", "with", "from",
    "how", "when", "where", "why", "there", "here", "if", "about", "into", "than",
    "tell", "please", "policy", "policies", "document",
    "ما", "ماذا", "من", "في", "على", "هذا", "هذه", "ذلك", "التي", "الذي", "الى",
    "هل", "و", "ف", "ل", "ب", "ك", "إن", "إلى", "عن", "أي", "لم", "لن", "قد",
})


def _snippet_query_tokens(query: str) -> list:
    """Meaningful tokens for matching (2+ chars); drops stopwords."""
    raw = re.findall(r"[\w\u0600-\u06FF]{2,}", (query or "").lower())
    out = []
    seen = set()
    for w in raw:
        if w in _SNIPPET_STOPWORDS or len(w) < 2:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _wrap_snippet(page_text: str, start: int, end: int) -> str:
    start = max(0, start)
    end = min(len(page_text), end)
    snippet = page_text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(page_text):
        snippet = snippet + "…"
    return snippet


def extract_snippet(page_text: str, query: str, window: int = 400) -> str:
    """
    Extracts the most relevant snippet from a page's text for the query.

    Chooses a fixed-length window that maximizes how many query tokens appear
    inside it (sliding scan). Stops words are ignored for anchoring. Falls back
    to the start of the page if nothing matches.
    """
    if not page_text:
        return ""
    if window < 1:
        window = 400

    n = len(page_text)
    if n <= window:
        return page_text.strip()

    query_words = _snippet_query_tokens(query)
    if not query_words:
        return _wrap_snippet(page_text, 0, window)

    text_lower = page_text.lower()

    def score_window(s: int) -> int:
        e = min(n, s + window)
        chunk = text_lower[s:e]
        return sum(1 for w in query_words if w in chunk)

    step = max(8, window // 16)
    best_start = 0
    best_score = -1
    for s in range(0, n - window + 1, step):
        sc = score_window(s)
        if sc > best_score:
            best_score = sc
            best_start = s

    # Refine around the best coarse window (finer step) for a better fit.
    if best_score > 0:
        refine_lo = max(0, best_start - step)
        refine_hi = min(n - window, best_start + step)
        for s in range(refine_lo, refine_hi + 1):
            sc = score_window(s)
            if sc > best_score:
                best_score = sc
                best_start = s

    if best_score == 0:
        # No token matched anywhere: try span-based centering on any non-stopword hit.
        positions = []
        for w in query_words:
            pos = text_lower.find(w)
            if pos != -1:
                positions.append((pos, pos + len(w)))
        if positions:
            span_lo = min(p[0] for p in positions)
            span_hi = max(p[1] for p in positions)
            center = (span_lo + span_hi) // 2
            best_start = max(0, min(center - window // 2, n - window))
        else:
            best_start = 0

    return _wrap_snippet(page_text, best_start, best_start + window)


def anchor_for_pdf_search(
    chunk_text: str,
    query_en: str,
    query_ar: str = "",
    answer_excerpt: str = "",
    max_len: int = 120,
) -> str:
    """
    Build a phrase from the retrieved chunk that overlaps the question/answer
    and is long enough for PyMuPDF search_for() (avoids tiny irrelevant clips).
    """
    if not chunk_text:
        return ""
    blob = " ".join(p for p in (query_en or "", query_ar or "", answer_excerpt or "") if p)
    win = min(max(500, max_len * 5), max(len(chunk_text), max_len + 40))
    snippet = extract_snippet(chunk_text, blob, window=win)
    s = snippet.strip()
    if s.startswith("…"):
        s = s[1:].lstrip()
    if s.endswith("…"):
        s = s[:-1].rstrip()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 35:
        s = re.sub(r"\s+", " ", chunk_text[:500]).strip()
    if len(s) > max_len:
        lo = min(len(s) // 6, max(0, len(s) - max_len))
        s = s[lo : lo + max_len].strip()
    return s[:max_len]


# ─── Confidence / Similarity Scores ──────────────────────────
def get_doc_scores(query: str, docs: list, reranker) -> dict:
    """
    Returns {id(doc): score} for each doc using the cross-encoder reranker.
    """
    if not docs or reranker is None:
        return {}
    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = reranker.predict(pairs)
        return {id(d): float(s) for d, s in zip(docs, scores)}
    except Exception:
        return {}


def batch_rerank_query_excerpts(reranker, query: str, docs: list, window: int = 480) -> list[float]:
    """
    Scores (query vs excerpt) where excerpt is the best-matching window of each chunk.
    Full 1000-token chunks often score artificially low; excerpts align with what the UI shows.
    """
    if not docs or reranker is None:
        return []
    excerpts = []
    for d in docs:
        ex = extract_snippet(d.page_content, query, window=window)
        if not ex.strip():
            ex = (d.page_content or "")[: window + 80]
        excerpts.append(ex)
    try:
        pairs = list(zip([query] * len(docs), excerpts))
        return [float(s) for s in reranker.predict(pairs)]
    except Exception:
        return []


def score_to_confidence(raw_score: float) -> tuple:
    """
    Converts a raw cross-encoder score to (label, hex_color).
    Prefer confidence_badge() for UI — it compares peers and uses looser bands.
    """
    if raw_score >= -4.0:
        return "High", "#2e7d32"
    if raw_score >= -9.0:
        return "Medium", "#f57c00"
    return "Low", "#c62828"


def confidence_badge(raw_score: float, peer_scores: list[float] | None) -> tuple[str, str]:
    """
    Labels for the policy excerpt score. With 2+ cited pages, ranks this page
    against the others so badges are not all Low. Single page: generous absolute
    bands (mmarco-mMiniLM logits are often negative even for good matches).
    """
    peers = [float(p) for p in (peer_scores or []) if p is not None]
    if len(peers) >= 2:
        lo, hi = min(peers), max(peers)
        span = max(hi - lo, 1e-6)
        norm = (float(raw_score) - lo) / span
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