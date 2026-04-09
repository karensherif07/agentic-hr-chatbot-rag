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
def extract_snippet(page_text: str, query: str, window: int = 400) -> str:
    """
    Extracts the most relevant snippet from a page's text around the query terms.
    Falls back to the first `window` characters if no match found.
    """
    if not page_text:
        return ""
    query_words = [w for w in re.findall(r"[\w\u0600-\u06FF]{3,}", query.lower()) if len(w) > 2]
    best_pos = -1
    best_count = 0
    text_lower = page_text.lower()

    step = max(1, window // 4)
    for start in range(0, max(1, len(page_text) - window), step):
        chunk = text_lower[start:start + window]
        hits = sum(1 for w in query_words if w in chunk)
        if hits > best_count:
            best_count = hits
            best_pos = start

    if best_pos >= 0 and best_count > 0:
        snippet = page_text[best_pos: best_pos + window].strip()
    else:
        snippet = page_text[:window].strip()

    if best_pos > 0:
        snippet = "…" + snippet
    if (best_pos if best_pos >= 0 else 0) + window < len(page_text):
        snippet = snippet + "…"
    return snippet


# ─── Confidence / Similarity Scores ──────────────────────────
def get_doc_scores(query: str, docs: list, reranker) -> dict:
    """
    Returns {doc_key: score} for each doc using the cross-encoder reranker.
    doc_key = first 120 chars of page_content.
    """
    if not docs or reranker is None:
        return {}
    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = reranker.predict(pairs)
        return {d.page_content[:120]: float(s) for d, s in zip(docs, scores)}
    except Exception:
        return {}


def score_to_confidence(raw_score: float) -> tuple:
    """
    Converts a raw cross-encoder score to (label, hex_color).
    Cross-encoder mmarco scores are roughly in range [-5, 5].
    """
    if raw_score >= 2.0:
        return "High", "#2e7d32"
    elif raw_score >= 0.0:
        return "Medium", "#f57c00"
    else:
        return "Low", "#c62828"