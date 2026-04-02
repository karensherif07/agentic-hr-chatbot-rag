import re

ARABIC_PDF_PATH = "ar_policy.pdf"

def translate(llm, text: str, target_language: str) -> str:
    try:
        res = llm.invoke(
            f"Translate the following text to {target_language}. "
            f"Return ONLY the translation, nothing else.\n\nText: {text}"
        )
        return res.content.strip()
    except Exception:
        return text


def build_context(docs: list) -> str:
    out = []
    for d in docs:
        page_num = d.metadata.get("page", 0) + 1
        source = d.metadata.get("source", "")
        lang_tag = "AR" if ARABIC_PDF_PATH in source else "EN"
        out.append(f"[Page {page_num} | {lang_tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)

# ─── Validation & Citation Utilities ─────────────────────────

def validate(ans: str, lang: str, has_citations: bool = False) -> str:
    ans = ans.strip()
    not_found_phrases = [
        "not available in the policy",
        "غير متوفرة في وثائق",
        "not found in",
        "mesh mawgoda f el policy",
    ]
    is_not_found = any(p.lower() in ans.lower() for p in not_found_phrases)
    if not has_citations and not is_not_found:
        if lang == "arabic":
            ans += "\n\n⚠️ لم يتم ذكر أرقام الصفحات في هذه الإجابة."
        else:
            ans += "\n\n⚠️ Page citations were not produced for this answer."
    return ans


def get_cited_pages(answer: str) -> set:
    return {int(n) for n in re.findall(r"\[Page\s*(\d+)", answer, re.IGNORECASE)}


def strip_citations(answer: str) -> str:
    cleaned = re.sub(r"\s*\[Page\s*\d+(?:\s*\|\s*(?:AR|EN))?\]", "", answer, flags=re.IGNORECASE)
    return re.sub(r"  +", " ", cleaned).strip()


def filter_cited_chunks(docs: list, cited_pages: set) -> list:
    if not cited_pages:
        return docs
    return [d for d in docs if (d.metadata.get("page", 0) + 1) in cited_pages]