import re

ARABIC_PDF_PATH = "policies/ar_policy.pdf"

def translate(llm, text: str, target_language: str) -> str:
    try:
        res = llm.invoke(
            f"Translate the following text to {target_language} using ONLY formal Modern Standard Arabic (no dialect). "
            f"Convert any dialect words into proper MSA.\n\n"
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

NO_INFO_PATTERNS = [
    "this information is not available in the policy documents",
    "information is not available in the policy",
    "not available in the policy",
    "not found in",
    "zis information is not available",
    "هذه المعلومات غير متوفرة في وثائق السياسة",
    "هذه المعلومات غير متاحة في وثائق السياسة",
    "معلومات غير متوفرة",
    "mesh mawgoda f el policy",
    "el ma3loma di mesh mawgoda f el policy",
    "el ma3loma mesh mawgoda",
    "mawgoda f el policy"
]


def is_no_info_answer(ans: str) -> bool:
    normalized = ans.strip().lower()
    return any(pattern in normalized for pattern in NO_INFO_PATTERNS)


def validate(ans: str, lang: str, has_citations: bool = False) -> str:
    ans = ans.strip()
    is_not_found = is_no_info_answer(ans)
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