import re
from functools import lru_cache

# ─────────────────────────────────────────────────────────────
# intent.py — LLM-BASED INTENT CLASSIFIER
# (FIXED: no unhashable ChatGroq in cache)
# ─────────────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """You are a classifier for an HR chatbot. Your only job is to classify the user's question into one of three intents.

=== INTENT DEFINITIONS ===

PERSONAL — The question asks about THIS specific employee's own live data.
The answer must come from a database (not policy documents).
Signs: possessive words (my, I have, بتاعي, عندي, راتبي, bta3i, bta3ti),
first-person ownership of a specific value.
Examples:
- "How many annual leave days do I have left?" → personal / leave
- "What is my current net salary?" → personal / salary
- "What was my last performance rating?" → personal / performance
- "كم يوم إجازة متبقي عندي؟" → personal / leave
- "أنا لسه في فترة التجربة والا خلصت؟" → personal / profile
- "الاجازات البتاعتي الواقفة دلوقتي إيه؟" → personal / leave
- "راتبي الصافي بتاع الشهر ده كام؟" → personal / salary
- "Do I have any active disciplinary actions against me?" → personal / disciplinary
- "raseed agaza bta3i kamet yom?" → personal / leave
- "ana lesa fi probation wla 5alaset?" → personal / profile
- "el okrs bta3ti 3amela ezay?" → personal / okr

HYBRID — The question needs BOTH the employee's personal data AND company policy to answer correctly.
Signs: eligibility questions ("am I eligible", "can I", "do I qualify"),
working hours (needs DB work_model + policy standard hours),
leave availability (needs DB balance + policy entitlement).
Examples:
- "What are my working hours?" → hybrid / profile
- "Am I eligible for a bonus this year?" → hybrid / all
- "هل يحق لي التقدم للترقية؟" → hybrid / all
- "ساعات شغلي إيه؟" → hybrid / profile
- "Can I take study leave?" → hybrid / leave
- "mawa3id shoghl bta3ti eh?" → hybrid / profile
- "momken akhod bonus el sana di?" → hybrid / all
- "هل أقدر آخد إجازة دلوقتي؟" → hybrid / leave

POLICY — General question about company rules answered from policy documents only.
No personal data needed. The same answer applies to any employee.
Signs: general "how many days", "what is the", "who approves", "what are the criteria".
Examples:
- "What are the promotion criteria?" → policy
- "How many days is Hajj leave?" → policy
- "What is the overtime rate?" → policy
- "كم يوم إجازة سنوية يحق لموظف بخبرة أكثر من 10 سنوات؟" → policy
- "ما مراحل الإجراء التأديبي التدريجي؟" → policy
- "emta el bonus bta3i byigi w ana lazem 3amel eh?" → policy
- "law 3andi inzar maktub a2dar a3terid 3aleih?" → policy
- "What happens to my personal data if I leave the company?" → policy

=== TOPICS (only for personal and hybrid) ===
leave, salary, performance, training, okr, profile, disciplinary, all

=== OUTPUT FORMAT (strict — no other text) ===
INTENT: personal|hybrid|policy
TOPIC: leave|salary|performance|training|okr|profile|disciplinary|all|none

Question: {question}"""


# ─────────────────────────────────────────────────────────────
# ✅ LLM REGISTRY (prevents caching unhashable objects)
# ─────────────────────────────────────────────────────────────
_LLM_REGISTRY = {}


def _parse_response(text: str) -> tuple:
    """Parse 'INTENT: X\\nTOPIC: Y' from LLM response."""
    intent = "policy"
    topic = None
    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("INTENT:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("personal", "hybrid", "policy"):
                intent = val
        elif line.upper().startswith("TOPIC:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in ("leave", "salary", "performance", "training",
                       "okr", "profile", "disciplinary", "all"):
                topic = val
    return intent, topic


def _llm_classify(question: str, llm) -> tuple:
    """Single structured LLM call."""
    try:
        prompt = _CLASSIFICATION_PROMPT.format(question=question)
        res = llm.invoke(prompt)
        return _parse_response(res.content)
    except Exception as e:
        print(f"[intent] LLM classify failed: {e} — defaulting to policy")
        return "policy", None


# ─────────────────────────────────────────────────────────────
# ✅ FIXED CACHE (ONLY HASHABLE INPUTS)
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=500)
def _cached_classify(question: str, llm_id: str) -> tuple:
    llm = _LLM_REGISTRY.get(llm_id)
    if llm is None:
        return "policy", None
    return _llm_classify(question, llm)


def classify_intent(question: str, llm=None) -> tuple:
    """
    Classify the user's question using the LLM.
    """
    if llm is None:
        return _fallback_classify(question)

    llm_id = str(id(llm))

    # store LLM outside cache
    _LLM_REGISTRY[llm_id] = llm

    return _cached_classify(question, llm_id)


# ─────────────────────────────────────────────────────────────
# fallback (unchanged)
# ─────────────────────────────────────────────────────────────
_FB_PERSONAL = re.compile(
    r"(\bmy\b.{0,25}\b(salary|leave|rating|performance|okr|goal|grade"
    r"|probation|disciplinary|training)\b"
    r"|راتبي|إجازتي|تقييمي|bta3i|bta3ti|raseed|عندي\b)",
    re.IGNORECASE
)

_FB_HYBRID = re.compile(
    r"(my working hours|my hours|am i eligible|can i be promoted"
    r"|can i take study leave|ساعات شغلي|ساعات عملي|هل يحق لي"
    r"|momken akhod bonus|mawa3id shoghl)",
    re.IGNORECASE
)


def _fallback_classify(question: str) -> tuple:
    if _FB_HYBRID.search(question):
        return "hybrid", "all"
    if _FB_PERSONAL.search(question):
        return "personal", "all"
    return "policy", None