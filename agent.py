"""
agent.py — Agentic HR Assistant  (two-stage architecture)

Architecture:
  Stage 1 — Lightweight intent router (routing_llm / 8b)
             Classifies: policy | personal | hybrid | out_of_scope

  Stage 2 — Dynamic planner  (routing_llm / 8b)
             hybrid:   retrieve_policy FIRST → planner reads context → selects DB tools
             personal: planner selects DB tools directly (no policy lookup)
             policy:   retrieve_policy only
             OOS:      returns immediately

Token budget:
  routing call  (8b):   ~200 tokens  (was ~300)
  planner call  (8b):   ~350 tokens  (was ~500)
  answer call   (70b):  ~2,000 tokens
  answer call   (qwen): ~2,000 tokens
  critique call (8b):   ~150 tokens  — skipped in test mode
"""

import json
import re
import time
from datetime import date

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from personal_data import (
    get_employee_profile, get_leave_balance, get_leave_requests,
    get_pending_leave, get_performance_history, get_latest_review,
    get_performance_trend, get_okrs, get_latest_salary,
    get_payroll_history, get_training_record, get_active_disciplinary,
)
from retrieval import retrieve, rerank, rrf
from nlp_utils import (
    egyptian_to_msa, get_semantic_dialect,
    normalize_arabic, normalize_english,
    detect_language_type, franco_to_arabic,
)
from utils import (
    build_context, is_no_info_answer, validate,
    get_cited_pages, strip_citations, filter_cited_chunks,
    ARABIC_PDF_PATH, _is_arabic_source,
)
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt
from personal_prompts import get_personal_prompt, get_hybrid_prompt, format_personal_data


# =============================================================================
# STAGE 1 — Intent Router prompt  (trimmed ~30% vs v1)
# =============================================================================
_ROUTER_SYSTEM = """HR chatbot intent router for Horizon Tech. Classify into one intent:

policy      — company rule/entitlement/procedure, same for all employees
              Signal: no first-person pronouns; asks about "the company", a grade, or a role.
              Examples: "What is the notice period for G3?", "What travel class does a G4 get?",
              "What does health insurance cover for children?", "What are the gift acceptance limits?",
              "What are the scholarship conditions?", "What is the per diem for business travel?"
personal    — requires THIS employee's own DB data; no policy lookup needed
              Signal: first-person pronouns — I/my/me/ana/bta3y/3andy/maratby/agaza bta3ty/
              kam yom fadel 3andy/eih maratby/el rating bta3y/el OKRs bta3ty.
              Examples: "How many leave days do I have?", "What is my net salary?",
              "Am I on a PIP?", "What was my last performance rating?",
              "kam yom agaza fadel 3andy?", "eih maratby el net?"
hybrid      — requires BOTH a policy lookup AND this employee's personal data to answer
              Signal: eligibility / entitlement / calculation that depends on the employee's
              personal grade, rating, tenure, or status AND a policy rule.
              Examples: "Am I eligible for the annual bonus?", "How much notice do I need to give?",
              "Can I apply for a promotion?", "What gratuity would I get if I resign?",
              "Can I switch to full remote?", "Am I eligible for the savings plan?",
              "lw esta2elt hakhod end of service ad eih?", "ana mosta7e2 el bonus?",
              "Am I on a PIP and what does that mean for my salary increment?"
out_of_scope — completely unrelated to HR, company policy, or the employee's work data
              Examples: weather, stock prices, coding tasks, writing personal emails

Question may be in English, Arabic (MSA), Egyptian dialect, or Franco-Arabic.
Reply with JSON only: {"intent": "<policy|personal|hybrid|out_of_scope>"}"""


# =============================================================================
# STAGE 2 — DB Tool Planner prompt  (trimmed ~30% vs v1)
# =============================================================================
_PLANNER_SYSTEM = """HR chatbot data-fetching planner.
Policy context is shown below (if any). Select only the personal data tools strictly needed to answer the question. Do not call retrieve_policy. Call each tool at most once. Do not speculate."""


# =============================================================================
# Constants
# =============================================================================
_DB_KEYS = ["profile", "leave_data", "salary_data", "performance_data",
            "training_data", "disciplinary_data"]

_MAX_CHUNK_CHARS    = 600
_MAX_CONTEXT_CHUNKS = 7   # matches _RERANKED_TOP_N

# Candidate pool and reranker top-n — kept in sync with generate_answers.py
_CANDIDATE_K    = 20
_RERANKED_TOP_N = 7

_NO_INFO_MARKERS = ("no relevant policy found", "not available")

_KEY_MAP = {
    "get_profile":           "profile",
    "get_leave_data":        "leave_data",
    "get_salary_data":       "salary_data",
    "get_performance_data":  "performance_data",
    "get_training_data":     "training_data",
    "get_disciplinary_data": "disciplinary_data",
}

_TOPIC_MAP = {
    "get_leave_data":        "leave",
    "get_salary_data":       "salary",
    "get_performance_data":  "performance",
    "get_training_data":     "training",
    "get_disciplinary_data": "disciplinary",
    "get_profile":           "profile",
}

_DB_TOOL_NAMES = frozenset(_KEY_MAP)


# =============================================================================
# HELPERS
# =============================================================================

def _build_context_truncated(docs: list) -> str:
    chunks  = []
    skipped = 0
    for d in docs[:_MAX_CONTEXT_CHUNKS]:
        content = d.page_content
        if (_is_boilerplate(content)
                or _is_corrupted_chunk(content)
                or _looks_like_ocr_loop(content)):
            skipped += 1
            continue
        page_num = d.metadata.get("page", 0) + 1
        lang_tag = "AR" if _is_arabic_source(d.metadata.get("source", "")) else "EN"
        truncated = content[:_MAX_CHUNK_CHARS]
        if len(content) > _MAX_CHUNK_CHARS:
            truncated += "…"
        chunks.append(f"[Page {page_num}|{lang_tag}]\n{truncated}")
    if skipped:
        print(f"[agent] Skipped {skipped} boilerplate/corrupted chunks")
    return "\n---\n".join(chunks) if chunks else ""


def _strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks.
    Handles:
    1. Well-formed <think>...</think> — strip the block.
    2. Unclosed <think> (truncated by max_tokens) — strip everything from <think> onward.
    3. No block — return unchanged.
    """
    # Case 1: well-formed block
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if cleaned != text:
        return re.sub(r"\s{3,}", "\n\n", cleaned).strip()

    # Case 2: unclosed <think> — everything after is reasoning, not an answer
    unclosed = re.search(r"<think\b[^>]*>", text, re.IGNORECASE)
    if unclosed:
        before = text[:unclosed.start()].strip()
        return before if len(before) > 10 else ""

    return re.sub(r"\s{3,}", "\n\n", text).strip()


# Backward-compat alias
_strip_qwen_thinking = _strip_think_blocks


def _invoke_with_retry(llm, prompt, max_retries=3):
    """Exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                m    = re.search(r"try again in (\d+)m(\d+)s", msg)
                wait = (int(m.group(1)) * 60 + int(m.group(2)) + 5) if m else 60 * (attempt + 1)
                print(f"[agent] Rate limit, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("LLM call failed after retries")


# =============================================================================
# CHUNK QUALITY FILTERS  (mirrors generate_answers.py exactly)
# =============================================================================

_BOILERPLATE_PATTERNS = [
    r"^Confidential\s*[—-]\s*Internal Use Only",
    r"^سري\s*[—-]\s*للاستخدام الداخلي فقط",
    r"^Mbps\s*\.",
    r"^Page\s*\d+\s*$",
    r"تاريخ النفاذ\s*سري",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.DOTALL)

_GARBAGE_PATTERNS = [
    r"[^\w\s\u0600-\u06FF]{15,}",
    r"(.)\1{10,}",
    r"\b[a-zA-Z]{1,2}\b(?:\s+\b[a-zA-Z]{1,2}\b){10,}",
]
_GARBAGE_RE = re.compile("|".join(_GARBAGE_PATTERNS), re.DOTALL)


def _is_boilerplate(text: str) -> bool:
    stripped = text.strip()
    stripped_check = re.sub(r'^[.\s]+', '', stripped)
    if len(stripped_check) < 50:
        return True
    if _BOILERPLATE_RE.search(stripped_check[:300]):
        return True
    _SIRI_ONLY = re.compile(
        r'^[\.\s]*سري\s*[—\-–]\s*للاستخدام الداخلي فقط',
        re.IGNORECASE
    )
    if _SIRI_ONLY.match(stripped):
        return True
    if 'سري' in stripped and 'للاستخدام الداخلي فقط' in stripped:
        _POLICY_WORDS = re.compile(
            r'(إجازة|راتب|موظف|بدل|تأمين|اشتراك|مكافأة|عمل إضافي|فترة|اختبار|تقييم|شهادة|تدريب|نفقة|مصروف)'
        )
        if not _POLICY_WORDS.search(stripped):
            return True
    _EN_STAMP = re.compile(
        r'^[\.\s]*(Confidential\s*[—\-–]\s*Internal Use Only|'
        r'Horizon Tech\s*[—\-–]\s*Human Resources)',
        re.IGNORECASE
    )
    if _EN_STAMP.match(stripped) and len(stripped) < 300:
        return True
    return False


def _looks_like_ocr_loop(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    repeated = sum(1 for i in range(len(lines) - 1) if lines[i] == lines[i + 1])
    if repeated >= 2:
        return True
    numeric_ratio = sum(
        1 for tok in text.split() if any(c.isdigit() for c in tok)
    ) / max(len(text.split()), 1)
    return numeric_ratio > 0.35


def _is_corrupted_chunk(text: str) -> bool:
    t = text.strip()
    if len(t) < 80:
        return True
    weird_ratio = sum(
        1 for c in t
        if not (c.isalnum() or c.isspace() or c in ".,:;!?-%()/")
    ) / max(len(t), 1)
    if weird_ratio > 0.30:
        return True
    if _GARBAGE_RE.search(t):
        return True
    words = t.split()
    if len(words) < 15 and len(t) > 300:
        return True
    return False


# ── Context instruction preambles (prepended to context in each prompt) ───────
_AR_CONTEXT_INSTRUCTION = (
    "استخدم جميع المقاطع التالية معاً للإجابة، بما فيها المقاطع باللغة الإنجليزية. "
    "لا تتجاهل أي مقطع. أجب بشكل مباشر ومختصر بناءً على المعلومات المتاحة فقط.\n\n"
)

_EN_CONTEXT_INSTRUCTION = (
    "ONLY use the passages labeled [Page N|EN] or [Page N|AR] below. "
    "Some passages are in Arabic — read them too. "
    "Do NOT use any knowledge from training. "
    "If the answer isn't in these passages, say: "
    "'This information is not available in the policy documents.' "
    "Answer in 1-3 sentences maximum.\n\n"
)

_FRANCO_CONTEXT_INSTRUCTION = (
    "Esta3mel kol el passages el gaya di — sawa el 3arabi w el inglizi. "
    "Matb2ash ay passage. "
    "Law el ma3loma mesh mawgoda: '2ol mesh mawgoda fel policy.' "
    "Egabtak: Franco bass — la 3arabi wala inglizi.\n\n"
)


# =============================================================================
# LOOP + FRANCO DETECTION  (mirrors generate_answers.py)
# =============================================================================

def _is_empty_policy(ctx: str) -> bool:
    """True when policy retrieval returned nothing useful."""
    if not ctx:
        return True
    return any(m in ctx.lower() for m in _NO_INFO_MARKERS)


_LOOP_PATTERNS = [
    r"(\[Page \d+\|(?:EN|AR)\]\n?){3,}",
    r"(التسجيل\n){5,}",
    r"(ليلة\n){5,}",
    r"(الخدمات التالية\n){3,}",
    r"(however:? the salary){3,}",
    r"(التحديد\n\d+\n){5,}",
    r"(سياسة الادخار\n){3,}",
    r"(تنفيذ برنامج){3,}",
    r"(هياخدك\s*تبعتك){3,}",
    r"(تبعتك\s*وتبعتك){3,}",
    r"(.{15,50})\1{4,}",
]
_LOOP_RE = re.compile("|".join(_LOOP_PATTERNS), re.DOTALL)


def _is_looping(text: str) -> bool:
    return bool(_LOOP_RE.search(text))


_FRANCO_WORD_RE = re.compile(
    r'\b(el|al|fe|fi|law|lw|bas|bs|msh|mesh|mish|ana|enta|enti|howa|hya|'
    r'lazem|yalla|mashy|tamam|tayeb|momken|ya3ni|3ashan|keda|aywa|la2|'
    r'leih|fein|emta|ezay|meen|eih|eh|da|di|dol|aho|ahi|'
    r'biyedi|biyebda2|biyestamr|biyedfa3|byet7aseb|'
    r'le7ad|men|3ala|mafish|feeh|fieh|walla|wala|'
    r'shoghl|rateb|agaza|ta2min|muwazaf|gedid|'
    r'[a-z]+[237][a-z]*|[a-z]*[237][a-z]+)\b',
    re.IGNORECASE
)


def _is_franco(text: str) -> bool:
    if not text or len(text) < 5:
        return False
    latin_chars  = len(re.findall(r'[a-zA-Z0-9 ]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    if latin_chars < 3 * max(arabic_chars, 1):
        return False
    return bool(_FRANCO_WORD_RE.search(text))



    """True when policy retrieval returned nothing useful."""
    if not ctx:
        return True
    return any(m in ctx.lower() for m in _NO_INFO_MARKERS)


# =============================================================================
# POLICY RETRIEVAL  — mirrors run_reranker_experiment.py exactly
#
# Language dispatch logic (all branches):
#   franco   : transliterate → MSA-expand both, RRF AR variants, RRF with EN
#   egyptian : MSA-expand, RRF both AR variants, RRF with EN
#   arabic   : dialect-detect; if egyptian → MSA-expand + RRF, then RRF with EN
#              if MSA → AR only, then RRF with EN
#   english  : EN first, then AR, RRF(EN, AR)
#   fallback : EN only
#
# Candidate pool : _CANDIDATE_K = 20  (matches generate_answers.py)
# Reranker top-n : _RERANKED_TOP_N = 7 (matches generate_answers.py)
# No score threshold — let reranker ranking decide; don't silently drop results.
# =============================================================================

def _retrieve_policy(query, ar_index, en_index, reranker, ara_tokenizer, dialect_pipe=None):
    """
    Language-aware hybrid retrieval + reranking.
    Mirrors generate_answers.py _retrieve_policy() exactly.
    Returns (top_docs, scores_dict, context_str).
    Returns ([], {}, "") when nothing useful is found.
    """
    from deep_translator import GoogleTranslator

    ar_vs, ar_bm25, ar_docs = ar_index
    en_vs, en_bm25, en_docs = en_index

    norm_ar = lambda t: normalize_arabic(t, ara_tokenizer)
    q_lang  = detect_language_type(query)

    # ── language-aware candidate retrieval ───────────────────────────────────
    if q_lang == "franco":
        franco_ar    = franco_to_arabic(query)
        franco_norm  = egyptian_to_msa(franco_ar)
        msa_query    = franco_norm
        docs_ar = rrf(
            retrieve(franco_ar,  ar_vs, ar_bm25, ar_docs, norm_ar),
            retrieve(msa_query,  ar_vs, ar_bm25, ar_docs, norm_ar),
        )
        try:
            en_query2 = GoogleTranslator(source='ar', target='en').translate(franco_ar)
        except Exception:
            en_query2 = query
        docs_en = rrf(
            retrieve(en_query2,  en_vs, en_bm25, en_docs, normalize_english),
            retrieve(msa_query,  en_vs, en_bm25, en_docs, normalize_english),
        )
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "egyptian":
        ar_msa   = egyptian_to_msa(query)
        docs_ar  = rrf(
            retrieve(query,  ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
            retrieve(ar_msa, ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
        )
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "arabic":
        # NOTE: get_semantic_dialect requires dialect_pipe, not ara_tokenizer
        if get_semantic_dialect(query, dialect_pipe) == "egyptian":
            ar_msa  = egyptian_to_msa(query)
            docs_ar = rrf(
                retrieve(query,  ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
                retrieve(ar_msa, ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K),
            )
        else:
            docs_ar = retrieve(query, ar_vs, ar_bm25, ar_docs, norm_ar, k=_CANDIDATE_K)
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)
        combined = rrf(docs_ar, docs_en)

    elif q_lang == "english":
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)
        docs_ar  = retrieve(query, ar_vs, ar_bm25, ar_docs, norm_ar,           k=_CANDIDATE_K)
        combined = rrf(docs_en, docs_ar)

    else:
        # Fallback: English only
        combined = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=_CANDIDATE_K)

    # ── rerank — franco uses max(franco_ar, msa) scores ──────────────────────
    if q_lang == "franco":
        rerank_query     = franco_to_arabic(query)
        rerank_query_msa = egyptian_to_msa(rerank_query)
        pairs_ar  = [(rerank_query,     d.page_content) for d in combined]
        pairs_msa = [(rerank_query_msa, d.page_content) for d in combined]
        scores_ar  = reranker.predict(pairs_ar)
        scores_msa = reranker.predict(pairs_msa)
        scores = [max(a, b) for a, b in zip(scores_ar, scores_msa)]
        ranked   = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
        top_docs = [d for d, _ in ranked[:_RERANKED_TOP_N]]
        scores_dict = {id(d): float(s) for d, s in ranked[:_RERANKED_TOP_N]}
    else:
        top_docs, scores_dict = rerank(query, combined, reranker, top_n=_RERANKED_TOP_N)

    if not top_docs:
        return [], {}, ""

    ctx = _build_context_truncated(top_docs)
    return top_docs, scores_dict, ctx


# =============================================================================
# TOOL FACTORY  — DB tools only; retrieve_policy is never a planner tool
# =============================================================================

def _make_db_tools(employee_id):

    @tool
    def get_profile() -> str:
        """Employee profile: name, grade, department, hire date, employment type,
        work model, probation status."""
        data = get_employee_profile(employee_id)
        return json.dumps(data, default=str, ensure_ascii=False) if data else "Profile not found."

    @tool
    def get_leave_data() -> str:
        """Leave balances (remaining days per type, current year), pending requests,
        and recent requests."""
        today = date.today()
        return json.dumps({
            "leave_balances":   get_leave_balance(employee_id, today.year),
            "pending_requests": get_pending_leave(employee_id),
            "recent_requests":  get_leave_requests(employee_id, limit=3),
        }, default=str, ensure_ascii=False)

    @tool
    def get_salary_data() -> str:
        """Latest salary breakdown (net, gross, base, allowances, deductions)
        and 3-month payroll history."""
        return json.dumps({
            "latest_salary":  get_latest_salary(employee_id),
            "salary_history": get_payroll_history(employee_id, months=3),
        }, default=str, ensure_ascii=False)

    @tool
    def get_performance_data() -> str:
        """Employee's latest review rating, performance trend, current OKRs,
        and recent review history. Use for personal rating/OKR questions."""
        return json.dumps({
            "latest_review":     get_latest_review(employee_id),
            "performance_trend": get_performance_trend(employee_id),
            "okrs":              get_okrs(employee_id),
            "history":           get_performance_history(employee_id, limit=2),
        }, default=str, ensure_ascii=False)

    @tool
    def get_training_data() -> str:
        """Training budget (total/used/remaining in USD), training days, and
        courses completed this year."""
        data = get_training_record(employee_id, date.today().year)
        return json.dumps(data or {}, default=str, ensure_ascii=False)

    @tool
    def get_disciplinary_data() -> str:
        """Active disciplinary actions: verbal/written warnings or PIP not yet
        expired. Use for personal disciplinary status questions only."""
        data = get_active_disciplinary(employee_id)
        return json.dumps(data, default=str, ensure_ascii=False)

    return [
        get_profile, get_leave_data, get_salary_data,
        get_performance_data, get_training_data, get_disciplinary_data,
    ]


# =============================================================================
# STAGE 1 — Intent classification
# =============================================================================

def _classify_intent(question: str, routing_llm) -> str:
    """
    Calls routing_llm with the router prompt.
    Returns: policy | personal | hybrid | out_of_scope
    Falls back to 'hybrid' on parse error (safe default).
    """
    messages = [
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=question),
    ]
    res  = _invoke_with_retry(routing_llm, messages)
    text = _strip_qwen_thinking(res.content).strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()

    try:
        intent = json.loads(text).get("intent", "hybrid").lower()
        if intent in ("policy", "personal", "hybrid", "out_of_scope"):
            return intent
    except Exception:
        pass

    # Fuzzy fallback
    for candidate in ("out_of_scope", "hybrid", "personal", "policy"):
        if candidate in text.lower():
            return candidate

    return "hybrid"


# =============================================================================
# STAGE 2 — DB tool planner
# =============================================================================

def _run_planner(question: str, policy_context: str,
                 routing_llm, db_tools: list) -> tuple[list, dict]:
    """
    Sends planner prompt + policy_context to routing_llm with DB tools bound.
    Executes chosen tool calls (each at most once).
    Returns (tools_called: list[str], tool_results: dict).
    """
    tool_map       = {t.name: t for t in db_tools}
    llm_with_tools = routing_llm.bind_tools(db_tools)

    # Keep planner user message concise — policy_context already trimmed upstream
    planner_user = f"Question: {question}"
    if policy_context:
        planner_user += f"\n\nPolicy context:\n{policy_context}"
    planner_user += "\n\nCall only the strictly needed personal data tools."

    messages = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=planner_user),
    ]

    tools_called: list = []
    tool_results: dict = {}
    called_set:   set  = set()

    for _ in range(3):
        response = _invoke_with_retry(llm_with_tools, messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tname = tc["name"]

            if tname in called_set:
                messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content="Already called.",
                ))
                continue

            called_set.add(tname)
            tools_called.append(tname)

            try:
                fn     = tool_map[tname]
                targs  = tc.get("args", {})
                result = fn.invoke(targs) if targs else fn.invoke({})
                tool_results[_KEY_MAP.get(tname, tname)] = result
            except Exception as e:
                result = f"Error: {e}"

            messages.append(ToolMessage(tool_call_id=tc["id"], content=result))

    return tools_called, tool_results


# =============================================================================
# ANSWER FORMATTER
# =============================================================================

def _merge_db(tool_results: dict) -> dict:
    combined = {}
    for key in _DB_KEYS:
        if key in tool_results:
            try:
                combined.update(json.loads(tool_results[key]))
            except Exception:
                pass
    return combined


def _pick_llm(lang, en_llm, ar_llm):
    return en_llm if lang in ["english", "franco"] else ar_llm


def _format_answer(lang, dialect, question, tool_results, history_str,
                   en_llm, ar_llm, top_docs):
    has_policy = bool(tool_results.get("policy_context"))
    has_db     = any(k in tool_results for k in _DB_KEYS)
    llm        = _pick_llm(lang, en_llm, ar_llm)
    ctx        = (_build_context_truncated(top_docs)
                  if top_docs else tool_results.get("policy_context", ""))

    if has_db and has_policy:
        res = _invoke_with_retry(llm, get_hybrid_prompt(lang, dialect).format(
            personal_data=format_personal_data(_merge_db(tool_results)),
            policy_context=ctx,
            question=question,
            history=history_str,
        ))
    elif has_db:
        res = _invoke_with_retry(llm, get_personal_prompt(lang, dialect).format(
            personal_data=format_personal_data(_merge_db(tool_results)),
            question=question,
            history=history_str,
        ))
    else:
        if not ctx:
            return "This information is not available in the policy documents."
        # Prepend context instruction preamble (mirrors generate_answers.py)
        if lang == "english":
            augmented_ctx = _EN_CONTEXT_INSTRUCTION + ctx
            prompt = english_prompt
        elif lang == "franco":
            augmented_ctx = _FRANCO_CONTEXT_INSTRUCTION + ctx
            prompt = franco_prompt
        else:
            augmented_ctx = _AR_CONTEXT_INSTRUCTION + ctx
            prompt = egy_prompt if dialect == "egyptian" else msa_prompt
        res = _invoke_with_retry(llm, prompt.format(
            context=augmented_ctx, question=question, history=history_str,
        ))

    return _strip_think_blocks(res.content)


# =============================================================================
# CRITIQUE  (answer truncation raised to 800 chars for coverage)
# =============================================================================

# Tighter prompt — same information, ~20% fewer tokens
_CRITIQUE_PROMPT = (
    "Is this HR answer complete?\n"
    "Answer: {answer}\n"
    'JSON only: {{"adequate":true,"missing":""}} or {{"adequate":false,"missing":"what is missing"}}'
)


def _critique(critique_llm, answer: str) -> dict:
    try:
        res  = _invoke_with_retry(
            critique_llm,
            _CRITIQUE_PROMPT.format(answer=answer[:800])   # raised from 400
        )
        text = re.sub(
            r"^```json|^```|```$", "",
            _strip_qwen_thinking(res.content),
            flags=re.MULTILINE
        ).strip()
        return json.loads(text)
    except Exception:
        return {"adequate": True, "missing": ""}


# =============================================================================
# HELPERS — intent / topic inference
# =============================================================================

def _infer_topic(tools_called: list) -> str:
    for t in tools_called:
        if t in _TOPIC_MAP:
            return _TOPIC_MAP[t]
    return "none"


def _build_personal_data_str(tool_results: dict) -> str:
    combined = _merge_db(tool_results)
    if not combined:
        return ""
    try:
        return format_personal_data(combined)
    except Exception:
        return ""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_agent(
    question, employee_id, lang, dialect, history_str,
    ar_index, en_index, routing_llm, en_llm, ar_llm, critique_llm,
    reranker, ara_tokenizer, dialect_pipe=None,
    max_iterations=3,
    skip_critique=True,
):
    """
    Two-stage agentic orchestration:

      Stage 1: _classify_intent()   — cheap router, single JSON token output
      Stage 2: policy retrieval + _run_planner() — policy first, then DB tools

    tools_called order:
      hybrid   → ["retrieve_policy", <db_tools…>]
      policy   → ["retrieve_policy"]
      personal → [<db_tools…>]
      OOS      → ["out_of_scope"]
    """
    db_tools     = _make_db_tools(employee_id)
    tool_results: dict = {}
    tools_called: list = []
    top_docs:     list = []
    scores_dict:  dict = {}

    # ── Stage 1: classify intent ──────────────────────────────────────────────
    intent = _classify_intent(question, routing_llm)

    # ── Stage 2: fetch data based on intent ───────────────────────────────────

    if intent == "out_of_scope":
        tools_called = ["out_of_scope"]

    elif intent == "policy":
        top_docs, scores_dict, ctx = _retrieve_policy(
            question, ar_index, en_index, reranker, ara_tokenizer, dialect_pipe
        )
        tools_called = ["retrieve_policy"]
        tool_results["policy_context"] = ctx or ""

    elif intent == "personal":
        db_tools_called, db_results = _run_planner(
            question, "",
            routing_llm, db_tools,
        )
        tools_called = db_tools_called
        tool_results.update(db_results)

    else:  # hybrid
        top_docs, scores_dict, ctx = _retrieve_policy(
            question, ar_index, en_index, reranker, ara_tokenizer, dialect_pipe
        )
        # Normalise: treat empty string same as "no policy found"
        ctx = ctx or ""
        tool_results["policy_context"] = ctx

        tools_called = [] if _is_empty_policy(ctx) else ["retrieve_policy"]

        db_tools_called, db_results = _run_planner(
            question, ctx,
            routing_llm, db_tools,
        )
        tools_called.extend(db_tools_called)
        tool_results.update(db_results)

    # ── Format answer ─────────────────────────────────────────────────────────
    raw   = _format_answer(lang, dialect, question, tool_results,
                           history_str, en_llm, ar_llm, top_docs)
    cited = get_cited_pages(raw)
    cdocs = filter_cited_chunks(top_docs, cited)
    clean = strip_citations(raw)

    db_only = (any(k in tool_results for k in _DB_KEYS)
               and not tool_results.get("policy_context"))
    answer  = clean if db_only else validate(clean, lang, has_citations=bool(cited))

    # ── Optional critique pass ────────────────────────────────────────────────
    if not skip_critique and not is_no_info_answer(answer):
        c = _critique(critique_llm, answer)
        if not c.get("adequate") and c.get("missing"):
            # Re-run the full language-aware pipeline for the enriched query,
            # not bare retrieve() calls — mirrors the main retrieval path.
            rq = f"{question} {c['missing']}"
            t2, s2, ctx2 = _retrieve_policy(
                rq, ar_index, en_index, reranker, ara_tokenizer, dialect_pipe
            )
            if t2:
                tool_results["policy_context"] = _build_context_truncated(t2)
                r2  = _format_answer(lang, dialect, question, tool_results,
                                     history_str, en_llm, ar_llm, t2)
                c2  = get_cited_pages(r2)
                cd2 = filter_cited_chunks(t2, c2)
                cl2 = strip_citations(r2)
                a2  = cl2 if db_only else validate(cl2, lang, has_citations=bool(c2))
                if not is_no_info_answer(a2):
                    answer, top_docs, scores_dict, cdocs = a2, t2, s2, cd2

    # ── Resolve final intent label ────────────────────────────────────────────
    has_db_data = any(k in tool_results for k in _DB_KEYS)
    if intent not in ("out_of_scope",) and is_no_info_answer(answer) and not has_db_data:
        intent = "out_of_scope"

    topic = _infer_topic(tools_called)
    pdata = (_build_personal_data_str(tool_results)
             if intent in ("personal", "hybrid") else "")

    return {
        "answer":        answer,
        "docs":          top_docs,
        "cited_docs":    cdocs,
        "scores":        scores_dict,
        "intent":        intent,
        "topic":         topic,
        "tools_called":  tools_called,
        "personal_data": pdata,
    }


# =============================================================================
# BACK-COMPAT: _make_tools + _infer_intent for run_intent_experiment.py
# _ORCHESTRATOR_SYSTEM is a SEPARATE prompt from _ROUTER_SYSTEM.
# _ROUTER_SYSTEM  → used in _classify_intent(): outputs JSON, no tool calls.
# _ORCHESTRATOR_SYSTEM → used in routing-only eval: LLM is bound with tools
#   and MUST call them — never answer directly from memory.
# =============================================================================

_ORCHESTRATOR_SYSTEM = """You are an HR assistant for Horizon Tech. You MUST answer every question by calling one or more tools — never respond from memory or training knowledge.

Tool selection rules:
- retrieve_policy  → ANY question about a company rule, rate, procedure, entitlement, or policy
                     (same answer for all employees). Always call this for policy questions even
                     if you think you know the answer — the authoritative source is the PDF.
                     Examples: notice period, travel class, health insurance coverage, gift limits,
                     per diem rates, scholarship conditions, leave entitlements.
- get_profile      → employee's grade, department, hire date, work model, probation status.
- get_leave_data   → employee's personal leave balance and pending requests.
- get_salary_data  → employee's personal salary breakdown and payroll history.
- get_performance_data → employee's personal rating, OKRs, and review history.
- get_training_data    → employee's personal training budget usage and courses.
- get_disciplinary_data → employee's active warnings or PIP status.
- out_of_scope     → ONLY when the question has absolutely nothing to do with HR,
                     company policy, or the employee's work data (e.g. weather, stock price,
                     coding tasks, personal email drafting).

For questions that need BOTH a policy rule AND personal data (eligibility, calculations,
entitlements that depend on grade/rating/tenure), call retrieve_policy AND the relevant
personal data tool(s).

Question language may be English, Arabic (MSA), Egyptian dialect, or Franco-Arabic."""


def _make_tools(employee_id, ar_index, en_index, reranker, ara_tokenizer, dialect_pipe=None):
    """Legacy shim for run_intent_experiment.py (ROUTING_ONLY eval mode)."""

    @tool
    def retrieve_policy(query: str) -> str:
        """Search Horizon Tech HR policy PDFs (Arabic + English).
        Use when the answer is a company rule, rate, procedure, or entitlement
        that is the same for every employee."""
        top_docs, scores, ctx = _retrieve_policy(
            query, ar_index, en_index, reranker, ara_tokenizer, dialect_pipe
        )
        retrieve_policy._last_docs   = top_docs
        retrieve_policy._last_scores = scores
        return ctx if ctx else "No relevant policy found."

    @tool
    def out_of_scope() -> str:
        """Call when the question has nothing to do with HR, company policy,
        or the employee's personal work data."""
        return "out_of_scope"

    db_tools  = _make_db_tools(employee_id)
    all_tools = [retrieve_policy] + db_tools + [out_of_scope]
    return all_tools, retrieve_policy


def _infer_intent(tools_called, policy_result: str = "") -> tuple[str, str]:
    """Legacy shim for run_intent_experiment.py."""
    has_policy = "retrieve_policy" in tools_called
    has_db     = bool(set(tools_called) & _DB_TOOL_NAMES)

    # Explicit out_of_scope tool call always wins.
    # Do NOT infer OOS from retrieval returning no-info — that is a retrieval
    # quality issue, not proof the topic is out of scope.
    if "out_of_scope" in tools_called:
        intent = "out_of_scope"
    elif has_db and has_policy:
        intent = "hybrid"
    elif has_db:
        intent = "personal"
    elif has_policy:
        intent = "policy"
    else:
        intent = "out_of_scope"  # nothing called at all

    return intent, _infer_topic(tools_called)