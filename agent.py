"""
agent.py — Agentic HR Assistant
================================
Replaces the hardcoded intent → fetch → retrieve pipeline in app.py
with a tool-calling LLM that decides what it needs.

HOW IT WORKS:
1. LLM receives the employee's question + available tools
2. It decides which tools to call (policy retrieval, DB lookups, or both)
3. Results are fed back → LLM generates the final answer using existing
   language-specific prompt templates (citations, dialect, Franco all preserved)
4. A self-critique loop triggers one targeted retry if the answer is incomplete
"""

import json
import re
from datetime import date

from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool

from personal_data import (
    get_employee_profile,
    get_leave_balance,
    get_leave_requests,
    get_pending_leave,
    get_performance_history,
    get_latest_review,
    get_performance_trend,
    get_okrs,
    get_latest_salary,
    get_payroll_history,
    get_training_record,
    get_active_disciplinary,
)
from retrieval import retrieve, rerank, rrf
from nlp_utils import normalize_arabic, normalize_english
from utils import (
    build_context, is_no_info_answer, validate,
    get_cited_pages, strip_citations, filter_cited_chunks,
)
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt
from personal_prompts import get_personal_prompt, get_hybrid_prompt, format_personal_data


# ── Orchestrator system prompt ───────────────────────────────────────────────
_ORCHESTRATOR_SYSTEM = """You are an HR assistant orchestrator for Horizon Tech.
Your job is to decide WHICH data to fetch to answer the employee's question.

AVAILABLE TOOLS:
- retrieve_policy: fetch HR policy document chunks (company rules, entitlements, procedures)
- get_profile: employee's basic info (grade, department, hire date, work model, probation status)
- get_leave_data: leave balances + pending/recent leave requests
- get_salary_data: latest payslip + salary history
- get_performance_data: performance reviews, ratings, trend, OKRs
- get_training_data: training budget and courses
- get_disciplinary_data: active disciplinary records (warnings, PIP)

DECISION RULES:
1. POLICY question (same answer for everyone) → retrieve_policy only
2. PERSONAL question (this employee's own data) → relevant DB tool(s) only
3. HYBRID (needs both) → retrieve_policy AND relevant DB tool(s)

EXAMPLES:
- "what is the overtime rate?" → retrieve_policy only
- "what is my net salary?" → get_salary_data only
- "am I still on probation?" → get_profile only
- "how many leave days do I have left?" → get_leave_data only
- "am I eligible for a bonus?" → get_performance_data + get_disciplinary_data + retrieve_policy
- "can I get promoted?" → get_profile + get_performance_data + retrieve_policy
- "what are my working hours?" → get_profile + retrieve_policy

Call tools now. Do not generate a final answer yet — just call the tools."""


# ── DB key list (shared across helpers) ──────────────────────────────────────
_DB_KEYS = [
    "profile", "leave_data", "salary_data",
    "performance_data", "training_data", "disciplinary_data",
]


# ── Tool factory (per-request closure over employee_id) ─────────────────────

def _make_tools(employee_id: int, ar_index, en_index, reranker, ara_tokenizer):
    """
    Build the tool set for this specific employee and request.
    retrieve_policy is returned separately so _last_docs/_last_scores
    can be read back after the agentic loop.
    """

    @tool
    def retrieve_policy(query: str) -> str:
        """
        Search Horizon Tech HR policy documents (Arabic + English PDFs).
        Use for company rules, entitlements, procedures, disciplinary steps.
        Pass the search query in English for best results.
        """
        ar_vs, ar_bm25, ar_docs = ar_index
        en_vs, en_bm25, en_docs = en_index

        docs_ar  = retrieve(query, ar_vs, ar_bm25, ar_docs,
                            lambda t: normalize_arabic(t, ara_tokenizer))
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english)
        combined = rrf(docs_ar, docs_en)
        top_docs, scores = rerank(query, combined, reranker, top_n=8)

        if not top_docs:
            return "No relevant policy found."

        retrieve_policy._last_docs   = top_docs
        retrieve_policy._last_scores = scores
        return build_context(top_docs)

    @tool
    def get_profile() -> str:
        """Get this employee's profile: name, grade, department, hire date,
        work model (remote/hybrid/office), and probation status."""
        data = get_employee_profile(employee_id)
        if not data:
            return "Profile not found."
        return json.dumps(data, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_leave_data() -> str:
        """Get this employee's leave balances (remaining days per leave type)
        and their recent and pending leave requests."""
        today    = date.today()
        balances = get_leave_balance(employee_id, today.year)
        pending  = get_pending_leave(employee_id)
        recent   = get_leave_requests(employee_id, limit=5)
        return json.dumps({
            "leave_balances":   balances,
            "pending_requests": pending,
            "recent_requests":  recent,
        }, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_salary_data() -> str:
        """Get this employee's latest salary breakdown (base, gross, net,
        allowances, deductions) and 6-month payroll history."""
        latest  = get_latest_salary(employee_id)
        history = get_payroll_history(employee_id, months=6)
        return json.dumps({
            "latest_salary":  latest,
            "salary_history": history,
        }, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_performance_data() -> str:
        """Get this employee's performance reviews, latest rating, trend,
        bonus multiplier, and current OKRs."""
        return json.dumps({
            "latest_review":     get_latest_review(employee_id),
            "performance_trend": get_performance_trend(employee_id),
            "okrs":              get_okrs(employee_id),
            "history":           get_performance_history(employee_id, limit=4),
        }, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_training_data() -> str:
        """Get this employee's training budget (total/used/remaining),
        training days, and completed courses for the current year."""
        data = get_training_record(employee_id, date.today().year)
        return json.dumps(data or {}, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_disciplinary_data() -> str:
        """Get this employee's active disciplinary records:
        verbal warnings, written warnings, and PIP status."""
        data = get_active_disciplinary(employee_id)
        return json.dumps(data, default=str, ensure_ascii=False, indent=2)

    tools = [
        retrieve_policy,
        get_profile,
        get_leave_data,
        get_salary_data,
        get_performance_data,
        get_training_data,
        get_disciplinary_data,
    ]
    return tools, retrieve_policy


# ── Helpers ──────────────────────────────────────────────────────────────────

def _merge_db_results(tool_results: dict) -> dict:
    """Merge all DB tool result JSONs into one combined dict."""
    combined = {}
    for key in _DB_KEYS:
        if key in tool_results:
            try:
                combined.update(json.loads(tool_results[key]))
            except Exception:
                pass
    return combined


def _build_personal_data_str(tool_results: dict) -> str:
    """
    Formats collected DB tool results for the UI 'Your data used' expander.
    Returns empty string if no DB data was fetched.
    """
    combined = _merge_db_results(tool_results)
    if not combined:
        return ""
    try:
        return format_personal_data(combined)
    except Exception:
        return ""


# ── Final answer generation via language-specific prompt templates ────────────

def _format_final_answer(
    llm,
    question: str,
    lang: str,
    dialect: str | None,
    tool_results: dict,
    history_str: str,
) -> str:
    """
    Generates the final answer using the existing language-specific prompt
    templates. This ensures citations, dialect style (Egyptian, Franco, MSA),
    and tone are fully controlled — not left to the tool-calling LLM.
    """
    has_policy = bool(tool_results.get("policy_context"))
    has_db     = any(k in tool_results for k in _DB_KEYS)

    if has_db and has_policy:
        personal_data_str = format_personal_data(_merge_db_results(tool_results))
        prompt = get_hybrid_prompt(lang, dialect)
        res = llm.invoke(prompt.format(
            personal_data=personal_data_str,
            policy_context=tool_results["policy_context"],
            question=question,
            history=history_str,
        ))

    elif has_db:
        personal_data_str = format_personal_data(_merge_db_results(tool_results))
        prompt = get_personal_prompt(lang, dialect)
        res = llm.invoke(prompt.format(
            personal_data=personal_data_str,
            question=question,
            history=history_str,
        ))

    else:
        policy_context = tool_results.get("policy_context", "")
        if not policy_context:
            return "This information is not available in the policy documents."
        prompt = (
            english_prompt if lang == "english" else
            franco_prompt  if lang == "franco"  else
            egy_prompt     if dialect == "egyptian" else
            msa_prompt
        )
        res = llm.invoke(prompt.format(
            context=policy_context,
            question=question,
            history=history_str,
        ))

    return res.content


# ── Self-critique ────────────────────────────────────────────────────────────

_CRITIQUE_PROMPT = """Review this HR assistant answer.
Answer: {answer}

Is this answer complete and accurate for the question asked?
Reply with JSON only — no extra text:
{{"adequate": true, "missing": ""}}
or
{{"adequate": false, "missing": "brief description of what is missing"}}"""


def _self_critique(llm, answer: str) -> dict:
    """Returns dict with 'adequate' bool and 'missing' str. Fails safe."""
    try:
        res  = llm.invoke(_CRITIQUE_PROMPT.format(answer=answer[:800]))
        text = res.content.strip()
        text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
        return json.loads(text)
    except Exception:
        return {"adequate": True, "missing": ""}


# ── Intent inference for analytics ───────────────────────────────────────────

def _infer_intent(tools_called: list[str]) -> tuple[str, str]:
    """Returns (intent, topic) for analytics logging."""
    db_tool_set = {
        "get_profile", "get_leave_data", "get_salary_data",
        "get_performance_data", "get_training_data", "get_disciplinary_data",
    }
    has_policy = "retrieve_policy" in tools_called
    has_db     = bool(set(tools_called) & db_tool_set)

    if has_db and has_policy:
        intent = "hybrid"
    elif has_db:
        intent = "personal"
    else:
        intent = "policy"

    topic_map = {
        "get_leave_data":        "leave",
        "get_salary_data":       "salary",
        "get_performance_data":  "performance",
        "get_training_data":     "training",
        "get_disciplinary_data": "disciplinary",
        "get_profile":           "profile",
    }
    for t in tools_called:
        if t in topic_map:
            return intent, topic_map[t]

    return intent, "all" if has_db else "none"


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent(
    question: str,
    employee_id: int,
    lang: str,
    dialect: str | None,
    history_str: str,
    ar_index: tuple,
    en_index: tuple,
    llm,
    reranker,
    ara_tokenizer,
    max_iterations: int = 5,
) -> dict:
    """
    Run the agentic HR assistant for one question.

    Returns:
        answer        — final clean answer string
        docs          — all retrieved policy docs (for source evidence UI)
        cited_docs    — only the docs actually cited in the answer
        scores        — reranker scores keyed by doc id (for confidence badge)
        intent        — personal | hybrid | policy  (for analytics)
        topic         — leave | salary | ...         (for analytics)
        tools_called  — list of tool names called    (for Query Info expander)
        personal_data — formatted personal data str  (for data expander)
    """
    tools, retrieve_policy_ref = _make_tools(
        employee_id, ar_index, en_index, reranker, ara_tokenizer
    )
    tool_map       = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=_ORCHESTRATOR_SYSTEM),
        HumanMessage(content=question),
    ]

    tool_results_collected: dict[str, str] = {}
    tools_called:  list[str] = []
    top_docs:      list      = []
    scores_dict:   dict      = {}

    # ── Agentic loop ──────────────────────────────────────────────────────────
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc.get("args", {})
            tools_called.append(tool_name)

            try:
                fn         = tool_map[tool_name]
                result_str = fn.invoke(tool_args) if tool_args else fn.invoke({})

                if tool_name == "retrieve_policy":
                    top_docs    = getattr(retrieve_policy_ref, "_last_docs",   [])
                    scores_dict = getattr(retrieve_policy_ref, "_last_scores", {})
                    tool_results_collected["policy_context"] = result_str
                else:
                    key_map = {
                        "get_profile":           "profile",
                        "get_leave_data":        "leave_data",
                        "get_salary_data":       "salary_data",
                        "get_performance_data":  "performance_data",
                        "get_training_data":     "training_data",
                        "get_disciplinary_data": "disciplinary_data",
                    }
                    tool_results_collected[key_map.get(tool_name, tool_name)] = result_str

            except Exception as e:
                result_str = f"Error calling {tool_name}: {e}"

            messages.append(ToolMessage(
                tool_call_id=tc["id"],
                content=result_str,
            ))

    # ── Generate final answer ─────────────────────────────────────────────────
    raw_answer = _format_final_answer(
        llm, question, lang, dialect, tool_results_collected, history_str
    )

    # ── Post-process ──────────────────────────────────────────────────────────
    cited_pages = get_cited_pages(raw_answer)
    cited_docs  = filter_cited_chunks(top_docs, cited_pages)
    clean_ans   = strip_citations(raw_answer)
    has_db      = any(k in tool_results_collected for k in _DB_KEYS)
    is_personal = has_db and not tool_results_collected.get("policy_context")
    answer      = (clean_ans if is_personal
                   else validate(clean_ans, lang, has_citations=bool(cited_pages)))

    # ── Self-critique retry (once) ────────────────────────────────────────────
    if not is_no_info_answer(answer):
        critique = _self_critique(llm, answer)
        if not critique.get("adequate") and critique.get("missing"):
            retry_query = f"{question} {critique['missing']}"
            ar_vs, ar_bm25, ar_docs = ar_index
            en_vs, en_bm25, en_docs = en_index
            docs_ar2  = retrieve(retry_query, ar_vs, ar_bm25, ar_docs,
                                 lambda t: normalize_arabic(t, ara_tokenizer))
            docs_en2  = retrieve(retry_query, en_vs, en_bm25, en_docs, normalize_english)
            combined2 = rrf(docs_ar2, docs_en2)
            top2, scores2 = rerank(retry_query, combined2, reranker, top_n=10)

            if top2:
                tool_results_collected["policy_context"] = build_context(top2)
                raw2    = _format_final_answer(
                    llm, question, lang, dialect, tool_results_collected, history_str
                )
                cited2  = get_cited_pages(raw2)
                cdocs2  = filter_cited_chunks(top2, cited2)
                clean2  = strip_citations(raw2)
                answer2 = (clean2 if is_personal
                           else validate(clean2, lang, has_citations=bool(cited2)))

                if not is_no_info_answer(answer2):
                    answer      = answer2
                    top_docs    = top2
                    scores_dict = scores2
                    cited_docs  = cdocs2

    # ── Finalise ──────────────────────────────────────────────────────────────
    intent, topic     = _infer_intent(tools_called)
    personal_data_str = _build_personal_data_str(tool_results_collected)

    return {
        "answer":        answer,
        "docs":          top_docs,
        "cited_docs":    cited_docs,
        "scores":        scores_dict,
        "intent":        intent,
        "topic":         topic,
        "tools_called":  tools_called,
        "personal_data": personal_data_str,
    }