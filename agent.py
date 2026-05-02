"""
agent.py — Agentic HR Assistant

Token-efficient design:
  routing call  (8b):   ~300 tokens  — 14,400 RPD quota
  answer call   (70b):  ~2,000 tokens — 1,000 RPD quota (English only)
  answer call   (qwen): ~2,000 tokens — separate quota (Arabic/Franco)
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
from nlp_utils import normalize_arabic, normalize_english, detect_language_type, franco_to_arabic
from utils import (
    build_context, is_no_info_answer, validate,
    get_cited_pages, strip_citations, filter_cited_chunks, ARABIC_PDF_PATH,
)
from prompts import english_prompt, msa_prompt, egy_prompt, franco_prompt
from personal_prompts import get_personal_prompt, get_hybrid_prompt, format_personal_data


_ORCHESTRATOR_SYSTEM = """You are the routing layer of an HR chatbot for Horizon Tech.
Call the right tools for the employee's question, then stop. Do not write an answer.

DECISION RULE:
Ask: "Is the answer the same for every employee at this company?"
  YES → call retrieve_policy only. Never add personal data tools.
  NO  → call the relevant personal tool(s). Also call retrieve_policy if you need policy to check eligibility.

Tool descriptions tell you exactly what each returns. Use them to pick the right tool.
Language (Arabic, Franco, English) does not change the routing logic."""

_DB_KEYS = ["profile", "leave_data", "salary_data", "performance_data", "training_data", "disciplinary_data"]

# Context truncation — keeps answer prompt within ~2,000 tokens
_MAX_CHUNK_CHARS   = 800
_MAX_CONTEXT_CHUNKS = 5


def _build_context_truncated(docs: list) -> str:
    sorted_docs = sorted(
        docs[:_MAX_CONTEXT_CHUNKS],
        key=lambda d: (d.metadata.get("source", ""), d.metadata.get("page", 0))
    )
    out = []
    for d in sorted_docs:
        page_num = d.metadata.get("page", 0) + 1
        lang_tag = "AR" if ARABIC_PDF_PATH in d.metadata.get("source", "") else "EN"
        content  = d.page_content[:_MAX_CHUNK_CHARS]
        if len(d.page_content) > _MAX_CHUNK_CHARS:
            content += "…"
        out.append(f"[Page {page_num} | {lang_tag}]\n{content}")
    return "\n\n---\n\n".join(out)


def _strip_qwen_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"\s{3,}", "\n\n", text).strip()


def _invoke_with_retry(llm, prompt, max_retries=3):
    """Exponential backoff on 429 rate limit errors."""
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


def _make_tools(employee_id, ar_index, en_index, reranker, ara_tokenizer):

    @tool
    def retrieve_policy(query: str) -> str:
        """Search Horizon Tech HR policy PDFs (Arabic + English).
        Use when the answer is a company rule, rate, procedure, or entitlement
        that is the same for every employee — e.g. leave policy, overtime rate,
        promotion criteria, disciplinary steps, data retention periods."""
        ar_vs, ar_bm25, ar_docs = ar_index
        en_vs, en_bm25, en_docs = en_index
        q_lang   = detect_language_type(query)
        ar_query = franco_to_arabic(query) if q_lang == "franco" else query
        
        # Retrieve from both indexes with higher k to avoid missing relevant chunks
        docs_ar  = retrieve(ar_query, ar_vs, ar_bm25, ar_docs, lambda t: normalize_arabic(t, ara_tokenizer), k=15)
        docs_en  = retrieve(query, en_vs, en_bm25, en_docs, normalize_english, k=15)
        
        # Merge with RRF to combine signals
        combined = rrf(docs_ar, docs_en)
        
        # Rerank with more candidates to ensure quality
        # top_n increased from 4 to 6 to capture more relevant context
        top_docs, scores = rerank(query, combined, reranker, top_n=6)
        
        if not top_docs:
            return "No relevant policy found."
        retrieve_policy._last_docs   = top_docs
        retrieve_policy._last_scores = scores
        return _build_context_truncated(top_docs)

    @tool
    def get_profile() -> str:
        """Returns THIS employee's personal profile: name, grade, department,
        hire date, employment type, work model (remote/hybrid/in-office),
        and probation status. Use when the question is about the employee's
        own identity, status, or role attributes."""
        data = get_employee_profile(employee_id)
        if not data: return "Profile not found."
        return json.dumps(data, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_leave_data() -> str:
        """Returns THIS employee's leave balances (remaining days per leave type
        for the current year) and their recent and pending leave requests.
        Use when the question asks about their specific remaining leave days
        or leave request history — not about the general leave policy."""
        today = date.today()
        return json.dumps({
            "leave_balances":   get_leave_balance(employee_id, today.year),
            "pending_requests": get_pending_leave(employee_id),
            "recent_requests":  get_leave_requests(employee_id, limit=3),
        }, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_salary_data() -> str:
        """Returns THIS employee's latest salary breakdown: net salary, gross,
        base, allowances, and deductions. Use when the question asks about
        their specific salary amount or components."""
        return json.dumps({
            "latest_salary":  get_latest_salary(employee_id),
            "salary_history": get_payroll_history(employee_id, months=3),
        }, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_performance_data() -> str:
        """Returns THIS employee's performance review history, latest rating,
        performance trend, bonus multiplier, and current OKRs with progress.
        Use when the question asks about their actual rating, review, or OKR status."""
        return json.dumps({
            "latest_review":     get_latest_review(employee_id),
            "performance_trend": get_performance_trend(employee_id),
            "okrs":              get_okrs(employee_id),
            "history":           get_performance_history(employee_id, limit=2),
        }, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_training_data() -> str:
        """Returns THIS employee's training budget (total, used, remaining in USD),
        training days used/remaining, and courses completed this year."""
        data = get_training_record(employee_id, date.today().year)
        return json.dumps(data or {}, default=str, ensure_ascii=False, indent=2)

    @tool
    def get_disciplinary_data() -> str:
        """Returns THIS employee's active disciplinary records: warnings and PIP.
        Use when checking if they have active disciplinary actions,
        or when checking bonus/promotion eligibility (PIP disqualifies)."""
        data = get_active_disciplinary(employee_id)
        return json.dumps(data, default=str, ensure_ascii=False, indent=2)

    tools = [retrieve_policy, get_profile, get_leave_data, get_salary_data,
             get_performance_data, get_training_data, get_disciplinary_data]
    return tools, retrieve_policy


def _merge_db(tool_results):
    combined = {}
    for key in _DB_KEYS:
        if key in tool_results:
            try: combined.update(json.loads(tool_results[key]))
            except: pass
    return combined


def _pick_llm(lang, en_llm, ar_llm):
    return en_llm if lang == "english" else ar_llm


def _format_answer(lang, dialect, question, tool_results, history_str, en_llm, ar_llm, top_docs):
    has_policy = bool(tool_results.get("policy_context"))
    has_db     = any(k in tool_results for k in _DB_KEYS)
    llm        = _pick_llm(lang, en_llm, ar_llm)
    ctx        = _build_context_truncated(top_docs) if top_docs else tool_results.get("policy_context", "")

    if has_db and has_policy:
        res = _invoke_with_retry(llm, get_hybrid_prompt(lang, dialect).format(
            personal_data=format_personal_data(_merge_db(tool_results)),
            policy_context=ctx, question=question, history=history_str,
        ))
    elif has_db:
        res = _invoke_with_retry(llm, get_personal_prompt(lang, dialect).format(
            personal_data=format_personal_data(_merge_db(tool_results)),
            question=question, history=history_str,
        ))
    else:
        if not ctx:
            return "This information is not available in the policy documents."
        prompt = (english_prompt if lang == "english" else
                  franco_prompt  if lang == "franco"  else
                  egy_prompt     if dialect == "egyptian" else msa_prompt)
        res = _invoke_with_retry(llm, prompt.format(context=ctx, question=question, history=history_str))

    return _strip_qwen_thinking(res.content)


_CRITIQUE_PROMPT = """Is this HR answer complete?
Answer: {answer}
JSON only: {{"adequate": true, "missing": ""}} or {{"adequate": false, "missing": "what"}}"""


def _critique(critique_llm, answer):
    try:
        res  = _invoke_with_retry(critique_llm, _CRITIQUE_PROMPT.format(answer=answer[:400]))
        text = re.sub(r"^```json|^```|```$", "", _strip_qwen_thinking(res.content), flags=re.MULTILINE).strip()
        return json.loads(text)
    except:
        return {"adequate": True, "missing": ""}


def _infer_intent(tools_called):
    db_set     = {"get_profile","get_leave_data","get_salary_data","get_performance_data","get_training_data","get_disciplinary_data"}
    has_policy = "retrieve_policy" in tools_called
    has_db     = bool(set(tools_called) & db_set)
    intent     = "hybrid" if has_db and has_policy else ("personal" if has_db else "policy")
    topic_map  = {"get_leave_data":"leave","get_salary_data":"salary","get_performance_data":"performance",
                  "get_training_data":"training","get_disciplinary_data":"disciplinary","get_profile":"profile"}
    for t in tools_called:
        if t in topic_map: return intent, topic_map[t]
    return intent, ("all" if has_db else "none")


def run_agent(
    question, employee_id, lang, dialect, history_str,
    ar_index, en_index, routing_llm, en_llm, ar_llm, critique_llm,
    reranker, ara_tokenizer,
    max_iterations=3, skip_critique=True,   # skip_critique=True by default saves tokens
):
    tools, retrieve_policy_ref = _make_tools(employee_id, ar_index, en_index, reranker, ara_tokenizer)
    tool_map       = {t.name: t for t in tools}
    llm_with_tools = routing_llm.bind_tools(tools)

    messages = [SystemMessage(content=_ORCHESTRATOR_SYSTEM), HumanMessage(content=question)]

    tool_results: dict = {}
    tools_called: list = []
    top_docs:     list = []
    scores_dict:  dict = {}

    for _ in range(max_iterations):
        response = _invoke_with_retry(llm_with_tools, messages)
        messages.append(response)
        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            tname = tc["name"]
            targs = tc.get("args", {})
            tools_called.append(tname)
            try:
                fn     = tool_map[tname]
                result = fn.invoke(targs) if targs else fn.invoke({})
                if tname == "retrieve_policy":
                    top_docs    = getattr(retrieve_policy_ref, "_last_docs",   [])
                    scores_dict = getattr(retrieve_policy_ref, "_last_scores", {})
                    tool_results["policy_context"] = result
                else:
                    km = {"get_profile":"profile","get_leave_data":"leave_data","get_salary_data":"salary_data",
                          "get_performance_data":"performance_data","get_training_data":"training_data",
                          "get_disciplinary_data":"disciplinary_data"}
                    tool_results[km.get(tname, tname)] = result
            except Exception as e:
                result = f"Error: {e}"
            messages.append(ToolMessage(tool_call_id=tc["id"], content=result))

    raw    = _format_answer(lang, dialect, question, tool_results, history_str, en_llm, ar_llm, top_docs)
    cited  = get_cited_pages(raw)
    cdocs  = filter_cited_chunks(top_docs, cited)
    clean  = strip_citations(raw)
    db_only = any(k in tool_results for k in _DB_KEYS) and not tool_results.get("policy_context")
    answer = clean if db_only else validate(clean, lang, has_citations=bool(cited))

    if not skip_critique and not is_no_info_answer(answer):
        c = _critique(critique_llm, answer)
        if not c.get("adequate") and c.get("missing"):
            rq = f"{question} {c['missing']}"
            av, bv, adocs = ar_index
            ev, bv2, edocs = en_index
            da2 = retrieve(rq, av, bv, adocs, lambda t: normalize_arabic(t, ara_tokenizer))
            de2 = retrieve(rq, ev, bv2, edocs, normalize_english)
            t2, s2 = rerank(rq, rrf(da2, de2), reranker, top_n=4)
            if t2:
                tool_results["policy_context"] = _build_context_truncated(t2)
                r2   = _format_answer(lang, dialect, question, tool_results, history_str, en_llm, ar_llm, t2)
                c2   = get_cited_pages(r2)
                cd2  = filter_cited_chunks(t2, c2)
                cl2  = strip_citations(r2)
                a2   = cl2 if db_only else validate(cl2, lang, has_citations=bool(c2))
                if not is_no_info_answer(a2):
                    answer, top_docs, scores_dict, cdocs = a2, t2, s2, cd2

    intent, topic = _infer_intent(tools_called)
    pdata = (_build_personal_data_str(tool_results) if intent in ("personal","hybrid") else "")

    return {
        "answer": answer, "docs": top_docs, "cited_docs": cdocs,
        "scores": scores_dict, "intent": intent, "topic": topic,
        "tools_called": tools_called, "personal_data": pdata,
    }


def _build_personal_data_str(tool_results):
    combined = _merge_db(tool_results)
    if not combined: return ""
    try: return format_personal_data(combined)
    except: return ""