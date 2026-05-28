"""
run_hybrid_eval.py
==================
Evaluates hybrid answer generation for the Horizon Tech HR chatbot.

KEY DESIGN FIXES vs previous version:
  1. strip_citations(raw) is called inside run_agent before returning — so the
     final answer NEVER contains [Page N | AR/EN] citations. Checking for them
     was always going to give 0%. M2 is now pure LLM judge on policy correctness.

  2. M1 is split into two independent sub-metrics:
     M1a — retrieve_policy called  (did the system fetch policy context?)
     M1b — correct DB tool called  (did the planner also fetch personal data?)
     Both reported separately. The planner intentionally skips DB tools when
     policy context alone is enough — M1b reflects this design choice.

  3. M2 — Policy Correctness (LLM judge only, ar_llm for all languages)
     Checks that the answer correctly states the policy rule for the question.

  4. M3 — Personal Data Grounding (rule-based)
     Expected DB value present in answer. Same logic as personal eval.

Mirrors agent.py EXACTLY:
  - Same run_agent() call signature
  - Same detect_language_type() + get_semantic_dialect()
  - Same _KEY_MAP, format_personal_data() via get_full_personal_context()
  - skip_critique=True, history_str=""

Usage:
    python run_hybrid_eval.py --employee_id 42
    python run_hybrid_eval.py --employee_id 42 --lang EN --verbose
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from nlp_utils import detect_language_type, get_semantic_dialect


# =============================================================================
# M2 — POLICY CORRECTNESS (LLM judge, ar_llm for all languages)
# =============================================================================

def _policy_judge(ar_llm, question: str, answer: str,
                  policy_keyword: str, personal_data_str: str) -> dict:
    """
    Checks that the answer correctly states the policy rule relevant to the
    question. Language of answer does not affect score.
    """
    from agent import _invoke_with_retry, _strip_think_blocks

    prompt = (
        "You are evaluating the POLICY part of a hybrid HR chatbot answer.\n"
        "A hybrid answer should combine the employee's personal data AND a company policy rule.\n\n"
        "YOUR JOB: check only whether the POLICY information in the answer is correct.\n"
        "Do NOT penalise for missing personal data — focus only on the policy rule.\n\n"
        "SCORING RULES:\n"
        "- Score 1: answer states a correct and relevant policy rule for the question.\n"
        "- Score 1: answer uses policy to give a concrete answer "
        "(e.g. 'G3 employees get 60 days notice period').\n"
        "- Score 0: answer ignores policy, invents a rule, contradicts the policy, "
        "or says unavailable when policy exists.\n"
        "- Answer language does NOT affect score.\n\n"
        f"=== EMPLOYEE CONTEXT (reference only) ===\n{personal_data_str[:600]}\n\n"
        f"=== QUESTION ===\n{question}\n\n"
        f"=== POLICY TOPIC EXPECTED ===\n{policy_keyword}\n\n"
        f"=== CHATBOT ANSWER ===\n{answer[:800]}\n\n"
        "Reply with JSON only:\n"
        '{"score": 1, "reason": "one sentence"}\n'
        '{"score": 0, "reason": "one sentence"}\n'
    )
    try:
        res     = _invoke_with_retry(ar_llm, prompt)
        content = _strip_think_blocks(res.content).strip()
        content = re.sub(r"```json|```", "", content).strip()
        # Extract first valid JSON object even if there's trailing text
        m = re.search(r'\{[^{}]*"score"\s*:\s*[01][^{}]*\}', content)
        if m:
            content = m.group(0)
        parsed  = json.loads(content)
        return {"score": int(bool(parsed.get("score"))), "reason": parsed.get("reason", "")}
    except Exception as e:
        return {"score": 0, "reason": f"Judge call failed: {e}"}


# =============================================================================
# M3 — PERSONAL DATA GROUNDING (rule-based)
# =============================================================================

def _extract_personal_value(personal_data_str: str, grounding_field: str) -> str | None:
    if not personal_data_str:
        return None

    eligibility_patterns = {
        "bonus_eligible": r"Bonus eligible:\s*(YES|NO)",
        "promo_eligible": r"Promotion eligible:\s*(YES|NO)",
        "schol_eligible": r"Scholarship eligible:\s*(YES|NO)",
    }
    if grounding_field in eligibility_patterns:
        m = re.search(eligibility_patterns[grounding_field], personal_data_str, re.IGNORECASE)
        return m.group(1).upper() if m else None

    numeric_patterns = {
        "net_salary":           r"Net:\s*([\d,]+)",
        "gross_salary":         r"Gross:\s*([\d,]+)",
        "base_salary":          r"Base:\s*([\d,]+)",
        "budget_remaining_usd": r"\$([\d,]+(?:\.\d+)?)\s+remaining",
        "remaining_days":       r"remaining=([\d.]+)",
        "rating":               r"Rating:\s*(\d)/5",
    }
    if grounding_field in numeric_patterns:
        m = re.search(numeric_patterns[grounding_field], personal_data_str)
        return m.group(1).replace(",", "") if m else None

    text_patterns = {
        "grade":           r"Grade:\s*(\S+)",
        "work_model":      r"Work model:\s*(\S+)",
        "employment_type": r"Type:\s*(\S+)",
        "hire_date":       r"Hire date:\s*(\S+)",
        "manager_name":    r"Manager:\s*([^\n|]+)",
    }
    if grounding_field in text_patterns:
        m = re.search(text_patterns[grounding_field], personal_data_str, re.IGNORECASE)
        return m.group(1).strip() if m else None

    # job_title: extract from personal_data_str but check against answer more flexibly
    # since Arabic/EGY/FR answers translate the title rather than repeat it in English
    if grounding_field == "job_title":
        m = re.search(r"Title:\s*([^\n|]+)", personal_data_str, re.IGNORECASE)
        if not m:
            return None
        title = m.group(1).strip().lower()
        # Return the most distinctive word (last word of title) for partial matching
        # e.g. "senior software engineer" -> check if "engineer" OR full title in answer
        words = title.split()
        # Return tuple hint as pipe-separated so check_personal_grounding can try both
        return title  # full title for exact match attempt; fallback handled below

    # probation_status: format_personal_data emits either
    #   "Probation status: ACTIVE — ends 2024-06-30"  or  "Probation status: NOT IN PROBATION"
    # The full string never appears verbatim in an answer, so normalise to a canonical
    # token the answer will actually contain.
    if grounding_field == "probation_status":
        m = re.search(r"Probation status:\s*(.+)", personal_data_str, re.IGNORECASE)
        if not m:
            return None
        raw = m.group(1).strip().upper()
        if raw.startswith("ACTIVE"):
            return "probation"       # answer will contain "probation" or "on probation"
        return "not in probation"    # answer will contain "not in probation" or "completed"

    return None


def check_personal_grounding(answer: str, personal_data_str: str,
                               grounding_field: str) -> tuple[int, str]:
    expected = _extract_personal_value(personal_data_str, grounding_field)

    if expected is None:
        # Field not found in personal_data_str — could be a regex mismatch or missing data.
        # Return 0 with a clear label so it shows up in failures for debugging,
        # rather than silently granting a free pass that masks real gaps.
        return 0, "field_absent_in_db"

    norm_ans = answer.replace(",", "").lower()
    norm_exp = expected.replace(",", "").lower().strip()

    # job_title: accept partial match (last meaningful word) or Arabic equivalents
    if grounding_field == "job_title":
        # Full title match
        if norm_exp in norm_ans:
            return 1, f"expected='{norm_exp}' found=True"
        # Last word match (e.g. 'engineer' from 'senior software engineer')
        last_word = norm_exp.split()[-1] if norm_exp.split() else norm_exp
        if last_word and last_word in norm_ans:
            return 1, f"expected='{norm_exp}' found=True (partial '{last_word}')"
        # Arabic/Egyptian equivalents for common titles
        arabic_equivalents = {
            "engineer": ["مهندس", "engineer"],
            "manager":  ["مدير", "manager"],
            "specialist": ["متخصص", "specialist", "mota5ases"],
            "analyst":  ["محلل", "analyst"],
            "director": ["مدير", "director"],
            "lead":     ["قائد", "lead"],
        }
        for key, alts in arabic_equivalents.items():
            if key in norm_exp:
                if any(alt in answer.lower() for alt in alts):
                    return 1, f"expected='{norm_exp}' found=True (arabic equiv)"
        return 0, f"expected='{norm_exp}' found=False"

    # Eligibility verdicts
    if grounding_field in ("bonus_eligible", "promo_eligible", "schol_eligible"):
        if norm_exp == "yes":
            yes_words = ["yes", "eligible", "مؤهل", "مستحق", "aywa",
                         "you are eligible", "na3am", "mosta7e2", "تستحق"]
            score = 1 if any(w in norm_ans for w in yes_words) else 0
        else:
            no_words = ["not eligible", "not qualify", "لا يمكن", "غير مؤهل",
                        "mesh mosta7e2", "laa", "no,", "no —", "unfortunately",
                        "لست مؤهلاً", "مش مستحق", "لا تستحق", "mesh mosta7e2a"]
            score = 1 if any(w in norm_ans for w in no_words) else 0
        return score, f"eligibility expected={norm_exp} found={bool(score)}"

    # Numeric ±1 tolerance
    try:
        val   = float(norm_exp)
        check = set()
        for delta in (-1, 0, 1):
            check.add(str(int(round(val + delta))))
            check.add(f"{round(val + delta, 1)}")
        score = 1 if any(c in norm_ans for c in check) else 0
        return score, f"expected={norm_exp} found={bool(score)}"
    except ValueError:
        pass

    score = 1 if norm_exp in norm_ans else 0
    return score, f"expected='{norm_exp}' found={bool(score)}"


# =============================================================================
# SINGLE QUERY RUNNER
# =============================================================================

def evaluate_one(query: dict, employee_id: int,
                 personal_data_str: str,
                 ar_llm, routing_llm, en_llm, critique_llm,
                 ar_index, en_index, reranker, dialect_pipe, ara_tokenizer,
                 verbose: bool = False) -> dict:

    from agent import run_agent, _KEY_MAP

    question = query["query"]
    qid      = query["id"]

    lang    = detect_language_type(question)
    dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else None

    t0 = time.time()
    try:
        result = run_agent(
            question      = question,
            employee_id   = employee_id,
            lang          = lang,
            dialect       = dialect,
            history_str   = "",
            ar_index      = ar_index,
            en_index      = en_index,
            routing_llm   = routing_llm,
            en_llm        = en_llm,
            ar_llm        = ar_llm,
            critique_llm  = critique_llm,
            reranker      = reranker,
            ara_tokenizer = ara_tokenizer,
            dialect_pipe  = dialect_pipe,
            skip_critique = True,
        )
    except Exception as e:
        if verbose:
            print(f"\n  AGENT ERROR on {qid}: {e}")
        result = {"answer": "", "tools_called": [], "intent": "error"}

    elapsed      = round(time.time() - t0, 2)
    answer       = result.get("answer", "")
    tools_called = result.get("tools_called", [])
    intent       = result.get("intent", "")

    # ── M1a: retrieve_policy called ───────────────────────────────────────────
    m1a = 1 if "retrieve_policy" in tools_called else 0

    # ── M1b: at least one expected DB tool called ─────────────────────────────
    expected_db = [t for t in query["expected_tools"] if t != "retrieve_policy"]
    actual_db   = [t for t in tools_called if t in _KEY_MAP]
    m1b         = 1 if any(t in actual_db for t in expected_db) else 0

    m1_detail = (f"retrieve_policy={'YES' if m1a else 'NO'} | "
                 f"expected_db={expected_db} | called_db={actual_db}")

    # ── M2: Policy Correctness (LLM judge) ───────────────────────────────────
    if answer.strip():
        judge     = _policy_judge(ar_llm, question, answer,
                                  query["policy_keyword"], personal_data_str)
        m2        = judge["score"]
        m2_detail = judge["reason"]
    else:
        m2, m2_detail = 0, "empty_answer"

    # ── M3: Personal Data Grounding (rule-based) ─────────────────────────────
    m3, m3_detail = check_personal_grounding(
        answer            = answer,
        personal_data_str = personal_data_str,
        grounding_field   = query["grounding_field"],
    )

    row = {
        "id":              qid,
        "language":        qid.split("-")[1],
        "topic":           query["topic"],
        "query":           question,
        "answer_snippet":  answer[:300],
        "M1a_policy_retrieved": m1a,
        "M1b_db_tool_called":   m1b,
        "M2_policy_correct":    m2,
        "M3_data_grounding":    m3,
        "m1_detail":  m1_detail,
        "m2_detail":  m2_detail,
        "m3_detail":  m3_detail,
        "intent_detected":  intent,
        "lang_detected":    lang,
        "tools_called":     tools_called,
        "response_time_s":  elapsed,
    }

    if verbose:
        total = m1a + m1b + m2 + m3
        icon  = "✓" if total == 4 else ("~" if total >= 2 else "✗")
        print(f"  {icon} {qid:14s}  M1a={m1a} M1b={m1b} M2={m2} M3={m3}  "
              f"intent={intent:8s}  ({elapsed}s)")
        if m1a == 0 or m1b == 0:
            print(f"         tool:   {m1_detail}")
        if m2 == 0:
            print(f"         policy: {m2_detail[:100]}")
        if m3 == 0:
            print(f"         data:   {m3_detail}")

    return row


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(results: list[dict]) -> dict:

    def _agg(rows):
        n = len(rows)
        if not n:
            return {}
        return {
            "n": n,
            "M1a_policy_retrieved_%": round(sum(r["M1a_policy_retrieved"] for r in rows) / n * 100, 1),
            "M1b_db_tool_called_%":   round(sum(r["M1b_db_tool_called"]   for r in rows) / n * 100, 1),
            "M2_policy_correct_%":    round(sum(r["M2_policy_correct"]    for r in rows) / n * 100, 1),
            "M3_data_grounding_%":    round(sum(r["M3_data_grounding"]    for r in rows) / n * 100, 1),
        }

    by_lang  = defaultdict(list)
    by_topic = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
        by_topic[r["topic"]].append(r)

    return {
        "overall":     _agg(results),
        "by_language": {k: _agg(v) for k, v in by_lang.items()},
        "by_topic":    {k: _agg(v) for k, v in by_topic.items()},
    }


# =============================================================================
# REPORT
# =============================================================================

def print_report(agg: dict, results: list[dict]):
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  HYBRID EVALUATION  —  Horizon Tech HR Chatbot")
    print(SEP)

    o = agg["overall"]
    print(f"\nOVERALL  (n={o['n']})")
    print(f"  M1a  Policy Retrieved       : {o['M1a_policy_retrieved_%']:>6.1f}%  "
          "(retrieve_policy called)")
    print(f"  M1b  DB Tool Called         : {o['M1b_db_tool_called_%']:>6.1f}%  "
          "(personal data fetched by planner)")
    print(f"  M2   Policy Correctness     : {o['M2_policy_correct_%']:>6.1f}%  "
          "(LLM judge)")
    print(f"  M3   Personal Data Grounding: {o['M3_data_grounding_%']:>6.1f}%  "
          "(rule-based)")

    print(f"\nBY LANGUAGE")
    hdr = (f"  {'Lang':<6}  {'n':>3}  {'M1a Pol%':>9}  "
           f"{'M1b DB%':>8}  {'M2 Correct%':>12}  {'M3 Data%':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lang, s in agg["by_language"].items():
        print(f"  {lang:<6}  {s['n']:>3}  "
              f"{s['M1a_policy_retrieved_%']:>8.1f}%  "
              f"{s['M1b_db_tool_called_%']:>7.1f}%  "
              f"{s['M2_policy_correct_%']:>11.1f}%  "
              f"{s['M3_data_grounding_%']:>8.1f}%")

    print(f"\nBY TOPIC")
    hdr2 = (f"  {'Topic':<16}  {'n':>3}  {'M1a':>6}  "
            f"{'M1b':>6}  {'M2':>6}  {'M3':>6}")
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for topic, s in agg["by_topic"].items():
        print(f"  {topic:<16}  {s['n']:>3}  "
              f"{s['M1a_policy_retrieved_%']:>5.0f}%  "
              f"{s['M1b_db_tool_called_%']:>5.0f}%  "
              f"{s['M2_policy_correct_%']:>5.0f}%  "
              f"{s['M3_data_grounding_%']:>5.0f}%")

    # Intent routing breakdown
    intent_counts = defaultdict(int)
    for r in results:
        intent_counts[r["intent_detected"]] += 1
    print(f"\nINTENT ROUTING BREAKDOWN  (n={len(results)})")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = "█" * int(pct / 5)
        print(f"  {intent:<14}  {count:>3} ({pct:>5.1f}%)  {bar}")

    # Misrouted
    misrouted = [r for r in results if r["intent_detected"] != "hybrid"]
    if misrouted:
        print(f"\nMIS-ROUTED  (intent != hybrid)  — {len(misrouted)}")
        for r in misrouted:
            print(f"  {r['id']:16s}  intent={r['intent_detected']:14s}  "
                  f"tools={r['tools_called']}")

    # Failures
    failures = [r for r in results
                if r["M1a_policy_retrieved"] + r["M2_policy_correct"] + r["M3_data_grounding"] < 2]
    if failures:
        print(f"\nFAILED  (M1a+M2+M3 < 2)  — {len(failures)}")
        for r in failures:
            print(f"  {r['id']:16s}  "
                  f"M1a={r['M1a_policy_retrieved']} M1b={r['M1b_db_tool_called']} "
                  f"M2={r['M2_policy_correct']} M3={r['M3_data_grounding']}  "
                  f"intent={r['intent_detected']}")
            if r["M2_policy_correct"] == 0:
                print(f"             policy: {r['m2_detail'][:100]}")
            if r["M3_data_grounding"] == 0:
                print(f"             data:   {r['m3_detail']}")

    print(f"\n{SEP}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid eval — Horizon Tech HR chatbot")
    parser.add_argument("--employee_id", type=int, required=True)
    parser.add_argument("--benchmark",   default="hybrid_eval_benchmark.json")
    parser.add_argument("--lang",        choices=["EN", "AR", "EGY", "FR"])
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    path = Path(args.benchmark)
    if not path.exists():
        print(f"ERROR: benchmark not found at {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        bench = json.load(f)

    lang_key_map = {
        "EN":  "english",
        "AR":  "arabic_msa",
        "EGY": "egyptian_arabic",
        "FR":  "franco_arabic",
    }

    all_queries = []
    for lang_key, queries in bench["queries"].items():
        if args.lang and lang_key != lang_key_map[args.lang]:
            continue
        all_queries.extend(queries)

    print(f"\nLoaded {len(all_queries)} queries  |  employee_id={args.employee_id}")

    print("Running setup()…")
    try:
        from setup import setup
        (ar_index, en_index,
         routing_llm, en_llm, ar_llm, critique_llm,
         reranker, dialect_pipe, ara_tokenizer) = setup()
        print("Setup complete.")
    except Exception as e:
        print(f"ERROR in setup(): {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching employee data for employee_id={args.employee_id}…")
    try:
        from personal_data import get_full_personal_context
        from personal_prompts import format_personal_data
        personal_context  = get_full_personal_context(args.employee_id)
        personal_data_str = format_personal_data(personal_context)
        print(f"personal_data_str: {len(personal_data_str)} chars\n")
        if len(personal_data_str) < 50:
            print("WARNING: very short — check employee_id exists in DB.")
    except Exception as e:
        print(f"ERROR fetching employee data: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total   = len(all_queries)

    for i, query in enumerate(all_queries, 1):
        print(f"[{i:3}/{total}] {query['id']}", end="  ", flush=True)

        row = evaluate_one(
            query             = query,
            employee_id       = args.employee_id,
            personal_data_str = personal_data_str,
            ar_llm            = ar_llm,
            routing_llm       = routing_llm,
            en_llm            = en_llm,
            critique_llm      = critique_llm,
            ar_index          = ar_index,
            en_index          = en_index,
            reranker          = reranker,
            dialect_pipe      = dialect_pipe,
            ara_tokenizer     = ara_tokenizer,
            verbose           = args.verbose,
        )
        results.append(row)

        if not args.verbose:
            s    = row["M1a_policy_retrieved"] + row["M2_policy_correct"] + row["M3_data_grounding"]
            icon = "✓" if s == 3 else ("~" if s == 2 else "✗")
            print(f"{icon}  M1a={row['M1a_policy_retrieved']} "
                  f"M1b={row['M1b_db_tool_called']} "
                  f"M2={row['M2_policy_correct']} "
                  f"M3={row['M3_data_grounding']}  "
                  f"intent={row['intent_detected']:8s}  ({row['response_time_s']}s)")

    agg = aggregate(results)
    print_report(agg, results)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = f"hybrid_eval_results_{ts}.json"
    txt_out  = f"hybrid_eval_report_{ts}.txt"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"aggregates": agg, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"Results → {json_out}")

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report(agg, results)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Report  → {txt_out}")


if __name__ == "__main__":
    main()