"""
run_personal_eval.py
====================
Evaluates personal-intent query answering for the Horizon Tech HR chatbot.

Key design decisions:
  - M2 judge: ar_llm (qwen3-32b) for ALL languages.
    Reason: llama-70b was far stricter than qwen on identical correct answers,
    causing EN=10% while AR=90% on the same data. qwen3-32b reads all four
    languages (EN, AR, EGY, Franco) and gives consistent judgements.
  - personal_data_str: fetched once via get_full_personal_context() +
    format_personal_data() to avoid the _merge_db() key-flattening bug.
  - M2 judge prompt: explicit about what counts as correct — the answer must
    contain the right value, even if phrased differently or in another language.

Three metrics:
  M1 — Tool Selection Accuracy  (rule-based)
  M2 — Answer Correctness       (LLM-as-judge, ar_llm for all languages)
  M3 — Data Grounding           (rule-based)

Usage:
    python run_personal_eval.py --employee_id 5
    python run_personal_eval.py --employee_id 5 --lang FR --verbose
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# =============================================================================
# M2 — LLM-AS-JUDGE
# Single model (ar_llm = qwen3-32b) for all languages.
# Reads English, Arabic MSA, Egyptian Arabic, and Franco equally well.
# =============================================================================

_JUDGE_PROMPT = """\
You are evaluating an HR chatbot answer. Your job is to check if the answer \
is factually correct based on the employee data.

SCORING RULES:
- Score 1 if the answer contains the correct value(s) from the employee data, \
even if phrased differently, translated, or in a different language/dialect.
- Score 1 if the question asks about absence of something (e.g. no pending leaves, \
no disciplinary actions) and the answer correctly states there are none.
- Score 0 if the answer gives a wrong value, avoids the question, says \
"information not available" when data exists, or is too vague to be useful.
- The answer language does NOT affect the score. An Arabic answer to an English \
question is fine if it is factually correct.

=== EMPLOYEE DATA ===
{personal_data}

=== QUESTION ===
{question}

=== CHATBOT ANSWER ===
{answer}

Reply with JSON only — no markdown, no explanation outside JSON:
{{"score": 1, "reason": "one sentence explaining why correct"}}
{{"score": 0, "reason": "one sentence explaining what is wrong or missing"}}
"""


def _llm_judge(ar_llm, question: str, answer: str, personal_data_str: str) -> dict:
    """Returns {"score": 0|1, "reason": str}. Uses ar_llm for all languages."""
    from agent import _invoke_with_retry, _strip_think_blocks

    prompt = _JUDGE_PROMPT.format(
        personal_data = personal_data_str[:1500],
        question      = question,
        answer        = answer[:800],
    )
    try:
        res     = _invoke_with_retry(ar_llm, prompt)
        content = _strip_think_blocks(res.content).strip()
        content = re.sub(r"```json|```", "", content).strip()
        parsed  = json.loads(content)
        return {"score": int(bool(parsed.get("score"))), "reason": parsed.get("reason", "")}
    except Exception as e:
        return {"score": 0, "reason": f"Judge call failed: {e}"}


# =============================================================================
# M3 — DATA GROUNDING
# =============================================================================

_TEXT_GROUNDING_FIELDS = {
    "probation_status", "work_model", "manager_name",
    "job_title", "hire_date", "trend_direction",
    "pending_leaves", "active_disciplinary", "courses",
    "employment_type", "grade",
}


def _extract_leave_field(personal_data_str: str, field: str,
                          grounding_filter: dict | None) -> str | None:
    """Extract used_days / carried_over_days / entitled_days from a specific leave type line."""
    leave_type = (grounding_filter or {}).get("leave_type", "").upper()
    field_key = {
        "used_days":          "used",
        "carried_over_days":  "carried",
        "entitled_days":      "entitled",
    }.get(field)
    if not field_key:
        return None
    for line in personal_data_str.splitlines():
        if (not leave_type or leave_type in line.upper()) and f"{field_key}=" in line:
            m = re.search(rf"{field_key}=([\d.]+)", line)
            if m:
                return m.group(1)
    return None


def _extract_expected_value(personal_data_str: str,
                             grounding_field: str,
                             grounding_filter: dict | None) -> str | None:
    if not personal_data_str:
        return None

    if grounding_field == "remaining_days":
        leave_type = (grounding_filter or {}).get("leave_type", "").upper()
        for line in personal_data_str.splitlines():
            if (not leave_type or leave_type in line.upper()) and "remaining=" in line:
                m = re.search(r"remaining=([\d.]+)", line)
                if m:
                    return m.group(1)
        return None

    if grounding_field in ("used_days", "carried_over_days", "entitled_days"):
        return _extract_leave_field(personal_data_str, grounding_field, grounding_filter)

    if grounding_field == "pending_leaves":
        return "PENDING" if "PENDING LEAVE REQUESTS" in personal_data_str else "no_pending"

    if grounding_field == "active_disciplinary":
        return "ACTIVE DISCIPLINARY" if "ACTIVE DISCIPLINARY" in personal_data_str else "no_disciplinary"

    if grounding_field == "courses":
        in_training = False
        for line in personal_data_str.splitlines():
            if "TRAINING" in line:
                in_training = True
            if in_training and line.strip().startswith("•"):
                return line.strip()[1:].strip()[:40]
        return None

    numeric_patterns = {
        "net_salary":            r"Net:\s*([\d,]+)",
        "gross_salary":          r"Gross:\s*([\d,]+)",
        "base_salary":           r"Base:\s*([\d,]+)",
        "transport_allowance":   r"Transport=([\d,]+)",
        "remote_allowance":      r"Remote=([\d,]+)",
        "income_tax":            r"Tax=([\d,]+)",
        "social_insurance":      r"Social ins\.=([\d,]+)",
        "salary_increment_pct":  r"Salary increment:\s*([\d.]+)%",
        "bonus_multiplier":      r"Bonus multiplier:\s*([\d.]+)x",
        # budget_used_usd: match "$X,XXX used" in the TRAINING section only
        # Use a tighter pattern anchored to the training budget line format
        "budget_used_usd":       r"\$([\d,]+(?:\.\d+)?)\s+used\s+\|",
        "budget_remaining_usd":  r"\$([\d,]+(?:\.\d+)?)\s+remaining",
        "budget_total_usd":      r"\$([\d,]+(?:\.\d+)?)\s+total",
        # training_days_used: "Days: 8 total | 2 used | 6 remaining"
        # Match the days line specifically (starts with "Days:")
        "training_days_used":    r"Days:\s*\d+\s+total\s+\|\s*(\d+)\s+used",
        "rating":                r"Rating:\s*(\d)/5",
        "overall_progress_pct":  r"—\s*(\d+)%",
    }
    if grounding_field in numeric_patterns:
        m = re.search(numeric_patterns[grounding_field], personal_data_str)
        return m.group(1).replace(",", "") if m else None

    text_patterns = {
        "probation_status":  r"Probation status:\s*(.+)",
        "work_model":        r"Work model:\s*(\S+)",
        "manager_name":      r"Manager:\s*([^\n|]+)",
        "job_title":         r"Title:\s*([^\n|]+)",
        "department":        r"Dept:\s*([^\n|]+)",
        "hire_date":         r"Hire date:\s*(\S+)",
        "trend_direction":   r"Direction:\s*(\S+)",
        "employment_type":   r"Type:\s*([^\n|]+)",
        "grade":             r"Grade:\s*(\S+)",
    }
    if grounding_field in text_patterns:
        m = re.search(text_patterns[grounding_field], personal_data_str, re.IGNORECASE)
        return m.group(1).strip() if m else None

    return None


def check_data_grounding(answer: str, personal_data_str: str,
                          grounding_field: str,
                          grounding_filter: dict | None) -> tuple[int, str]:
    expected = _extract_expected_value(personal_data_str, grounding_field, grounding_filter)

    if expected is None:
        return 1, "field_absent_in_db"

    norm_ans = answer.replace(",", "").lower()
    norm_exp = expected.replace(",", "").lower().strip()

    if grounding_field == "pending_leaves":
        if norm_exp == "no_pending":
            no_words = ["no pending", "none", "ليس", "لا توجد", "mafish", "mafesh",
                        "no requests", "لا يوجد", "no active", "لا طلبات",
                        "you have no", "you don't have", "mesh 3andak", "mafeesh"]
            score = 1 if any(w in norm_ans for w in no_words) else 0
        else:
            score = 1 if any(w in norm_ans for w in
                             ["pending", "طلب", "request", "agaza", "إجازة"]) else 0
        return score, f"pending_check={'pass' if score else 'fail'}"

    if grounding_field == "active_disciplinary":
        if norm_exp == "no_disciplinary":
            no_words = ["no active", "no disciplinary", "none", "ليس", "لا توجد",
                        "mafish", "no actions", "لا يوجد", "لا توجد إجراءات",
                        "you have no", "you don't have", "no warnings", "clean record",
                        "no current", "mesh 3andak"]
            score = 1 if any(w in norm_ans for w in no_words) else 0
        else:
            score = 1 if any(w in norm_ans for w in
                             ["disciplinary", "warning", "إنذار", "تأديبي", "pip"]) else 0
        return score, f"disciplinary_check={'pass' if score else 'fail'}"

    if grounding_field == "courses":
        first_word = norm_exp.split()[0] if norm_exp.split() else norm_exp
        score = 1 if first_word in norm_ans else 0
        return score, f"course_word='{first_word}' found={bool(score)}"

    if grounding_field in _TEXT_GROUNDING_FIELDS:
        score = 1 if norm_exp in norm_ans else 0
        return score, f"expected='{norm_exp}' found={bool(score)}"

    try:
        val   = float(norm_exp)
        check = set()
        for delta in (-1, 0, 1):
            check.add(str(int(round(val + delta))))
            check.add(f"{round(val + delta, 1)}")
        score = 1 if any(c in norm_ans for c in check) else 0
        return score, f"expected={norm_exp} found={bool(score)}"
    except ValueError:
        score = 1 if norm_exp in norm_ans else 0
        return score, f"expected='{norm_exp}' found={bool(score)}"


# =============================================================================
# SINGLE QUERY RUNNER
# =============================================================================

def evaluate_one(query: dict, employee_id: int,
                 personal_data_str: str,
                 ar_llm,                          # judge for ALL languages
                 routing_llm, en_llm, critique_llm,
                 ar_index, en_index, reranker, dialect_pipe, ara_tokenizer,
                 verbose: bool = False) -> dict:

    from agent import run_agent, _KEY_MAP
    from nlp_utils import detect_language_type, get_semantic_dialect

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
        result = {"answer": "", "tools_called": [], "personal_data": "",
                  "intent": "error", "topic": "none"}

    elapsed      = round(time.time() - t0, 2)
    answer       = result.get("answer", "")
    tools_called = result.get("tools_called", [])

    # ── M1 ────────────────────────────────────────────────────────────────────
    actual_db = [t for t in tools_called if t in _KEY_MAP]
    expected  = query["expected_tool"]
    m1        = 1 if expected in actual_db else 0
    m1_detail = f"expected={expected} | called={actual_db}"

    # ── M2 — same judge (ar_llm) for all languages ────────────────────────────
    if answer.strip():
        judge     = _llm_judge(ar_llm, question, answer, personal_data_str)
        m2        = judge["score"]
        m2_detail = judge["reason"]
    else:
        m2, m2_detail = 0, "empty_answer"

    # ── M3 ────────────────────────────────────────────────────────────────────
    m3, m3_detail = check_data_grounding(
        answer            = answer,
        personal_data_str = personal_data_str,
        grounding_field   = query["grounding_field"],
        grounding_filter  = query.get("grounding_filter"),
    )

    row = {
        "id":               qid,
        "language":         qid.split("-")[0],
        "topic":            query["topic"],
        "query":            question,
        "answer_snippet":   answer[:250],
        "M1_tool_accuracy":  m1,
        "M2_answer_correct": m2,
        "M3_data_grounding": m3,
        "m1_detail":        m1_detail,
        "m2_detail":        m2_detail,
        "m3_detail":        m3_detail,
        "response_time_s":  elapsed,
        "intent_detected":  result.get("intent", ""),
        "lang_detected":    lang,
        "dialect_detected": dialect or "",
        "tools_called":     tools_called,
    }

    if verbose:
        total = m1 + m2 + m3
        icon  = "✓" if total == 3 else ("~" if total == 2 else "✗")
        print(f"  {icon} {qid:10s}  M1={m1} M2={m2} M3={m3}  lang={lang}  {elapsed}s")
        if m1 == 0:
            print(f"         tool:      {m1_detail}")
        if m2 == 0:
            print(f"         judge:     {m2_detail[:100]}")
        if m3 == 0:
            print(f"         grounding: {m3_detail}")

    return row


# =============================================================================
# AGGREGATION + REPORT
# =============================================================================

def aggregate(results: list[dict]) -> dict:
    def _agg(rows):
        n = len(rows)
        if not n:
            return {}
        return {
            "n": n,
            "M1_tool_accuracy_%":  round(sum(r["M1_tool_accuracy"]  for r in rows) / n * 100, 1),
            "M2_answer_correct_%": round(sum(r["M2_answer_correct"] for r in rows) / n * 100, 1),
            "M3_data_grounding_%": round(sum(r["M3_data_grounding"] for r in rows) / n * 100, 1),
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


def print_report(agg: dict, results: list[dict]):
    SEP = "=" * 66
    print(f"\n{SEP}")
    print("  PERSONAL QUERY EVALUATION  —  Horizon Tech HR Chatbot")
    print(SEP)

    o = agg["overall"]
    print(f"\nOVERALL  (n={o['n']})")
    print(f"  M1  Tool Selection Accuracy : {o['M1_tool_accuracy_%']:>6.1f}%")
    print(f"  M2  Answer Correctness (LLM): {o['M2_answer_correct_%']:>6.1f}%")
    print(f"  M3  Data Grounding          : {o['M3_data_grounding_%']:>6.1f}%")

    print(f"\nBY LANGUAGE")
    hdr = f"  {'Language':<12}  {'n':>3}  {'M1 Tool':>8}  {'M2 Correct':>10}  {'M3 Ground':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lang, s in agg["by_language"].items():
        print(f"  {lang:<12}  {s['n']:>3}  {s['M1_tool_accuracy_%']:>7.1f}%  "
              f"{s['M2_answer_correct_%']:>9.1f}%  {s['M3_data_grounding_%']:>8.1f}%")

    print(f"\nBY TOPIC")
    hdr2 = f"  {'Topic':<16}  {'n':>3}  {'M1 Tool':>8}  {'M2 Correct':>10}  {'M3 Ground':>9}"
    print(hdr2)
    print("  " + "-" * (len(hdr2) - 2))
    for topic, s in agg["by_topic"].items():
        print(f"  {topic:<16}  {s['n']:>3}  {s['M1_tool_accuracy_%']:>7.1f}%  "
              f"{s['M2_answer_correct_%']:>9.1f}%  {s['M3_data_grounding_%']:>8.1f}%")

    fr_rows = [r for r in results if r["language"] == "FR"]
    if fr_rows:
        detected_counts = defaultdict(int)
        for r in fr_rows:
            detected_counts[r["lang_detected"]] += 1
        print(f"\nFRANCO LANGUAGE DETECTION  (n={len(fr_rows)})")
        for detected, count in sorted(detected_counts.items()):
            print(f"  detected as '{detected}': {count}/{len(fr_rows)}")
        misclassified = [r for r in fr_rows if r["lang_detected"] != "franco"]
        if misclassified:
            print("  Misclassified queries:")
            for r in misclassified:
                print(f"    {r['id']:10s}  detected={r['lang_detected']}  "
                      f"tools={r['tools_called']}  query={r['query'][:60]}")

    failures = [r for r in results
                if r["M1_tool_accuracy"] + r["M2_answer_correct"] + r["M3_data_grounding"] < 2]
    if failures:
        print(f"\nFAILED QUERIES  (<2 metrics passed)  — {len(failures)}")
        for r in failures:
            print(f"  {r['id']:10s}  M1={r['M1_tool_accuracy']} "
                  f"M2={r['M2_answer_correct']} M3={r['M3_data_grounding']}  "
                  f"lang={r['lang_detected']}  tools={r['tools_called']}")
            if r["M2_answer_correct"] == 0:
                print(f"             judge:     {r['m2_detail'][:100]}")
            if r["M3_data_grounding"] == 0:
                print(f"             grounding: {r['m3_detail']}")

    print(f"\n{SEP}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--employee_id", type=int, required=True)
    parser.add_argument("--benchmark",   default="personal_eval_benchmark.json")
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

    # ── Pre-fetch employee data once with correct nested structure ────────────
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
            s    = row["M1_tool_accuracy"] + row["M2_answer_correct"] + row["M3_data_grounding"]
            icon = "✓" if s == 3 else ("~" if s == 2 else "✗")
            print(f"{icon}  M1={row['M1_tool_accuracy']} "
                  f"M2={row['M2_answer_correct']} "
                  f"M3={row['M3_data_grounding']}  "
                  f"lang={row['lang_detected']}  ({row['response_time_s']}s)")

    agg = aggregate(results)
    print_report(agg, results)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = f"personal_eval_results_{ts}.json"
    txt_out  = f"personal_eval_report_{ts}.txt"

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