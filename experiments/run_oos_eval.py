"""
run_oos_eval.py
===============
Evaluates out-of-scope refusal quality for the Horizon Tech HR chatbot.

Mirrors agent.py EXACTLY:
  - Same run_agent() call signature
  - Same _classify_intent() via _ROUTER_SYSTEM (routing_llm)
  - Same detect_language_type() + get_semantic_dialect() from nlp_utils
  - Same is_no_info_answer() from utils.py
  - OOS path in run_agent: tools_called = ["out_of_scope"], then _format_answer
    returns "This information is not available in the policy documents." (or
    equivalent) because tool_results has no policy_context and no DB keys.
  - skip_critique=True, history_str="" (eval mode)

REFUSAL DETECTION (rule-based, mirrors system exactly):
  A response is considered a REFUSAL if ANY of the following:
  1. "out_of_scope" in tools_called  (router classified it as OOS)
  2. is_no_info_answer(answer) == True  (from utils.py — same function the app uses)
  3. answer matches known OOS deflection phrases (multilingual, covers all 4 langs)

Three metrics (all rule-based):
  M1 — Refusal Accuracy   : correctly refused truly OOS queries
                             = TP / (TP + FN)   where TP = refused OOS, FN = answered OOS
  M2 — False Refusal Rate : refused valid HR queries (lower is better)
                             = FP / (FP + TN)   where FP = refused HR, TN = answered HR
  M3 — Hallucination Rate : answered substantively on truly OOS queries (lower is better)
                             = FN / (TP + FN)   = 1 - M1

Usage:
    python run_oos_eval.py --employee_id 42
    python run_oos_eval.py --employee_id 42 --lang EN --verbose
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
# REFUSAL DETECTION
# Mirrors the system's own refusal signals exactly.
# =============================================================================

# These are the exact patterns from utils.py NO_INFO_PATTERNS —
# copied here so the evaluator uses the SAME detection as the app.
_NO_INFO_PATTERNS = [
    "this information is not available in the policy documents",
    "information is not available in the policy",
    "not available in the policy",
    "not found in",
    "هذه المعلومات غير متوفرة في وثائق السياسة",
    "هذه المعلومات غير متاحة في وثائق السياسة",
    "معلومات غير متوفرة في وثائق",
    "الموضوع ده مش موجود في السياسة",
    "mesh mawgoda f el policy",
    "ma3loma mesh mawgoda fel policy",
    "no relevant policy documents found",
]

# Additional OOS deflection phrases the system may produce in each language.
# These cover the _format_answer() fallback path when no context and no DB data.
_OOS_DEFLECTION_PHRASES = [
    # English
    "not related to hr",
    "outside the scope",
    "out of scope",
    "cannot help with that",
    "i can only help with hr",
    "this is not an hr",
    "not an hr-related",
    "unrelated to hr",
    "not within my scope",
    "only assist with hr",
    "only answer hr",
    # Arabic MSA
    "هذا السؤال خارج نطاق",
    "لا يتعلق بالموارد البشرية",
    "لا يندرج ضمن نطاق",
    "غير ذي صلة بالموارد البشرية",
    "لا أستطيع المساعدة في ذلك",
    "هذا ليس سؤالاً يخص الموارد البشرية",
    "خارج اختصاص",
    "لا يتعلق بسياسات الشركة",
    # Egyptian Arabic
    "مش من اختصاصي",
    "ده مش موضوع موارد بشرية",
    "مش بقدر أساعد في ده",
    "ده بره نطاقي",
    "مش من شغلي",
    "الكلام ده مش بتاعي",
    # Franco
    "mesh men ekhtesasy",
    "da mesh mawdoo3 hr",
    "mesh 2adir asa3ed fi da",
    "da barra neta2y",
    "mesh men shoghlena",
    "da mesh bta3na",
]


def _is_refusal(tools_called: list, answer: str) -> bool:
    """
    Returns True if the system refused to answer the query.

    Checks in order:
    1. Router explicitly classified as OOS → tools_called = ["out_of_scope"]
    2. is_no_info_answer() patterns (same function as utils.py)
    3. OOS deflection phrases (multilingual)
    """
    # 1. Router-level OOS signal
    if tools_called == ["out_of_scope"]:
        return True

    # 2. is_no_info_answer() — exact copy of utils.py logic
    norm = answer.strip().lower()
    if any(p in norm for p in _NO_INFO_PATTERNS):
        return True

    # 3. OOS deflection phrases
    if any(p in norm for p in _OOS_DEFLECTION_PHRASES):
        return True

    return False


# =============================================================================
# SINGLE QUERY RUNNER
# =============================================================================

def evaluate_one(query: dict, employee_id: int,
                 routing_llm, en_llm, ar_llm, critique_llm,
                 ar_index, en_index, reranker, dialect_pipe, ara_tokenizer,
                 verbose: bool = False) -> dict:

    from agent import run_agent
    from nlp_utils import detect_language_type, get_semantic_dialect

    question     = query["query"]
    qid          = query["id"]
    should_refuse = query["should_refuse"]

    # Language detection — exactly as app.py does
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

    refused = _is_refusal(tools_called, answer)

    # ── Metric assignments ────────────────────────────────────────────────────
    # For truly OOS queries (should_refuse=True):
    #   correct = refused (TP),  incorrect = not refused (FN = hallucination)
    # For valid HR queries (should_refuse=False):
    #   correct = not refused (TN),  incorrect = refused (FP = false refusal)

    if should_refuse:
        # TP = correctly refused, FN = hallucinated an answer
        correct  = refused
        outcome  = "TP" if refused else "FN"
    else:
        # TN = correctly answered, FP = wrongly refused
        correct  = not refused
        outcome  = "TN" if not refused else "FP"

    row = {
        "id":           qid,
        "language":     qid.split("-")[0],
        "category":     query["category"],
        "should_refuse": should_refuse,
        "refused":       refused,
        "outcome":       outcome,
        "correct":       correct,
        "tools_called":  tools_called,
        "intent_detected": intent,
        "lang_detected": lang,
        "answer_snippet": answer[:200],
        "response_time_s": elapsed,
    }

    if verbose:
        icon = "✓" if correct else "✗"
        tag  = f"[{outcome}]"
        print(f"  {icon} {qid:14s} {tag:5s}  refused={refused}  "
              f"intent={intent:12s}  tools={tools_called}  ({elapsed}s)")
        if not correct:
            print(f"         answer: {answer[:120]}")

    return row


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(results: list[dict]) -> dict:

    def _compute(rows):
        n = len(rows)
        if not n:
            return {}

        oos_rows  = [r for r in rows if r["should_refuse"]]
        hr_rows   = [r for r in rows if not r["should_refuse"]]

        tp = sum(1 for r in oos_rows if r["outcome"] == "TP")
        fn = sum(1 for r in oos_rows if r["outcome"] == "FN")
        fp = sum(1 for r in hr_rows  if r["outcome"] == "FP")
        tn = sum(1 for r in hr_rows  if r["outcome"] == "TN")

        n_oos = len(oos_rows)
        n_hr  = len(hr_rows)

        return {
            "n":                   n,
            "n_oos":               n_oos,
            "n_hr":                n_hr,
            "TP":                  tp,
            "FN":                  fn,
            "FP":                  fp,
            "TN":                  tn,
            # M1: Refusal Accuracy (on OOS queries)
            "M1_refusal_acc_%":    round(tp / n_oos * 100, 1) if n_oos else 0.0,
            # M2: False Refusal Rate (on valid HR queries) — lower is better
            "M2_false_refusal_%":  round(fp / n_hr  * 100, 1) if n_hr  else 0.0,
            # M3: Hallucination Rate (on OOS queries) — lower is better = 1 - M1
            "M3_hallucination_%":  round(fn / n_oos * 100, 1) if n_oos else 0.0,
        }

    by_lang  = defaultdict(list)
    by_cat   = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
        by_cat[r["category"]].append(r)

    return {
        "overall":      _compute(results),
        "by_language":  {k: _compute(v) for k, v in by_lang.items()},
        "by_category":  {k: _compute(v) for k, v in by_cat.items()},
    }


# =============================================================================
# REPORT
# =============================================================================

def print_report(agg: dict, results: list[dict]):
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  OOS REFUSAL EVALUATION  —  Horizon Tech HR Chatbot")
    print(SEP)

    o = agg["overall"]
    print(f"\nOVERALL  (n={o['n']}  |  OOS={o['n_oos']}  |  valid HR={o['n_hr']})")
    print(f"  M1  Refusal Accuracy (OOS queries)  : {o['M1_refusal_acc_%']:>6.1f}%  "
          f"  TP={o['TP']} FN={o['FN']}")
    print(f"  M2  False Refusal Rate (HR queries) : {o['M2_false_refusal_%']:>6.1f}%  "
          f"  FP={o['FP']} TN={o['TN']}  ← lower is better")
    print(f"  M3  Hallucination Rate (OOS queries): {o['M3_hallucination_%']:>6.1f}%  "
          f"  FN={o['FN']}           ← lower is better")

    print(f"\nBY LANGUAGE")
    hdr = (f"  {'Language':<6}  {'n':>3}  "
           f"{'M1 Refusal%':>12}  {'M2 FalseRef%':>13}  {'M3 Halluc%':>11}  "
           f"{'TP':>3}{'FN':>3}{'FP':>3}{'TN':>3}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lang, s in agg["by_language"].items():
        print(f"  {lang:<6}  {s['n']:>3}  "
              f"{s['M1_refusal_acc_%']:>11.1f}%  "
              f"{s['M2_false_refusal_%']:>12.1f}%  "
              f"{s['M3_hallucination_%']:>10.1f}%  "
              f"{s['TP']:>3}{s['FN']:>3}{s['FP']:>3}{s['TN']:>3}")

    print(f"\nBY CATEGORY  (OOS categories only)")
    oos_cats = {k: v for k, v in agg["by_category"].items()
                if any(r["should_refuse"] for r in results if r["category"] == k)}
    for cat, s in sorted(oos_cats.items()):
        if s["n_oos"] == 0:
            continue
        print(f"  {cat:<22}  M1={s['M1_refusal_acc_%']:>5.1f}%  "
              f"M3={s['M3_hallucination_%']:>5.1f}%  "
              f"TP={s['TP']} FN={s['FN']}")

    # ── Failures ──────────────────────────────────────────────────────────────
    fn_rows = [r for r in results if r["outcome"] == "FN"]
    fp_rows = [r for r in results if r["outcome"] == "FP"]

    if fn_rows:
        print(f"\nFALSE NEGATIVES — Hallucinated answers on OOS queries  ({len(fn_rows)})")
        for r in fn_rows:
            print(f"  {r['id']:16s}  intent={r['intent_detected']:12s}  "
                  f"tools={r['tools_called']}")
            print(f"    query:  {r['query'][:80]}")
            print(f"    answer: {r['answer_snippet'][:100]}")

    if fp_rows:
        print(f"\nFALSE POSITIVES — Wrongly refused valid HR queries  ({len(fp_rows)})")
        for r in fp_rows:
            print(f"  {r['id']:16s}  intent={r['intent_detected']:12s}  "
                  f"tools={r['tools_called']}")
            print(f"    query:  {r['query'][:80]}")
            print(f"    answer: {r['answer_snippet'][:100]}")

    print(f"\n{SEP}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="OOS refusal eval — Horizon Tech HR chatbot")
    parser.add_argument("--employee_id", type=int, required=True)
    parser.add_argument("--benchmark",   default="oos_eval_benchmark.json")
    parser.add_argument("--lang",        choices=["EN", "AR", "EGY", "FR"])
    parser.add_argument("--batch",       type=int, choices=[1, 2],
                        help="Run only batch 1 (original) or batch 2 (harder). Default: all.")
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
        for q in queries:
            if args.batch and q.get("batch") != args.batch:
                continue
            all_queries.append(q)

    print(f"\nLoaded {len(all_queries)} queries  |  employee_id={args.employee_id}")

    print("Running setup()…")
    try:
        from setup import setup
        (ar_index, en_index,
         routing_llm, en_llm, ar_llm, critique_llm,
         reranker, dialect_pipe, ara_tokenizer) = setup()
        print("Setup complete.\n")
    except Exception as e:
        print(f"ERROR in setup(): {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    total   = len(all_queries)

    for i, query in enumerate(all_queries, 1):
        tag = "OOS" if query["should_refuse"] else "HR "
        print(f"[{i:3}/{total}] {query['id']:16s} [{tag}]", end="  ", flush=True)

        row = evaluate_one(
            query         = query,
            employee_id   = args.employee_id,
            routing_llm   = routing_llm,
            en_llm        = en_llm,
            ar_llm        = ar_llm,
            critique_llm  = critique_llm,
            ar_index      = ar_index,
            en_index      = en_index,
            reranker      = reranker,
            dialect_pipe  = dialect_pipe,
            ara_tokenizer = ara_tokenizer,
            verbose       = args.verbose,
        )
        results.append(row)

        if not args.verbose:
            icon = "✓" if row["correct"] else "✗"
            print(f"{icon}  [{row['outcome']}]  refused={row['refused']}  "
                  f"intent={row['intent_detected']:12s}  ({row['response_time_s']}s)")

    agg = aggregate(results)
    print_report(agg, results)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = f"oos_eval_results_{ts}.json"
    txt_out  = f"oos_eval_report_{ts}.txt"

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