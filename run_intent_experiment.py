import json, re, sys, time
from datetime import datetime, timedelta
from pathlib  import Path
from typing   import List
from dotenv   import load_dotenv

load_dotenv()

import pandas as pd
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
INTENT_QUERY_FILE       = "intent_queries.json"
OUTPUT_XLSX             = "intent_results.xlsx"
CHECKPOINT_FILE         = "intent_results_checkpoint.jsonl"
TEST_EMPLOYEE_ID        = "EMP001"
SKIP_CRITIQUE           = True
ROUTING_ONLY            = True
DELAY_BETWEEN_QUERIES_S = 4
RATE_LIMIT_WAIT_S       = 90
MAX_RATE_LIMIT_RETRIES  = 10

# ---------------------------------------------------------------------------
# OOS detection phrases — used in routing-only mode where there is no final
# answer. We check the raw retrieve_policy tool output instead.
# In full-agent mode the answer text is used (oos_no_info logic below).
# ---------------------------------------------------------------------------
_OOS_POLICY_PHRASES = [
    "no relevant policy found",   # exact string returned by retrieve_policy
    "not relevant",
    "no information",
]

# Phrases checked against the *answer* in full-agent mode
_NO_INFO_PHRASES = [
    "not available", "no information", "not in the policy",
    "لا تتوفر", "لا يوجد", "not found", "غير متاح",
    "outside", "not an hr", "cannot find",
]
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def _load_checkpoint() -> dict:
    """Return {query_id: row_dict} for every already-completed query."""
    done = {}
    p = Path(CHECKPOINT_FILE)
    if not p.exists():
        return done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            done[row["query_id"]] = row
        except Exception:
            pass
    return done


def _save_checkpoint(row: dict):
    """Append one completed row to the checkpoint file (never overwrites)."""
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# =============================================================================
# GAP DETECTION
# =============================================================================

def _natural_sort_key(qid: str):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", qid)]


def _audit_coverage(all_queries: List[dict], done: dict) -> List[dict]:
    all_ids  = [q["id"] for q in all_queries]
    done_ids = set(done.keys())
    pending  = [q for q in all_queries if q["id"] not in done_ids]
    gap_ids  = [qid for qid in all_ids if qid not in done_ids]

    print(f"\n{'─'*60}")
    print(f"  Coverage audit")
    print(f"  Total in query file : {len(all_ids)}")
    print(f"  Done (checkpoint)   : {len(done_ids)}")
    print(f"  Pending             : {len(pending)}")

    if gap_ids:
        sorted_gaps = sorted(gap_ids, key=_natural_sort_key)
        print(f"  Gap IDs             : {', '.join(sorted_gaps)}")
    else:
        print(f"  No gaps — all query-file IDs are covered ✓")

    orphan_ids = done_ids - set(all_ids)
    if orphan_ids:
        sorted_orphans = sorted(orphan_ids, key=_natural_sort_key)
        print(f"  ⚠  Checkpoint has {len(orphan_ids)} ID(s) not in query file "
              f"(orphans): {', '.join(sorted_orphans)}")

    print(f"{'─'*60}\n")
    return pending


# =============================================================================
# RATE-LIMIT HELPERS
# =============================================================================

def _parse_wait_seconds(error_msg: str) -> int:
    m = re.search(r"try again in\s+(?:(\d+)m\s*)?(\d+(?:\.\d+)?)s", error_msg, re.I)
    if m:
        return int(int(m.group(1) or 0) * 60 + float(m.group(2))) + 10
    m2 = re.search(r"(\d+)m(\d+)s", error_msg)
    if m2:
        return int(m2.group(1)) * 60 + int(m2.group(2)) + 10
    return RATE_LIMIT_WAIT_S


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower()


def _eta(seconds: int) -> str:
    return (datetime.now() + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def _invoke_rl(fn, *args, **kwargs):
    for attempt in range(MAX_RATE_LIMIT_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit(e):
                wait = _parse_wait_seconds(str(e))
                print(f"\n  ⏳ Rate limit (attempt {attempt+1}/{MAX_RATE_LIMIT_RETRIES}). "
                      f"Waiting {wait}s — ready at {_eta(wait)} …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Still rate-limited after {MAX_RATE_LIMIT_RETRIES} retries.")


# =============================================================================
# OOS DETECTION — routing-only mode
# =============================================================================

def _detect_oos_from_tool_outputs(tools_called: List[str], tool_outputs: dict) -> bool:
    """
    In ROUTING_ONLY mode there is no final answer to inspect, so we check
    the raw retrieve_policy output for the no-info sentinel string.

    Returns True only when:
      - retrieve_policy was the only tool called (no DB tools)
      - its output looks like a no-info / empty result
    """
    db_set = {
        "get_profile", "get_leave_data", "get_salary_data",
        "get_performance_data", "get_training_data", "get_disciplinary_data",
    }
    has_db = bool(set(tools_called) & db_set)
    if has_db:
        return False   # has personal data → not OOS

    policy_out = tool_outputs.get("retrieve_policy", "").lower()
    return any(phrase in policy_out for phrase in _OOS_POLICY_PHRASES)


# =============================================================================
# ROUTING-ONLY AGENT RUN
# Returns intent, topic, tools_called, AND the raw tool outputs so OOS
# detection can inspect the retrieve_policy result.
# =============================================================================

def _run_routing_only(question, employee_id, ar_index, en_index,
                      routing_llm, reranker, ara_tokenizer) -> dict:
    from agent import _make_tools, _infer_intent, _ORCHESTRATOR_SYSTEM
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    tools, _       = _make_tools(employee_id, ar_index, en_index, reranker, ara_tokenizer)
    tool_map       = {t.name: t for t in tools}
    llm_with_tools = routing_llm.bind_tools(tools)

    messages     = [SystemMessage(content=_ORCHESTRATOR_SYSTEM),
                    HumanMessage(content=question)]
    tools_called  = []
    tool_outputs  = {}   # ← NEW: captures raw output per tool name

    for _ in range(3):
        response = _invoke_rl(llm_with_tools.invoke, messages)
        messages.append(response)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            tname = tc["name"]
            targs = tc.get("args", {})
            tools_called.append(tname)
            try:
                fn     = tool_map[tname]
                result = str(fn.invoke(targs) if targs else fn.invoke({}))[:400]
            except Exception as e:
                result = f"Error: {e}"
            tool_outputs[tname] = result   # ← store raw output
            messages.append(ToolMessage(tool_call_id=tc["id"], content=result))

    # ── Intent inference ────────────────────────────────────────────────────
    # In routing-only mode we pass the raw policy output to let _infer_intent
    # detect OOS.  If your agent.py version of _infer_intent does not yet
    # accept policy_result, the local override below is used instead.
    intent, topic = _infer_intent_local(tools_called, tool_outputs)

    return {
        "intent":       intent,
        "topic":        topic,
        "tools_called": tools_called,
        "tool_outputs": tool_outputs,
    }


# =============================================================================
# LOCAL _infer_intent — owns OOS logic for the eval harness.
# Mirrors agent.py's _infer_intent but adds the OOS branch so the eval
# works even before you update agent.py.
# =============================================================================

def _infer_intent_local(tools_called: List[str], tool_outputs: dict) -> tuple:
    db_set = {
        "get_profile", "get_leave_data", "get_salary_data",
        "get_performance_data", "get_training_data", "get_disciplinary_data",
    }
    has_policy = "retrieve_policy" in tools_called
    has_db     = bool(set(tools_called) & db_set)

    if has_policy and not has_db and _detect_oos_from_tool_outputs(tools_called, tool_outputs):
        intent = "out_of_scope"
    else:
        intent = "hybrid" if has_db and has_policy else ("personal" if has_db else "policy")

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
    return intent, ("all" if has_db else "none")


# =============================================================================
# HELPERS
# =============================================================================

def load_queries() -> List[dict]:
    p = Path(INTENT_QUERY_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{INTENT_QUERY_FILE} not found.")
    return json.loads(p.read_text(encoding="utf-8"))


def _detect_dialect(question: str, ara_tokenizer) -> str:
    from nlp_utils import detect_language_type, get_semantic_dialect
    lang = detect_language_type(question)
    return get_semantic_dialect(question, ara_tokenizer) if lang == "arabic" else lang


def intent_match(inferred: str, expected: str) -> bool:
    return inferred.strip().lower() == expected.strip().lower()


def tools_match(called: List[str], expected: List[str]) -> bool:
    return all(t in called for t in expected)


def is_out_of_scope(q: dict) -> bool:
    # Support both old style (topic=out_of_scope) and new style (expected_intent=out_of_scope)
    return (
        q.get("topic", "") == "out_of_scope"
        or q.get("expected_intent", "") == "out_of_scope"
    )


# =============================================================================
# REPORT BUILDER
# =============================================================================

def _print_and_save(df: pd.DataFrame):
    total = len(df)
    if total == 0:
        print("No data to report.")
        return

    n_pass  = int(df["intent_pass"].sum())
    n_tools = int(df["tools_pass"].sum())

    # ── Separate OOS from main ───────────────────────────────────────────────
    # Support rows written by both old code (out_of_scope boolean column) and
    # new code (expected_intent == "out_of_scope").
    def _is_oos_row(row):
        return (
            row.get("out_of_scope", False)
            or row.get("expected_intent", "") == "out_of_scope"
        )

    oos_mask = df.apply(_is_oos_row, axis=1)
    oos_df   = df[oos_mask]
    main_df  = df[~oos_mask]
    n_oos    = len(oos_df)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 3 RESULTS  ({total} queries completed)")
    print(f"{'='*70}\n")

    print("── OVERALL ───────────────────────────────────────────────────────")
    print(f"  Intent accuracy (all)    : {n_pass}/{total}  = {n_pass/total:.1%}")
    print(f"  Tools accuracy (all)     : {n_tools}/{total} = {n_tools/total:.1%}")
    if len(main_df):
        mp = main_df["intent_pass"].mean()
        print(f"  Intent accuracy (non-OOS): "
              f"{int(main_df['intent_pass'].sum())}/{len(main_df)} = {mp:.1%}")
    if n_oos:
        # "correctly handled" = inferred intent is out_of_scope
        oos_correct = int((oos_df["inferred_intent"] == "out_of_scope").sum())
        # Fallback: old metric — routed to policy (pre-OOS-intent era rows)
        oos_routed  = int((oos_df["inferred_intent"] == "policy").sum())
        oos_ni      = int(oos_df["oos_no_info"].fillna(False).astype(bool).sum())

        print(f"  OOS correctly detected   : {oos_correct}/{n_oos}  = {oos_correct/n_oos:.1%}")
        print(f"  OOS routed to policy     : {oos_routed}/{n_oos}  = {oos_routed/n_oos:.1%}  "
              f"(legacy metric — pre-OOS-intent rows)")
        if not ROUTING_ONLY:
            print(f"  OOS returned no-info msg : {oos_ni}/{n_oos}  = {oos_ni/n_oos:.1%}")
    print()

    def _pivot(frame, col):
        if frame.empty or col not in frame.columns:
            return pd.DataFrame()
        return (
            frame.groupby(col)
            .agg(total=("intent_pass", "count"),
                 correct=("intent_pass", "sum"),
                 accuracy=("intent_pass", "mean"),
                 tools_acc=("tools_pass", "mean"))
            .round(3)
        )

    p_intent     = _pivot(main_df, "expected_intent")
    p_lang       = _pivot(df,      "language")
    p_complexity = _pivot(df,      "complexity")
    p_topic      = _pivot(df,      "topic")

    for label, piv in [("BY INTENT TYPE", p_intent),
                        ("BY LANGUAGE",    p_lang),
                        ("BY COMPLEXITY",  p_complexity),
                        ("BY TOPIC",       p_topic)]:
        print(f"── {label} {'─'*(57-len(label))}")
        print(piv.to_string() if not piv.empty else "  (no data)", "\n")

    failures = df[~df["intent_pass"].astype(bool)]
    if not failures.empty:
        print(f"── FAILURES ({len(failures)}) ─────────────────────────────────────────")
        for _, row in failures.iterrows():
            print(f"  {row['query_id']:5s} [{row['language']:8s}] "
                  f"expected={row['expected_intent']:12s} "
                  f"got={row['inferred_intent']:12s} "
                  f"tools={row['tools_called']}")
        print()

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw", index=False)

        pd.DataFrame([
            {"metric": "Intent Accuracy (all)",        "value": f"{n_pass/total:.1%}"},
            {"metric": "Intent Accuracy (non-OOS)",    "value": f"{main_df['intent_pass'].mean():.1%}" if len(main_df) else "N/A"},
            {"metric": "Tools Match Accuracy",          "value": f"{n_tools/total:.1%}"},
            {"metric": "Total Queries Completed",       "value": total},
            {"metric": "OOS Queries",                   "value": int(n_oos)},
            {"metric": "OOS Correctly Detected",        "value": int((oos_df["inferred_intent"] == "out_of_scope").sum()) if n_oos else 0},
            {"metric": "ROUTING_ONLY mode",             "value": str(ROUTING_ONLY)},
            {"metric": "Report generated",              "value": datetime.now().isoformat()},
        ]).to_excel(writer, sheet_name="Overall", index=False)

        if not p_intent.empty:     p_intent.to_excel(writer,     sheet_name="By_Intent")
        if not p_lang.empty:       p_lang.to_excel(writer,       sheet_name="By_Language")
        if not p_complexity.empty: p_complexity.to_excel(writer, sheet_name="By_Complexity")
        if not p_topic.empty:      p_topic.to_excel(writer,      sheet_name="By_Topic")

        if not oos_df.empty:
            cols = ["query_id", "language", "expected_intent", "inferred_intent",
                    "tools_called", "oos_no_info", "answer_snippet"]
            oos_df[[c for c in cols if c in oos_df.columns]].to_excel(
                writer, sheet_name="OOS_Detail", index=False)

        if not failures.empty:
            cols = ["query_id", "language", "complexity", "topic",
                    "expected_intent", "inferred_intent",
                    "expected_tools", "tools_called", "answer_snippet"]
            failures[[c for c in cols if c in failures.columns]].to_excel(
                writer, sheet_name="Failures", index=False)

    print(f"✓ Results saved to {OUTPUT_XLSX}")


# =============================================================================
# ANALYTICS LOG INSERT
# Writes one row to analytics_log after each query so the eval run is
# reflected in the admin dashboard just like production traffic.
#
# Schema columns used:
#   employee_id, intent, topic, language, unanswered,
#   question_text, resolved, response_time_ms, error
#
# Call insert_analytics_row() only when you have a live DB connection.
# Set LOG_TO_DB = True to enable; False to skip silently (safe default).
# =============================================================================

LOG_TO_DB = False   # ← flip to True once DB is reachable from eval machine

def insert_analytics_row(
    employee_id: int,
    intent: str,
    topic: str,
    language: str,
    question: str,
    unanswered: bool,
    response_time_ms: float | None = None,
    error: bool = False,
):
    """
    Insert one row into analytics_log.

    intent   — one of: personal | policy | hybrid | out_of_scope
    topic    — leave | salary | performance | training | disciplinary |
               profile | promotion | offboarding | … | out_of_scope
    unanswered — True when the bot said it couldn't answer (OOS / no-info)
    resolved   — always False on insert; an admin marks it True later
    """
    if not LOG_TO_DB:
        return

    # Import here so the eval still runs without a DB when LOG_TO_DB=False
    from database import get_db
    from sqlalchemy import text

    # analytics_log.intent is VARCHAR(20) — truncate safely
    intent_val = (intent or "")[:20]
    topic_val  = (topic  or "")[:40]
    lang_val   = (language or "")[:20]
    q_text     = (question or "")[:300]

    with get_db() as db:
        db.execute(text("""
            INSERT INTO analytics_log
                (employee_id, intent, topic, language, unanswered,
                 question_text, resolved, response_time_ms, error)
            VALUES
                (:eid, :intent, :topic, :language, :unanswered,
                 :question_text, FALSE, :response_time_ms, :error)
        """), {
            "eid":              employee_id,
            "intent":          intent_val,
            "topic":           topic_val,
            "language":        lang_val,
            "unanswered":      unanswered,
            "question_text":   q_text,
            "response_time_ms": response_time_ms,
            "error":           error,
        })
        db.commit()


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiment():
    print("\n🚀 Experiment 3 — Intent Classification Accuracy")
    print(f"   ROUTING_ONLY={ROUTING_ONLY}  SKIP_CRITIQUE={SKIP_CRITIQUE}")
    print(f"   Checkpoint : {CHECKPOINT_FILE}")
    print(f"   Delay      : {DELAY_BETWEEN_QUERIES_S}s between queries")
    print(f"   DB logging : {'ON' if LOG_TO_DB else 'OFF'}")

    done    = _load_checkpoint()
    queries = load_queries()
    pending = _audit_coverage(queries, done)

    if not pending:
        print("All queries already completed. Building report from checkpoint.\n")
        df = pd.DataFrame(list(done.values()))
        _print_and_save(df)
        return df

    (ar_index, en_index,
     routing_llm, en_llm, ar_llm, critique_llm,
     reranker, dialect_pipe, ara_tokenizer) = setup()

    for q in tqdm(pending, desc="Intent eval"):
        qid        = q["id"]
        question   = q["question"]
        lang       = q["language"]
        dialect    = _detect_dialect(question, ara_tokenizer)
        exp_intent = q["expected_intent"]
        exp_tools  = q.get("expected_tools", [])
        topic      = q.get("topic", "")
        complexity = q.get("complexity", "")
        oos        = is_out_of_scope(q)

        t_start = time.monotonic()

        try:
            if ROUTING_ONLY:
                result = _run_routing_only(
                    question, TEST_EMPLOYEE_ID,
                    ar_index, en_index,
                    routing_llm, reranker, ara_tokenizer,
                )
                inf_intent   = result["intent"]
                tools_called = result["tools_called"]
                answer       = ""

                # OOS no-info: in routing-only mode, flag when policy tool
                # returned a no-info sentinel (same data used for intent above)
                oos_no_info = (
                    _detect_oos_from_tool_outputs(tools_called, result["tool_outputs"])
                    if oos else None
                )

            else:
                from agent import run_agent
                result = run_agent(
                    question=question, employee_id=TEST_EMPLOYEE_ID,
                    lang=lang if lang != "franco" else "arabic",
                    dialect=dialect, history_str="",
                    ar_index=ar_index, en_index=en_index,
                    routing_llm=routing_llm, en_llm=en_llm,
                    ar_llm=ar_llm, critique_llm=critique_llm,
                    reranker=reranker, ara_tokenizer=ara_tokenizer,
                    skip_critique=True,
                )
                inf_intent   = result["intent"]
                tools_called = result["tools_called"]
                answer       = result.get("answer", "")

                oos_no_info = (
                    any(ph in answer.lower() for ph in _NO_INFO_PHRASES)
                    if oos and answer else None
                )

            elapsed_ms = (time.monotonic() - t_start) * 1000

            i_match = intent_match(inf_intent, exp_intent)
            t_match = tools_match(tools_called, exp_tools)

            # ── analytics_log insert ────────────────────────────────────────
            insert_analytics_row(
                employee_id=int(TEST_EMPLOYEE_ID.replace("EMP", "")) if TEST_EMPLOYEE_ID.startswith("EMP") else 0,
                intent=inf_intent,
                topic=topic,
                language=lang,
                question=question,
                unanswered=bool(oos_no_info) if oos_no_info is not None else (inf_intent == "out_of_scope"),
                response_time_ms=elapsed_ms,
                error=False,
            )

            row = {
                "query_id":        qid,
                "language":        lang,
                "complexity":      complexity,
                "topic":           topic,
                "out_of_scope":    oos,
                "expected_intent": exp_intent,
                "inferred_intent": inf_intent,
                "intent_pass":     i_match,
                "expected_tools":  ", ".join(sorted(exp_tools)),
                "tools_called":    ", ".join(tools_called),
                "tools_pass":      t_match,
                "oos_no_info":     oos_no_info,
                "answer_snippet":  answer[:120],
                "timestamp":       datetime.now().isoformat(),
            }
            _save_checkpoint(row)
            done[qid] = row

            status = "✓" if i_match else "✗"
            tqdm.write(f"  [{status}] {qid} | {exp_intent:12s} → {inf_intent:12s} | {tools_called}")

            time.sleep(DELAY_BETWEEN_QUERIES_S)

        except Exception as e:
            elapsed_ms = (time.monotonic() - t_start) * 1000

            insert_analytics_row(
                employee_id=int(TEST_EMPLOYEE_ID.replace("EMP", "")) if TEST_EMPLOYEE_ID.startswith("EMP") else 0,
                intent="error",
                topic=topic,
                language=lang,
                question=question,
                unanswered=False,
                response_time_ms=elapsed_ms,
                error=True,
            )

            print(f"\n  [ERR] {qid}: {e}")
            row = {
                "query_id":        qid,
                "language":        lang,
                "complexity":      complexity,
                "topic":           topic,
                "out_of_scope":    oos,
                "expected_intent": exp_intent,
                "inferred_intent": "ERROR",
                "intent_pass":     False,
                "expected_tools":  ", ".join(exp_tools),
                "tools_called":    "ERROR",
                "tools_pass":      False,
                "oos_no_info":     None,
                "answer_snippet":  str(e)[:120],
                "timestamp":       datetime.now().isoformat(),
            }
            _save_checkpoint(row)
            done[qid] = row

    df = pd.DataFrame(list(done.values()))
    _print_and_save(df)
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from setup import setup

    if "--report" in sys.argv:
        done = _load_checkpoint()
        if not done:
            print(f"No checkpoint found at {CHECKPOINT_FILE}. Run experiment first.")
            sys.exit(1)
        queries = load_queries()
        _audit_coverage(queries, done)
        print(f"Report-only mode: {len(done)} rows from {CHECKPOINT_FILE}")
        df = pd.DataFrame(list(done.values()))
        _print_and_save(df)

    elif "--rerun-oos" in sys.argv:
        # ── Selective OOS re-evaluation ──────────────────────────────────────
        # Strips only OOS rows from the checkpoint (by expected_intent OR topic)
        # then falls through to run_experiment() which fills the gaps.
        #
        # Usage:  python eval.py --rerun-oos
        #
        p = Path(CHECKPOINT_FILE)
        if p.exists():
            kept, removed = [], []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    is_oos = (
                        row.get("topic") == "out_of_scope"
                        or row.get("expected_intent") == "out_of_scope"
                        or row.get("out_of_scope") is True
                    )
                    (removed if is_oos else kept).append(line)
                except Exception:
                    kept.append(line)

            # Back up before modifying
            backup = CHECKPOINT_FILE + ".bak"
            p.rename(backup)
            print(f"  Backed up checkpoint → {backup}")

            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(kept) + ("\n" if kept else ""))

            print(f"  Stripped {len(removed)} OOS row(s), kept {len(kept)} row(s).")
        else:
            print(f"  No checkpoint found — will run all queries.")

        # Now run normally; gap detector picks up the stripped IDs
        run_experiment()

    else:
        run_experiment()