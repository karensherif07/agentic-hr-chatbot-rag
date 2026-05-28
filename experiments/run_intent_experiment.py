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
ROUTING_ONLY            = True
DELAY_BETWEEN_QUERIES_S = 4
RATE_LIMIT_WAIT_S       = 90
MAX_RATE_LIMIT_RETRIES  = 10

INTENT_LABELS = ["policy", "personal", "hybrid", "out_of_scope"]

_NO_INFO_PHRASES = [
    "not available", "no information", "not in the policy",
    "لا تتوفر", "لا يوجد", "not found", "غير متاح",
    "outside", "not an hr", "cannot find",
]


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def _load_checkpoint() -> dict:
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
        print(f"  Gap IDs             : {', '.join(sorted(gap_ids, key=_natural_sort_key))}")
    else:
        print(f"  No gaps — all IDs covered ✓")
    orphan_ids = done_ids - set(all_ids)
    if orphan_ids:
        print(f"  ⚠  Orphan IDs in checkpoint: {', '.join(sorted(orphan_ids, key=_natural_sort_key))}")
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
# ROUTING-ONLY AGENT RUN
# =============================================================================

def _run_routing_only(question, employee_id, ar_index, en_index,
                      routing_llm, reranker, dialect_pipe, ara_tokenizer) -> dict:
    from agent import _make_tools, _ORCHESTRATOR_SYSTEM
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    tools, _       = _make_tools(employee_id, ar_index, en_index, reranker, ara_tokenizer)
    tool_map       = {t.name: t for t in tools}
    llm_with_tools = routing_llm.bind_tools(tools)

    messages     = [SystemMessage(content=_ORCHESTRATOR_SYSTEM),
                    HumanMessage(content=question)]
    tools_called = []
    tool_outputs = {}

    for _ in range(3):
        try:
            response = _invoke_rl(llm_with_tools.invoke, messages)
        except Exception as e:
            # "Failed to call a function" — model generated malformed tool args.
            # Retry once without tools so we still get an intent signal.
            if "failed_generation" in str(e).lower() or "failed to call a function" in str(e).lower():
                try:
                    response = _invoke_rl(routing_llm.invoke, messages)
                    # No tool calls on plain invoke — treat as out_of_scope
                except Exception:
                    break
            else:
                raise
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
            tool_outputs[tname] = result
            messages.append(ToolMessage(tool_call_id=tc["id"], content=result))

    intent, topic = _infer_intent_local(tools_called, tool_outputs)
    return {"intent": intent, "topic": topic,
            "tools_called": tools_called, "tool_outputs": tool_outputs}


# =============================================================================
# INTENT INFERENCE
# =============================================================================

def _infer_intent_local(tools_called: List[str], tool_outputs: dict) -> tuple:
    db_set = {
        "get_profile", "get_leave_data", "get_salary_data",
        "get_performance_data", "get_training_data", "get_disciplinary_data",
    }
    has_policy = "retrieve_policy" in tools_called
    has_db     = bool(set(tools_called) & db_set)

    if "out_of_scope" in tools_called:
        intent = "out_of_scope"
    elif has_db and has_policy:
        intent = "hybrid"
    elif has_db:
        intent = "personal"
    elif has_policy:
        intent = "policy"
    else:
        intent = "out_of_scope"

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


def _detect_dialect(question: str, dialect_pipe) -> str:
    from nlp_utils import detect_language_type, get_semantic_dialect
    lang = detect_language_type(question)
    return get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else lang


def intent_match(inferred: str, expected: str) -> bool:
    return inferred.strip().lower() == expected.strip().lower()


# =============================================================================
# REPORT BUILDER
# All 4 intents treated equally — no OOS separation, no tools metrics
# =============================================================================

def _print_and_save(df: pd.DataFrame):
    total = len(df)
    if total == 0:
        print("No data to report.")
        return

    n_pass = int(df["intent_pass"].sum())

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 3 RESULTS  ({total} queries)")
    print(f"{'='*70}\n")

    # ── OVERALL ───────────────────────────────────────────────────────────────
    print("── OVERALL ───────────────────────────────────────────────────────")
    print(f"  Intent accuracy : {n_pass}/{total} = {n_pass/total:.1%}\n")

    # ── BY INTENT ─────────────────────────────────────────────────────────────
    def _pivot(frame, col):
        if frame.empty or col not in frame.columns:
            return pd.DataFrame()
        return (
            frame.groupby(col)
            .agg(total=("intent_pass", "count"),
                 correct=("intent_pass", "sum"),
                 accuracy=("intent_pass", "mean"))
            .round(3)
        )

    p_intent     = _pivot(df, "expected_intent")
    p_lang       = _pivot(df, "language")
    p_complexity = _pivot(df, "complexity")
    p_topic      = _pivot(df, "topic")

    for label, piv in [("BY INTENT",     p_intent),
                        ("BY LANGUAGE",   p_lang),
                        ("BY COMPLEXITY", p_complexity),
                        ("BY TOPIC",      p_topic)]:
        print(f"── {label} {'─'*(57-len(label))}")
        print(piv.to_string() if not piv.empty else "  (no data)", "\n")

    # ── FAILURES ──────────────────────────────────────────────────────────────
    failures = df[~df["intent_pass"].astype(bool)]
    if not failures.empty:
        print(f"── FAILURES ({len(failures)}) ─────────────────────────────────────────")
        for _, row in failures.iterrows():
            print(f"  {row['query_id']:5s} [{row['language']:8s}] "
                  f"expected={row['expected_intent']:12s} "
                  f"got={row['inferred_intent']:12s}")
        print()

    # ── PRECISION / RECALL / F1 ───────────────────────────────────────────────
    clf_df = df[df["inferred_intent"] != "ERROR"].copy()
    prf_rows = []
    for lbl in INTENT_LABELS:
        tp = int(((clf_df["expected_intent"] == lbl) & (clf_df["inferred_intent"] == lbl)).sum())
        fp = int(((clf_df["expected_intent"] != lbl) & (clf_df["inferred_intent"] == lbl)).sum())
        fn = int(((clf_df["expected_intent"] == lbl) & (clf_df["inferred_intent"] != lbl)).sum())
        prec    = tp / (tp + fp) if (tp + fp) else 0.0
        rec     = tp / (tp + fn) if (tp + fn) else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int((clf_df["expected_intent"] == lbl).sum())
        prf_rows.append({"intent": lbl, "precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3), "support": support})

    prf_df = pd.DataFrame(prf_rows).set_index("intent")
    macro  = prf_df[["precision", "recall", "f1"]].mean().round(3)
    prf_df.loc["macro_avg"] = [macro["precision"], macro["recall"], macro["f1"],
                               int(clf_df.shape[0])]

    print("── PRECISION / RECALL / F1 ───────────────────────────────────────")
    print(prf_df.to_string(), "\n")

    # ── CONFUSION MATRIX ──────────────────────────────────────────────────────
    all_labels = INTENT_LABELS + (["ERROR"] if (df["inferred_intent"] == "ERROR").any() else [])
    conf_df = pd.DataFrame(0, index=INTENT_LABELS, columns=all_labels)
    for _, row in df.iterrows():
        exp = row["expected_intent"]
        got = row["inferred_intent"]
        if exp in conf_df.index and got in conf_df.columns:
            conf_df.loc[exp, got] += 1
    conf_df.index.name   = "expected \\ predicted"
    conf_df.columns.name = None
    print("── CONFUSION MATRIX (rows=expected, cols=predicted) ─────────────")
    print(conf_df.to_string(), "\n")

    # ── SAVE EXCEL ────────────────────────────────────────────────────────────
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw", index=False)

        pd.DataFrame([
            {"metric": "Intent Accuracy",    "value": f"{n_pass/total:.1%}"},
            {"metric": "Total Queries",       "value": total},
            {"metric": "Macro Precision",     "value": f"{macro['precision']:.3f}"},
            {"metric": "Macro Recall",        "value": f"{macro['recall']:.3f}"},
            {"metric": "Macro F1",            "value": f"{macro['f1']:.3f}"},
            {"metric": "ROUTING_ONLY mode",   "value": str(ROUTING_ONLY)},
            {"metric": "Report generated",    "value": datetime.now().isoformat()},
        ]).to_excel(writer, sheet_name="Overall", index=False)

        for piv, sheet in [(p_intent,     "By_Intent"),
                           (p_lang,       "By_Language"),
                           (p_complexity, "By_Complexity"),
                           (p_topic,      "By_Topic")]:
            if not piv.empty:
                piv.to_excel(writer, sheet_name=sheet)

        prf_df.to_excel(writer, sheet_name="Precision_Recall_F1")
        conf_df.to_excel(writer, sheet_name="Confusion_Matrix")

        if not failures.empty:
            cols = ["query_id", "language", "complexity", "topic",
                    "expected_intent", "inferred_intent", "answer_snippet"]
            failures[[c for c in cols if c in failures.columns]].to_excel(
                writer, sheet_name="Failures", index=False)

    print(f"✓ Results saved to {OUTPUT_XLSX}")


# =============================================================================
# ANALYTICS LOG INSERT
# =============================================================================

LOG_TO_DB = False

def insert_analytics_row(employee_id, intent, topic, language, question,
                          unanswered, response_time_ms=None, error=False):
    if not LOG_TO_DB:
        return
    from database import get_db
    from sqlalchemy import text
    with get_db() as db:
        db.execute(text("""
            INSERT INTO analytics_log
                (employee_id, intent, topic, language, unanswered,
                 question_text, resolved, response_time_ms, error)
            VALUES (:eid,:intent,:topic,:language,:unanswered,
                    :question_text,FALSE,:response_time_ms,:error)
        """), {
            "eid": employee_id, "intent": (intent or "")[:20],
            "topic": (topic or "")[:40], "language": (language or "")[:20],
            "unanswered": unanswered, "question_text": (question or "")[:300],
            "response_time_ms": response_time_ms, "error": error,
        })
        db.commit()


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiment():
    print("\n🚀 Experiment 3 — Intent Classification Accuracy")
    print(f"   ROUTING_ONLY={ROUTING_ONLY}")
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

    # Correctly unpack all 9 values from setup()
    (ar_index, en_index,
     routing_llm, en_llm, ar_llm, critique_llm,
     reranker, dialect_pipe, ara_tokenizer) = setup()

    for q in tqdm(pending, desc="Intent eval"):
        qid        = q["id"]
        question   = q["question"]
        lang       = q["language"]
        dialect    = _detect_dialect(question, dialect_pipe)
        exp_intent = q["expected_intent"]
        exp_tools  = q.get("expected_tools", [])
        topic      = q.get("topic", "")
        complexity = q.get("complexity", "")

        t_start = time.monotonic()

        try:
            if ROUTING_ONLY:
                result = _run_routing_only(
                    question, TEST_EMPLOYEE_ID,
                    ar_index, en_index,
                    routing_llm, reranker, dialect_pipe, ara_tokenizer,
                )
                inf_intent   = result["intent"]
                tools_called = result["tools_called"]
                answer       = ""

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

            elapsed_ms = (time.monotonic() - t_start) * 1000
            i_match    = intent_match(inf_intent, exp_intent)

            insert_analytics_row(
                employee_id=int(TEST_EMPLOYEE_ID.replace("EMP", "")) if TEST_EMPLOYEE_ID.startswith("EMP") else 0,
                intent=inf_intent, topic=topic, language=lang, question=question,
                unanswered=(inf_intent == "out_of_scope"),
                response_time_ms=elapsed_ms, error=False,
            )

            row = {
                "query_id":        qid,
                "language":        lang,
                "complexity":      complexity,
                "topic":           topic,
                "expected_intent": exp_intent,
                "inferred_intent": inf_intent,
                "intent_pass":     i_match,
                "answer_snippet":  answer[:120],
                "timestamp":       datetime.now().isoformat(),
            }
            _save_checkpoint(row)
            done[qid] = row

            status = "✓" if i_match else "✗"
            tqdm.write(f"  [{status}] {qid} | {exp_intent:12s} → {inf_intent:12s}")

            time.sleep(DELAY_BETWEEN_QUERIES_S)

        except Exception as e:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            insert_analytics_row(
                employee_id=int(TEST_EMPLOYEE_ID.replace("EMP", "")) if TEST_EMPLOYEE_ID.startswith("EMP") else 0,
                intent="error", topic=topic, language=lang, question=question,
                unanswered=False, response_time_ms=elapsed_ms, error=True,
            )
            print(f"\n  [ERR] {qid}: {e}")
            row = {
                "query_id":        qid,
                "language":        lang,
                "complexity":      complexity,
                "topic":           topic,
                "expected_intent": exp_intent,
                "inferred_intent": "ERROR",
                "intent_pass":     False,
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
            print(f"No checkpoint found at {CHECKPOINT_FILE}.")
            sys.exit(1)
        queries = load_queries()
        _audit_coverage(queries, done)
        df = pd.DataFrame(list(done.values()))
        _print_and_save(df)

    elif "--rerun-errors" in sys.argv:
        p = Path(CHECKPOINT_FILE)
        if p.exists():
            kept, removed = [], []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    is_error = row.get("inferred_intent") == "ERROR"
                    (removed if is_error else kept).append(line)
                except Exception:
                    kept.append(line)
            backup = CHECKPOINT_FILE + ".bak"
            p.rename(backup)
            print(f"  Backed up → {backup}")
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(kept) + ("\n" if kept else ""))
            print(f"  Stripped {len(removed)} ERROR row(s), kept {len(kept)} rows.")
            print(f"  Will rerun: {[json.loads(l)['query_id'] for l in removed]}")
        else:
            print(f"  No checkpoint found — will run all queries.")
        run_experiment()

    elif "--rerun-failed" in sys.argv:
        p = Path(CHECKPOINT_FILE)
        if p.exists():
            kept, removed = [], []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    (removed if not row.get("intent_pass", False) else kept).append(line)
                except Exception:
                    kept.append(line)
            backup = CHECKPOINT_FILE + ".bak"
            p.rename(backup)
            print(f"  Backed up → {backup}")
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(kept) + ("\n" if kept else ""))
            print(f"  Stripped {len(removed)} failed rows, kept {len(kept)} passing.")
        run_experiment()

    else:
        run_experiment()