"""
run_intent_baseline.py

Baseline intent classifier for Experiment 3.

Architecture:
  Query embedding
       ↓
  Cosine similarity against class prototypes
       ↓
  Highest-scoring intent selected

Produces the same output format / Excel as run_intent_experiment.py so
results can be compared directly in the same report.

Usage:
    python run_intent_baseline.py              # full run
    python run_intent_baseline.py --report     # report from checkpoint only
    python run_intent_baseline.py --rerun-failed  # strip failures and rerun
"""

import json, re, sys, time
from datetime import datetime, timedelta
from pathlib  import Path
from typing   import List

import numpy  as np
import pandas as pd
from tqdm     import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
INTENT_QUERY_FILE       = "intent_queries.json"
OUTPUT_XLSX             = "intent_baseline_results.xlsx"
CHECKPOINT_FILE         = "intent_baseline_checkpoint.jsonl"

# Embedding model — same one your retrieval stack uses (sentence-transformers).
# multilingual-e5-large is a strong multilingual choice; swap if you use another.
EMBEDDING_MODEL         = "intfloat/multilingual-e5-large"

INTENT_LABELS = ["policy", "personal", "hybrid", "out_of_scope"]

# ── PROTOTYPE SEEDS ───────────────────────────────────────────────────────────
# Each intent gets a list of representative seed phrases across all 4 languages.
# These are averaged into a single prototype vector per class.
# The seeds deliberately mirror the language mix of the query file.
PROTOTYPE_SEEDS = {
    "policy": [
        # English
        "What is the overtime rate?",
        "What are the rules for probation?",
        "How many days of annual leave does an employee get?",
        "What is the notice period for resignation?",
        "What travel class does a manager get?",
        "What are the gross misconduct offences?",
        "What is the company policy on gifts from suppliers?",
        "What does health insurance cover for children?",
        "What are the mandatory compliance training modules?",
        "What is the salary payment day?",
        # Arabic (MSA)
        "ما هي نسبة تعويض العمل الإضافي؟",
        "ما هي قواعد فترة الاختبار؟",
        "كم يوم إجازة سنوية يستحقه الموظف؟",
        "ما هي مدة الإشعار عند الاستقالة؟",
        "ما هي وحدات التدريب الإلزامية؟",
        "ما هي الجرائم التي تستوجب الفصل الفوري؟",
        # Egyptian
        "إيه نسبة الأوفر تايم؟",
        "إيه قواعد فترة الاختبار؟",
        "إيه إجراءات الفصل الفوري؟",
        "إيه مدة الإشعار لو الموظف استقال؟",
        # Franco
        "eih el overtime rate?",
        "eih el notice period lel esta2ala?",
        "eih el 7ad el a2sa lel 2ard el taware2?",
        "eih el mandatory training modules?",
    ],

    "personal": [
        # English
        "How many leave days do I have left?",
        "What is my current salary?",
        "What was my performance rating?",
        "What is my job grade?",
        "Do I have any active disciplinary warnings?",
        "Have I completed my compliance training?",
        "Show me my OKRs.",
        "Am I on a PIP?",
        "What are my remaining training budget?",
        "Do I have pending leave requests?",
        # Arabic
        "كم يوم إجازة تبقى لي؟",
        "ما راتبي الصافي؟",
        "ما كان تقييمي في آخر دورة أداء؟",
        "هل لدي إنذارات تأديبية نشطة؟",
        "ما مستواي الوظيفي الحالي؟",
        # Egyptian
        "كام يوم إجازة فاضلي؟",
        "إيه مرتبي الصافي؟",
        "إيه تقييمي في آخر مراجعة؟",
        "عندي إنذارات تأديبية نشطة؟",
        "أنا على PIP دلوقتي؟",
        # Franco
        "kam yom agaza fadel 3andy?",
        "eih maratby el net?",
        "eih el rating bta3y?",
        "3andy indarat ta2dibiya active?",
        "ana 3ala PIP delwa2ty?",
    ],

    "hybrid": [
        # English
        "Am I eligible for the annual bonus this year?",
        "Can I apply for a promotion right now?",
        "What end-of-service gratuity would I get if I resign?",
        "Based on my rating, what salary increment will I get?",
        "How much notice do I need to give if I resign?",
        "Do I qualify for an emergency interest-free loan?",
        "How much of my training budget have I used vs the company limit?",
        "If I take unpaid leave, how does it affect my gratuity?",
        # Arabic
        "هل أنا مؤهل للمكافأة السنوية بناءً على وضعي؟",
        "ما مكافأة نهاية الخدمة التي سأحصل عليها لو استقلت اليوم؟",
        "هل يمكنني التقدم للترقية الآن؟",
        "بناءً على تقييمي، ما نسبة الزيادة التي سأحصل عليها؟",
        "هل أستطيع الحصول على قرض طوارئ بدون فوائد؟",
        # Egyptian
        "أنا مستحق بونص السنه دي ولا لأ؟",
        "لو استقلت النهارده هاخد مكافأة نهاية خدمة قد إيه؟",
        "بناءً على تقييمي هاخد قد إيه زيادة؟",
        "أقدر آخد قرض طوارئ دلوقتي؟",
        # Franco
        "ana mosta7e2 el bonus el sana di wala la2?",
        "lw esta2elt en naharda hakhod end of service ad eih?",
        "3ala asas el rating bta3y hakhod 2ad eih zeyada?",
        "a2dar akhod 2ard taware2 delwa2ty?",
    ],

    "out_of_scope": [
        # English
        "What is the weather in Cairo today?",
        "Can you write a Python script to sort a list?",
        "What is the stock price of Amazon?",
        "What are the latest football results?",
        "Translate this sentence into French.",
        "What is the best restaurant near the office?",
        "Who is the CEO of Apple?",
        "What is 15% of 3500?",
        # Arabic
        "ما سعر الدولار اليوم؟",
        "ما أفضل مطعم في القاهرة؟",
        "اكتب لي قصيدة عن الربيع.",
        "ما هي نتائج مباريات كرة القدم أمس؟",
        # Egyptian
        "إيه سعر الدولار النهارده؟",
        "إيه أحسن مطعم في المعادي؟",
        "إيه نتيجة ماتش الأهلي إمبارح؟",
        "إزاي أعمل كيكة شوكولاتة؟",
        # Franco
        "eih se3r el dollar en naharda?",
        "eih a7san mat3am fel ma3adi?",
        "eih nateget match el ahly embareh?",
        "ezay a3mel chocolate cake?",
    ],
}


# =============================================================================
# EMBEDDING LAYER
# =============================================================================

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading embedding model: {EMBEDDING_MODEL} …")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("  Model loaded ✓")
    return _embedder


def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts. Prepends 'query: ' prefix for e5 models."""
    model   = _get_embedder()
    prefixed = [f"query: {t}" for t in texts]
    vecs    = model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-normalised vectors."""
    return float(np.dot(a, b))


# =============================================================================
# PROTOTYPE BUILDER
# =============================================================================

_prototypes: dict = {}     # intent -> np.ndarray (unit vector)


def build_prototypes():
    """
    Embed all seed phrases for each intent class and average them into a
    single prototype vector per class. Vectors are L2-normalised so cosine
    similarity == dot product.
    """
    global _prototypes
    print("\n── Building class prototypes …")
    for intent, seeds in PROTOTYPE_SEEDS.items():
        vecs = _embed(seeds)          # shape (n_seeds, dim)
        proto = vecs.mean(axis=0)
        norm  = np.linalg.norm(proto)
        if norm > 0:
            proto /= norm
        _prototypes[intent] = proto
        print(f"   {intent:15s}: {len(seeds)} seeds → proto dim={proto.shape[0]}")
    print("── Prototypes built ✓\n")


# =============================================================================
# CLASSIFIER
# =============================================================================

def classify(question: str) -> dict:
    """
    Embed the question and return the closest intent prototype.
    Returns a dict with intent, scores (per class), and margin.
    """
    if not _prototypes:
        raise RuntimeError("Call build_prototypes() before classify().")

    q_vec  = _embed([question])[0]   # shape (dim,)
    scores = {intent: _cosine_sim(q_vec, proto)
              for intent, proto in _prototypes.items()}

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_intent    = sorted_scores[0][0]
    margin        = sorted_scores[0][1] - sorted_scores[1][1]

    return {
        "intent":    top_intent,
        "scores":    scores,
        "margin":    round(margin, 4),
        "top_score": round(sorted_scores[0][1], 4),
    }


# =============================================================================
# CHECKPOINT HELPERS  (identical contract to experiment script)
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
        print(f"  ⚠  Orphan IDs: {', '.join(sorted(orphan_ids, key=_natural_sort_key))}")
    print(f"{'─'*60}\n")
    return pending


# =============================================================================
# HELPERS
# =============================================================================

def load_queries() -> List[dict]:
    p = Path(INTENT_QUERY_FILE)
    if not p.exists():
        raise FileNotFoundError(f"{INTENT_QUERY_FILE} not found.")
    return json.loads(p.read_text(encoding="utf-8"))


def intent_match(inferred: str, expected: str) -> bool:
    return inferred.strip().lower() == expected.strip().lower()


# =============================================================================
# REPORT BUILDER  (identical structure to experiment script for easy comparison)
# =============================================================================

def _print_and_save(df: pd.DataFrame):
    total = len(df)
    if total == 0:
        print("No data to report.")
        return

    n_pass = int(df["intent_pass"].sum())

    print(f"\n{'='*70}")
    print(f"BASELINE RESULTS  ({total} queries)  [embedding + prototype cosine sim]")
    print(f"{'='*70}\n")

    print("── OVERALL ───────────────────────────────────────────────────────")
    print(f"  Intent accuracy : {n_pass}/{total} = {n_pass/total:.1%}\n")

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

    failures = df[~df["intent_pass"].astype(bool)]
    if not failures.empty:
        print(f"── FAILURES ({len(failures)}) ─────────────────────────────────────────")
        for _, row in failures.iterrows():
            print(f"  {row['query_id']:5s} [{row['language']:8s}] "
                  f"expected={row['expected_intent']:12s} "
                  f"got={row['inferred_intent']:12s}  "
                  f"score={row.get('top_score','?')}  margin={row.get('margin','?')}")
        print()

    # Precision / Recall / F1
    prf_rows = []
    for lbl in INTENT_LABELS:
        tp = int(((df["expected_intent"] == lbl) & (df["inferred_intent"] == lbl)).sum())
        fp = int(((df["expected_intent"] != lbl) & (df["inferred_intent"] == lbl)).sum())
        fn = int(((df["expected_intent"] == lbl) & (df["inferred_intent"] != lbl)).sum())
        prec    = tp / (tp + fp) if (tp + fp) else 0.0
        rec     = tp / (tp + fn) if (tp + fn) else 0.0
        f1      = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = int((df["expected_intent"] == lbl).sum())
        prf_rows.append({"intent": lbl, "precision": round(prec, 3),
                         "recall": round(rec, 3), "f1": round(f1, 3), "support": support})

    prf_df = pd.DataFrame(prf_rows).set_index("intent")
    macro  = prf_df[["precision", "recall", "f1"]].mean().round(3)
    prf_df.loc["macro_avg"] = [macro["precision"], macro["recall"], macro["f1"],
                                int(df.shape[0])]

    print("── PRECISION / RECALL / F1 ───────────────────────────────────────")
    print(prf_df.to_string(), "\n")

    # Confusion matrix
    conf_df = pd.DataFrame(0, index=INTENT_LABELS, columns=INTENT_LABELS)
    for _, row in df.iterrows():
        exp = row["expected_intent"]
        got = row["inferred_intent"]
        if exp in conf_df.index and got in conf_df.columns:
            conf_df.loc[exp, got] += 1
    conf_df.index.name   = "expected \\ predicted"
    conf_df.columns.name = None
    print("── CONFUSION MATRIX (rows=expected, cols=predicted) ─────────────")
    print(conf_df.to_string(), "\n")

    # Cosine score distribution per intent
    if "top_score" in df.columns:
        score_dist = df.groupby("expected_intent")["top_score"].agg(["mean", "min", "max"]).round(4)
        print("── COSINE SCORE DISTRIBUTION (top score per query) ──────────────")
        print(score_dist.to_string(), "\n")

    # Save Excel
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw", index=False)

        pd.DataFrame([
            {"metric": "Method",          "value": "Embedding + Prototype Cosine Similarity"},
            {"metric": "Embedding model", "value": EMBEDDING_MODEL},
            {"metric": "Intent accuracy", "value": f"{n_pass/total:.1%}"},
            {"metric": "Total queries",   "value": total},
            {"metric": "Macro Precision", "value": f"{macro['precision']:.3f}"},
            {"metric": "Macro Recall",    "value": f"{macro['recall']:.3f}"},
            {"metric": "Macro F1",        "value": f"{macro['f1']:.3f}"},
            {"metric": "Report generated","value": datetime.now().isoformat()},
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
                    "expected_intent", "inferred_intent",
                    "top_score", "margin"]
            failures[[c for c in cols if c in failures.columns]].to_excel(
                writer, sheet_name="Failures", index=False)

    print(f"✓ Results saved to {OUTPUT_XLSX}")


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_experiment():
    print("\n🚀 Baseline — Embedding + Prototype Cosine Similarity")
    print(f"   Model      : {EMBEDDING_MODEL}")
    print(f"   Checkpoint : {CHECKPOINT_FILE}")

    done    = _load_checkpoint()
    queries = load_queries()
    pending = _audit_coverage(queries, done)

    if not pending:
        print("All queries already completed. Building report from checkpoint.\n")
        df = pd.DataFrame(list(done.values()))
        _print_and_save(df)
        return df

    # Build prototypes once before the loop
    build_prototypes()

    for q in tqdm(pending, desc="Baseline eval"):
        qid        = q["id"]
        question   = q["question"]
        lang       = q["language"]
        exp_intent = q["expected_intent"]
        topic      = q.get("topic", "")
        complexity = q.get("complexity", "")

        t_start = time.monotonic()

        try:
            result     = classify(question)
            inf_intent = result["intent"]
            elapsed_ms = (time.monotonic() - t_start) * 1000
            i_match    = intent_match(inf_intent, exp_intent)

            row = {
                "query_id":        qid,
                "language":        lang,
                "complexity":      complexity,
                "topic":           topic,
                "expected_intent": exp_intent,
                "inferred_intent": inf_intent,
                "intent_pass":     i_match,
                "top_score":       result["top_score"],
                "margin":          result["margin"],
                "scores_policy":           round(result["scores"].get("policy",        0), 4),
                "scores_personal":         round(result["scores"].get("personal",      0), 4),
                "scores_hybrid":           round(result["scores"].get("hybrid",        0), 4),
                "scores_out_of_scope":     round(result["scores"].get("out_of_scope",  0), 4),
                "elapsed_ms":      round(elapsed_ms, 1),
                "timestamp":       datetime.now().isoformat(),
            }
            _save_checkpoint(row)
            done[qid] = row

            status = "✓" if i_match else "✗"
            tqdm.write(
                f"  [{status}] {qid:5s} [{lang:8s}] "
                f"expected={exp_intent:12s} got={inf_intent:12s} "
                f"score={result['top_score']:.3f} margin={result['margin']:.3f}"
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            print(f"\n  [ERR] {qid}: {e}")
            row = {
                "query_id":        qid,
                "language":        lang,
                "complexity":      complexity,
                "topic":           topic,
                "expected_intent": exp_intent,
                "inferred_intent": "ERROR",
                "intent_pass":     False,
                "top_score":       None,
                "margin":          None,
                "elapsed_ms":      round(elapsed_ms, 1),
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
    if "--report" in sys.argv:
        done = _load_checkpoint()
        if not done:
            print(f"No checkpoint found at {CHECKPOINT_FILE}.")
            sys.exit(1)
        queries = load_queries()
        _audit_coverage(queries, done)
        df = pd.DataFrame(list(done.values()))
        _print_and_save(df)

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