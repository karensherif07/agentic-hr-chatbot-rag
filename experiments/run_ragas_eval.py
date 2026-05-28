"""
run_ragas_eval.py – RAGAS evaluation from a generated answers checkpoint.
"""

import argparse
import asyncio
import json
import os
import re
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--resume",       action="store_true")
parser.add_argument("--generated-checkpoint", default="generation_checkpoint.json")
parser.add_argument("--checkpoint",   default="ragas_checkpoint.json")
parser.add_argument("--output",       default="ragas_results.json")
parser.add_argument("--errors-log",   default="ragas_errors.json")
parser.add_argument("--batch-size",   type=int,   default=2)
parser.add_argument("--batch-sleep",  type=float, default=30.0)
args = parser.parse_args()

BASE_WAIT = 20
MAX_WAIT  = 120

# ── Windows asyncio fix ───────────────────────────────────────────────────────
# Python 3.12 on Windows has broken timeout handling with the default
# ProactorEventLoop. Force SelectorEventLoop to fix it.
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ═════════════════════════════════════════════════════════════════════════════
# 1.  LLM SETUP
# ═════════════════════════════════════════════════════════════════════════════
GROQ_KEY   = os.environ["GROQ_API_KEY"]
GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")

from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

RUN_CONFIG = RunConfig(timeout=180, max_retries=5, max_wait=60)

def _groq(model: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_KEY, model_name=model, temperature=0)

if GEMINI_KEY:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        _eval_base = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GEMINI_KEY,
            temperature=0,
        )
        # Force Groq — Gemini free tier exhausted
        EVAL_LLM = LangchainLLMWrapper(langchain_llm=_groq("llama-3.3-70b-versatile"))
        print("✓ Evaluator: llama-3.3-70b-versatile via Groq")
    except ImportError:
        EVAL_LLM = LangchainLLMWrapper(langchain_llm=_groq("llama-3.1-8b-instant"))
        print("WARNING: langchain-google-genai not installed → llama-3.1-8b-instant")
else:
    EVAL_LLM = LangchainLLMWrapper(langchain_llm=_groq("llama-3.1-8b-instant"))
    print("INFO: No GOOGLE_API_KEY → llama-3.1-8b-instant")

try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HFEmb
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HFEmb

_hf_emb = _HFEmb(model_name="intfloat/multilingual-e5-large")
EVAL_EMBEDDINGS = LangchainEmbeddingsWrapper(embeddings=_hf_emb)
print("✓ Embeddings: intfloat/multilingual-e5-large")

# ── Smoke-test the LLM before starting ───────────────────────────────────────
print("Testing evaluator LLM connectivity...")
try:
    _test = EVAL_LLM.langchain_llm.invoke("Reply with the single word OK")
    print(f"✓ LLM reachable: {str(_test.content)[:60]}")
except Exception as _e:
    print(f"✗ LLM test FAILED: {_e}")
    raise SystemExit(1)

# ═════════════════════════════════════════════════════════════════════════════
# 2.  RAGAS IMPORTS
# ═════════════════════════════════════════════════════════════════════════════
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from ragas import SingleTurnSample

METRICS = [faithfulness, answer_relevancy, answer_correctness]

# Wire up llm/embeddings on each metric object directly — required when
# calling single_turn_ascore() outside of evaluate()
from ragas.metrics import AnswerSimilarity

# Wire up llm/embeddings on each metric directly
for _m in METRICS:
    _m.llm = EVAL_LLM
    if hasattr(_m, "embeddings"):
        _m.embeddings = EVAL_EMBEDDINGS

# answer_correctness internally uses AnswerSimilarity — must be set explicitly
_ans_sim = AnswerSimilarity(embeddings=EVAL_EMBEDDINGS)
answer_correctness.answer_similarity = _ans_sim

# ═════════════════════════════════════════════════════════════════════════════
# 3.  SCORING — direct async calls, no evaluate(), no nest_asyncio
# ═════════════════════════════════════════════════════════════════════════════
async def _score_all(sample: SingleTurnSample) -> dict:
    """Score all metrics concurrently with a hard timeout per metric."""
    async def _one(metric):
        try:
            return metric.name, await asyncio.wait_for(
                metric.single_turn_ascore(sample),
                timeout=180,
            )
        except asyncio.TimeoutError:
            print(f"\n  ⚠️  {metric.name} timed out")
            return metric.name, None
        except Exception as exc:
            print(f"\n  ⚠️  {metric.name} error: {exc}")
            return metric.name, None

    pairs = await asyncio.gather(*[_one(m) for m in METRICS])
    return dict(pairs)


def ragas_score(question: str, answer: str,
                contexts: list, reference: str) -> dict:
    if not contexts:
        contexts = ["(no context retrieved)"]

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )

    wait = BASE_WAIT
    for attempt in range(5):
        try:
            # Fresh event loop every call — safe because we're single-threaded
            scores_raw = asyncio.run(_score_all(sample))

            scores = {
                k: (None if v is None or v != v else round(float(v), 4))
                for k, v in scores_raw.items()
            }

            if all(v is None for v in scores.values()):
                raise RuntimeError("All metrics returned None — LLM unreachable?")

            return scores

        except RuntimeError as exc:
            # "All None" or similar transient failure → retry with backoff
            if attempt < 4:
                print(f"\n  ⏳ Attempt {attempt+1} failed ({exc}) — waiting {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, MAX_WAIT)
                continue
            raise

        except Exception as exc:
            msg = str(exc).lower()
            is_transient = any(t in msg for t in ("429", "rate_limit", "rate limit",
                                                   "timeout", "timed out", "503",
                                                   "unavailable", "overloaded"))
            if is_transient and attempt < 4:
                print(f"\n  ⏳ Transient error ({type(exc).__name__}: {exc}) — waiting {wait}s...")
                time.sleep(wait)
                wait = min(wait * 2, MAX_WAIT)
                continue
            raise

    raise RuntimeError("RAGAS scoring retries exhausted.")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  CHECKPOINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

def _save(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ═════════════════════════════════════════════════════════════════════════════
# 5.  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════
def main():
    gen_path = Path(args.generated_checkpoint)
    if not gen_path.exists():
        print(f"ERROR: Generated checkpoint not found: {gen_path}")
        return

    gen_data = _load(gen_path, {})
    if not gen_data:
        print(f"ERROR: {gen_path} is empty or invalid.")
        return

    records = list(gen_data.values())
    print(f"Loaded {len(records)} generated answers from {gen_path}")

    done   = _load(Path(args.checkpoint), {}) if args.resume else {}
    errors = _load(Path(args.errors_log),  []) if args.resume else []

    remaining = [rec for rec in records if rec["query_id"] not in done]
    print(f"Queries — total: {len(records)} | done: {len(done)} | remaining: {len(remaining)}\n")

    if not remaining:
        print("No queries to evaluate. Exiting.")
        return

    batch_n = 0

    for rec in tqdm(remaining, desc="Evaluating"):
        qid = rec["query_id"]
        try:
            question  = rec["question"]
            answer    = rec["answer"]
            contexts  = rec.get("contexts", [])
            reference = rec["reference"]

            scores = ragas_score(question, answer, contexts, reference)

            result_rec = {
                "query_id":   qid,
                "concept_id": rec.get("concept_id"),
                "language":   rec.get("language"),
                "complexity": rec.get("complexity"),
                "source_doc": rec.get("source_doc"),
                "topic":      rec.get("topic"),
                "question":   question,
                "answer":     answer,
                "reference":  reference,
                "scores":     scores,
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }

            done[qid] = result_rec
            _save(Path(args.checkpoint), done)

            score_str = "  ".join(
                f"{k}={v:.3f}" if v is not None else f"{k}=None"
                for k, v in scores.items()
            )
            tqdm.write(f"  ✓ {qid} | {rec.get('language','?')} | {score_str}")

        except Exception as exc:
            errors.append({
                "query_id":  qid,
                "language":  rec.get("language", ""),
                "question":  rec.get("question", ""),
                "error":     str(exc),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            _save(Path(args.errors_log), errors)
            tqdm.write(f"  ✗ {qid} — {exc}")
            continue

        batch_n += 1
        if batch_n % args.batch_size == 0:
            tqdm.write(f"\n  💤 Batch of {args.batch_size} done. Sleeping {args.batch_sleep}s...\n")
            time.sleep(args.batch_sleep)

    results = list(done.values())
    _save(Path(args.output), results)
    print(f"\n✅ Saved {len(results)} scored results → {args.output}")
    if errors:
        print(f"⚠️  {len(errors)} failed → {args.errors_log}  (re-run with --resume to retry)")

    _print_summary(results)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
def _print_summary(results: list):
    import statistics

    METRIC_COLS  = ["faithfulness", "answer_relevancy", "answer_correctness"]
    LANGS        = ["english", "arabic", "egyptian", "franco"]
    COMPLEXITIES = ["L1", "L2", "L3", "L4"]

    def _avg(vals):
        v = [x for x in vals if x is not None]
        return statistics.mean(v) if v else None

    def _std(vals):
        v = [x for x in vals if x is not None]
        return statistics.stdev(v) if len(v) > 1 else 0.0

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    sep = "=" * 70

    print(f"\n{sep}")
    print("TABLE 1 — By Language")
    print(sep)
    header = f"{'Language':<12}" + "".join(f"{m:>22}" for m in METRIC_COLS)
    print(header)
    print("-" * len(header))
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r["language"]].append(r)
    for lang in LANGS:
        recs = by_lang[lang]
        row  = f"{lang:<12}"
        for m in METRIC_COLS:
            vals = [r["scores"].get(m) for r in recs]
            row += f"{_fmt(_avg(vals)):>22}"
        print(row)

    print(f"\n{sep}")
    print("TABLE 2 — By Complexity")
    print(sep)
    by_comp = defaultdict(list)
    for r in results:
        by_comp[r["complexity"]].append(r)
    print(f"{'Complexity':<12}" + "".join(f"{m:>22}" for m in METRIC_COLS) + f"{'N':>6}")
    for c in COMPLEXITIES:
        recs = by_comp[c]
        row  = f"{c:<12}"
        for m in METRIC_COLS:
            vals = [r["scores"].get(m) for r in recs]
            row += f"{_fmt(_avg(vals)):>22}"
        row += f"{len(recs):>6}"
        print(row)

    print(f"\n{sep}")
    print("TABLE 3 — Answer Correctness: Language × Complexity")
    print(sep)
    print(f"{'':12}" + "".join(f"{c:>10}" for c in COMPLEXITIES))
    for lang in LANGS:
        row = f"{lang:<12}"
        for c in COMPLEXITIES:
            recs = [r for r in results
                    if r["language"] == lang and r["complexity"] == c]
            vals = [r["scores"].get("answer_correctness") for r in recs]
            row += f"{_fmt(_avg(vals)):>10}"
        print(row)

    print(f"\n{sep}")
    print(f"OVERALL (n={len(results)})")
    print(sep)
    for m in METRIC_COLS:
        vals = [r["scores"].get(m) for r in results]
        print(f"  {m:<25} mean={_fmt(_avg(vals))}  std={_fmt(_std(vals))}")
    print(sep)


if __name__ == "__main__":
    main()