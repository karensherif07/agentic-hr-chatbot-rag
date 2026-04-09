"""
HR Chatbot — Manual Test Grader 

"""
import json
import os
import sys
import datetime
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

QUESTIONS_FILE = "test_questions.json"
RESULTS_DIR = "results"

POLICY_SUMMARY = """
Horizon Tech HR Policy — Key Facts for Grading: 

LEAVE: 14 days (0-1yr), 21 days (1-5yr), 25 days (5-10yr), 30 days (10+yr).
Carryover: 7 days (1-5yr), 10 days (5-10yr), 14 days (10+yr).
Maternity: 90 days paid. Paternity: 10 days paid (birth cert required).
Marriage: 5 days. Bereavement: 3-5 days. Hajj: 20 days once. Study: up to 10 days/yr.

TRAVEL: Short flights (<6hrs): Premium Economy all. Long flights: Premium Economy up to 3G,
Business 4G+. Daily allowance: $100 domestic, $200 Arab region, $250 international, $350 London/NYC.
Expense report due within 7 working days of return.

REMOTE WORK: Hybrid (3 days office) needs manager approval. Full remote needs CEO + HR.
Requirements: 20 Mbps internet, camera on, respond within 30 mins. Allowance: 800 EGP/month.

DISCIPLINE: Step1 verbal warning (3mo), Step2 first written (6mo), Step3 final written (12mo),
Step4 PIP (90 days), Step5 termination. Gross misconduct = immediate termination (theft, fraud,
assault, data breach, alcohol/drugs, disclosing confidential info).

GIFTS: <$50 acceptable with logging. $50-$200 mandatory report to HR. >$200 refuse/donate.
Cash gifts (any amount): always refuse. Violations = disciplinary action.

LEARNING: Budget: $2,500 (0-1yr), $5,000 (1-5yr), $8,000 (manager), $12,000 (exec 4G+).
Scholarship: up to $15,000/yr for relevant postgrad. Must stay 2 years post-grant. Apply Oct-Nov.

PERFORMANCE: Bi-annual reviews (June + December). Rating 5: 20% raise, 1.5x bonus.
Rating 4: 15% raise, 1.25x bonus. Rating 3: 8% raise, 1x bonus. Rating 1-2: 0%.
Bonus requires 6+ months service. PIP = no bonus.

SALARY GRADES (EGP/month): G1: 8k-15k, G2: 15k-28k, G3: 28k-45k, G4: 45k-70k, G5: 70k-120k.
Allowances: Transport 1,500 EGP (all), Mobile 500 EGP (G2+), Housing 20% base (G4+).

NOTICE PERIODS: 1G-2G: 30 days. 3G-4G: 60 days. 5G+: 90 days.
GRATUITY: 1-3yrs: 0.5 salary/yr. 3-5yrs: 1 salary/yr. 5+yrs: 1.5 salary/yr.
Not paid for gross misconduct dismissal.

PROBATION: 3 months (extendable to 6). Either party: 7 days notice. Health insurance from day 1.

AI TOOLS: Allowed: drafting, summarizing public docs, research. Banned: client data, proprietary code,
unapproved tools. Report security breaches within 1 hour. Change passwords every 90 days.
"""


def grade_with_groq(client, question, chatbot_answer, expected_summary, language):
    prompt = f"""You are grading a chatbot's response to an HR policy question.

POLICY REFERENCE:
{POLICY_SUMMARY}

LANGUAGE OF TEST: {language}
QUESTION ASKED: {question}
EXPECTED ANSWER (summary): {expected_summary}
CHATBOT'S ACTUAL RESPONSE: {chatbot_answer}

Grade this response strictly. Return ONLY a JSON object with these exact fields:
{{
  "verdict": "PASS" | "PARTIAL" | "FAIL",
  "score": 0-10,
  "reason": "one sentence explanation",
  "missing": "what key info was missing or wrong (empty string if PASS)"
}}

Grading criteria:
- PASS (8-10): Correct answer with the right numbers/details, appropriate language.
- PARTIAL (4-7): Mostly correct but missing a key detail, number, or condition.
- FAIL (0-3): Wrong answer, hallucinated numbers, or missed the point entirely.
- If the chatbot said it doesn't know but the policy covers it: FAIL.
- Language mismatch (e.g. asked in Egyptian Arabic but answered in English): deduct 2 points.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def color(text, code):
    return f"\033[{code}m{text}\033[0m"


def print_banner():
    print("\n" + "="*60)
    print("  Horizon Tech HR Chatbot — Test Grader")
    print("="*60)
    print(color("  Auto-graded by Groq (Llama 3) | Manual mode", "36"))
    print("="*60 + "\n")


def run_tests():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(color("ERROR: GROQ_API_KEY not set.", "31"))
        print('Run: $env:GROQ_API_KEY="gsk_your_key_here"')
        sys.exit(1)

    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(color(f"ERROR: {QUESTIONS_FILE} not found.", "31"))
        sys.exit(1)

    chains = data["chains"]

    print("Available test chains:\n")
    for i, chain in enumerate(chains):
        print(f"  [{i+1:2d}] {chain['id']:12s} | {chain['language']:15s} | {chain['topic']}")

    print("\nEnter chain numbers (e.g. 1,3,5) or press Enter for ALL:")
    choice = input(">>> ").strip()

    if choice:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            selected = [chains[i] for i in indices]
        except (ValueError, IndexError):
            print(color("Invalid selection. Running all.", "33"))
            selected = chains
    else:
        selected = chains

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")
    report_file = os.path.join(RESULTS_DIR, f"report_{timestamp}.txt")

    all_results = []
    total_pass = total_partial = total_fail = 0

    for chain in selected:
        print("\n" + color(f"─── Chain {chain['id']}: {chain['topic']} ({chain['language']}) ───", "33"))
        chain_results = {"chain_id": chain["id"], "language": chain["language"],
                         "topic": chain["topic"], "turns": []}
        prev_qa = []

        for turn in chain["turns"]:
            print(f"\n  Turn {turn['turn']} of {len(chain['turns'])}")

            if prev_qa:
                print(color("  [Context — previous turns in this chain]", "36"))
                for p in prev_qa:
                    print(color(f"  Q: {p['q']}", "36"))
                    print(color(f"  A: {p['a'][:80]}{'...' if len(p['a'])>80 else ''}", "36"))

            print(color(f"\n  ┌─ Copy this question into your chatbot:", "32"))
            print(color(f"  │  {turn['question']}", "97"))
            print(color(f"  └─ (Follow-up)" if turn["turn"] > 1 else "  └─ (New chat session)", "90"))

            print("\n  Paste response (Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "" and (not lines or lines[-1] == ""):
                    if lines: break
                    else: continue
                lines.append(line)
            chatbot_answer = "\n".join(lines).strip()

            if not chatbot_answer:
                print(color("  Skipped.", "33"))
                continue

            print(color("  Grading...", "36"), end=" ", flush=True)
            
            try:
                grade = grade_with_groq(client, turn["question"], chatbot_answer,
                                       turn["expected_answer_summary"], chain["language"])

                verdict = grade["verdict"]
                score = grade["score"]
                reason = grade["reason"]
                missing = grade.get("missing", "")

                if verdict == "PASS":
                    total_pass += 1
                    v_colored = color(f"✓ PASS  ({score}/10)", "32")
                elif verdict == "PARTIAL":
                    total_partial += 1
                    v_colored = color(f"~ PARTIAL ({score}/10)", "33")
                else:
                    total_fail += 1
                    v_colored = color(f"✗ FAIL  ({score}/10)", "31")

                print(f"\r  Grade: {v_colored}")
                print(f"  Reason: {reason}")
                if missing:
                    print(color(f"  Missing: {missing}", "31"))

                chain_results["turns"].append({
                    "turn": turn["turn"],
                    "question": turn["question"],
                    "chatbot_answer": chatbot_answer,
                    "verdict": verdict,
                    "score": score,
                    "reason": reason,
                    "missing": missing
                })
                prev_qa.append({"q": turn["question"], "a": chatbot_answer})
            except Exception as e:
                print(color(f"\r  Error during grading: {e}", "31"))

        all_results.append(chain_results)

    # Final Summary Logic
    total = total_pass + total_partial + total_fail
    print("\n" + "="*60)
    print(color("  FINAL SUMMARY", "97"))
    print("="*60)
    print(f"  Total questions: {total}")
    print(color(f"  PASS    : {total_pass}", "32"))
    print(color(f"  PARTIAL : {total_partial}", "33"))
    print(color(f"  FAIL    : {total_fail}", "31"))

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"summary": {"pass": total_pass, "partial": total_partial, "fail": total_fail, "total": total},
                   "chains": all_results}, f, ensure_ascii=False, indent=2)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Horizon Tech Chatbot Test Report\nGenerated: {datetime.datetime.now()}\n")
        f.write(f"PASS: {total_pass} | PARTIAL: {total_partial} | FAIL: {total_fail}\n")

    print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    print_banner()
    run_tests()