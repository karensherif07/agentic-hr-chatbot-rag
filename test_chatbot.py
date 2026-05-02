"""
test_chatbot.py — HR Chatbot Test Runner
=========================================
Runs test cases through agent.py. Saves results to CSV + HTML.

Usage:
    python test_chatbot.py                    # all sections
    python test_chatbot.py --section A        # one section only
    python test_chatbot.py --section A --delay 5   # 5s between questions

TOKEN BUDGET (Groq free tier):
    llama-3.3-70b:    100,000 TPD  → ~37 questions/day (policy only)
    gemma2-9b-it:     100,000 TPD  → used for routing only (separate quota)
    qwen3-32b:        14,400 TPM   → used for Arabic answers
    llama-3.1-8b:     500,000 TPD  → critique only

STRATEGY: Run 1-2 sections per day to stay within limits.
    Day 1: --section A --section B
    Day 2: --section C --section D
    etc.
"""

import os
import csv
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

EMPLOYEE_IDS = {
    "karen":   6,
    "junior":  11,
    "senior":  5,
    "analyst": 9,
    "eng2":    8,
}
DEFAULT_EMPLOYEE = "karen"
OUTPUT_CSV  = "test_results.csv"
OUTPUT_HTML = "test_results.html"

# ── Test cases ────────────────────────────────────────────────────────────────
# (section, id, question, employee_key, expected_intent, expected_contains, notes)

TEST_CASES = [

    # SECTION A — POLICY · ENGLISH
    ("A","A01","What is the overtime rate on weekdays?","karen","policy",["1.5"],"EN p.2"),
    ("A","A02","How many days of annual leave do I get after 3 years of service?","karen","policy",["21"],"AR p.9"),
    ("A","A03","What documents do I need to submit for paternity leave?","karen","policy",["birth certificate"],"AR p.9"),
    ("A","A04","How many days is Hajj leave and is it paid?","karen","policy",["20","paid"],"AR p.9"),
    ("A","A05","What is the daily allowance for international travel outside the Arab region?","karen","policy",["250"],"AR p.4"),
    ("A","A06","Who must approve fully remote work?","karen","policy",["ceo","hr"],"AR p.5"),
    ("A","A07","What happens to my personal data if I leave the company?","karen","policy",["7","10","5"],"AR p.12"),
    ("A","A08","Can I get a bonus if I am on a PIP?","karen","policy",["no","not eligible","pip"],"EN p.4"),
    ("A","A09","What is the end-of-service gratuity for someone who worked 7 years?","karen","policy",["1.5"],"AR p.11"),
    ("A","A10","What are the promotion criteria?","karen","policy",["1 year","rating","3","manager","hr"],"EN p.5-6"),
    ("A","A11","How long does a verbal warning stay on record?","karen","policy",["3 month"],"EN p.9"),
    ("A","A12","What is the salary increment for a rating of 4?","karen","policy",["15","up to"],"EN p.5"),

    # SECTION B — POLICY · MSA ARABIC
    ("B","B01","كم يوم إجازة سنوية يحق لموظف بخبرة أكثر من 10 سنوات؟","karen","policy",["30"],"AR p.9"),
    ("B","B02","ما هي مدة إجازة الأمومة وهل هي مدفوعة؟","karen","policy",["90"],"AR p.9"),
    ("B","B03","ما قيمة بدل العمل عن بُعد الشهري؟","karen","policy",["800"],"AR p.5"),
    ("B","B04","ما شروط الحصول على المنحة الدراسية؟","karen","policy",["15,000","سنتين","أكتوبر"],"AR p.3"),
    ("B","B05","ما مراحل الإجراء التأديبي التدريجي؟","karen","policy",["تحذير","إنذار","PIP"],"AR/EN p.9"),
    ("B","B06","ما نطاق راتب موظف من الفئة G2؟","karen","policy",["15,000","28,000"],"EN p.3"),
    ("B","B07","متى تبدأ تغطية التأمين الصحي للموظف الجديد؟","karen","policy",["أول يوم"],"EN p.2"),
    ("B","B08","ما معايير الترقية الداخلية؟","karen","policy",["سنة","تقييم","مدير"],"EN p.5-6"),
    ("B","B09","ما مدة صلاحية الإنذار الكتابي الأول؟","karen","policy",["6"],"EN p.9"),
    ("B","B10","ما الخدمات النفسية المجانية المتاحة للموظفين؟","karen","policy",["جلسات","خط","إجازة"],"AR p.2"),
    ("B","B11","من يُصدر التحذير الشفهي؟","karen","policy",["المدير المباشر","المدير"],"EN p.9"),
    ("B","B12","ما فترة الإشعار عند استقالة موظف G4؟","karen","policy",["60"],"AR p.10"),

    # SECTION C — POLICY · EGYPTIAN ARABIC
    ("C","C01","الأجازة السنوية بتاعتي كام يوم لو أنا شغال 3 سنين؟","karen","policy",["21"],"AR p.9"),
    ("C","C02","لو اتجوزت محتاج كام يوم إجازة وبيتدفعوا؟","karen","policy",["5"],"AR p.9"),
    ("C","C03","الأوفرتايم بيتحسب إزاي لو اشتغلت في الويكند؟","karen","policy",["2"],"EN p.2"),
    ("C","C04","الراتب بتاع G3 بيبدأ من كام؟","karen","policy",["28,000","45,000"],"EN p.3"),
    ("C","C05","التأمين الصحي بيشمل إيه للأولاد؟","karen","policy",["80","90","50"],"EN p.3"),
    ("C","C06","لو rating بتاعي 5، هاخد كام زيادة في الراتب؟","karen","policy",["20","لحد"],"EN p.5"),
    ("C","C07","إيه شروط الترقية عندنا في الشركة؟","karen","policy",["سنة","تقييم","مدير"],"EN p.5-6"),
    ("C","C08","لو مش عارف أشتكي من حاجة في الشغل أعمل إيه؟","karen","policy",["hr","hotline","10"],"EN p.7"),
    ("C","C09","الموظف اللي بيشتغل remote بياخد بدل قد إيه؟","karen","policy",["800"],"AR p.5"),
    ("C","C10","لو اشتغلت 4 سنين واستقلت، هاخد مكافأة نهاية خدمة قد إيه؟","karen","policy",["راتب","كامل","سنة"],"AR p.11"),
    ("C","C11","إجازة الحج كام يوم وبتتاخد إزاي؟","karen","policy",["20","تصريح","مرة"],"AR p.9"),
    ("C","C12","لو عندي إنذار مكتوب ممكن أعترض عليه؟","karen","policy",["5","أيام","hr"],"EN p.9"),

    # SECTION D — POLICY · FRANCO ARABIC
    ("D","D01","emta el bonus bta3i byigi w ana lazem 3amel eh?","karen","policy",["6","shohoor","pip"],"EN p.4"),
    ("D","D02","law rating bta3i 4, el salary raise bta3i kamet?","karen","policy",["15","le7ad"],"EN p.5"),
    ("D","D03","mawa3id el shoghl el rasmeya eh?","karen","policy",["8","sa3at","7ad"],"EN p.2"),
    ("D","D04","el overtime byta3ed ezay law ashtaghalt yom agaza rasmi?","karen","policy",["2"],"EN p.2"),
    ("D","D05","agaza el gawaz kamet yom w lazem agib eh?","karen","policy",["5","3aqd","gawaz"],"AR p.9"),
    ("D","D06","el scholarship bta3et el dirasa shar6ha eh?","karen","policy",["15,000","seneteen","oktober"],"AR p.3"),
    ("D","D07","law ashtaghalt 7 sneen w 5arabt, hakhod compensation qad eh?","karen","policy",["1.5","rateb"],"AR p.11"),
    ("D","D08","el remote work badal bta3o kamet?","karen","policy",["800"],"AR p.5"),
    ("D","D09","eih elli momken yewdi le fasl fehri fel shoghl?","karen","policy",["serega","3enf","kohol"],"EN p.9"),
    ("D","D10","law 3andi inzar maktub, a2dar a3terid 3aleih?","karen","policy",["5","ayyam","hr"],"EN p.9"),

    # SECTION E — PERSONAL
    ("E","E01","How many annual leave days do I have left?","karen","personal",["27"],"remaining=27"),
    ("E","E02","What is my current net salary?","karen","personal",["53"],"net ~53570"),
    ("E","E03","What was my last performance rating?","karen","personal",[],"no reviews → say no data"),
    ("E","E04","كم يوم إجازة متبقي عندي؟","karen","personal",["27"],"MSA"),
    ("E","E05","أنا لسه في فترة التجربة ولا خلصت؟","karen","personal",["لا","not","probation"],"NOT IN PROBATION"),
    ("E","E06","الأجازات البتاعتي الواقفة دلوقتي إيه؟","karen","personal",[],"no pending leaves"),
    ("E","E07","راتبي الصافي بتاع الشهر ده كام؟","karen","personal",["53"],"Egyptian net salary"),
    ("E","E08","Do I have any active disciplinary actions against me?","karen","personal",["no","none"],"no records"),
    ("E","E09","raseed agaza bta3i kamet yom?","karen","personal",["27"],"Franco remaining days"),
    ("E","E10","ana lesa fi probation wla 5alaset?","karen","personal",[],"Franco not on probation"),
    ("E","E11","el okrs bta3ti 3amela ezay?","karen","personal",[],"Franco no OKRs"),
    ("E","E12","Am I still on probation?","junior","personal",["yes","probation","2026"],"junior IS on probation"),
    ("E","E13","How many annual leave days do I have left?","junior","personal",["14"],"junior remaining=14"),

    # SECTION F — HYBRID
    ("F","F01","What are my working hours?","karen","hybrid",["remote","8","hour"],"work_model + policy"),
    ("F","F02","Am I eligible for a bonus this year?","karen","hybrid",["eligible","yes"],"karen eligible"),
    ("F","F03","هل يحق لي التقدم للترقية؟","karen","hybrid",[],"MSA promotion check"),
    ("F","F04","ساعات شغلي إيه؟","karen","hybrid",["remote","8","ساعات"],"Egyptian hours"),
    ("F","F05","Can I take study leave?","karen","hybrid",["10","full-time","exam"],"karen eligible"),
    ("F","F06","mawa3id shoghl bta3ti eh?","karen","hybrid",["remote","8"],"Franco hours"),
    ("F","F07","momken akhod bonus el sana di?","karen","hybrid",["aywa","eligible"],"Franco bonus"),
    ("F","F08","هل أقدر آخد إجازة دلوقتي؟","junior","hybrid",["probation","فترة التجربة","لا"],"junior probation"),
    ("F","F09","Am I eligible for a bonus this year?","junior","hybrid",["no","probation","not eligible"],"junior not eligible"),

    # SECTION G — CALCULATIONS
    ("G","G01","If my rating is 4, what salary raise will I get?","karen","policy",["15","up to"],"up to 15%"),
    ("G","G02","If I've worked 7 years, what is my end-of-service gratuity?","karen","policy",["1.5"],"1.5x"),
    ("G","G03","What is my bonus with a rating of 5?","karen","policy",["1.5","multiplier"],"1.5x multiplier"),
    ("G","G04","My annual leave entitlement is 21 days. I used 8 and have 3 pending. How many are left?","karen","policy",["10"],"21-8-3=10"),
    ("G","G05","If I've worked 4 years and resign, what is my notice period?","karen","policy",[],"60 days G3"),

    # SECTION H — OUT OF SCOPE
    ("H","H01","What is the company's stock option plan?","karen","policy",["not available","policy"],"OOS"),
    ("H","H02","How do I reset my VPN?","karen","policy",["not available","policy"],"OOS"),
    ("H","H03","What is the company canteen menu?","karen","policy",["not available","policy"],"OOS"),
    ("H","H04","What is the exact bonus pool amount this year?","karen","policy",["not available","policy"],"OOS"),
    ("H","H05","What is the weather today?","karen","policy",["not available","policy"],"OOS"),

    # SECTION J — EDGE CASES
    ("J","J01","What are my working hours?","karen","hybrid",["remote"],"must be hybrid"),
    ("J","J02","What is the annual leave policy?","karen","policy",["14","21","25","30"],"policy only"),
    ("J","J03","Tell me about my grade","karen","personal",["G3","senior"],"personal profile"),
    ("J","J04","Am I on probation?","karen","personal",["not","no"],"personal status"),
    ("J","J05","raseed bta3i kamet","karen","personal",[],"Franco personal"),
    ("J","J06","My performance review — how am I doing?","karen","personal",[],"personal no data"),
    ("J","J07","If my rating is 4, what salary raise will I get?","karen","policy",["15","up to"],"policy rate"),
    ("J","J08","Can I work from home every day?","karen","hybrid",["remote","ceo","hr"],"hybrid"),
]

MULTITURN_CASES = [
    {
        "id": "MT01", "employee": "karen",
        "turns": [
            ("What is my annual leave balance?","personal",["27"],"T1: balance"),
            ("Can I take all of it at once?","hybrid",[],"T2: check policy"),
        ]
    },
    {
        "id": "MT02", "employee": "karen",
        "turns": [
            ("What are the promotion criteria?","policy",["1 year","rating","3"],"T1: criteria"),
            ("Do I meet them?","hybrid",[],"T2: check my data"),
        ]
    },
    {
        "id": "MT03", "employee": "karen",
        "turns": [
            ("My rating is 4.","policy",[],"T1: user states rating"),
            ("What raise will I get?","policy",["15","up to"],"T2: policy rate"),
        ]
    },
]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tests(section_filter=None, delay=8):
    print("Loading models...")
    from setup import setup
    from agent import run_agent
    from nlp_utils import detect_language_type, get_semantic_dialect
    from utils import build_history_str, is_no_info_answer

    (ar_index, en_index,
     routing_llm, en_llm, ar_llm, critique_llm,
     reranker, dialect_pipe, ara_tokenizer) = setup()

    print(f"Models loaded. Delay between questions: {delay}s\n")

    cases = TEST_CASES
    if section_filter:
        sections = [s.strip().upper() for s in section_filter.split(",")]
        cases = [c for c in TEST_CASES if c[0] in sections]
        print(f"Running sections {sections}: {len(cases)} cases\n")

    results = []

    for section, qid, question, emp_key, expected_intent, expected_contains, notes in cases:
        employee_id = EMPLOYEE_IDS.get(emp_key, EMPLOYEE_IDS[DEFAULT_EMPLOYEE])
        print(f"[{qid}] {question[:65]}...")

        start = time.time()
        try:
            lang    = detect_language_type(question)
            dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else None

            result = run_agent(
                question=question,
                employee_id=employee_id,
                lang=lang,
                dialect=dialect,
                history_str="",
                ar_index=ar_index,
                en_index=en_index,
                routing_llm=routing_llm,
                en_llm=en_llm,
                ar_llm=ar_llm,
                critique_llm=critique_llm,
                reranker=reranker,
                ara_tokenizer=ara_tokenizer,
                skip_critique=True,   # saves ~200 tokens per question in test mode
            )

            answer        = result["answer"]
            actual_intent = result["intent"]
            tools_called  = result["tools_called"]
            elapsed       = round(time.time() - start, 1)
            no_info       = is_no_info_answer(answer)

            answer_lower    = answer.lower()
            contains_checks = [
                f"{'✅' if t.lower() in answer_lower else '❌'} '{t}'"
                for t in expected_contains
            ]
            contains_pass = all(t.lower() in answer_lower for t in expected_contains)
            intent_match  = actual_intent == expected_intent
            overall       = "✅ PASS" if (intent_match and contains_pass) else "❌ FAIL"

            results.append({
                "section": section, "id": qid, "employee": emp_key,
                "question": question,
                "expected_intent": expected_intent,
                "actual_intent": actual_intent,
                "intent_ok": "✅" if intent_match else f"❌ got={actual_intent}",
                "lang": f"{lang}" + (f"/{dialect}" if dialect else ""),
                "tools_called": ", ".join(tools_called),
                "answer": answer, "no_info": no_info,
                "contains_checks": " | ".join(contains_checks) if contains_checks else "manual review",
                "contains_pass": "✅" if contains_pass else "❌",
                "overall": overall,
                "elapsed_sec": elapsed,
                "notes": notes,
            })

            icon = "✅" if overall == "✅ PASS" else "❌"
            print(f"   {icon} intent:{actual_intent} lang:{lang} {elapsed}s")
            if expected_contains and not contains_pass:
                missing = [t for t in expected_contains if t.lower() not in answer_lower]
                print(f"   Missing terms: {missing}")
            print(f"   {answer[:120]}\n")

        except Exception as e:
            results.append({
                "section": section, "id": qid, "employee": emp_key,
                "question": question,
                "expected_intent": expected_intent, "actual_intent": "ERROR",
                "intent_ok": "❌ ERROR", "lang": "", "tools_called": "",
                "answer": str(e), "no_info": True,
                "contains_checks": "", "contains_pass": "❌",
                "overall": "❌ ERROR",
                "elapsed_sec": round(time.time() - start, 1),
                "notes": notes,
            })
            print(f"   ❌ ERROR: {str(e)[:120]}\n")

        # Delay between questions to spread token usage over time
        time.sleep(delay)

    # Multi-turn tests (only if no section filter or section I requested)
    if not section_filter or "I" in (section_filter or "").upper():
        mt = run_multiturn_tests(
            ar_index, en_index,
            routing_llm, en_llm, ar_llm, critique_llm,
            reranker, dialect_pipe, ara_tokenizer,
            delay=delay,
        )
        results.extend(mt)

    _save_csv(results)
    _save_html(results)
    _print_summary(results)


def run_multiturn_tests(ar_index, en_index, routing_llm, en_llm, ar_llm, critique_llm,
                        reranker, dialect_pipe, ara_tokenizer, delay=8):
    from agent import run_agent
    from nlp_utils import detect_language_type, get_semantic_dialect
    from utils import build_history_str, is_no_info_answer

    results = []
    print("\n── Multi-turn conversations ──")

    for conv in MULTITURN_CASES:
        emp_id  = EMPLOYEE_IDS.get(conv["employee"], EMPLOYEE_IDS[DEFAULT_EMPLOYEE])
        history = []
        summary = ""

        for turn_idx, (question, expected_intent, expected_contains, notes) in enumerate(conv["turns"]):
            qid = f"{conv['id']}-T{turn_idx+1}"
            print(f"[{qid}] {question[:60]}...")
            start   = time.time()
            lang    = detect_language_type(question)
            dialect = get_semantic_dialect(question, dialect_pipe) if lang == "arabic" else None
            history_str = build_history_str(history, summary)

            try:
                result = run_agent(
                    question=question, employee_id=emp_id, lang=lang, dialect=dialect,
                    history_str=history_str, ar_index=ar_index, en_index=en_index,
                    routing_llm=routing_llm, en_llm=en_llm, ar_llm=ar_llm,
                    critique_llm=critique_llm, reranker=reranker,
                    ara_tokenizer=ara_tokenizer, skip_critique=True,
                )
                answer        = result["answer"]
                actual_intent = result["intent"]
                elapsed       = round(time.time() - start, 1)
                history.append({"role": "user",      "content": question})
                history.append({"role": "assistant",  "content": answer})

                answer_lower  = answer.lower()
                contains_pass = all(t.lower() in answer_lower for t in expected_contains)
                intent_match  = actual_intent == expected_intent
                overall       = "✅ PASS" if (intent_match and contains_pass) else "❌ FAIL"

                results.append({
                    "section": "I", "id": qid, "employee": conv["employee"],
                    "question": question,
                    "expected_intent": expected_intent, "actual_intent": actual_intent,
                    "intent_ok": "✅" if intent_match else f"❌ got={actual_intent}",
                    "lang": lang, "tools_called": ", ".join(result["tools_called"]),
                    "answer": answer, "no_info": is_no_info_answer(answer),
                    "contains_checks": " | ".join(
                        f"{'✅' if t.lower() in answer_lower else '❌'} '{t}'"
                        for t in expected_contains
                    ) or "manual review",
                    "contains_pass": "✅" if contains_pass else "❌",
                    "overall": overall, "elapsed_sec": elapsed, "notes": notes,
                })
                print(f"   {'✅' if overall == '✅ PASS' else '❌'} {actual_intent} {elapsed}s\n")

            except Exception as e:
                results.append({
                    "section": "I", "id": qid, "employee": conv["employee"],
                    "question": question,
                    "expected_intent": expected_intent, "actual_intent": "ERROR",
                    "intent_ok": "❌ ERROR", "lang": "", "tools_called": "",
                    "answer": str(e), "no_info": True,
                    "contains_checks": "", "contains_pass": "❌",
                    "overall": "❌ ERROR",
                    "elapsed_sec": round(time.time() - start, 1), "notes": notes,
                })
                print(f"   ❌ ERROR: {str(e)[:100]}\n")

            time.sleep(delay)

    return results


def _print_summary(results):
    total  = len(results)
    passed = sum(1 for r in results if r["overall"] == "✅ PASS")
    failed = sum(1 for r in results if "FAIL" in r["overall"])
    errors = sum(1 for r in results if "ERROR" in r["overall"])
    avg_t  = round(sum(r["elapsed_sec"] for r in results) / max(total, 1), 1)
    sections = {}
    for r in results:
        s = r["section"]
        sections.setdefault(s, {"pass": 0, "total": 0})
        sections[s]["total"] += 1
        if r["overall"] == "✅ PASS":
            sections[s]["pass"] += 1
    print(f"\n{'='*50}")
    print(f"SUMMARY — {total} questions")
    print(f"  PASS:  {passed}/{total} ({round(passed/max(total,1)*100)}%)")
    print(f"  FAIL:  {failed}  ERROR: {errors}  Avg: {avg_t}s")
    for s in sorted(sections):
        d   = sections[s]
        pct = round(d["pass"] / d["total"] * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        print(f"  {s}: {bar} {d['pass']}/{d['total']} ({pct}%)")
    print(f"\n→ {OUTPUT_HTML}")


def _save_csv(results):
    if not results:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def _save_html(results):
    snames = {
        "A":"Policy·English","B":"Policy·MSA","C":"Policy·Egyptian",
        "D":"Policy·Franco","E":"Personal","F":"Hybrid",
        "G":"Calculations","H":"Out-of-Scope","I":"Multi-turn","J":"Edge Cases",
    }
    rows = ""
    cur  = None
    for r in results:
        if r["section"] != cur:
            cur   = r["section"]
            label = snames.get(cur, cur)
            rows += f'<tr><td colspan="9" style="background:#1565c0;color:#fff;font-weight:bold;padding:6px">Section {cur} — {label}</td></tr>\n'
        bg  = "#e8f5e9" if r["overall"] == "✅ PASS" else "#ffebee"
        ans = r["answer"].replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
        rows += f"""<tr style="background:{bg}">
<td>{r['id']}</td><td>{r['employee']}</td>
<td style="max-width:200px">{r['question']}</td>
<td><b>{r['overall']}</b></td><td>{r['intent_ok']}</td>
<td>{r['lang']}</td><td style="font-size:.8em">{r['tools_called']}</td>
<td style="max-width:350px;font-size:.82em">{ans[:400]}</td>
<td style="font-size:.78em">{r['contains_checks']}</td></tr>\n"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>HR Chatbot Results {datetime.now().strftime('%Y-%m-%d')}</title>
<style>body{{font-family:Arial;font-size:13px;padding:20px}}
table{{border-collapse:collapse;width:100%}}
th{{background:#1565c0;color:#fff;padding:7px;text-align:left;position:sticky;top:0}}
td{{border:1px solid #ddd;padding:5px 7px;vertical-align:top}}</style></head>
<body><h1>HR Chatbot Test Results</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total: {len(results)} |
Pass: {sum(1 for r in results if r['overall']=='✅ PASS')} |
Fail: {sum(1 for r in results if 'FAIL' in r['overall'])}</p>
<table><tr><th>ID</th><th>Emp</th><th>Question</th><th>Result</th>
<th>Intent</th><th>Lang</th><th>Tools</th><th>Answer</th><th>Checks</th></tr>
{rows}</table></body></html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=str, default=None,
                        help="Section(s) to run, comma-separated. E.g. A or A,B")
    parser.add_argument("--delay", type=float, default=8,
                        help="Seconds between questions (default 8)")
    args = parser.parse_args()
    run_tests(section_filter=args.section, delay=args.delay)