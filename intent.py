"""
intent.py
Classifies a user question into:
  - "personal"  → answer from DB only (employee's own data)
  - "policy"    → answer from RAG only (existing PDF flow)
  - "hybrid"    → needs both DB data AND policy context

Also extracts the intent_topic so personal_data.py can run only the
relevant queries instead of fetching everything.

topic values: leave | performance | salary | training | okr |
              profile | disciplinary | all
"""

import re


# ── Keyword maps ───────────────────────────────────────────────
# Each entry: (pattern, topic)
# Checked in order — first match wins.

PERSONAL_PATTERNS = [
    # Leave / absence
    (r"my (leave|vacation|days off|remaining|balance|annual|sick|maternity|paternity|hajj|bereavement|absence)",  "leave"),
    (r"(how many|how much).*(leave|days|vacation|off).*(i have|left|remaining|got|balance)",                       "leave"),
    (r"(رصيد|إجازت|إجازتي|أيام متبقية|يتبقى لي|كم يوم)",                                                         "leave"),
    (r"(agazti|raseed bta3i|raseed agaza|kam yom 3andi|el balance bta3i)",                                         "leave"),
    (r"(pending leave|طلب إجازة|talab agaza|my leave request)",                                                    "leave"),

    # Performance
    (r"my (performance|rating|review|score|appraisal|evaluation)",                                                 "performance"),
    (r"(describe|summarise|show).*(my performance|my ratings|how i.ve been doing)",                                "performance"),
    (r"(تقييمي|أدائي|تقرير أدائي|درجتي|مراجعتي)",                                                                "performance"),
    (r"(taqyimi|adai|el rating bta3i|describe adai|kif adai)",                                                     "performance"),
    (r"(am i improving|my trend|how have i improved|my scores over)",                                              "performance"),

    # Salary / payroll
    (r"my (salary|pay|payslip|net|gross|allowance|payroll|income|compensation)",                                   "salary"),
    (r"(راتبي|مرتبي|بدلاتي|صافي الراتب|إجمالي الراتب)",                                                          "salary"),
    (r"(ratbi|mashy|el rateb bta3i|flous bta3ti|salary bta3i)",                                                    "salary"),

    # Training & development
    (r"my (training|budget|course|learning|development|certification)",                                            "training"),
    (r"(how much|how many).*(training|budget|courses).*(left|remaining|used|spent|have)",                          "training"),
    (r"(ميزانيتي التدريبية|دوراتي|رصيد التدريب)",                                                                 "training"),
    (r"(budget bta3i|training bta3i|courses bta3ti|sneen el tdrb)",                                                "training"),

    # OKRs / goals
    (r"my (okr|goal|objective|target|kpi|key result)",                                                             "okr"),
    (r"(أهدافي|مؤشراتي|okr بتاعتي|okr الخاصة بي)",                                                              "okr"),
    (r"(okrs bta3ti|goals bta3ti|objectives bta3ti)",                                                              "okr"),

    # Profile / general
    (r"(my profile|my grade|my department|my manager|my hire date|when did i join|my role|my title)",             "profile"),
    (r"(درجتي الوظيفية|قسمي|مديري|تاريخ التحاقي)",                                                              "profile"),
    (r"(grade bta3i|2ismi|my info|my details|manager bta3i)",                                                      "profile"),

    # Disciplinary
    (r"(my warning|my pip|my disciplinary|any action against me|am i on pip)",                                     "disciplinary"),
    (r"(إنذاراتي|هل عليّ إنذار|خطة تحسين أدائي)",                                                               "disciplinary"),
]

HYBRID_PATTERNS = [
    r"can i (take|request|apply).*(more leave|another|this type)",
    r"(am i eligible|do i qualify|هل يحق لي|هل أستحق|momken|ayezna3raf).*(leave|bonus|scholarship|promotion|training)",
    r"(entitled to|استحق|bystahel).*(based on|according to)",
    r"(how much more|كم يتبقى|remaining).*(allowed|policy|يسمح)",
    r"(my .* compared to policy|does my .* meet)",
    r"(can i still get|هل يمكنني الحصول على|momken a5od).*(bonus|scholarship|leave|promotion)",
]


def _match(text: str, patterns: list) -> tuple[bool, str]:
    """Returns (matched, topic). topic is None for HYBRID_PATTERNS."""
    t = text.lower()
    for entry in patterns:
        if isinstance(entry, tuple):
            pattern, topic = entry
            if re.search(pattern, t):
                return True, topic
        else:
            if re.search(entry, t):
                return True, None
    return False, None


def classify_intent(question: str) -> tuple[str, str]:
    """
    Returns (intent, topic).
    intent: "personal" | "policy" | "hybrid"
    topic:  "leave" | "performance" | "salary" | "training" |
            "okr" | "profile" | "disciplinary" | "all" | None
    """
    is_hybrid, _ = _match(question, HYBRID_PATTERNS)
    if is_hybrid:
        # Still need to know which topic to fetch from DB
        _, topic = _match(question, PERSONAL_PATTERNS)
        return "hybrid", (topic or "all")

    is_personal, topic = _match(question, PERSONAL_PATTERNS)
    if is_personal:
        return "personal", (topic or "all")

    return "policy", None