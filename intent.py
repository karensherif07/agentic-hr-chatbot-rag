import re

# ─────────────────────────────────────────────────────────────
# intent.py
# Classifies a question into: personal | hybrid | policy
# Also extracts the data topic so personal_data.py fetches only
# what's needed: leave | performance | salary | training |
#                okr | profile | disciplinary | all
# ─────────────────────────────────────────────────────────────

PERSONAL_PATTERNS = [
    # Leave
    (r"my (leave|vacation|days off|remaining|balance|annual|sick|maternity|paternity|hajj|bereavement|absence|study)", "leave"),
    (r"(how many|how much).*(leave|days|vacation|off|study).*(i have|left|remaining|got|balance)", "leave"),
    (r"(رصيد|إجازت|إجازتي|أيام متبقية|يتبقى لي|كم يوم|دراسي)", "leave"),
    (r"(agazti|raseed bta3i|raseed agaza|kam yom 3andi|el balance bta3i)", "leave"),
    (r"(pending leave|طلب إجازة|talab agaza|my leave request)", "leave"),
    # Performance
    (r"my (performance|rating|review|score|appraisal|evaluation)", "performance"),
    (r"(describe|summarise|show).*(my performance|my ratings|how i.ve been doing)", "performance"),
    (r"(تقييمي|أدائي|تقرير أدائي|درجتي|مراجعتي)", "performance"),
    (r"(taqyimi|adai|el rating bta3i|describe adai)", "performance"),
    (r"(am i improving|my trend|my scores over)", "performance"),
    # Salary
    (r"my (salary|pay|payslip|net salary|gross salary|allowance|payroll|income|compensation)", "salary"),
    (r"(راتبي|مرتبي|بدلاتي|صافي الراتب|إجمالي الراتب)", "salary"),
    (r"(ratbi|el rateb bta3i|flous bta3ti|salary bta3i)", "salary"),
    # Training
    (r"my (training|budget|course|learning|development|certification)", "training"),
    (r"(how much|how many).*(training|budget|courses).*(left|remaining|used|spent)", "training"),
    (r"(ميزانيتي التدريبية|دوراتي|رصيد التدريب)", "training"),
    (r"(budget bta3i|training bta3i|courses bta3ti)", "training"),
    # OKRs
    (r"my (okr|goal|objective|target|kpi|key result)", "okr"),
    (r"(أهدافي|مؤشراتي|okr بتاعتي)", "okr"),
    (r"(okrs bta3ti|goals bta3ti)", "okr"),
    # Profile (grade, title, department, manager, hire date)
    # NOTE: working hours / shift / schedule NOT here — routed to hybrid below
    (r"(my profile|my grade|my job grade|my department|my manager|my hire date|when did i join|my role|my job title|my title)", "profile"),
    (r"(درجتي الوظيفية|قسمي|مديري|تاريخ التحاقي)", "profile"),
    (r"(grade bta3i|manager bta3i|my info|my details)", "profile"),
    # Disciplinary
    (r"(my warning|my pip|my disciplinary|any action against me|am i on pip)", "disciplinary"),
    (r"(إنذاراتي|هل عليّ إنذار|خطة تحسين أدائي)", "disciplinary"),
    # Probation — personal only
    (r"(my probation|am i on probation|probation status|probation requirements|فترة التجربة|فترة الاختبار)", "profile"),
]

HYBRID_PATTERNS = [
    # Working hours / schedule: DB (work_model) + policy (shift / hours) → hybrid
    r"(what are my working hours|what are my hours|my working hours|my work hours|hours do i work|how many hours"
    r".*(per day|a day|do i work|weekly)|when do i (start|finish|clock in|work)|my shift|my schedule|my start time"
    r"|my work schedule|am i (remote|hybrid|office|on-?site)|work from home|ساعات عملي|مواعيد عملي|عدد ساعات|جدول عملي"
    r"|mawa3id shoghl|working hours bta3i|kam sa3a|sa3at el shoghl)",
    # Can I take more leave — needs balance + policy entitlement
    r"can i (take|get|have|request|apply).*(more|additional|extra|another|any).*(leave|agaza)",
    r"(can i).*(study|sick|annual|maternity|paternity|hajj).*(leave|agaza)",
    r"(am i eligible|do i qualify|هل يحق لي|هل أستحق|momken|ayezna3raf).*(leave|bonus|scholarship|promotion|training)",
    r"(هل يمكنني|هل استطيع).*(إجازة|اجازه)",
    r"(how much more|كم يتبقى|remaining).*(allowed|policy|يسمح)",
    r"(can i still get|هل يمكنني الحصول على|momken a5od).*(bonus|scholarship|leave|promotion)",
    # Remind me style (needs both profile from DB and policy context)
    r"(remind me|tell me|what is|what are|وريني|فكرني).*(my).*(grade|hours|schedule|salary|leave|balance|shift)",
]


def _match(text: str, patterns: list) -> tuple[bool, str | None]:
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


def classify_intent(question: str) -> tuple[str, str | None]:
    """
    Returns (intent, topic).
    intent : "personal" | "hybrid" | "policy"
    topic  : "leave" | "performance" | "salary" | "training" |
             "okr" | "profile" | "disciplinary" | "all" | None
    """
    # Hybrid check first (working hours, eligibility questions, etc.)
    is_hybrid, _ = _match(question, HYBRID_PATTERNS)
    if is_hybrid:
        _, topic = _match(question, PERSONAL_PATTERNS)
        return "hybrid", (topic or "all")

    # Personal check
    is_personal, topic = _match(question, PERSONAL_PATTERNS)
    if is_personal:
        return "personal", (topic or "all")

    # Default: policy RAG
    return "policy", None