import json
from langchain_core.prompts import PromptTemplate


def format_personal_data(data: dict) -> str:
    """
    Converts personal data dict to a compact readable block.
    Only emits sections that have actual data — no empty headers.
    Skips zero-value leave types to reduce noise.
    """
    lines = []

    profile = data.get("profile", {})
    if profile:
        lines.append("PROFILE")
        lines.append(f"Name: {profile.get('full_name')} | Grade: {profile.get('grade')} | Title: {profile.get('job_title')}")
        lines.append(f"Dept: {profile.get('department')} | Manager: {profile.get('manager_name', 'N/A')}")
        lines.append(f"Hire date: {profile.get('hire_date')} | Type: {profile.get('employment_type')} | Work model: {profile.get('work_model')}")
        if profile.get("probation_end_date"):
            lines.append(f"Probation status: ACTIVE — ends {profile['probation_end_date']}")
        else:
            lines.append("Probation status: NOT IN PROBATION")
        wm = (profile.get("work_model") or "").strip().lower()
        if wm == "remote":
            lines.append(
                "Work arrangement: REMOTE — you are not expected on-site daily; work location is primarily off-site. "
                "Standard company clock hours and break rules in policy still apply unless your manager agrees different core hours."
            )
            lines.append("Allowances: remote work allowance 800 EGP/month (if listed in payroll).")
        elif wm == "hybrid":
            lines.append(
                "Work arrangement: HYBRID — mix of office and remote as defined by your department; "
                "standard company clock hours apply on working days unless agreed otherwise."
            )
        else:
            lines.append(
                "Work arrangement: ON-SITE / office-based unless policy or your contract says otherwise; "
                "standard company clock hours apply."
            )
        lines.append("")

    leave_balances = data.get("leave_balances", [])
    non_zero = [lb for lb in leave_balances if float(lb.get("remaining_days", 0)) > 0 or float(lb.get("used_days", 0)) > 0]
    if non_zero:
        lines.append("LEAVE BALANCES (current year)")
        for lb in non_zero:
            lines.append(
                f"  {lb['leave_type'].upper()}: entitled={lb['entitled_days']} "
                f"carried={lb['carried_over_days']} used={lb['used_days']} "
                f"pending={lb['pending_days']} remaining={lb['remaining_days']}"
            )
        lines.append("")

    pending = data.get("pending_leaves", [])
    if pending:
        lines.append("PENDING LEAVE REQUESTS")
        for p in pending:
            lines.append(f"  {p['leave_type']} {p['start_date']}→{p['end_date']} ({p['days_count']} days)")
        lines.append("")

    recent_requests = data.get("recent_requests", [])
    if recent_requests:
        lines.append("RECENT LEAVE REQUESTS (last 5)")
        for r in recent_requests:
            status_str = r['status'].upper()
            lines.append(
                f"  [{status_str}] {r['leave_type']} {r['start_date']}→{r['end_date']} "
                f"({r['days_count']} days)"
                + (f" — rejected: {r['rejection_reason']}" if r.get('rejection_reason') else "")
            )
        lines.append("")

    latest = data.get("latest_review")
    if latest:
        lines.append("LATEST PERFORMANCE REVIEW")
        lines.append(f"  Period: {latest['review_period']} | Rating: {latest['rating']}/5 ({latest['rating_label']})")
        lines.append(f"  Salary increment: {latest['salary_increment_pct']}% | Bonus multiplier: {latest['bonus_multiplier']}x")
        if latest.get("strengths"):    lines.append(f"  Strengths: {latest['strengths']}")
        if latest.get("areas_for_growth"): lines.append(f"  Areas for growth: {latest['areas_for_growth']}")
        if latest.get("overall_comments"): lines.append(f"  Comments: {latest['overall_comments']}")
        lines.append("")

    trend = data.get("performance_trend", {})
    if trend.get("ratings_oldest_first"):
        lines.append("PERFORMANCE TREND")
        for period, rating, label in zip(trend["review_periods"], trend["ratings_oldest_first"], trend["labels_oldest_first"]):
            lines.append(f"  {period}: {rating}/5 ({label})")
        lines.append(f"  Average: {trend['average_rating']} | Direction: {trend['trend_direction']}")
        lines.append("")

    okrs = data.get("current_okrs", [])
    if okrs:
        lines.append("CURRENT OKRs")
        for okr in okrs:
            lines.append(f"  [{okr['status'].upper()}] {okr['title']} — {okr['overall_progress_pct']}%")
            krs = okr.get("key_results", [])
            if isinstance(krs, str):
                krs = json.loads(krs)
            for kr in krs:
                lines.append(f"    • {kr.get('kr')} → {kr.get('progress_pct', 0)}% ({kr.get('status', '')})")
        lines.append("")

    salary = data.get("latest_salary")
    if salary:
        lines.append("LATEST SALARY")
        lines.append(f"  Month: {salary.get('month', 'N/A')}")
        lines.append(f"  Base: {float(salary['base_salary']):,.0f} EGP | Gross: {float(salary['gross_salary']):,.0f} | Net: {float(salary['net_salary']):,.0f}")
        allowances = []
        for k, label in [
            ("transport_allowance", "Transport"),
            ("housing_allowance", "Housing"),
            ("mobile_allowance", "Mobile"),
            ("remote_allowance", "Remote"),
            ("shift_allowance", "Shift"),
            ("on_call_allowance", "On-call"),
        ]:
            v = float(salary.get(k, 0))
            if v > 0:
                allowances.append(f"{label}={v:,.0f}")
        if allowances:
            lines.append(f"  Allowances: {' | '.join(allowances)}")
        deductions = []
        for k, label in [("income_tax", "Tax"), ("social_insurance", "Social ins."), ("other_deductions", "Other")]:
            v = float(salary.get(k, 0))
            if v > 0:
                deductions.append(f"{label}={v:,.0f}")
        if deductions:
            lines.append(f"  Deductions: {' | '.join(deductions)}")
        lines.append("")

    # Salary history (last 3 months trend) — was previously not rendered
    salary_history = data.get("salary_history", [])
    if salary_history and len(salary_history) > 1:
        lines.append("SALARY HISTORY (recent months)")
        for rec in salary_history[:3]:
            lines.append(
                f"  {rec.get('month', '?')}: Base={float(rec['base_salary']):,.0f} | "
                f"Gross={float(rec['gross_salary']):,.0f} | Net={float(rec['net_salary']):,.0f}"
            )
        lines.append("")

    training = data.get("training")
    if training:
        lines.append("TRAINING (current year)")
        lines.append(f"  Budget: ${float(training['budget_total_usd']):,.0f} total | ${float(training['budget_used_usd']):,.0f} used | ${float(training['budget_remaining_usd']):,.0f} remaining")
        lines.append(f"  Days: {training['training_days_total']} total | {training['training_days_used']} used | {training['days_remaining']} remaining")
        courses = training.get("courses", [])
        if isinstance(courses, str):
            courses = json.loads(courses)
        for c in courses:
            cert = " [certified]" if c.get("certificate") else ""
            lines.append(f"  • {c['name']} ({c['provider']}, {c['date']}, ${c.get('cost_usd',0):,.0f}){cert}")
        lines.append("")

    disc = data.get("active_disciplinary", [])
    if disc:
        lines.append("ACTIVE DISCIPLINARY")
        for d in disc:
            lines.append(f"  {d['action_type']} issued {d['issued_date']} expires {d.get('expiry_date','N/A')}: {d['reason']}")
        lines.append("")


    # ── Eligibility pre-check block ──────────────────────────────────────
    # Structured YES/NO verdicts computed in Python so the LLM never has to
    # reason over raw dates or do arithmetic itself. The LLM must use these
    # verdicts as ground truth and never override them.
    from datetime import date as _date
    lines.append("ELIGIBILITY PRE-CHECK (use these verdicts exactly — do not recalculate)")
    in_probation = bool(profile.get("probation_end_date")) if profile else False
    on_pip       = any(d.get("action_type") == "pip" for d in (data.get("active_disciplinary") or []))

    tenure_months = 0
    hire_str = profile.get("hire_date") if profile else None
    if hire_str:
        try:
            hire  = _date.fromisoformat(str(hire_str))
            today = _date.today()
            tenure_months = (today.year - hire.year) * 12 + (today.month - hire.month)
        except Exception:
            tenure_months = 0

    latest_rating = None
    latest_rev = data.get("latest_review")
    if latest_rev:
        try:
            latest_rating = int(latest_rev.get("rating", 0))
        except Exception:
            latest_rating = None

    annual_remaining = 0
    for lb in data.get("leave_balances", []):
        if lb.get("leave_type", "").lower() == "annual":
            try:
                annual_remaining = float(lb.get("remaining_days", 0))
            except Exception:
                annual_remaining = 0

    bonus_eligible = (
        tenure_months >= 6 and not in_probation
        and not on_pip and latest_rating is not None and latest_rating >= 3
    )
    bonus_reason = []
    if tenure_months < 6:   bonus_reason.append(f"tenure only {tenure_months} months (need 6+)")
    if in_probation:        bonus_reason.append("still in probation")
    if on_pip:              bonus_reason.append("currently on PIP")
    if latest_rating is not None and latest_rating < 3:
        bonus_reason.append(f"rating {latest_rating}/5 below minimum")
    lines.append("  Bonus eligible: " + ("YES" if bonus_eligible else "NO") +
                 ((" — " + ", ".join(bonus_reason)) if bonus_reason else ""))

    promo_eligible = tenure_months >= 12 and not in_probation and (latest_rating or 0) >= 4
    promo_reason = []
    if tenure_months < 12:  promo_reason.append(f"tenure only {tenure_months} months (need 12+)")
    if in_probation:        promo_reason.append("still in probation")
    if (latest_rating or 0) < 4:
        promo_reason.append(f"rating {latest_rating or 0}/5 (need 4+)")
    lines.append("  Promotion eligible: " + ("YES" if promo_eligible else "NO") +
                 ((" — " + ", ".join(promo_reason)) if promo_reason else ""))

    schol_eligible = tenure_months >= 12 and not in_probation
    schol_reason = []
    if tenure_months < 12:  schol_reason.append(f"tenure only {tenure_months} months (need 12+)")
    if in_probation:        schol_reason.append("still in probation")
    lines.append("  Scholarship eligible: " + ("YES" if schol_eligible else "NO") +
                 ((" — " + ", ".join(schol_reason)) if schol_reason else ""))

    leave_verdict = f"YES ({annual_remaining:.0f} days remaining)" if annual_remaining > 0 else "NO — balance is zero"
    lines.append("  Can take more annual leave: " + leave_verdict)
    lines.append("")
    return "\n".join(lines).strip()


# ── Personal rules — answers come from DB data, not from policy docs ──────
_PERSONAL_RULES_EN = (
    "RULES:\n"
    "1. Answer ONLY from the employee data block below. Never guess or invent.\n"
    "2. State exact numbers — days, amounts, ratings — as they appear in the data.\n"
    "3. PROBATION: Read the 'Probation status' field directly. Never infer it from other fields.\n"
    "4. WORK MODEL / HOURS: Read 'Work arrangement' and related lines in the data block; never contradict work_model (e.g. do not say daily office attendance if remote).\n"
    "5. If a field is missing: say 'This information is not available in your record.'\n"
    "6. Mirror pronouns: user says I/my → reply with you/your.\n"
    "7. No page citations — this data is from the live HR database.\n"
    "8. LANGUAGE LOCK: Reply in English only.\n"
    "9. ELIGIBILITY: When asked about bonus, promotion, scholarship, or leave eligibility, read the ELIGIBILITY PRE-CHECK block and state the YES/NO verdict directly with the reason given. Do not re-derive eligibility from raw fields.\n"
)

_PERSONAL_RULES_AR = (
    "القواعد:\n"
    "1. أجب من بيانات الموظف المرفقة فقط. لا تخمّن.\n"
    "2. اذكر الأرقام الدقيقة — أيام، مبالغ، تقييمات — كما هي في البيانات.\n"
    "3. التجربة: اقرأ حقل 'Probation status' مباشرة. لا تستنتجه من حقول أخرى.\n"
    "4. نموذج العمل / الساعات: اتبع سطور 'Work arrangement'؛ لا تقل إن الحضور اليومي للمكتب مطلوب إذا كان العمل عن بُعد.\n"
    "5. إذا غاب حقل: 'هذه المعلومة غير متوفرة في سجلك.'\n"
    "6. طابق الضمائر: أنا/لي → أنت/لك.\n"
    "7. لا أرقام صفحات — البيانات من قاعدة البيانات الحية.\n"
    "8. قفل اللغة: أجب بالعربية فقط.\n"
)

_PERSONAL_RULES_EGY = (
    "القواعد:\n"
    "1. جاوب من بيانات الموظف دي بس. متخمنش.\n"
    "2. اذكر الأرقام الصح — أيام، فلوس، تقييمات — زي ما هي في البيانات.\n"
    "3. التجربة: اقرأ حقل 'Probation status' على طول. متستنتجش من حاجة تانية.\n"
    "4. نموذج العمل / الساعات: اتبع سطور 'Work arrangement' في البيانات؛ لا تقل إن الحضور اليومي للمكتب مطلوب إذا كان العمل عن بُعد.\n"
    "5. لو المعلومة مش موجودة: 'المعلومة دي مش متاحة في سجلك.'\n"
    "6. الموظف بيقول أنا/بتاعي → رد بـ إنت/بتاعك.\n"
    "7. متستخدمش أرقام صفحات.\n"
    "8. قفل اللغة: أجب بالعامية المصرية فقط. لا فصحى.\n"
)

_PERSONAL_RULES_FRANCO = (
    "OUTPUT: Franco Arabi — Egyptian Arabic in Latin letters, like real WhatsApp messages. "
    "Use 3 7 5 2 4 for ع ح خ أ/ء ش when it reads naturally. "
    "Keep sentences short and clear. No formal MSA paragraphs. Do not reply in English.\n"
    "RULES:\n"
    "1. Use ONLY the employee data block. No guessing.\n"
    "2. Numbers and facts exactly as in the data (days, money, ratings).\n"
    "3. Probation: read 'Probation status' as written — do not infer from other fields.\n"
    "4. Work model / hours: follow the 'Work arrangement' and allowance lines — say remote/hybrid/office clearly.\n"
    "5. If a field is missing: 'El ma3loma di mesh mawgoda f segelak.'\n"
    "6. User says I/my → you/your in Franco.\n"
    "7. No page numbers — data is from the live HR database.\n"
)

# ── Hybrid rules — combines DB data with policy context ───────────────────
_HYBRID_RULES_EN = (
    "RULES:\n"
    "1. State the employee's actual data FIRST (exact fields from the data block — especially work_model).\n"
    "2. WORKING HOURS / SCHEDULE: Always lead with 'Work arrangement' / work_model (remote, hybrid, or on-site). "
    "If remote: make clear they are not required to be in the office daily; standard start/end times from policy still apply "
    "unless an exception is stated. Do NOT describe them like a full-time office worker if work_model is remote.\n"
    "3. Then cite the relevant policy lines (shift length, breaks, typical corporate hours) with [Page N | AR/EN].\n"
    "4. Combine into one concise answer — do not repeat the same fact in multiple sentences.\n"
    "5. No page citations for the personal/database part — only for policy sentences.\n"
    "6. LANGUAGE LOCK: Reply in English only.\n"
)

_HYBRID_RULES_AR = (
    "القواعد:\n"
    "1. اذكر بيانات الموظف أولاً (خصوصاً نموذج العمل work_model).\n"
    "2. ساعات العمل / الجدول: ابدأ بتوضيح طريقة العمل (عن بُعد، هجين، أو في المكتب). "
    "إذا كان عن بُعد: أوضح أن الحضور اليومي للمكتب غير مطلوب؛ أوقات بداية ونهاية الدوام من السياسة تظل سارية ما لم يُستثنَ. "
    "لا تصِف الموظف كمن يعمل يومياً من المكتب إذا كان العمل عن بُعد.\n"
    "3. ثم استشهد بما ينطبق من السياسة مع [Page N | AR/EN].\n"
    "4. إجابة موجزة دون تكرار نفس الجملة.\n"
    "5. لا أرقام صفحات لجزء قاعدة البيانات — فقط لجمل السياسة.\n"
    "6. قفل اللغة: أجب بالعربية فقط.\n"
)

_HYBRID_RULES_EGY = (
    "القواعد:\n"
    "1. اذكر بيانات الموظف الأول — وخصوصاً شغلك عن بُعد ولا هجين ولا من المكتب (work_model).\n"
    "2. ساعات الشغل: لو أنت remote متقولش إنك لازم تقعد في الشركة كل يوم؛ الأوقات الرسمية من السياسة لسه بتتطبق. "
    "لو hybrid أو office وضّح الفرق باختصار.\n"
    "3. بعدين استشهد بالسياسة مع [Page N | AR/EN].\n"
    "4. من غير تكرار نفس الكلام مرتين.\n"
    "5. لا أرقام صفحات للبيانات الشخصية — بس للسياسة.\n"
    "6. قفل اللغة: عامية مصرية بس.\n"
)

_HYBRID_RULES_FRANCO = (
    "OUTPUT: Franco Arabi — Egyptian Arabic in Latin letters, natural texting style "
    "(3=ع, 7=ح, 5=خ, 2=أ/ء, 4=ش). Short sentences; يعني، كده، عشان، لو where natural. "
    "No formal فصحى paragraphs. Do not answer in English.\n"
    "RULES:\n"
    "1. 2ol el bayanat el shakhseya awwel — khosousan work_model (remote / hybrid / office).\n"
    "2. Law remote: wa7yed en el attendance el yomiyyan fel ma7kam mesh matloob; el mawa3id el rasmeya men el policy lissa matlooba. "
    "Mat2olsh en el shoghl zay employee office kamel law howwa remote.\n"
    "3. Ba3den el policy ma3 [Page N | AR/EN] lel ta2seel (sa3at, breaks, shift).\n"
    "4. Mat3awdsh takrar nafs el fekra maratein.\n"
    "5. Mafish [Page …] lel gomel el database — bas lel policy.\n"
)

# ── Prompt templates ───────────────────────────────────────────
PERSONAL_EN = PromptTemplate(
    template=(
        "You are a personal HR assistant.\n"
        + _PERSONAL_RULES_EN
        + "\nRecent conversation:\n{history}\n\n"
        "=== EMPLOYEE DATA ===\n{personal_data}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["personal_data", "question", "history"]
)

PERSONAL_AR = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية شخصي.\n"
        + _PERSONAL_RULES_AR
        + "\nالمحادثة الأخيرة:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "question", "history"]
)

PERSONAL_EGY = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية شخصي.\n"
        + _PERSONAL_RULES_EGY
        + "\nالمحادثة اللي فاتت:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "question", "history"]
)

PERSONAL_FRANCO = PromptTemplate(
    template=(
        "You are a personal HR assistant.\n"
        + _PERSONAL_RULES_FRANCO
        + "\nEl kalam el fat:\n{history}\n\n"
        "=== El bayanat el shakhseya ===\n{personal_data}\n\n"
        "El so2al: {question}\nEl egaba (Franco bass):"
    ),
    input_variables=["personal_data", "question", "history"]
)

HYBRID_EN = PromptTemplate(
    template=(
        "You are an HR assistant.\n"
        + _HYBRID_RULES_EN
        + "\nRecent conversation:\n{history}\n\n"
        "=== EMPLOYEE DATA ===\n{personal_data}\n\n"
        "=== POLICY CONTEXT ===\n{policy_context}\n\n"
        "Question: {question}\nAnswer:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

HYBRID_AR = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية.\n"
        + _HYBRID_RULES_AR
        + "\nالمحادثة الأخيرة:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "=== سياق السياسة ===\n{policy_context}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

HYBRID_EGY = PromptTemplate(
    template=(
        "أنت مساعد موارد بشرية.\n"
        + _HYBRID_RULES_EGY
        + "\nالمحادثة اللي فاتت:\n{history}\n\n"
        "=== بيانات الموظف ===\n{personal_data}\n\n"
        "=== سياق السياسة ===\n{policy_context}\n\n"
        "السؤال: {question}\nالإجابة:"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)

HYBRID_FRANCO = PromptTemplate(
    template=(
        "You are an HR assistant.\n"
        + _HYBRID_RULES_FRANCO
        + "\nEl kalam el fat:\n{history}\n\n"
        "=== El bayanat el shakhseya ===\n{personal_data}\n\n"
        "=== El policy context ===\n{policy_context}\n\n"
        "El so2al: {question}\nEl egaba (Franco bass):"
    ),
    input_variables=["personal_data", "policy_context", "question", "history"]
)


def get_personal_prompt(lang: str, dialect: str = None):
    if lang == "english":       return PERSONAL_EN
    if lang == "franco":        return PERSONAL_FRANCO
    if dialect == "egyptian":   return PERSONAL_EGY
    return PERSONAL_AR


def get_hybrid_prompt(lang: str, dialect: str = None):
    if lang == "english":       return HYBRID_EN
    if lang == "franco":        return HYBRID_FRANCO
    if dialect == "egyptian":   return HYBRID_EGY
    return HYBRID_AR