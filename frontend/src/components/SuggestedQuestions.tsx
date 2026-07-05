const CATEGORIES: { label: string; icon: string; questions: string[] }[] = [
  {
    label: "Leave",
    icon: "🏖️",
    questions: [
      "How many annual leave days do I have left?",
      "What's the process to request sick leave?",
      "How many days of maternity/paternity leave am I entitled to?",
    ],
  },
  {
    label: "Payroll",
    icon: "💰",
    questions: [
      "What allowances am I eligible for?",
      "When is salary paid each month?",
      "What was my net salary last month?",
    ],
  },
  {
    label: "Performance",
    icon: "📈",
    questions: [
      "What was my last performance review rating?",
      "How is the salary raise percentage calculated?",
      "What are my current OKRs?",
    ],
  },
  {
    label: "Training",
    icon: "🎓",
    questions: [
      "How much training budget do I have left this year?",
      "Does the company reimburse certifications?",
      "How many training days am I allowed per year?",
    ],
  },
  {
    label: "Policies",
    icon: "📋",
    questions: [
      "What's the remote work policy?",
      "What's the code of conduct on conflicts of interest?",
      "What's the probation period length?",
    ],
  },
];

export default function SuggestedQuestions({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div style={{ marginTop: 18 }}>
      <div style={{ fontSize: 12.5, color: "var(--text-lo)", marginBottom: 12, fontWeight: 600, letterSpacing: "0.02em" }}>
        Not sure what to ask? Try one of these:
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {CATEGORIES.map((cat) => (
          <div key={cat.label}>
            <div style={{ fontSize: 12, color: "var(--text-mid)", marginBottom: 8, fontWeight: 600 }}>
              {cat.icon} {cat.label}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {cat.questions.map((q) => (
                <button key={q} className="chip" onClick={() => onPick(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}