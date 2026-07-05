export interface Employee {
  id: number;
  full_name: string;
  full_name_ar?: string;
  email: string;
  grade: string;
  job_title: string;
  department: string;
  hire_date: string;
  employment_type: string;
  work_model: string;
  admin_role: "hr_admin" | "super_admin" | null;
  is_admin: boolean;
  in_probation: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  is_arabic: boolean;
  is_franco: boolean;
}

export interface CitedDoc {
  page: number;
  source: string;
  doc_name: string;
  lang: string;
  content: string;
}

export interface ChatResponse {
  answer: string;
  lang: "english" | "arabic" | "franco";
  dialect: string | null;
  intent: "policy" | "personal" | "hybrid" | "out_of_scope";
  topic: string;
  tools_called: string[];
  no_info: boolean;
  personal_data: string;
  cited_docs: CitedDoc[];
  chat_history: ChatMessage[];
  conversation_summary: string;
}

export interface AnalyticsRow {
  id: number;
  asked_at: string;
  full_name: string;
  department: string;
  grade: string;
  intent: string;
  topic: string;
  language: string;
  unanswered: boolean;
  question_text: string;
}

export interface AnalyticsResponse {
  rows: AnalyticsRow[];
  total: number;
  unanswered: number;
  unanswered_pct: number;
  by_language: Record<string, number>;
  by_intent: Record<string, number>;
}

export interface EscalationRow {
  id: number;
  asked_at: string;
  full_name: string;
  email: string;
  department: string;
  language: string;
  question_text: string;
}

export interface AuditRow {
  admin_id: number;
  action: string;
  resource_type: string | null;
  resource_id: number | null;
  performed_at: string;
  notes: string | null;
}

export interface PolicyDoc {
  id: number;
  file_path: string;
  doc_name: string;
  lang: "arabic" | "english";
  is_active: boolean;
  uploaded_at: string;
  uploaded_by_name: string | null;
}
