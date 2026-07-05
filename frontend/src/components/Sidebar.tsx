import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import { api } from "../api/client";

export default function Sidebar({ onClearChat }: { onClearChat: () => void }) {
  const { employee, logout } = useAuthStore();
  const navigate = useNavigate();
  const [contactOpen, setContactOpen] = useState(false);
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [sendState, setSendState] = useState<"idle" | "sending" | "sent" | "failed">("idle");

  if (!employee) return null;

  async function sendContactHr(e: React.FormEvent) {
    e.preventDefault();
    if (!subject.trim() || !body.trim()) return;
    setSendState("sending");
    try {
      const res = await api.post<{ sent: boolean }>("/api/chat/contact-hr", { subject, body });
      setSendState(res.sent ? "sent" : "failed");
      if (res.sent) {
        setSubject("");
        setBody("");
      }
    } catch {
      setSendState("failed");
    }
  }

  return (
    <div
      style={{
        width: 260,
        flexShrink: 0,
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        padding: "24px 18px",
        borderInlineEnd: "1px solid var(--line)",
      }}
    >
      <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--brass-300)", fontWeight: 600 }}>
        Horizon Tech
      </div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: "1.3rem", margin: "4px 0 14px" }}>
        HR Assistant
      </div>
      <div className="horizon-rule" style={{ marginBottom: 20 }} />

      <div style={{ fontWeight: 600, fontSize: 14.5 }}>{employee.full_name}</div>
      <div style={{ fontSize: 12.5, color: "var(--text-mid)", marginTop: 2 }}>
        {employee.grade} · {employee.department}
        {employee.admin_role && (
          <>
            {" "}· <span style={{ color: "var(--brass-300)" }}>🔑 {employee.admin_role}</span>
          </>
        )}
      </div>

      <div style={{ marginTop: 20 }}>
        {employee.is_admin && (
          <button className="btn btn-ghost" style={{ width: "100%", marginBottom: 8 }} onClick={() => navigate("/admin")}>
            ⚙️ Admin Portal
          </button>
        )}

        {!employee.is_admin && (
          <div className="glass-panel" style={{ padding: 12, marginBottom: 12 }}>
            <button
              className="btn btn-ghost"
              style={{ width: "100%", fontSize: 13 }}
              onClick={() => setContactOpen((v) => !v)}
            >
              📧 Contact HR {contactOpen ? "▲" : "▼"}
            </button>
            {contactOpen && (
              <form onSubmit={sendContactHr} style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
                <input
                  className="input"
                  placeholder="Subject"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  style={{ fontSize: 13 }}
                />
                <textarea
                  className="input"
                  placeholder="Write your message to HR here…"
                  value={body}
                  onChange={(e) => setBody(e.target.value)}
                  rows={4}
                  style={{ fontSize: 13, resize: "vertical", fontFamily: "inherit" }}
                />
                <button className="btn btn-primary" type="submit" style={{ fontSize: 13 }} disabled={sendState === "sending"}>
                  {sendState === "sending" ? "Sending…" : "Send"}
                </button>
                {sendState === "sent" && <div style={{ fontSize: 12, color: "var(--success)" }}>✅ Sent to HR.</div>}
                {sendState === "failed" && <div style={{ fontSize: 12, color: "var(--danger)" }}>Failed to send.</div>}
              </form>
            )}
          </div>
        )}

        <button className="btn btn-ghost" style={{ width: "100%" }} onClick={onClearChat}>
          🗑 Clear chat history
        </button>
      </div>

      <div style={{ flex: 1 }} />

      <button className="btn btn-ghost" style={{ width: "100%" }} onClick={() => logout().then(() => navigate("/login"))}>
        Sign out
      </button>
    </div>
  );
}
