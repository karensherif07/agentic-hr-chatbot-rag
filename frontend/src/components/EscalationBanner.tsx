import { useState } from "react";
import { api } from "../api/client";

export default function EscalationBanner({
  question,
  onDismiss,
}: {
  question: string;
  onDismiss: () => void;
}) {
  const [status, setStatus] = useState<"idle" | "sending" | "sent" | "failed">("idle");

  async function notify() {
    setStatus("sending");
    try {
      const res = await api.post<{ sent: boolean }>("/api/chat/escalate", { question });
      setStatus(res.sent ? "sent" : "failed");
    } catch {
      setStatus("failed");
    }
  }

  if (status === "sent") {
    return (
      <div className="glass-panel" style={{ padding: "12px 16px", fontSize: 13.5, color: "var(--success)", marginTop: 10 }}>
        ✅ The HR team has been notified.
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{ padding: "12px 16px", marginTop: 10 }}>
      <div style={{ fontSize: 13.5, marginBottom: 10, color: "var(--text-mid)" }}>
        I couldn't find this in the policy documents. Would you like me to notify the HR team?
      </div>
      {status === "failed" && (
        <div style={{ fontSize: 12.5, color: "var(--danger)", marginBottom: 8 }}>
          Email not sent — check SMTP settings.
        </div>
      )}
      <div style={{ display: "flex", gap: 8 }}>
        <button className="btn btn-primary" style={{ fontSize: 12.5, padding: "6px 14px" }} onClick={notify} disabled={status === "sending"}>
          {status === "sending" ? "Sending…" : "📧 Notify HR"}
        </button>
        <button className="btn btn-ghost" style={{ fontSize: 12.5, padding: "6px 14px" }} onClick={onDismiss}>
          Dismiss
        </button>
      </div>
    </div>
  );
}
