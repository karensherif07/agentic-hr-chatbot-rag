import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { EscalationRow } from "../../api/types";

export default function EscalationsTab() {
  const [rows, setRows] = useState<EscalationRow[] | null>(null);
  const [openId, setOpenId] = useState<number | null>(null);

  function load() {
    api.get<{ rows: EscalationRow[] }>("/api/admin/escalations").then((r) => setRows(r.rows));
  }

  useEffect(load, []);

  async function resolve(id: number) {
    await api.post(`/api/admin/escalations/${id}/resolve`);
    load();
  }

  if (!rows) return <div style={{ color: "var(--text-mid)" }}>Loading…</div>;

  if (rows.length === 0) {
    return (
      <div className="glass-panel" style={{ padding: 20, color: "var(--success)" }}>
        ✅ No pending escalations
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: 16, fontSize: 13, color: "var(--text-mid)" }}>
        Pending escalations: <strong>{rows.length}</strong>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {rows.map((r) => (
          <div key={r.id} className="glass-panel" style={{ padding: 16 }}>
            <button
              onClick={() => setOpenId(openId === r.id ? null : r.id)}
              style={{ background: "none", border: "none", color: "var(--text-hi)", cursor: "pointer", fontSize: 13.5, textAlign: "left", width: "100%", padding: 0 }}
            >
              #{r.id} — {r.question_text.slice(0, 80)}… {openId === r.id ? "▲" : "▼"}
            </button>
            {openId === r.id && (
              <div style={{ marginTop: 12, fontSize: 13, color: "var(--text-mid)", display: "flex", flexDirection: "column", gap: 4 }}>
                <div>👤 {r.full_name} ({r.department})</div>
                <div>📧 {r.email}</div>
                <div>🕒 {new Date(r.asked_at).toLocaleString()}</div>
                <div className="glass-panel" style={{ padding: 10, marginTop: 6, background: "rgba(255,255,255,0.03)" }}>
                  {r.question_text}
                </div>
                <button className="btn btn-primary" style={{ marginTop: 10, width: "fit-content", fontSize: 12.5 }} onClick={() => resolve(r.id)}>
                  ✅ Mark as Resolved
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
