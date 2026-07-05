import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { AuditRow } from "../../api/types";

export default function AuditLogTab() {
  const [rows, setRows] = useState<AuditRow[] | null>(null);

  useEffect(() => {
    api.get<{ rows: AuditRow[] }>("/api/admin/audit-log").then((r) => setRows(r.rows));
  }, []);

  if (!rows) return <div style={{ color: "var(--text-mid)" }}>Loading…</div>;
  if (rows.length === 0) return <div className="glass-panel" style={{ padding: 20 }}>No audit entries yet.</div>;

  return (
    <div className="glass-panel" style={{ padding: 18, overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
        <thead>
          <tr style={{ textAlign: "left", color: "var(--text-lo)" }}>
            {["Admin ID", "Action", "Resource", "Resource ID", "When", "Notes"].map((h) => (
              <th key={h} style={{ padding: "6px 10px", borderBottom: "1px solid var(--line-strong)" }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} style={{ borderBottom: "1px solid var(--line)" }}>
              <td style={{ padding: "6px 10px" }}>{r.admin_id}</td>
              <td style={{ padding: "6px 10px" }}>{r.action}</td>
              <td style={{ padding: "6px 10px" }}>{r.resource_type || "—"}</td>
              <td style={{ padding: "6px 10px" }}>{r.resource_id ?? "—"}</td>
              <td style={{ padding: "6px 10px", whiteSpace: "nowrap" }}>{new Date(r.performed_at).toLocaleString()}</td>
              <td style={{ padding: "6px 10px" }}>{r.notes || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
