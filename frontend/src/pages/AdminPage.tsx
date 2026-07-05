import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";
import AnalyticsTab from "../components/admin/AnalyticsTab";
import EscalationsTab from "../components/admin/EscalationsTab";
import AuditLogTab from "../components/admin/AuditLogTab";
import PolicyManagerTab from "../components/admin/PolicyManagerTab";

const TABS = [
  { id: "analytics", label: "📊 Analytics" },
  { id: "escalations", label: "🔴 Escalations" },
  { id: "policies", label: "📁 Manage Policies" },
  { id: "audit", label: "📋 Audit Log" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function AdminPage() {
  const [tab, setTab] = useState<TabId>("analytics");
  const { employee, logout } = useAuthStore();
  const navigate = useNavigate();

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "32px 28px 60px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--indigo-600)", fontWeight: 600 }}>
            Horizon Tech
          </div>
          <h1 style={{ fontFamily: "var(--font-display)", fontSize: "1.8rem", fontWeight: 500, margin: "4px 0" }}>
            ⚙️ HR Admin Portal
          </h1>
          <div style={{ fontSize: 13, color: "var(--text-mid)" }}>
            Logged in as {employee?.full_name} ({employee?.admin_role})
          </div>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button className="btn btn-ghost" onClick={() => navigate("/")}>
            ← Back to chat
          </button>
          <button className="btn btn-ghost" onClick={() => logout().then(() => navigate("/login"))}>
            Sign out
          </button>
        </div>
      </div>

      <div className="horizon-rule" style={{ margin: "18px 0 24px" }} />

      <div style={{ display: "flex", gap: 6, marginBottom: 24, flexWrap: "wrap" }}>
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className="btn"
            style={{
              background: tab === t.id ? "rgba(91, 95, 227,0.16)" : "var(--surface-alt)",
              border: `1px solid ${tab === t.id ? "rgba(91, 95, 227,0.4)" : "var(--line-strong)"}`,
              color: tab === t.id ? "var(--indigo-600)" : "var(--text-mid)",
              fontSize: 13.5,
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "analytics" && <AnalyticsTab />}
      {tab === "escalations" && <EscalationsTab />}
      {tab === "policies" && <PolicyManagerTab />}
      {tab === "audit" && <AuditLogTab />}
    </div>
  );
}