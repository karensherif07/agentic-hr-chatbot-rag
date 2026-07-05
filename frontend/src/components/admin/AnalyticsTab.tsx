import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { AnalyticsResponse, AnalyticsRow } from "../../api/types";

const PAGE_SIZE = 20;

function isoDate(d: Date) {
  return d.toISOString().slice(0, 10);
}

const LANG_LABELS: Record<string, string> = {
  english: "English",
  arabic_msa: "Arabic (MSA)",
  arabic_egyptian: "Arabic (Egyptian)",
  franco: "Franco Arabic",
};

function Bar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = max ? (value / max) * 100 : 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5, marginBottom: 4, color: "var(--text-mid)" }}>
        <span>{label}</span>
        <span>{value}</span>
      </div>
      <div style={{ height: 8, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

// Same shape as the original df.to_csv(index=False) — header row + quoted values.
function toCsv(rows: AnalyticsRow[]): string {
  const headers = ["asked_at", "full_name", "department", "grade", "intent", "topic", "language", "unanswered", "question_text"];
  const escape = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [headers.join(",")];
  for (const r of rows) {
    lines.push(headers.map((h) => escape((r as any)[h])).join(","));
  }
  return lines.join("\n");
}

function downloadCsv(filename: string, csv: string) {
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function AnalyticsTab() {
  const [from, setFrom] = useState(isoDate(new Date(Date.now() - 30 * 86400000)));
  const [to, setTo] = useState(isoDate(new Date()));
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [langFilter, setLangFilter] = useState("All");
  const [intentFilter, setIntentFilter] = useState("All");
  const [unansFilter, setUnansFilter] = useState("All");
  const [search, setSearch] = useState("");
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  useEffect(() => {
    api.get<AnalyticsResponse>(`/api/admin/analytics?date_from=${from}&date_to=${to}`).then(setData);
  }, [from, to]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.rows.filter((r) => {
      if (langFilter !== "All" && r.language !== langFilter) return false;
      if (intentFilter !== "All" && r.intent !== intentFilter) return false;
      if (unansFilter === "Answered" && r.unanswered) return false;
      if (unansFilter === "Unanswered" && !r.unanswered) return false;
      if (search && !r.question_text.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    });
  }, [data, langFilter, intentFilter, unansFilter, search]);

  // Reset pagination whenever filters change so you don't get stranded past the end.
  useEffect(() => setVisibleCount(PAGE_SIZE), [langFilter, intentFilter, unansFilter, search, from, to]);

  if (!data) return <div style={{ color: "var(--text-mid)" }}>Loading…</div>;

  const maxLang = Math.max(1, ...Object.values(data.by_language));
  const maxIntent = Math.max(1, ...Object.values(data.by_intent));
  const visible = filtered.slice(0, visibleCount);

  return (
    <div>
      <div style={{ display: "flex", gap: 12, marginBottom: 20 }}>
        <label style={{ fontSize: 12.5, color: "var(--text-mid)" }}>
          From
          <input className="input" type="date" value={from} onChange={(e) => setFrom(e.target.value)} style={{ marginTop: 4 }} />
        </label>
        <label style={{ fontSize: 12.5, color: "var(--text-mid)" }}>
          To
          <input className="input" type="date" value={to} onChange={(e) => setTo(e.target.value)} style={{ marginTop: 4 }} />
        </label>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14, marginBottom: 24 }}>
        {[
          { label: "Total queries", value: data.total },
          { label: "Unanswered", value: data.unanswered },
          { label: "Unanswered %", value: `${data.unanswered_pct}%` },
        ].map((m) => (
          <div key={m.label} className="glass-panel" style={{ padding: 18 }}>
            <div style={{ fontSize: 12, color: "var(--text-lo)", marginBottom: 6 }}>{m.label}</div>
            <div style={{ fontSize: "1.8rem", fontFamily: "var(--font-display)" }}>{m.value}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
        <div className="glass-panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>By Language</div>
          {Object.entries(data.by_language)
            .filter(([k]) => k !== "arabic")
            .map(([k, v]) => (
              <Bar key={k} label={LANG_LABELS[k] || k} value={v} max={maxLang} color="var(--brass-500)" />
            ))}
        </div>
        <div className="glass-panel" style={{ padding: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>By Intent</div>
          {Object.entries(data.by_intent).map(([k, v]) => (
            <Bar key={k} label={k} value={v} max={maxIntent} color="var(--teal-400)" />
          ))}
        </div>
      </div>

      <div className="glass-panel" style={{ padding: 18, marginBottom: 24 }}>
        <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, color: "var(--danger)" }}>🔴 Unanswered Questions</div>
        {data.rows
          .filter((r) => r.unanswered)
          .slice(0, 20)
          .map((r) => (
            <div key={r.id} style={{ fontSize: 13, padding: "8px 0", borderBottom: "1px solid var(--line)" }}>
              <span style={{ color: "var(--text-lo)" }}>{new Date(r.asked_at).toLocaleString()}</span>{" "}
              <strong>{r.full_name}</strong> ({r.department}) — {r.question_text}
            </div>
          ))}
      </div>

      <div className="glass-panel" style={{ padding: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10, marginBottom: 14 }}>
          <div style={{ fontSize: 13, fontWeight: 600 }}>All Queries Log</div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "6px 12px" }} onClick={() => downloadCsv("analytics_all.csv", toCsv(data.rows))}>
              ⬇️ Download All (CSV)
            </button>
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "6px 12px" }} onClick={() => downloadCsv("analytics_filtered.csv", toCsv(filtered))}>
              ⬇️ Download Filtered (CSV)
            </button>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, marginBottom: 14, flexWrap: "wrap" }}>
          <select className="input" style={{ width: "auto" }} value={langFilter} onChange={(e) => setLangFilter(e.target.value)}>
            {["All", "english", "arabic_msa", "arabic_egyptian", "franco"].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <select className="input" style={{ width: "auto" }} value={intentFilter} onChange={(e) => setIntentFilter(e.target.value)}>
            {["All", "policy", "personal", "hybrid", "out_of_scope"].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <select className="input" style={{ width: "auto" }} value={unansFilter} onChange={(e) => setUnansFilter(e.target.value)}>
            {["All", "Answered", "Unanswered"].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <input
            className="input"
            style={{ flex: 1, minWidth: 200 }}
            placeholder="🔍 Search question text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <div style={{ fontSize: 12, color: "var(--text-lo)", marginBottom: 10 }}>
          Showing {visible.length} of {filtered.length} queries ({data.total} total)
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--text-lo)" }}>
                {["Time", "Employee", "Dept", "Grade", "Intent", "Topic", "Lang", "Unanswered?", "Question"].map((h) => (
                  <th key={h} style={{ padding: "6px 10px", borderBottom: "1px solid var(--line-strong)" }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.id} style={{ borderBottom: "1px solid var(--line)" }}>
                  <td style={{ padding: "6px 10px", whiteSpace: "nowrap" }}>{new Date(r.asked_at).toLocaleString()}</td>
                  <td style={{ padding: "6px 10px" }}>{r.full_name}</td>
                  <td style={{ padding: "6px 10px" }}>{r.department}</td>
                  <td style={{ padding: "6px 10px" }}>{r.grade}</td>
                  <td style={{ padding: "6px 10px" }}>{r.intent}</td>
                  <td style={{ padding: "6px 10px" }}>{r.topic}</td>
                  <td style={{ padding: "6px 10px" }}>{r.language}</td>
                  <td style={{ padding: "6px 10px" }}>{r.unanswered ? "✅" : ""}</td>
                  <td style={{ padding: "6px 10px", maxWidth: 320 }}>{r.question_text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {visibleCount < filtered.length && (
          <div style={{ textAlign: "center", marginTop: 14 }}>
            <button className="btn btn-ghost" onClick={() => setVisibleCount((v) => v + PAGE_SIZE)}>
              Load more ({filtered.length - visibleCount} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
