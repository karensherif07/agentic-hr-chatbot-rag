import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { PolicyDoc } from "../../api/types";

export default function PolicyManagerTab() {
  const [rows, setRows] = useState<PolicyDoc[] | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [docName, setDocName] = useState("");
  const [lang, setLang] = useState<"arabic" | "english">("english");
  const [uploading, setUploading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [notice, setNotice] = useState<{ type: "ok" | "err"; text: string } | null>(null);

  function load() {
    api.get<{ rows: PolicyDoc[] }>("/api/admin/policies").then((r) => setRows(r.rows));
  }

  useEffect(load, []);

  async function upload(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !docName.trim()) return;
    setUploading(true);
    setNotice(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("doc_name", docName.trim());
      form.append("lang", lang);
      const res = await api.post<{ note: string }>("/api/admin/policies", form);
      setNotice({ type: "ok", text: res.note });
      setFile(null);
      setDocName("");
      load();
    } catch (e: any) {
      setNotice({ type: "err", text: e.message || "Upload failed." });
    } finally {
      setUploading(false);
    }
  }

  async function toggleActive(p: PolicyDoc) {
    await api.post(`/api/admin/policies/${p.id}/${p.is_active ? "deactivate" : "activate"}`);
    load();
  }

  async function remove(p: PolicyDoc) {
    if (!confirm(`Delete "${p.doc_name}"? This removes the file permanently.`)) return;
    await api.del(`/api/admin/policies/${p.id}`);
    load();
  }

  async function rebuildIndex() {
    setRebuilding(true);
    setNotice(null);
    try {
      const res = await api.post<{ note: string }>("/api/admin/policies/rebuild-index");
      setNotice({ type: "ok", text: res.note });
    } catch (e: any) {
      setNotice({ type: "err", text: e.message || "Rebuild failed." });
    } finally {
      setRebuilding(false);
    }
  }

  return (
    <div>
      <div className="glass-panel" style={{ padding: 20, marginBottom: 22 }}>
        <div style={{ fontSize: 13.5, fontWeight: 600, marginBottom: 4 }}>Add a new policy PDF</div>
        <div style={{ fontSize: 12.5, color: "var(--text-lo)", marginBottom: 16 }}>
          Upload a PDF, tag its language, then rebuild the index so it's picked up by retrieval.
        </div>
        <form onSubmit={upload} style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
          <label style={{ fontSize: 12, color: "var(--text-mid)", flex: "1 1 220px" }}>
            PDF file
            <input
              className="input"
              type="file"
              accept="application/pdf"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              style={{ marginTop: 4 }}
            />
          </label>
          <label style={{ fontSize: 12, color: "var(--text-mid)", flex: "1 1 180px" }}>
            Display name
            <input
              className="input"
              placeholder="e.g. eng_remote_work.pdf"
              value={docName}
              onChange={(e) => setDocName(e.target.value)}
              style={{ marginTop: 4 }}
            />
          </label>
          <label style={{ fontSize: 12, color: "var(--text-mid)" }}>
            Language
            <select className="input" value={lang} onChange={(e) => setLang(e.target.value as any)} style={{ marginTop: 4 }}>
              <option value="english">English</option>
              <option value="arabic">Arabic</option>
            </select>
          </label>
          <button className="btn btn-primary" type="submit" disabled={uploading || !file || !docName.trim()}>
            {uploading ? "Uploading…" : "Upload"}
          </button>
        </form>
      </div>

      {notice && (
        <div
          className="glass-panel"
          style={{ padding: "10px 16px", marginBottom: 18, fontSize: 13, color: notice.type === "ok" ? "var(--success)" : "var(--danger)" }}
        >
          {notice.text}
        </div>
      )}

      <div className="glass-panel" style={{ padding: 20, marginBottom: 22 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
          <div style={{ fontSize: 13.5, fontWeight: 600 }}>Current policy documents</div>
          <button className="btn btn-ghost" onClick={rebuildIndex} disabled={rebuilding}>
            {rebuilding ? "Rebuilding…" : "🔄 Rebuild Index"}
          </button>
        </div>

        {!rows && <div style={{ color: "var(--text-mid)" }}>Loading…</div>}
        {rows && rows.length === 0 && <div style={{ color: "var(--text-lo)" }}>No policy documents yet.</div>}

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {rows?.map((p) => (
            <div
              key={p.id}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 14px",
                borderRadius: 10,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--line)",
                opacity: p.is_active ? 1 : 0.5,
              }}
            >
              <div>
                <div style={{ fontSize: 13.5 }}>
                  <span className="badge" style={{ marginInlineEnd: 8 }}>
                    {p.lang === "arabic" ? "AR" : "EN"}
                  </span>
                  {p.doc_name}
                  {!p.is_active && <span style={{ color: "var(--text-lo)" }}> — inactive</span>}
                </div>
                <div style={{ fontSize: 11.5, color: "var(--text-lo)", marginTop: 2 }}>
                  Uploaded {new Date(p.uploaded_at).toLocaleDateString()}
                  {p.uploaded_by_name ? ` by ${p.uploaded_by_name}` : ""}
                </div>
              </div>
              <div style={{ display: "flex", gap: 6 }}>
                <button className="btn btn-ghost" style={{ fontSize: 12, padding: "5px 10px" }} onClick={() => toggleActive(p)}>
                  {p.is_active ? "Deactivate" : "Activate"}
                </button>
                <button className="btn btn-danger" style={{ fontSize: 12, padding: "5px 10px" }} onClick={() => remove(p)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
