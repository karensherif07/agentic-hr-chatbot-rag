import { useEffect, useState } from "react";
import { BASE_URL, getToken } from "../api/client";
import type { CitedDoc } from "../api/types";

function PdfPage({ doc }: { doc: CitedDoc }) {
  const [imgFailed, setImgFailed] = useState(false);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const isAr = doc.lang === "arabic";

  // <img src="..."> can't attach an Authorization header, so we fetch the
  // image as a blob (with the token) and point the <img> at the resulting
  // local object URL instead. Same end result, just token-compatible.
  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;

    async function load() {
      try {
        const token = getToken();
        const url = `${BASE_URL}/api/policies/page-image?source=${encodeURIComponent(doc.source)}&page=${doc.page}`;
        const res = await fetch(url, {
          headers: token ? { Authorization: `Bearer ${token}` } : undefined,
        });
        if (!res.ok) throw new Error();
        const blob = await res.blob();
        objectUrl = URL.createObjectURL(blob);
        if (!cancelled) setBlobUrl(objectUrl);
      } catch {
        if (!cancelled) setImgFailed(true);
      }
    }
    load();

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [doc.source, doc.page]);

  if (imgFailed) {
    // Same fallback as the original chat_ui.py: plain text block, RTL/LTR aware.
    return (
      <div
        className={isAr ? "rtl" : "ltr"}
        style={{
          background: "#f9f9f9",
          color: "#1a1a1a",
          padding: 12,
          borderInlineStart: "3px solid #1976d2",
          borderRadius: 4,
          fontSize: 13.5,
          lineHeight: 1.7,
        }}
      >
        {doc.content}
      </div>
    );
  }

  if (!blobUrl) {
    return <div style={{ fontSize: 12.5, color: "var(--text-lo)" }}>Loading page…</div>;
  }

  return (
    <img
      src={blobUrl}
      alt={`Page ${doc.page} — ${doc.doc_name}`}
      onError={() => setImgFailed(true)}
      style={{ width: "100%", maxWidth: 700, borderRadius: 8, border: "1px solid var(--line)" }}
    />
  );
}

export default function SourceEvidence({ docs }: { docs: CitedDoc[] }) {
  const [open, setOpen] = useState(false);

  const unique = new Map<string, CitedDoc>();
  for (const d of docs) {
    const key = `${d.source}:${d.page}`;
    if (!unique.has(key)) unique.set(key, d);
  }
  const list = Array.from(unique.values());
  if (list.length === 0) return null;

  return (
    <div style={{ marginTop: 8 }}>
      <button
        className="btn btn-ghost"
        style={{ fontSize: 12.5, padding: "6px 12px" }}
        onClick={() => setOpen((v) => !v)}
      >
        📄 Source Evidence ({list.length}) {open ? "▲" : "▼"}
      </button>
      {open && (
        <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 14 }}>
          {list.map((d, i) => (
            <div key={i} className="glass-panel" style={{ padding: 14 }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 8, color: "var(--indigo-600)" }}>
                Page {d.page} — [{d.lang === "arabic" ? "Arabic" : "English"}] {d.doc_name}
              </div>
              <PdfPage doc={d} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}