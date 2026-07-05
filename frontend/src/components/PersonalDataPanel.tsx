import { useState } from "react";

export default function PersonalDataPanel({ data }: { data: string }) {
  const [open, setOpen] = useState(false);
  if (!data) return null;

  return (
    <div style={{ marginTop: 8 }}>
      <button
        className="btn btn-ghost"
        style={{ fontSize: 12.5, padding: "6px 12px" }}
        onClick={() => setOpen((v) => !v)}
      >
        📋 Your data used to answer this {open ? "▲" : "▼"}
      </button>
      {open && (
        <pre
          className="glass-panel"
          style={{
            marginTop: 10,
            padding: 14,
            fontSize: 12.5,
            fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
            color: "var(--text-mid)",
            whiteSpace: "pre-wrap",
            overflowX: "auto",
          }}
        >
          {data}
        </pre>
      )}
    </div>
  );
}
