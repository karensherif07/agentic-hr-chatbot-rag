import type { ChatMessage } from "../api/types";

export default function ChatBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";
  const isRtl = msg.is_arabic && !msg.is_franco;

  return (
    <div style={{ display: "flex", justifyContent: isUser ? "flex-end" : "flex-start", margin: "6px 0" }}>
      <div
        className={isRtl ? "rtl" : "ltr"}
        style={{
          maxWidth: "78%",
          padding: "12px 16px",
          borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
          background: isUser ? "var(--indigo-500)" : "var(--surface)",
          color: isUser ? "#ffffff" : "var(--text-hi)",
          border: isUser ? "none" : "1px solid var(--border)",
          boxShadow: isUser ? "var(--shadow-lift)" : "var(--shadow-soft)",
          lineHeight: 1.75,
          fontSize: "0.96rem",
          whiteSpace: "pre-wrap",
        }}
      >
        {msg.content}
      </div>
    </div>
  );
}