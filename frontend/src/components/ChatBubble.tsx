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
          background: isUser ? "rgba(201, 162, 39, 0.14)" : "rgba(255,255,255,0.055)",
          border: `1px solid ${isUser ? "rgba(201,162,39,0.28)" : "var(--line)"}`,
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
