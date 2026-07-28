import { useEffect, useRef, useState } from "react";
import Sidebar from "../components/Sidebar";
import ChatBubble from "../components/ChatBubble";
import SourceEvidence from "../components/SourceEvidence";
import PersonalDataPanel from "../components/PersonalDataPanel";
import EscalationBanner from "../components/EscalationBanner";
import VoiceRecorder from "../components/VoiceRecorder";
import { api, BASE_URL, getToken, setToken, clearToken } from "../api/client";
import { exportChatAsPdf } from "../utils/exportPdf";
import { useAuthStore } from "../store/authStore";
import type { ChatMessage, ChatResponse } from "../api/types";

interface TurnMeta {
  cited_docs: ChatResponse["cited_docs"];
  personal_data: string;
  no_info: boolean;
  question: string;
  lang: ChatResponse["lang"];
}

// Matches the server's own cap in sessions.py (_MAX_STORED_MESSAGES = 40) —
// no correctness reason to send more than this each turn, since retrieval
// only ever looks back 2 turns and the summary carries the rest of the
// context. Keeps every request lean regardless of how long the on-screen
// conversation gets.
const MAX_SENT_HISTORY = 40;

export default function ChatPage() {
  const employee = useAuthStore((s) => s.employee);
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [summary, setSummary] = useState("");
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [metaByIndex, setMetaByIndex] = useState<Record<number, TurnMeta>>({});
  const [escalationForIndex, setEscalationForIndex] = useState<number | null>(null);
  const [translated, setTranslated] = useState<{ text: string; lang: string } | null>(null);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.get<{ history: ChatMessage[]; summary: string }>("/api/chat/history").then((res) => {
      setHistory(res.history);
      setSummary(res.summary);
    });
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, thinking]);

  async function sendQuestion(question: string) {
    if (!question.trim() || thinking) return;
    setInput("");
    setThinking(true);
    setTranslated(null);
    setStreamingText("");

    const requestBody = {
      question,
      chat_history: history.slice(-MAX_SENT_HISTORY),
      conversation_summary: summary,
    };

    function finalize(res: ChatResponse) {
      const newTurns = res.chat_history.slice(-2);
      const fullHistory = [...history, ...newTurns];
      setHistory(fullHistory);
      setSummary(res.conversation_summary);
      const assistantIndex = fullHistory.length - 1;
      setMetaByIndex((prev) => ({
        ...prev,
        [assistantIndex]: {
          cited_docs: res.cited_docs,
          personal_data: res.personal_data,
          no_info: res.no_info,
          question,
          lang: res.lang,
        },
      }));
      if (res.no_info) setEscalationForIndex(assistantIndex);
    }

    async function fallbackNonStreaming() {
      try {
        const res = await api.post<ChatResponse>("/api/chat/message", requestBody);
        finalize(res);
      } catch {
        setHistory((h) => [
          ...h,
          { role: "assistant", content: "Sorry, something went wrong. Please try again.", is_arabic: false, is_franco: false },
        ]);
      }
    }

    try {
      const token = getToken();
      const res = await fetch(`${BASE_URL}/api/chat/message/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(requestBody),
      });

      if (res.status === 401) {
        clearToken();
        window.location.href = "/login";
        return;
      }
      const refreshed = res.headers.get("X-Refreshed-Token");
      if (refreshed) setToken(refreshed);
      if (!res.ok || !res.body) {
        await fallbackNonStreaming();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let liveText = "";
      let doneResult: ChatResponse | null = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by a blank line; each starts with "data: ".
        const events = buffer.split("\n\n");
        buffer = events.pop() || ""; // last item may be incomplete, keep for next read

        for (const evt of events) {
          const line = evt.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          let parsed: any;
          try {
            parsed = JSON.parse(jsonStr);
          } catch {
            continue;
          }
          if (parsed.type === "chunk") {
            liveText += parsed.text;
            setStreamingText(liveText);
          } else if (parsed.type === "done") {
            doneResult = parsed as ChatResponse;
          } else if (parsed.type === "error") {
            throw new Error(parsed.detail || "Stream error");
          }
        }
      }

      if (doneResult) {
        // One-time swap: the live-streamed text may still contain raw
        // citation markers; the "done" event's answer is fully cleaned.
        // finalize() adds it to `history`, which replaces the streamingText
        // bubble in the render below.
        finalize(doneResult);
      } else {
        await fallbackNonStreaming();
      }
    } catch {
      await fallbackNonStreaming();
    } finally {
      setStreamingText(null);
      setThinking(false);
    }
  }

  async function clearChat() {
    await api.del("/api/chat/history");
    setHistory([]);
    setSummary("");
    setMetaByIndex({});
    setEscalationForIndex(null);
  }

  const lastAssistantIndex = [...history].map((m, i) => (m.role === "assistant" ? i : -1)).filter((i) => i >= 0).pop();
  const lastMeta = lastAssistantIndex !== undefined ? metaByIndex[lastAssistantIndex] : undefined;
  const lastAnswer = lastAssistantIndex !== undefined ? history[lastAssistantIndex]?.content : undefined;

  async function handleTranslate() {
    if (!lastAnswer || !lastMeta) return;
    const res = await api.post<{ translated: string }>("/api/chat/translate", {
      text: lastAnswer,
      source_lang: lastMeta.lang,
    });
    setTranslated({ text: res.translated, lang: lastMeta.lang === "english" ? "arabic" : "english" });
  }

  async function handleReadAloud() {
    if (!lastAnswer || !lastMeta) return;
    const res = await api.post<{ available: boolean; mime?: string; audio_base64?: string }>("/api/chat/tts", {
      text: lastAnswer,
      lang: lastMeta.lang,
    });
    if (res.available && res.audio_base64) {
      const audio = new Audio(`data:${res.mime};base64,${res.audio_base64}`);
      audio.play();
    }
  }

  return (
    <div style={{ display: "flex", height: "100vh" }}>
      <Sidebar onClearChat={clearChat} onExportPdf={() => exportChatAsPdf(history, employee?.full_name)} />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", maxWidth: 900, margin: "0 auto", width: "100%" }}>
        <div style={{ padding: "20px 28px 0" }}>
          <div className="horizon-rule" />
        </div>

        <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>
          {history.length === 0 && (
            <div className="glass-panel ltr" style={{ padding: 20, lineHeight: 1.8, fontSize: 14.5 }}>
              👋 <strong>Welcome to the Employee Support Chatbot.</strong>
              <br />
              <br />
              I can help with HR policies and employee information. You may ask questions in:
              <br />
              &nbsp;&nbsp;🌐 English
              <br />
              &nbsp;&nbsp;📜 Modern Standard Arabic
              <br />
              &nbsp;&nbsp;🗣️ Egyptian Arabic
              <br />
              &nbsp;&nbsp;💬 Franco-Arabic
              <br />
              <br />
              How can I assist you today?
            </div>
          )}

          {history.map((msg, i) => {
            const meta = metaByIndex[i];
            return (
              <div key={i}>
                <ChatBubble msg={msg} />
                {msg.role === "assistant" && meta && (
                  <div style={{ marginInlineStart: 4, marginBottom: 14 }}>
                    <PersonalDataPanel data={meta.personal_data} />
                    <SourceEvidence docs={meta.cited_docs} />
                    {escalationForIndex === i && (
                      <EscalationBanner question={meta.question} onDismiss={() => setEscalationForIndex(null)} />
                    )}
                  </div>
                )}
              </div>
            );
          })}

          {thinking && (
            <>
              {streamingText ? (
                <ChatBubble
                  msg={{
                    role: "assistant",
                    content: streamingText,
                    is_arabic: /[\u0600-\u06FF]/.test(streamingText),
                    is_franco: false,
                  }}
                />
              ) : (
                <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "8px 4px", color: "var(--text-lo)", fontSize: 13.5 }}>
                  <span className="horizon-rule" style={{ width: 36, height: 2 }} />
                  Thinking…
                </div>
              )}
            </>
          )}
        </div>

        {lastAnswer && !thinking && (
          <div style={{ padding: "0 28px" }}>
            <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
              <button className="btn btn-ghost" style={{ fontSize: 12.5, padding: "6px 14px" }} onClick={handleTranslate}>
                🔄 Translate
              </button>
              <button className="btn btn-ghost" style={{ fontSize: 12.5, padding: "6px 14px" }} onClick={handleReadAloud}>
                🔊 Read
              </button>
            </div>
            {translated && (
              <div className={translated.lang === "arabic" ? "rtl" : "ltr"} style={{ marginBottom: 12, fontSize: 14, color: "var(--text-mid)" }}>
                {translated.text}
              </div>
            )}
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendQuestion(input);
          }}
          style={{ display: "flex", gap: 10, padding: "16px 28px 24px", alignItems: "center" }}
        >
          <VoiceRecorder onTranscript={(t) => sendQuestion(t)} />
          <input
            className="input"
            placeholder="Ask your question…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={thinking}
          />
          <button className="btn btn-primary" type="submit" disabled={thinking || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}