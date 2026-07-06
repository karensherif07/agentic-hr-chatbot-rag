import { useRef, useState } from "react";
import { BASE_URL, getToken } from "../api/client";

type Mode = "mic" | "upload" | null;

export default function VoiceRecorder({
  onTranscript,
}: {
  onTranscript: (text: string) => void;
}) {
  const [mode, setMode] = useState<Mode>(null);
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function toggleMode(next: Mode) {
    setError(null);
    setMode((current) => (current === next ? null : next));
  }

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await sendForTranscription(blob, "recording.webm");
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch {
      setError("Microphone access denied or unavailable.");
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop();
    setRecording(false);
  }

  function handleFilePicked(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    sendForTranscription(file, file.name);
    e.target.value = "";
  }

  async function sendForTranscription(blob: Blob, filename: string) {
    setTranscribing(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", blob, filename);
      const token = getToken();
      const res = await fetch(`${BASE_URL}/api/voice/transcribe`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : undefined,
        body: form,
      });
      if (!res.ok) throw new Error();
      const data = await res.json();
      onTranscript(data.transcript);
      setMode(null);
    } catch {
      setError("Could not transcribe — please try again.");
    } finally {
      setTranscribing(false);
    }
  }

  const iconBtnStyle = (active: boolean) => ({
    width: 36,
    height: 36,
    padding: 0,
    borderRadius: 8,
    fontSize: 14.5,
    background: active ? "rgba(91, 95, 227,0.18)" : "var(--surface-alt)",
    border: `1px solid ${active ? "rgba(91, 95, 227,0.45)" : "var(--line-strong)"}`,
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <button
          type="button"
          title="Record voice"
          className="btn"
          style={iconBtnStyle(mode === "mic")}
          onClick={() => toggleMode("mic")}
          disabled={transcribing}
        >
          🎙️
        </button>
        <button
          type="button"
          title="Upload audio"
          className="btn"
          style={iconBtnStyle(mode === "upload")}
          onClick={() => toggleMode("upload")}
          disabled={transcribing}
        >
          📎
        </button>
        {transcribing && <span style={{ fontSize: 12, color: "var(--text-lo)" }}>Transcribing…</span>}
        {error && <span style={{ fontSize: 11.5, color: "var(--danger)" }}>{error}</span>}
      </div>

      {mode === "mic" && (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <button
            type="button"
            className="btn"
            onClick={recording ? stopRecording : startRecording}
            disabled={transcribing}
            style={{
              fontSize: 12.5,
              padding: "6px 14px",
              background: recording ? "rgba(224, 86, 122,0.2)" : "var(--border)",
              border: `1px solid ${recording ? "rgba(224, 86, 122,0.5)" : "var(--line-strong)"}`,
            }}
          >
            {recording ? "⏹ Stop recording" : "● Start recording"}
          </button>
        </div>
      )}

      {mode === "upload" && (
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.m4a,audio/*"
            onChange={handleFilePicked}
            disabled={transcribing}
            style={{ fontSize: 12.5, color: "var(--text-mid)" }}
          />
        </div>
      )}
    </div>
  );
}