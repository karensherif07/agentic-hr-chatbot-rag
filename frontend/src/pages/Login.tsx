import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthStore } from "../store/authStore";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const { login, error } = useAuthStore();
  const navigate = useNavigate();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    const ok = await login(email, password);
    setSubmitting(false);
    if (ok) navigate("/");
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: 24,
      }}
    >
      <div className="glass-panel" style={{ width: "100%", maxWidth: 420, padding: "40px 36px" }}>
        <div style={{ marginBottom: 6, color: "var(--brass-300)", fontSize: 12, letterSpacing: "0.14em", textTransform: "uppercase", fontWeight: 600 }}>
          Horizon Tech
        </div>
        <h1
          style={{
            fontFamily: "var(--font-display)",
            fontSize: "2rem",
            margin: "0 0 18px",
            fontWeight: 500,
            color: "var(--text-hi)",
          }}
        >
          HR Assistant
        </h1>
        <div className="horizon-rule" style={{ marginBottom: 28 }} />

        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <label style={{ fontSize: 13, color: "var(--text-mid)" }}>
            Work email
            <input
              className="input"
              style={{ marginTop: 6 }}
              type="email"
              required
              autoFocus
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@horizontech.com"
            />
          </label>
          <label style={{ fontSize: 13, color: "var(--text-mid)" }}>
            Password
            <input
              className="input"
              style={{ marginTop: 6 }}
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </label>

          {error && (
            <div style={{ color: "var(--danger)", fontSize: 13.5 }} role="alert">
              {error}
            </div>
          )}

          <button className="btn btn-primary" type="submit" disabled={submitting} style={{ marginTop: 8, width: "100%" }}>
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
