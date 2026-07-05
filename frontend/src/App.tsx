import { useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { useAuthStore } from "./store/authStore";
import Login from "./pages/Login";
import ChatPage from "./pages/ChatPage";
import AdminPage from "./pages/AdminPage";

function Splash() {
  return (
    <div style={{ display: "grid", placeItems: "center", height: "100vh" }}>
      <div className="horizon-rule" style={{ width: 160 }} />
    </div>
  );
}

function RequireAuth({ children }: { children: JSX.Element }) {
  const { employee, loading } = useAuthStore();
  if (loading) return <Splash />;
  if (!employee) return <Navigate to="/login" replace />;
  // Original app.py let admins use the chat too — they just also get an
  // "Admin Portal" link in the sidebar (see Sidebar.tsx). No redirect here.
  return children;
}

function RequireAdmin({ children }: { children: JSX.Element }) {
  const { employee, loading } = useAuthStore();
  if (loading) return <Splash />;
  if (!employee) return <Navigate to="/login" replace />;
  if (!employee.is_admin) return <Navigate to="/" replace />;
  return children;
}

export default function App() {
  const init = useAuthStore((s) => s.init);
  const loading = useAuthStore((s) => s.loading);

  useEffect(() => {
    init();
  }, [init]);

  if (loading) return <Splash />;

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            <RequireAuth>
              <ChatPage />
            </RequireAuth>
          }
        />
        <Route
          path="/admin"
          element={
            <RequireAdmin>
              <AdminPage />
            </RequireAdmin>
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
