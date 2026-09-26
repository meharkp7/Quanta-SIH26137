import { useState } from "react";
import type { FormEvent } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { Building2, Eye, EyeOff, Lock, Mail, Route, ShieldCheck, ArrowRight } from "lucide-react";
import { supabase } from "@/workspace/supabase";
import { useWorkspace } from "@/workspace/Store";
import { Btn, SectionHead } from "./ui";

// Section 1 — the image's LOGIN card: logo, e-mail + password, remember me,
// forgot password, and the green Sign In button over a night-city backdrop.
export function Login() {
  const store = useWorkspace();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  const signedIn = store.demo || (!!store.session && !!store.company);
  if (signedIn && !store.error) return <Navigate to="/app" replace />;

  async function signIn(event: FormEvent) {
    event.preventDefault();
    setError("");
    setNote("");
    if (!supabase) {
      // No company backend configured → the workspace opens in demo mode so
      // the primary button still does something honest.
      store.enterDemo();
      setNote("Supabase is not configured — opening the demo workspace.");
      window.setTimeout(() => navigate("/app"), 400);
      return;
    }
    setBusy(true);
    try {
      const { error: authError } = await supabase.auth.signInWithPassword({ email, password });
      if (authError) throw authError;
      sessionStorage.removeItem("quanta.demo");
      navigate("/onboarding");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function forgot() {
    setError("");
    setNote("");
    if (!supabase) {
      setError("Password recovery needs Supabase configured (VITE_SUPABASE_URL).");
      return;
    }
    if (!email) {
      setError("Enter your work e-mail first, then press Forgot password.");
      return;
    }
    const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${location.origin}/reset-password`,
    });
    if (resetError) setError(resetError.message);
    else setNote("If that address has an account, a reset link is on its way.");
  }

  return (
    <div className="q-login">
      <main className="q-login-main">
        <SectionHead n={1} title="LOGIN" sub="Access your QUANTA workspace" />

        <div className="q-login-card">
          <div className="q-login-brand">
            <span className="q-logo-mark">
              <Route size={30} strokeWidth={2.6} />
            </span>
            <span className="q-logo-word">QUANTA</span>
            <small>Fleet Routing Intelligence</small>
          </div>

          <form onSubmit={signIn}>
            <label className="q-input">
              <Mail size={16} />
              <input
                type="email"
                aria-label="Work e-mail"
                autoComplete="email"
                placeholder="company@yourfleet.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>

            <label className="q-input">
              <Lock size={16} />
              <input
                type={show ? "text" : "password"}
                aria-label="Password"
                autoComplete="current-password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              <button type="button" className="q-input-eye" aria-label={show ? "Hide password" : "Show password"} onClick={() => setShow((v) => !v)}>
                {show ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </label>

            <div className="q-login-row">
              <label className="q-check">
                <input type="checkbox" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
                Remember me
              </label>
              <button type="button" className="q-link" onClick={() => void forgot()}>Forgot password?</button>
            </div>

            {error && <p className="q-form-error">{error}</p>}
            {note && <p className="q-form-note">{note}</p>}

            <Btn type="submit" full disabled={busy}>
              {busy ? "Signing in…" : "Sign In"}
              <ArrowRight size={16} />
            </Btn>
          </form>

          <p className="q-login-tag">Secure. Scalable. Sustainable.</p>

          {!supabase && (
            <p className="q-login-hint">
              <ShieldCheck size={14} /> Company sign-in activates with Supabase — otherwise Sign In opens the demo workspace.
            </p>
          )}
          {store.error && <p className="q-form-error">{store.error}</p>}
        </div>
      </main>

      {/* Night-city panel from the image: skyline + highway light trails. */}
      <aside className="q-login-aside" aria-hidden="true">
        <svg className="q-city" viewBox="0 0 640 420" preserveAspectRatio="none">
          <defs>
            <linearGradient id="skyGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#07202b" />
              <stop offset="60%" stopColor="#04121b" />
              <stop offset="100%" stopColor="#020a10" />
            </linearGradient>
            <linearGradient id="trail" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#22c55e" stopOpacity="0" />
              <stop offset="50%" stopColor="#4ade80" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect width="640" height="420" fill="url(#skyGlow)" />
          {[
            [30, 210, 60, 210], [95, 160, 46, 260], [150, 120, 70, 300], [230, 185, 54, 235],
            [292, 140, 64, 280], [365, 100, 52, 320], [425, 175, 74, 245], [508, 130, 58, 290], [574, 195, 52, 225],
          ].map(([x, y, w, h], i) => (
            <g key={i}>
              <rect x={x} y={y} width={w} height={h} fill="#071a24" stroke="#0e2c39" />
              {Array.from({ length: Math.floor(h / 26) }).map((_, r) =>
                Array.from({ length: Math.floor(w / 18) }).map((_, c) =>
                  (r * 7 + c * 3 + i) % 4 === 0 ? (
                    <rect key={`${r}-${c}`} x={x + 6 + c * 18} y={y + 8 + r * 26} width={8} height={11} fill={(r + c + i) % 5 === 0 ? "#38e08d" : "#1c4a5e"} opacity={0.8} />
                  ) : null,
                ),
              )}
            </g>
          ))}
          <path d="M0 370 C 140 340, 300 392, 640 352" fill="none" stroke="#0d2635" strokeWidth="26" />
          <path d="M0 372 C 140 344, 300 394, 640 356" fill="none" stroke="#08161f" strokeWidth="12" />
          {[0, 1, 2].map((i) => (
            <path key={i} d="M0 370 C 140 340, 300 392, 640 352" fill="none" stroke="url(#trail)" strokeWidth="3" strokeDasharray="60 40" style={{ strokeDashoffset: `${i * 90}` }} />
          ))}
          <text x="24" y="46" className="q-city-tag">SMARTER ROUTES · GREENER CITIES</text>
        </svg>
        <div className="q-login-power">
          <span className="q-eyebrow"><Building2 size={13} /> POWERING</span>
          <h2>Smarter Fleet Operations<br />in a Cleaner Tomorrow</h2>
          <p>
            <ShieldCheck size={14} /> Company data stays in your workspace · every route is independently validated
          </p>
        </div>
      </aside>
    </div>
  );
}
